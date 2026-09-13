"""Generate LLM-tailored adversarial resumes (attacks A4 and A5).

This is the modern threat the reviewed literature does not model. Attacks A1-A3
in ``generate_corpus`` are lexical: a keyword block, a repeated term, a pasted
job description in white text. All three leave artefacts a detector can find.

Asking a language model to rewrite a resume for a specific posting leaves none.
The output is fluent, unrepetitive, free of verbatim copying, and *genuinely*
closer to the job description in embedding space - so a dense retriever does not
merely fail to punish it, it actively rewards it. Resume-tailoring pipelines of
exactly this kind are published and widely deployed (ResumeFlow, SIGIR 2024),
so this is a realistic attacker, not a hypothetical one.

Two variants, and the distinction between them is the whole point:

``llm_tailored`` (A4)
    Rewrites only the SKILLS and SUMMARY sections to mirror the job
    description. The work history is left byte-identical. This is what a
    candidate does in thirty seconds with a chat window. The new claims have no
    supporting evidence, because the evidence section was not touched.

``llm_fabricated`` (A5)
    Also rewrites the experience bullets so they narrate the newly claimed
    skills. This fabricates supporting evidence, and is the honest hard case
    for evidence-grounding: if you invent the whole work history, grounding has
    nothing left to check against.

The expected result - which the benchmark tests rather than assumes - is that
grounding resists A4 and degrades on A5. That would mean grounding does not
eliminate the attack but raises its cost from "paste a keyword list" to
"fabricate a coherent employment history", which is a meaningful and defensible
security claim.

Usage::

    python benchmark/generate_adversarial.py --corpus benchmark/corpus
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generate_corpus import (  # noqa: E402
    ROLE_BY_KEY,
    TIER_PLAN,
    ResumeMeta,
    build_resume_text,
    write_pdf,
)

from talentlens.explain import NVIDIA_BASE_URL, load_env  # noqa: E402

LLM_ATTACKS = ("llm_tailored", "llm_fabricated")

_TAILOR_PROMPT = """You are helping a job applicant optimise their resume for an \
applicant tracking system.

Rewrite ONLY the SUMMARY and SKILLS sections so they align closely with the job \
description below. You may add any skill named in the job description to the \
SKILLS list.

Hard constraints:
- Reproduce the EXPERIENCE and EDUCATION sections EXACTLY as given, character for
  character. Do not add, remove or reword a single bullet.
- Keep the same section headings, in the same order, in the same plain-text layout.
- Do not invent employers, job titles, dates or degrees.
- Output the complete resume as plain text. No commentary, no markdown fences."""

_FABRICATE_PROMPT = """You are helping a job applicant optimise their resume for an \
applicant tracking system.

Rewrite the resume so it presents the candidate as a strong match for the job \
description below. Update the SUMMARY and SKILLS sections, and also rewrite the \
EXPERIENCE bullet points so that they describe hands-on work with the \
technologies the job description asks for.

Hard constraints:
- Keep the same employers, job titles and employment dates. Do not change them.
- Keep the EDUCATION section exactly as given.
- Keep the same section headings, in the same order, in the same plain-text layout.
- Output the complete resume as plain text. No commentary, no markdown fences."""

PROMPTS = {"llm_tailored": _TAILOR_PROMPT, "llm_fabricated": _FABRICATE_PROMPT}

#: Minimum plausible length of a rewritten resume, as a fraction of the
#: original. Guards against a truncated or refused generation being written to
#: the corpus as though it were a successful attack.
_MIN_LENGTH_RATIO = 0.55
_REQUIRED_SECTIONS = ("EXPERIENCE", "EDUCATION", "SKILLS")


class GenerationError(RuntimeError):
    pass


def complete(
    system: str,
    user: str,
    api_key: str,
    model: str,
    timeout: float = 180.0,
    max_tokens: int = 2000,
    temperature: float = 0.7,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        # Reasoning traces add ~20x latency for no benefit on a rewriting task.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{NVIDIA_BASE_URL}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.load(response)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        raise GenerationError(str(exc)) from exc
    return (data["choices"][0]["message"].get("content") or "").strip()


def _strip_fences(text: str) -> str:
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
    return "\n".join(lines).strip()


def _split_sections(text: str) -> dict[str, list[str]]:
    """Split a plain-text resume on its ALL-CAPS headings."""
    sections: dict[str, list[str]] = {"_header": []}
    current = "_header"
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and stripped.isupper() and len(stripped.split()) <= 3:
            current = stripped
            sections[current] = []
        else:
            sections[current].append(line)
    return sections


def enforce_evidence_preserved(generated: str, original: str) -> str:
    """Splice the ORIGINAL work history back into a tailored resume.

    A4 is supposed to isolate a single variable: claims are inflated while the
    evidence base is held constant. The model does not reliably honour that
    instruction - in testing it silently dropped an experience bullet - and a
    variant that also *removes* evidence would let grounding look effective for
    the wrong reason. So the constraint is enforced here rather than requested
    in the prompt: keep the model's rewritten summary and skills, restore the
    original EXPERIENCE and EDUCATION verbatim.
    """
    gen = _split_sections(generated)
    orig = _split_sections(original)

    rebuilt: list[str] = []
    for heading, body in gen.items():
        source = orig.get(heading, body) if heading in ("EXPERIENCE", "EDUCATION") else body
        if heading != "_header":
            rebuilt.append(heading)
        rebuilt.extend(source)

    # If the model omitted a section entirely, append the original so the
    # attacked resume is never missing evidence the honest one had.
    for heading in ("EXPERIENCE", "EDUCATION"):
        if heading not in gen and heading in orig:
            rebuilt.append(heading)
            rebuilt.extend(orig[heading])

    return "\n".join(rebuilt).strip()


def validate(generated: str, original: str) -> str:
    """Reject generations that are truncated, refused, or structurally broken.

    Without this, a refusal or a cut-off response would be written into the
    corpus as an "attack" and quietly weaken every downstream result.
    """
    cleaned = _strip_fences(generated)
    if len(cleaned) < len(original) * _MIN_LENGTH_RATIO:
        raise GenerationError(
            f"generation too short ({len(cleaned)} vs {len(original)} chars)"
        )
    upper = cleaned.upper()
    missing = [s for s in _REQUIRED_SECTIONS if s not in upper]
    if missing:
        raise GenerationError(f"generation missing sections: {missing}")
    return cleaned


def generate_attacks(
    corpus_dir: Path,
    attempts: int = 3,
    seed: int = 11,
    per_role: int = 2,
) -> dict:
    load_env()
    import os

    api_key = os.environ.get("NVIDIA_API_KEY", "")
    model = os.environ.get("NVIDIA_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
    if not api_key:
        raise SystemExit(
            "NVIDIA_API_KEY is not set. Put it in .env at the project root.\n"
            "These attacks require a language model; the lexical attacks in "
            "generate_corpus.py do not."
        )

    payload = json.loads((corpus_dir / "corpus.json").read_text(encoding="utf-8"))
    rng = random.Random(seed)
    resume_dir = corpus_dir / "resumes"

    existing = {m["candidate_id"] for m in payload["resumes"]}
    new_metas: list[dict] = []
    failures: list[str] = []

    for role_key, job in payload["jobs"].items():
        role = ROLE_BY_KEY[role_key]
        job_text = job["text"]

        for index in range(per_role):
            # Same base resume for both attacks, so A4 and A5 differ only by
            # whether the work history was fabricated.
            base_text, base_meta = build_resume_text(role, 1, rng, 80 + index)

            # Matched-pair control: the same resume, unmanipulated.
            control_id = f"{base_meta.candidate_id}_control"
            if control_id not in existing:
                write_pdf(resume_dir / f"{control_id}.pdf", base_text)
                from dataclasses import replace as _replace, asdict as _asdict

                new_metas.append(_asdict(_replace(base_meta, candidate_id=control_id)))

            for attack in LLM_ATTACKS:
                candidate_id = f"{base_meta.candidate_id}_{attack}"
                if candidate_id in existing:
                    continue

                user_message = (
                    f"JOB DESCRIPTION\n{job_text}\n\n"
                    f"CURRENT RESUME\n{base_text}"
                )
                generated = None
                for attempt in range(attempts):
                    try:
                        raw = complete(PROMPTS[attack], user_message, api_key, model)
                        generated = validate(raw, base_text)
                        if attack == "llm_tailored":
                            # A4 must inflate claims without touching evidence.
                            generated = enforce_evidence_preserved(generated, base_text)
                        break
                    except GenerationError as exc:
                        print(f"  attempt {attempt + 1} failed for {candidate_id}: {exc}")
                        time.sleep(2)

                if generated is None:
                    failures.append(candidate_id)
                    continue

                write_pdf(resume_dir / f"{candidate_id}.pdf", generated)
                meta = ResumeMeta(
                    candidate_id=candidate_id,
                    role=role_key,
                    tier=base_meta.tier,
                    name=base_meta.name,
                    years=base_meta.years,
                    degree=base_meta.degree,
                    skills=base_meta.skills,
                    narrated_skills=base_meta.narrated_skills,
                    listed_only_skills=base_meta.listed_only_skills,
                    stuffed=True,
                    attack=attack,
                    stuffing_techniques=[attack],
                    control_id=control_id,
                )
                from dataclasses import asdict

                new_metas.append(asdict(meta))
                print(f"  generated {candidate_id} ({len(generated)} chars)")

    payload["resumes"].extend(new_metas)

    # Relevance labels are unchanged by the attack: rewriting a resume does not
    # change what the candidate can actually do. That invariant is the entire
    # basis for measuring whether an attack "worked".
    for job_key, row in payload["labels"].items():
        for meta in new_metas:
            row[meta["candidate_id"]] = meta["tier"] if meta["role"] == job_key else 0

    (corpus_dir / "corpus.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"generated": len(new_metas), "failed": failures}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="benchmark/corpus", type=Path)
    parser.add_argument("--per-role", default=2, type=int)
    parser.add_argument("--seed", default=11, type=int)
    args = parser.parse_args()

    if not (args.corpus / "corpus.json").exists():
        raise SystemExit(
            f"no corpus at {args.corpus}. Run: python benchmark/generate_corpus.py"
        )

    result = generate_attacks(args.corpus, seed=args.seed, per_role=args.per_role)
    print(f"\ngenerated {result['generated']} LLM-tailored resumes")
    if result["failed"]:
        print(f"failed: {result['failed']}")


if __name__ == "__main__":
    main()
