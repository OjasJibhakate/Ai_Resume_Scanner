"""Explanations for why a candidate ranked where they did.

Two backends behind one interface:

``TemplateExplainer`` renders the arithmetic and the skill-gap breakdown
directly. It is the default and it always works - no key, no network, no cost,
and no possibility of inventing a qualification the candidate does not have.

``NvidiaExplainer`` turns the same facts into prose via NVIDIA NIM. Crucially it
is given **only the extracted facts**, never the raw resume text. That is the
grounding discipline Paper 4 uses to keep its RAG stage honest: the model's job
is to phrase evidence it has been handed, not to read a document and form its
own opinion. It cannot cite a skill that is not in the matched list, because it
never sees anything else.

Any backend failure - missing key, timeout, rate limit - falls back to the
template rather than raising, so a screening run never dies because an API had
a bad minute.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .config import DEFAULT_WEIGHTS, PROJECT_ROOT, ScoringWeights
from .schemas import GapStatus, JobSpec, RankedCandidate
from .scoring import explain_contributions
from .skillgap import upskill_priorities

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_NVIDIA_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"


def load_env(path: str | Path | None = None) -> None:
    """Load ``KEY=value`` pairs from a .env file into the environment.

    A 12-line loader beats adding python-dotenv for this. Existing environment
    variables always win, so a real deployment can override the file.
    """
    env_path = Path(path) if path else PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


# --------------------------------------------------------------------------
# Template backend
# --------------------------------------------------------------------------


def _bar(value: float, width: int = 20) -> str:
    filled = int(round(max(0.0, min(1.0, value)) * width))
    return "#" * filled + "." * (width - filled)


@dataclass
class TemplateExplainer:
    """Deterministic explanation rendered straight from the score components."""

    weights: ScoringWeights = DEFAULT_WEIGHTS
    name: str = "template"

    @property
    def available(self) -> bool:
        return True

    def facts(self, candidate: RankedCandidate, job: JobSpec) -> dict:
        """Structured evidence bundle - also what the LLM backends are given."""
        scores = candidate.scores
        contributions = explain_contributions(scores, self.weights)
        gap = candidate.gap
        resume = candidate.resume

        return {
            "candidate": candidate.display_name,
            "rank": candidate.rank,
            "job_title": job.title,
            "total_score": round(scores.total, 4),
            "sub_scores": {
                "semantic": round(scores.semantic, 4),
                "skill": round(scores.skill, 4),
                "experience": round(scores.experience, 4),
                "education": round(scores.education, 4),
            },
            "weighted_contributions": {k: round(v, 4) for k, v in contributions.items()},
            "penalty": round(scores.penalty, 4),
            "required_skill_coverage": round(scores.required_coverage, 4),
            "prerequisite_gate": round(scores.gate, 4),
            "experience_years": resume.years_experience,
            "experience_required": job.min_years,
            "degree": resume.degree.label,
            "degree_required": job.min_degree.label,
            "matched_skills": [i.skill for i in gap.matched],
            "transferable_skills": [
                {"skill": i.skill, "via": i.via, "path": list(i.path)}
                for i in gap.transferable
            ],
            "missing_skills": [
                {"skill": i.skill, "required": i.required} for i in gap.missing
            ],
            "gaming_flags": [
                {"signal": s.name, "evidence": s.evidence}
                for s in candidate.gaming.triggered_signals
            ],
        }

    def _evidence_facts(self, candidate: RankedCandidate) -> dict:
        """Per-skill evidence, for the LLM backends to quote rather than invent."""
        grounding = candidate.grounding
        if grounding is None:
            return {}
        return {
            "evidence_for_claimed_skills": [
                {
                    "skill": item.skill,
                    "demonstrated": grounding.get(item.skill).is_grounded,
                    "supporting_sentence": grounding.get(item.skill).evidence,
                }
                for item in candidate.gap.matched
            ],
            "fraction_of_claims_with_evidence": round(grounding.grounded_fraction, 3),
        }

    def explain(self, candidate: RankedCandidate, job: JobSpec) -> str:
        scores = candidate.scores
        contributions = explain_contributions(scores, self.weights)
        gap = candidate.gap
        resume = candidate.resume

        lines = [
            f"Rank #{candidate.rank} - {candidate.display_name} - overall {scores.total:.3f}",
            "",
        ]

        rows = [
            (
                "Semantic fit",
                scores.semantic,
                self.weights.semantic,
                contributions["semantic"],
                "SBERT similarity to the job description",
            ),
            (
                "Skill coverage",
                scores.skill,
                self.weights.skill,
                contributions["skill"],
                f"{len(gap.matched)}/{len(gap.items)} requirements met"
                if gap.items
                else "no skills specified in the job description",
            ),
            (
                "Experience",
                scores.experience,
                self.weights.experience,
                contributions["experience"],
                f"{resume.years_experience:.1f} yrs vs {job.min_years:.1f} required"
                if job.min_years > 0
                else f"{resume.years_experience:.1f} yrs (no minimum stated)",
            ),
            (
                "Education",
                scores.education,
                self.weights.education,
                contributions["education"],
                f"{resume.degree.label} vs {job.min_degree.label} required",
            ),
        ]

        for label, value, weight, contribution, note in rows:
            lines.append(
                f"  {label:<15} {_bar(value)} {value:.3f} x {weight:.2f} = {contribution:.3f}   {note}"
            )

        if scores.gate < 1.0:
            lines.append(
                f"  {'Prerequisite':<15} {'':<20} gate x{scores.gate:.2f}"
                f"          only {scores.required_coverage:.0%} of required skills held"
            )
        if scores.penalty > 0:
            lines.append(f"  {'Anti-gaming':<15} {'':<20} penalty            -{scores.penalty:.3f}")

        lines.append("")

        if gap.matched:
            grounding = candidate.grounding
            if grounding is not None:
                # With evidence available, cite it. "Matched: docker" tells a
                # recruiter nothing they could check; the sentence that
                # demonstrates it is the actual justification.
                lines.append(f"  Matched ({len(gap.matched)}) with evidence:")
                for item in gap.matched:
                    record = grounding.get(item.skill)
                    if record.is_grounded and record.evidence:
                        lines.append(f"    + {item.skill}: {record.evidence[:88]!r}")
                    else:
                        lines.append(f"    ? {item.skill}: claimed, not demonstrated in the work history")
                unevidenced = sum(
                    1 for i in gap.matched if not grounding.get(i.skill).is_grounded
                )
                if unevidenced:
                    lines.append(
                        f"    ({unevidenced} of {len(gap.matched)} matched skills are "
                        f"claims without supporting evidence)"
                    )
            else:
                lines.append(
                    f"  Matched ({len(gap.matched)}): "
                    + ", ".join(i.skill for i in gap.matched)
                )
        if gap.transferable:
            detail = "; ".join(
                f"{i.skill} via {i.via}" for i in gap.transferable
            )
            lines.append(f"  Transferable ({len(gap.transferable)}): {detail}")
        if gap.missing:
            detail = ", ".join(
                f"{i.skill}{' (required)' if i.required else ''}" for i in gap.missing
            )
            lines.append(f"  Missing ({len(gap.missing)}): {detail}")

        if candidate.gaming.flagged:
            lines.append("")
            lines.append("  INTEGRITY FLAGS:")
            for signal in candidate.gaming.triggered_signals:
                lines.append(f"    - {signal.name}: {signal.evidence}")

        priorities = upskill_priorities(gap, limit=3)
        if priorities:
            lines.append("")
            focus = ", ".join(i.skill for i in priorities)
            lines.append(f"  Interview focus: {focus}")

        return "\n".join(lines)


# --------------------------------------------------------------------------
# LLM backends
# --------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a recruiting assistant writing a short rationale for \
why a candidate received their ranking.

You will be given ONLY a structured JSON summary of a scoring result. Rules:
- Use only facts present in the JSON. Never invent skills, employers, or dates.
- If a skill is not in matched_skills, the candidate does not have it.
- Write 2-4 sentences of plain prose. No bullet points, no headings, no preamble.
- Lead with the overall verdict, then the single strongest reason, then the main gap.
- If gaming_flags is non-empty, state the integrity concern plainly in one sentence.
- If evidence_for_claimed_skills is present, prefer skills marked demonstrated=true
  and quote or paraphrase their supporting_sentence. If a required skill is claimed
  but demonstrated=false, say it is asserted without supporting experience.
- Be factual and neutral. Do not address the candidate directly."""


@dataclass
class NvidiaExplainer:
    """NVIDIA NIM backend (OpenAI-compatible chat completions)."""

    model: str = ""
    api_key: str = ""
    base_url: str = NVIDIA_BASE_URL
    timeout: float = 60.0
    #: Reasoning traces roughly 20x the latency for no gain on a task this
    #: constrained, so thinking is off unless explicitly requested.
    enable_thinking: bool = False
    max_tokens: int = 400
    temperature: float = 0.2
    name: str = "nvidia"

    def __post_init__(self) -> None:
        load_env()
        self.api_key = self.api_key or os.environ.get("NVIDIA_API_KEY", "")
        self.model = self.model or os.environ.get("NVIDIA_MODEL", DEFAULT_NVIDIA_MODEL)
        self._fallback = TemplateExplainer()

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _post(self, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def complete(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        data = self._post(payload)
        return (data["choices"][0]["message"].get("content") or "").strip()

    def explain(self, candidate: RankedCandidate, job: JobSpec) -> str:
        if not self.available:
            return self._fallback.explain(candidate, job)
        facts = self._fallback.facts(candidate, job)
        facts.update(self._fallback._evidence_facts(candidate))
        try:
            prose = self.complete(_SYSTEM_PROMPT, json.dumps(facts, indent=2))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, OSError):
            # Never let an API hiccup take down a screening run.
            return self._fallback.explain(candidate, job)
        return prose or self._fallback.explain(candidate, job)


@dataclass
class ClaudeExplainer:
    """Anthropic backend, active only when ANTHROPIC_API_KEY is set."""

    model: str = "claude-sonnet-5"
    api_key: str = ""
    max_tokens: int = 400
    name: str = "claude"

    def __post_init__(self) -> None:
        load_env()
        self.api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._fallback = TemplateExplainer()

    @property
    def available(self) -> bool:
        if not self.api_key:
            return False
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return True

    def explain(self, candidate: RankedCandidate, job: JobSpec) -> str:
        if not self.available:
            return self._fallback.explain(candidate, job)
        facts = self._fallback.facts(candidate, job)
        facts.update(self._fallback._evidence_facts(candidate))
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.api_key)
            message = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": json.dumps(facts, indent=2)}],
            )
            return "".join(
                block.text for block in message.content if block.type == "text"
            ).strip()
        except Exception:
            return self._fallback.explain(candidate, job)


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

_BACKENDS = {
    "template": TemplateExplainer,
    "nvidia": NvidiaExplainer,
    "claude": ClaudeExplainer,
}


def available_backends() -> dict[str, bool]:
    """Which backends can actually run right now, for the UI to grey out."""
    load_env()
    status = {}
    for name, factory in _BACKENDS.items():
        try:
            status[name] = factory().available
        except Exception:
            status[name] = False
    return status


def get_explainer(name: str = "auto", **kwargs):
    """Return an explainer by name.

    ``auto`` prefers a configured LLM backend and silently settles for the
    template when none is usable, so calling code never has to branch.
    """
    load_env()
    if name == "auto":
        for candidate in ("nvidia", "claude"):
            backend = _BACKENDS[candidate](**kwargs)
            if backend.available:
                return backend
        return TemplateExplainer()

    factory = _BACKENDS.get(name)
    if factory is None:
        raise ValueError(f"unknown explainer {name!r}; choose from {sorted(_BACKENDS)}")
    return factory(**kwargs)
