"""Adversarial robustness study: does evidence-grounding resist attacks it never saw?

This is the experiment the contribution stands or falls on.

**Design.** Matched pairs. Every attacked resume has an untouched control that is
the same candidate with the same true relevance, differing only by the
manipulation. Both sit in the same candidate pool and are ranked together, so
the difference in their rank percentiles isolates the effect of the attack from
every property of the candidate.

    attack_gain = percentile(control) - percentile(attacked)

Percentile 0.0 is the top of the list, so a **positive** gain means the attack
promoted the candidate and therefore worked. Zero means the manipulation bought
nothing. Negative means it backfired.

**Systems.** Two of the configurations isolate our claim:

``hybrid_grounded_only``
    Evidence-grounded scoring with the anti-gaming detector switched **off**.
    If grounding is genuinely structural rather than a disguised detector, this
    configuration must resist attacks without any attack-specific rule firing.

``hybrid_detector_only``
    Plain coverage scoring with the detector on - the prior-art defence.

**Hypothesis.** The detector resists the lexical attacks it was designed around
(A1-A3) and fails on LLM-tailored text (A4), which leaves no lexical artefact.
Grounding resists A1-A4 without knowing anything about any of them, and degrades
on A5, where the work history itself is fabricated.

A5 failing is a real result, not a caveat to bury: it locates the exact boundary
of what evidence-grounding can do.

Usage::

    python benchmark/generate_corpus.py
    python benchmark/generate_adversarial.py
    python benchmark/run_attack_study.py
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from talentlens.config import PrerequisiteGate  # noqa: E402
from talentlens.extraction import build_job_spec  # noqa: E402
from talentlens.pipeline import load_resumes  # noqa: E402
from talentlens.ranking import Bm25Ranker, HybridRanker, SbertRanker, TfidfRanker  # noqa: E402
from talentlens.schemas import DegreeLevel  # noqa: E402

DEGREE_LOOKUP = {
    "None": DegreeLevel.NONE,
    "Associate": DegreeLevel.ASSOCIATE,
    "Bachelor": DegreeLevel.BACHELOR,
    "Master": DegreeLevel.MASTER,
    "Doctorate": DegreeLevel.DOCTORATE,
}

ATTACK_ORDER = (
    "keyword_block",
    "repetition",
    "hidden_text",
    "llm_tailored",
    "llm_fabricated",
)

ATTACK_LABELS = {
    "keyword_block": "A1 keyword block",
    "repetition": "A2 repetition",
    "hidden_text": "A3 hidden text",
    "llm_tailored": "A4 LLM tailored",
    "llm_fabricated": "A5 LLM fabricated",
}

OFF = PrerequisiteGate(enabled=False)

#: Resamples for bootstrap confidence intervals. With 15 matched pairs per
#: attack a normal approximation is not safe, and the bootstrap makes no
#: distributional assumption.
BOOTSTRAP_RESAMPLES = 10000


def bootstrap_ci(
    values: list[float], alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean."""
    import numpy as np

    if not values:
        return (float("nan"), float("nan"))
    if len(values) == 1:
        return (values[0], values[0])
    rng = np.random.default_rng(seed)
    sample = np.asarray(values, dtype=float)
    draws = rng.choice(sample, size=(BOOTSTRAP_RESAMPLES, len(sample)), replace=True)
    means = draws.mean(axis=1)
    return (
        float(np.percentile(means, 100 * alpha / 2)),
        float(np.percentile(means, 100 * (1 - alpha / 2))),
    )


def paired_difference(
    treatment: dict[str, float], baseline: dict[str, float], seed: int = 0
) -> dict[str, float]:
    """Paired comparison of two systems over the same attack/control pairs.

    Pairing matters: the same 15 attacked resumes are scored by both systems, so
    comparing them pairwise removes between-resume variance that an unpaired
    test would leave in. Reports the mean difference, a bootstrap CI, and a
    two-sided permutation p-value on the sign of the differences.
    """
    import numpy as np

    shared = sorted(set(treatment) & set(baseline))
    if not shared:
        return {"n": 0, "mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "p": float("nan")}

    diffs = np.array([treatment[k] - baseline[k] for k in shared], dtype=float)
    lo, hi = bootstrap_ci(list(diffs), seed=seed)

    # Sign-flip permutation test: under the null the manipulation has no
    # systematic effect, so flipping the sign of any difference is equally likely.
    rng = np.random.default_rng(seed)
    observed = abs(diffs.mean())
    signs = rng.choice([-1.0, 1.0], size=(BOOTSTRAP_RESAMPLES, len(diffs)))
    null = np.abs((signs * diffs).mean(axis=1))
    p = float((null >= observed - 1e-12).mean())

    return {"n": len(shared), "mean": float(diffs.mean()), "lo": lo, "hi": hi, "p": p}


def build_systems() -> dict[str, object]:
    """Systems under test.

    The gate is disabled throughout. It damps candidates who lack required
    skills, and every attack works by *claiming* those skills - so leaving it on
    would let the gate absorb some of the attack and confound the comparison
    between the detector and grounding.
    """
    return {
        "tfidf": TfidfRanker(),
        "bm25": Bm25Ranker(),
        "sbert": SbertRanker(),
        "hybrid_plain": HybridRanker(
            skill_method="coverage", use_antigaming=False, gate=OFF
        ),
        "hybrid_detector_only": HybridRanker(
            skill_method="coverage", use_antigaming=True, gate=OFF
        ),
        "hybrid_grounded_only": HybridRanker(
            skill_method="grounded", use_antigaming=False, gate=OFF
        ),
        "hybrid_full": HybridRanker(
            skill_method="grounded", use_antigaming=True, gate=OFF
        ),
    }


def run(corpus_dir: Path, out_dir: Path) -> dict:
    payload = json.loads((corpus_dir / "corpus.json").read_text(encoding="utf-8"))
    meta_by_id = {m["candidate_id"]: m for m in payload["resumes"]}

    pairs = [
        (m["candidate_id"], m["control_id"], m["attack"], m["role"])
        for m in payload["resumes"]
        if m.get("control_id") and m["attack"] != "none"
    ]
    if not pairs:
        raise SystemExit(
            "no matched attack pairs in the corpus. Regenerate it:\n"
            "  python benchmark/generate_corpus.py\n"
            "  python benchmark/generate_adversarial.py"
        )

    resume_paths = sorted((corpus_dir / "resumes").glob("*.pdf"))
    print(f"parsing {len(resume_paths)} resumes ...")
    resumes, failures = load_resumes(resume_paths)
    for path, error in failures:
        print(f"  FAILED {path}: {error}")
    print(f"  parsed {len(resumes)}; {len(pairs)} matched attack pairs")

    jobs = {
        key: build_job_spec(
            spec["text"],
            title=spec["title"],
            min_years=float(spec["min_years"]),
            min_degree=DEGREE_LOOKUP[spec["min_degree"]],
        )
        for key, spec in payload["jobs"].items()
    }

    from talentlens.embedding import get_embedder

    print("warming embedding cache ...")
    started = time.perf_counter()
    embedder = get_embedder()
    embedder.encode_documents([r.text for r in resumes])
    embedder.encode_documents([j.text for j in jobs.values()])
    from talentlens.grounding import narrative_sentences

    every_sentence = [s for r in resumes for s in narrative_sentences(r)]
    embedder.encode(every_sentence)
    embedder.flush()
    print(f"  warmed in {time.perf_counter() - started:.1f}s")

    systems = build_systems()
    # system -> attack -> {pair_id: gain}. Keyed by pair so the comparison
    # between two systems is paired over identical resumes.
    gains: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    detector_flags: dict[str, list[int]] = defaultdict(list)
    grounded_fraction: dict[str, list[float]] = defaultdict(list)

    for name, system in systems.items():
        print(f"running {name} ...")
        for job_key, job in jobs.items():
            relevant_pairs = [p for p in pairs if p[3] == job_key]
            if not relevant_pairs:
                continue

            if isinstance(system, HybridRanker):
                detailed = system.rank_detailed(resumes, job)
                ranking = [(c.candidate_id, c.scores.total) for c in detailed]
                if name == "hybrid_full":
                    for candidate in detailed:
                        info = meta_by_id.get(candidate.candidate_id)
                        if info is None:
                            continue
                        detector_flags[info["attack"]].append(
                            1 if candidate.gaming.flagged else 0
                        )
                        if candidate.grounding is not None:
                            grounded_fraction[info["attack"]].append(
                                candidate.grounding.grounded_fraction
                            )
            else:
                ranking = system.rank(resumes, job)

            position = {cid: i for i, (cid, _) in enumerate(ranking)}
            size = max(1, len(ranking))

            for attacked_id, control_id, attack, _ in relevant_pairs:
                if attacked_id not in position or control_id not in position:
                    continue
                gain = (position[control_id] - position[attacked_id]) / size
                gains[name][attack][attacked_id] = gain

    rows = []
    for name in systems:
        row: dict[str, object] = {"system": name}
        per_attack = []
        for attack in ATTACK_ORDER:
            values = list(gains[name].get(attack, {}).values())
            mean = statistics.mean(values) if values else float("nan")
            lo, hi = bootstrap_ci(values, seed=17)
            row[attack] = mean
            row[f"{attack}_lo"] = lo
            row[f"{attack}_hi"] = hi
            row[f"{attack}_n"] = len(values)
            if values:
                per_attack.extend(values)
        row["mean_all"] = statistics.mean(per_attack) if per_attack else float("nan")
        lo, hi = bootstrap_ci(per_attack, seed=17)
        row["mean_all_lo"] = lo
        row["mean_all_hi"] = hi
        rows.append(row)

    # Paired tests against the undefended multi-factor ranker, which is the
    # configuration whose vulnerability the defences are supposed to fix.
    comparisons = []
    baseline_name = "hybrid_plain"
    for name in systems:
        if name == baseline_name or not name.startswith("hybrid"):
            continue
        for attack in ATTACK_ORDER:
            stats = paired_difference(
                gains[name].get(attack, {}), gains[baseline_name].get(attack, {}), seed=23
            )
            stats.update({"system": name, "attack": attack, "baseline": baseline_name})
            comparisons.append(stats)

    detector_recall = {
        attack: (statistics.mean(flags) if flags else float("nan"))
        for attack, flags in detector_flags.items()
    }
    grounding_by_attack = {
        attack: (statistics.mean(v) if v else float("nan"))
        for attack, v in grounded_fraction.items()
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "attack_study.csv", rows)
    _write_report(
        out_dir / "attack_study.md",
        rows,
        comparisons,
        detector_recall,
        grounding_by_attack,
        len(resumes),
        len(pairs),
    )
    _write_csv(out_dir / "attack_pairwise.csv", comparisons)
    return {
        "rows": rows,
        "detector_recall": detector_recall,
        "grounding": grounding_by_attack,
        "comparisons": comparisons,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value) -> str:
    try:
        if value != value:  # NaN
            return "  -  "
        return f"{value:+.3f}"
    except TypeError:
        return str(value)


def _write_report(
    path: Path,
    rows: list[dict],
    comparisons: list[dict],
    detector_recall: dict,
    grounding: dict,
    n_resumes: int,
    n_pairs: int,
) -> None:
    lines = [
        "# Adversarial robustness study",
        "",
        f"{n_resumes} resumes, {n_pairs} matched attack/control pairs, 5 jobs.",
        "",
        "## Attack gain by system and attack type",
        "",
        "Rank-percentile gained by the attacked resume over its identical control.",
        "**Positive = the attack worked.** 0.000 = the manipulation bought nothing.",
        "",
        "| System | " + " | ".join(ATTACK_LABELS[a] for a in ATTACK_ORDER) + " | Mean |",
        "|---" * (len(ATTACK_ORDER) + 2) + "|",
    ]
    for row in rows:
        cells = " | ".join(_fmt(row.get(a)) for a in ATTACK_ORDER)
        lines.append(f"| {row['system']} | {cells} | {_fmt(row.get('mean_all'))} |")

    lines += [
        "",
        "## Detector recall by attack type",
        "",
        "Share of attacked resumes the five-signal detector flagged.",
        "",
        "| Attack | Detector recall | Mean grounded fraction |",
        "|---|---|---|",
    ]
    for attack in ATTACK_ORDER:
        recall = detector_recall.get(attack, float("nan"))
        ground = grounding.get(attack, float("nan"))
        recall_s = "  -  " if recall != recall else f"{recall:.2f}"
        ground_s = "  -  " if ground != ground else f"{ground:.2f}"
        lines.append(f"| {ATTACK_LABELS[attack]} | {recall_s} | {ground_s} |")

    honest = grounding.get("none", float("nan"))
    if honest == honest:
        lines += ["", f"Honest baseline grounded fraction: **{honest:.2f}**."]

    lines += [
        "",
        "## Reading this",
        "",
        "`hybrid_detector_only` is the prior-art defence: attack-specific signals",
        "bolted onto a ranker. `hybrid_grounded_only` runs with the detector",
        "switched off, so anything it resists, it resists structurally rather than",
        "by recognising the attack.",
        "",
        "The corpus is synthetic and self-labelled, so absolute ranking quality",
        "here means little. The matched-pair design is what carries weight: a",
        "control and its attacked twin are the same candidate, so the difference",
        "between them is attributable to the manipulation and nothing else.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="benchmark/corpus", type=Path)
    parser.add_argument("--out", default="benchmark/results", type=Path)
    args = parser.parse_args()

    result = run(args.corpus, args.out)

    print()
    header = f"{'system':<24}" + "".join(f"{ATTACK_LABELS[a][:14]:>16}" for a in ATTACK_ORDER) + f"{'mean':>9}"
    print(header)
    print("-" * len(header))
    for row in result["rows"]:
        cells = "".join(f"{_fmt(row.get(a)):>16}" for a in ATTACK_ORDER)
        print(f"{row['system']:<24}{cells}{_fmt(row.get('mean_all')):>9}")

    print()
    print("detector recall / grounded fraction by attack:")
    for attack in ATTACK_ORDER:
        r = result["detector_recall"].get(attack, float("nan"))
        g = result["grounding"].get(attack, float("nan"))
        rs = " - " if r != r else f"{r:.2f}"
        gs = " - " if g != g else f"{g:.2f}"
        print(f"  {ATTACK_LABELS[attack]:<20} recall={rs:>5}   grounded={gs:>5}")
    honest = result["grounding"].get("none", float("nan"))
    if honest == honest:
        print(f"  {'honest baseline':<20} {'':>12}   grounded={honest:.2f}")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
