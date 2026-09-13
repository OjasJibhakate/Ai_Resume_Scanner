"""The robustness-fairness tradeoff curve for evidence grounding.

Evidence grounding rewards resumes that name the technologies they used. That is
the whole mechanism - and it is also the whole risk. A candidate who writes
"administered the campus network and resolved escalations" describes the same
work as one who writes "administered the Cisco campus network using BGP", but
only the second is grounded. Naming habits track seniority, first-language
background, and whether someone has had professional CV coaching.

So grounding buys adversarial robustness and spends fairness, and the exchange
rate is set by one parameter: ``GroundingConfig.floor``, the credit a claimed-but
-unevidenced skill still receives. At ``floor = 1.0`` grounding is off and the
system is maximally permissive; at ``floor = 0`` only demonstrated skills count
at all.

This script measures both sides of that exchange on the same axis:

* **Robustness** - attack gain on the LLM-tailored attack (A4), the attack no
  lexical detector sees, measured on matched synthetic pairs.
* **Fairness cost** - the skill-score penalty suffered by real resumes rewritten
  to describe identical work without naming the technologies.

Neither number alone justifies a setting. The curve is the deliverable: it lets
a deployer choose a point knowingly rather than inherit our default by accident.

Usage::

    python benchmark/run_tradeoff.py
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from talentlens import fairness  # noqa: E402
from talentlens.config import GroundingConfig, PrerequisiteGate  # noqa: E402
from talentlens.extraction import build_job_spec, build_resume  # noqa: E402
from talentlens.grounding import ground_skills, grounded_coverage  # noqa: E402
from talentlens.pipeline import build_job, load_resumes  # noqa: E402
from talentlens.ranking import HybridRanker  # noqa: E402
from talentlens.schemas import DegreeLevel, ParsedDocument  # noqa: E402

from run_fairness_audit import JOB_TEXT, load_real  # noqa: E402

OFF = PrerequisiteGate(enabled=False)
FLOORS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
TARGET_ATTACK = "llm_tailored"

DEGREE_LOOKUP = {
    "None": DegreeLevel.NONE,
    "Associate": DegreeLevel.ASSOCIATE,
    "Bachelor": DegreeLevel.BACHELOR,
    "Master": DegreeLevel.MASTER,
    "Doctorate": DegreeLevel.DOCTORATE,
}


def measure_robustness(corpus_dir: Path, floor: float) -> float:
    """Mean attack gain on A4 with grounding at the given floor.

    Positive means the attack still pays. Uses the matched-pair design: the
    attacked resume and its untouched control are the same candidate.
    """
    payload = json.loads((corpus_dir / "corpus.json").read_text(encoding="utf-8"))
    pairs = [
        (m["candidate_id"], m["control_id"], m["role"])
        for m in payload["resumes"]
        if m.get("control_id") and m["attack"] == TARGET_ATTACK
    ]
    if not pairs:
        return float("nan")

    resumes, _ = load_resumes(sorted((corpus_dir / "resumes").glob("*.pdf")))
    jobs = {
        key: build_job_spec(
            spec["text"],
            title=spec["title"],
            min_years=float(spec["min_years"]),
            min_degree=DEGREE_LOOKUP[spec["min_degree"]],
        )
        for key, spec in payload["jobs"].items()
    }

    ranker = HybridRanker(
        skill_method="grounded",
        use_antigaming=False,
        gate=OFF,
        grounding_config=GroundingConfig(floor=floor),
    )
    gains = []
    for job_key, job in jobs.items():
        relevant = [p for p in pairs if p[2] == job_key]
        if not relevant:
            continue
        ranking = ranker.rank(resumes, job)
        position = {cid: i for i, (cid, _) in enumerate(ranking)}
        size = max(1, len(ranking))
        for attacked, control, _ in relevant:
            if attacked in position and control in position:
                gains.append((position[control] - position[attacked]) / size)

    return statistics.mean(gains) if gains else float("nan")


def measure_fairness_cost(real_path: Path, category: str, limit: int, floor: float) -> dict:
    """Skill-score penalty for describing the same work without naming tools."""
    job = build_job(JOB_TEXT, title="Senior Network and Systems Administrator")
    originals = load_real(real_path, category, limit)
    implicits = [
        build_resume(
            ParsedDocument(
                path=f"{r.candidate_id}_i.txt",
                text=fairness.implicit_rewrite(r),
                source_format="txt",
            ),
            candidate_id=f"{r.candidate_id}__implicit",
        )
        for r in originals
    ]

    config = GroundingConfig(floor=floor)
    original_scores, implicit_scores = [], []
    for base, implicit in zip(originals, implicits):
        for resume, sink in ((base, original_scores), (implicit, implicit_scores)):
            report = ground_skills(resume, resume.skills_expanded, config=config)
            sink.append(
                grounded_coverage(
                    resume.skills_expanded,
                    job.required_skills,
                    job.preferred_skills,
                    report=report,
                    config=config,
                )
            )

    base_mean = statistics.mean(original_scores)
    implicit_mean = statistics.mean(implicit_scores)
    penalty = (implicit_mean - base_mean) / base_mean if base_mean else float("nan")
    return {
        "original_skill": base_mean,
        "implicit_skill": implicit_mean,
        "style_penalty": penalty,
    }


def run(corpus_dir: Path, real_path: Path, out_dir: Path, category: str, limit: int) -> list[dict]:
    rows = []
    for floor in FLOORS:
        print(f"floor = {floor:.1f} ...")
        robustness = measure_robustness(corpus_dir, floor)
        cost = measure_fairness_cost(real_path, category, limit, floor)
        rows.append(
            {
                "floor": floor,
                "a4_attack_gain": robustness,
                "original_skill": cost["original_skill"],
                "implicit_skill": cost["implicit_skill"],
                "style_penalty": cost["style_penalty"],
            }
        )
        print(
            f"  attack gain {robustness:+.4f}   style penalty {cost['style_penalty']:+.1%}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "tradeoff.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    _write_report(out_dir / "tradeoff.md", rows, limit)
    return rows


def _write_report(path: Path, rows: list[dict], limit: int) -> None:
    lines = [
        "# Robustness-fairness tradeoff",
        "",
        "`floor` is the credit a claimed-but-unevidenced skill receives.",
        "`floor = 1.0` disables grounding entirely.",
        "",
        "- **A4 attack gain** - rank percentiles an LLM-tailored resume gains over",
        "  its identical control. Lower is better. Measured on matched synthetic pairs.",
        f"- **Style penalty** - skill-score change when {limit} real resumes are rewritten",
        "  to describe the same work without naming the technologies. Nearer zero is fairer.",
        "",
        "| floor | A4 attack gain | Style penalty | Original skill | Implicit skill |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['floor']:.1f} | {row['a4_attack_gain']:+.4f} | "
            f"{row['style_penalty']:+.1%} | {row['original_skill']:.4f} | "
            f"{row['implicit_skill']:.4f} |"
        )
    lines += [
        "",
        "## Reading the curve",
        "",
        "These move in opposite directions, and that is the point: there is no",
        "setting that is simultaneously maximally robust and maximally fair. A",
        "deployer has to choose, and should choose explicitly.",
        "",
        "The fairness cost is real but its *interpretation* needs care. Whether a",
        "penalty for not naming technologies constitutes unfairness depends on",
        "whether that writing habit correlates with protected characteristics.",
        "This corpus carries no demographic labels and cannot answer that. What",
        "is established here is a quantified disparate-treatment risk on a",
        "writing-style axis - enough to require evaluation before deployment, not",
        "enough to conclude discrimination.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="benchmark/corpus", type=Path)
    parser.add_argument("--real", default="data/real/real_resumes.json", type=Path)
    parser.add_argument("--out", default="benchmark/results", type=Path)
    parser.add_argument("--category", default="INFORMATION-TECHNOLOGY")
    parser.add_argument("--limit", default=30, type=int)
    args = parser.parse_args()

    rows = run(args.corpus, args.real, args.out, args.category, args.limit)

    print()
    print(f"{'floor':>7}{'A4 gain':>12}{'style penalty':>16}")
    print("-" * 35)
    for row in rows:
        print(
            f"{row['floor']:>7.1f}{row['a4_attack_gain']:>+12.4f}"
            f"{row['style_penalty']:>+16.1%}"
        )
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
