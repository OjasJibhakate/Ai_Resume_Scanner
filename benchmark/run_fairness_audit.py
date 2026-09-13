"""Fairness audit: name-substitution and writing-style counterfactuals.

Runs on **real resumes** (see ``fetch_real_resumes.py``), because the risks under
test are properties of how real people write, and a synthetic corpus written by
one generator has none of that variation.

Three questions, in the order a practitioner would ask them:

1. **Does the candidate's name move the score?** Names are substituted and
   nothing else changes, so any movement is caused by the name. The pipeline
   never scores the name, but it embeds the whole document, so the name reaches
   the semantic term.

2. **Which component carries the disparity?** Because the scorer decomposes, the
   gap can be attributed to a specific term rather than to "the model". This is
   the step black-box audits cannot perform, and it is what makes the result
   actionable rather than merely alarming.

3. **Does the obvious mitigation work?** Re-run with the identifying header
   stripped before embedding, and measure whether the gap closes. A fairness
   claim that is not re-measured after the fix is not a fairness claim.

A fourth question is specific to this project's method. Evidence grounding
rewards resumes that *narrate* their work, and narrative richness is not evenly
distributed - it tracks writing confidence, first-language background, and access
to CV coaching. The style audit rewrites narratives tersely while preserving the
technologies named, so the facts are constant and only the prose changes. If
grounding drops those candidates, that is a defect in the method, not in them.

Usage::

    python benchmark/fetch_real_resumes.py --limit 500
    python benchmark/run_fairness_audit.py
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
from talentlens.config import PrerequisiteGate  # noqa: E402
from talentlens.extraction import build_resume  # noqa: E402
from talentlens.pipeline import build_job  # noqa: E402
from talentlens.ranking import HybridRanker  # noqa: E402
from talentlens.schemas import ParsedDocument  # noqa: E402

OFF = PrerequisiteGate(enabled=False)

#: The job description has to match the population actually being screened, or
#: the skill term goes inert and the audit measures nothing.
#:
#: The real corpus's INFORMATION-TECHNOLOGY category is dominated by network and
#: infrastructure staff, not software engineers - the most common extracted
#: skills are networking, cisco, firewall, vpn, linux and sql. A software
#: engineering posting produced a mean required-skill coverage of 0.133 across
#: these candidates, leaving the skill component near zero for everyone and the
#: disparity in it trivially 0.0000. This posting is written against the
#: population that is actually present.
JOB_TEXT = """Senior Network and Systems Administrator

We are hiring an experienced administrator to run and secure our network and
server infrastructure.

Requirements:
- 5+ years of professional experience in networking or systems administration
- Strong hands-on networking, routing and switching
- Firewall administration and network security
- Linux server administration
- Bachelor degree or equivalent practical experience

Preferred:
- Cisco equipment and CCNA certification
- VPN and remote access infrastructure
- SQL and database administration
- Monitoring and incident response
"""


def load_real(path: Path, category: str, limit: int) -> list:
    records = json.loads(path.read_text(encoding="utf-8"))
    selected = [r for r in records if r["category"] == category][:limit]
    resumes = []
    for record in selected:
        document = ParsedDocument(
            path=f"{record['id']}.txt", text=record["text"], source_format="txt"
        )
        resumes.append(build_resume(document, candidate_id=record["id"]))
    return resumes


def build_name_variants(resumes: list, group: str) -> list:
    """One copy of every resume with its name replaced by a group name."""
    names = fairness.name_variants(group, count=max(8, len(resumes)))
    variants = []
    for index, resume in enumerate(resumes):
        replacement = names[index % len(names)]
        original = resume.name or resume.text.strip().splitlines()[0][:40]
        text = fairness.substitute_name(resume.text, original, replacement)
        document = ParsedDocument(
            path=f"{resume.candidate_id}_{group}.txt", text=text, source_format="txt"
        )
        built = build_resume(document, candidate_id=f"{resume.candidate_id}__{group}")
        variants.append(built)
    return variants


def audit_names(resumes: list, job, anonymise: bool, skill_method: str) -> fairness.AuditReport:
    """Rank every group's variants together and compare group means."""
    ranker = HybridRanker(
        skill_method=skill_method,
        use_antigaming=False,
        gate=OFF,
        anonymise=anonymise,
    )

    pool, membership = [], {}
    for group in fairness.NAME_SETS:
        variants = build_name_variants(resumes, group)
        pool.extend(variants)
        for variant in variants:
            membership[variant.candidate_id] = group

    ranked = ranker.rank_detailed(pool, job)
    groups = {name: fairness.GroupResult(name) for name in fairness.NAME_SETS}
    for candidate in ranked:
        group = membership.get(candidate.candidate_id)
        if group:
            groups[group].add(candidate.scores, candidate.rank)

    label = f"names ({skill_method}{', anonymised' if anonymise else ''})"
    return fairness.AuditReport(groups=groups, label=label)


def audit_style(resumes: list, job, skill_method: str) -> dict:
    """Terse vs original narrative, facts held constant."""
    ranker = HybridRanker(
        skill_method=skill_method, use_antigaming=False, gate=OFF
    )

    pool, kind = [], {}
    for resume in resumes:
        pool.append(resume)
        kind[resume.candidate_id] = "original"

        implicit_text = fairness.implicit_rewrite(resume)
        document = ParsedDocument(
            path=f"{resume.candidate_id}_implicit.txt",
            text=implicit_text,
            source_format="txt",
        )
        built = build_resume(document, candidate_id=f"{resume.candidate_id}__implicit")
        pool.append(built)
        kind[built.candidate_id] = "implicit"

    ranked = ranker.rank_detailed(pool, job)
    buckets: dict[str, list[float]] = {"original": [], "implicit": []}
    skills: dict[str, list[float]] = {"original": [], "implicit": []}
    for candidate in ranked:
        bucket = kind.get(candidate.candidate_id)
        if bucket:
            buckets[bucket].append(candidate.scores.total)
            skills[bucket].append(candidate.scores.skill)

    return {
        "skill_method": skill_method,
        "original_total": statistics.mean(buckets["original"]) if buckets["original"] else float("nan"),
        "implicit_total": statistics.mean(buckets["implicit"]) if buckets["implicit"] else float("nan"),
        "original_skill": statistics.mean(skills["original"]) if skills["original"] else float("nan"),
        "implicit_skill": statistics.mean(skills["implicit"]) if skills["implicit"] else float("nan"),
    }


def run(real_path: Path, out_dir: Path, category: str, limit: int) -> dict:
    resumes = load_real(real_path, category, limit)
    print(f"loaded {len(resumes)} real {category} resumes")
    job = build_job(JOB_TEXT, title="Senior Software Engineer")

    reports = []
    for skill_method in ("coverage", "grounded"):
        for anonymise in (False, True):
            print(f"auditing names: {skill_method}, anonymise={anonymise} ...")
            reports.append(audit_names(resumes, job, anonymise, skill_method))

    print("auditing writing style ...")
    style = [audit_style(resumes, job, m) for m in ("coverage", "grounded")]

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for report in reports:
        row = {
            "audit": report.label,
            "max_total_gap": report.max_gap,
            "dominant_component": report.dominant_component,
            "disparate_impact@30": report.disparate_impact(30),
            "di_reliable": report.di_is_reliable(30),
        }
        row.update({f"gap_{k}": v for k, v in report.component_gaps.items()})
        row.update({f"mean_{k}": v for k, v in report.group_means.items()})
        rows.append(row)
    _write_csv(out_dir / "fairness_names.csv", rows)
    _write_csv(out_dir / "fairness_style.csv", style)
    _write_report(out_dir / "fairness.md", reports, style, len(resumes), category)

    return {"reports": reports, "style": style}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    path: Path, reports: list, style: list[dict], n: int, category: str
) -> None:
    lines = [
        "# Fairness audit",
        "",
        f"{n} real resumes ({category}), 6 name groups, one job description.",
        "",
        "> **Names are a weak proxy for demographic groups.** This measures",
        "> name-induced score disparity - the harm channel that actually exists",
        "> for an automated screener, since the name is all it sees. It is not a",
        "> measurement of outcomes for real demographic populations, and must not",
        "> be reported as one.",
        "",
        "## Name substitution",
        "",
        "Identical resumes, only the name changed. Gap = largest difference",
        "between any two group means.",
        "",
        "| Audit | Max total gap | Dominant component | DI@30 |",
        "|---|---|---|---|",
    ]
    for report in reports:
        ratio = report.disparate_impact(30)
        ratio_s = "-" if ratio != ratio else f"{ratio:.3f}"
        lines.append(
            f"| {report.label} | {report.max_gap:.4f} | {report.dominant_component} | {ratio_s} |"
        )

    lines += ["", "### Per-component gaps", "", "| Audit | semantic | skill | experience | education |", "|---|---|---|---|---|"]
    for report in reports:
        gaps = report.component_gaps
        lines.append(
            f"| {report.label} | {gaps['semantic']:.4f} | {gaps['skill']:.4f} | "
            f"{gaps['experience']:.4f} | {gaps['education']:.4f} |"
        )

    lines += [
        "",
        "### Group means (total score)",
        "",
        "| Audit | " + " | ".join(fairness.NAME_SETS) + " |",
        "|---" * (len(fairness.NAME_SETS) + 1) + "|",
    ]
    for report in reports:
        means = report.group_means
        cells = " | ".join(f"{means.get(g, float('nan')):.4f}" for g in fairness.NAME_SETS)
        lines.append(f"| {report.label} | {cells} |")

    lines += [
        "",
        "## Writing-style counterfactual",
        "",
        "Narratives rewritten tersely with the technologies they name preserved,",
        "so the facts are constant and only the prose changes. This targets the",
        "risk specific to evidence grounding: that it rewards fluent writing",
        "rather than demonstrated competence.",
        "",
        "| Skill measure | Original total | Implicit total | Original skill | Implicit skill | Skill delta |",
        "|---|---|---|---|---|---|",
    ]
    for row in style:
        delta = row["implicit_skill"] - row["original_skill"]
        lines.append(
            f"| {row['skill_method']} | {row['original_total']:.4f} | {row['implicit_total']:.4f} | "
            f"{row['original_skill']:.4f} | {row['implicit_skill']:.4f} | {delta:+.4f} |"
        )

    lines += [
        "",
        "A large negative skill delta under `grounded` but not under `coverage`",
        "means evidence grounding penalises candidates who describe the same work",
        "without naming the technology - a writing habit, not a competence gap.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", default="data/real/real_resumes.json", type=Path)
    parser.add_argument("--out", default="benchmark/results", type=Path)
    parser.add_argument("--category", default="INFORMATION-TECHNOLOGY")
    parser.add_argument("--limit", default=40, type=int)
    args = parser.parse_args()

    if not args.real.exists():
        raise SystemExit(
            f"no real corpus at {args.real}. Run:\n"
            "  python benchmark/fetch_real_resumes.py --limit 500"
        )

    result = run(args.real, args.out, args.category, args.limit)

    print()
    for report in result["reports"]:
        print(report.summary(top_k=30))
        print()
    print("writing-style counterfactual:")
    for row in result["style"]:
        delta = row["implicit_skill"] - row["original_skill"]
        print(
            f"  {row['skill_method']:<10} skill {row['original_skill']:.4f} -> "
            f"{row['implicit_skill']:.4f}  ({delta:+.4f})"
        )
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
