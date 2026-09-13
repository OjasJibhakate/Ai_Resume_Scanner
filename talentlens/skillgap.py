"""Explainable multi-dimensional skill-gap analyser (novel contribution #2).

A raw match percentage tells a recruiter that someone scored 0.62 on skills. It
does not tell them *which* requirements are unmet, whether the gaps are
fundamental or trivially closeable, or what to ask about in an interview. This
module turns the skill dimension into an itemised, auditable breakdown.

The useful idea is the middle category. Standard screening is binary - you have
the skill or you do not - which throws away the most actionable information in
the whole pipeline. A candidate who knows Django but not Flask is not "missing
Flask" in any meaningful sense; they are one short hop away through Python. So
every unmet requirement is classified as:

* ``MATCHED``      - held directly, or implied by the ontology.
* ``TRANSFERABLE`` - not held, but reachable from something they do hold within
  a couple of ontology hops, with the bridging skill and path reported.
* ``MISSING``      - no nearby relative; a genuine gap.
"""

from __future__ import annotations

from .ontology import SkillOntology, get_ontology
from .schemas import (
    GapStatus,
    JobSpec,
    ParsedResume,
    SkillGapItem,
    SkillGapReport,
)

#: How far to search for a bridging skill. Two hops covers the common
#: "sibling through a shared parent" case (Django -> Python -> Flask) without
#: drifting into claims like "they know CSS so they can pick up Kubernetes".
DEFAULT_MAX_HOPS = 2


def analyse(
    resume: ParsedResume,
    job: JobSpec,
    ontology: SkillOntology | None = None,
    max_hops: int = DEFAULT_MAX_HOPS,
) -> SkillGapReport:
    """Classify every job requirement against what the candidate holds."""
    ontology = ontology or get_ontology()

    held_raw = set(resume.skills)
    held_expanded = set(resume.skills_expanded)

    required = set(job.required_skills)
    preferred = set(job.preferred_skills) - required

    items: list[SkillGapItem] = []
    for skill in sorted(required | preferred):
        is_required = skill in required

        if skill in held_expanded:
            items.append(
                SkillGapItem(skill=skill, status=GapStatus.MATCHED, required=is_required)
            )
            continue

        bridge, path = ontology.nearest_held(skill, held_raw, max_hops=max_hops)
        if bridge is not None:
            items.append(
                SkillGapItem(
                    skill=skill,
                    status=GapStatus.TRANSFERABLE,
                    required=is_required,
                    via=bridge,
                    path=path,
                )
            )
            continue

        items.append(
            SkillGapItem(skill=skill, status=GapStatus.MISSING, required=is_required)
        )

    # Surplus is computed against the raw skill set, not the expanded one, so we
    # report things the candidate actually wrote rather than implied ancestors.
    surplus = held_raw - required - preferred

    return SkillGapReport(items=tuple(items), surplus=frozenset(surplus))


def upskill_priorities(
    report: SkillGapReport,
    ontology: SkillOntology | None = None,
    limit: int = 5,
) -> list[SkillGapItem]:
    """Order the unmet requirements by what to close first.

    Required beats preferred; among equals, a transferable gap outranks a
    genuine one, because it is the cheapest to close and therefore the most
    useful thing to raise in an interview.
    """
    ontology = ontology or get_ontology()

    def sort_key(item: SkillGapItem) -> tuple:
        return (
            0 if item.required else 1,
            0 if item.status is GapStatus.TRANSFERABLE else 1,
            len(item.path),
            item.skill,
        )

    unmet = [i for i in report.items if i.status is not GapStatus.MATCHED]
    return sorted(unmet, key=sort_key)[:limit]


def describe(item: SkillGapItem) -> str:
    """One-line human-readable rendering of a single gap item."""
    if item.status is GapStatus.MATCHED:
        return f"{item.skill}: held"
    if item.status is GapStatus.TRANSFERABLE:
        path = " -> ".join(item.path) if item.path else f"{item.via} -> {item.skill}"
        return f"{item.skill}: transferable via {item.via} ({path})"
    tier = "required" if item.required else "preferred"
    return f"{item.skill}: missing ({tier})"


def summary_counts(report: SkillGapReport) -> dict[str, int]:
    """Counts for the dashboard chips and the benchmark rows."""
    return {
        "matched": len(report.matched),
        "transferable": len(report.transferable),
        "missing": len(report.missing),
        "missing_required": len(report.missing_required),
        "surplus": len(report.surplus),
        "total_requirements": len(report.items),
    }
