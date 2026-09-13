"""The multi-factor compatibility score.

    S_total = w_sem*S_sem + w_skill*S_skill + w_exp*S_exp + w_edu*S_edu - P

A note on the skill term, because we deliberately depart from the literal
formula in the source study.

The study specifies Sorensen-Dice overlap::

    S_skill = 2 |K_cand & K_req| / (|K_cand| + |K_req|)

Dice is symmetric, and that is wrong for resume screening. A resume lists every
skill a person has; a JD lists the handful the role needs. Consider a JD asking
for 10 skills:

* Candidate A knows 50 skills, including all 10  -> Dice = 2*10/(50+10) = 0.33
* Candidate B knows exactly those 10             -> Dice = 2*10/(10+10) = 1.00

Dice ranks the broader, fully-qualified candidate *below* the narrower one,
purely for knowing extra things. That is a scoring artefact, not a hiring
signal. What a recruiter actually wants is **coverage**: what fraction of the
requirements does this person meet?

So ``coverage`` is the default, weighting required skills fully and preferred
skills at half. ``dice`` remains available and is reported side by side in the
benchmark, so the effect is measured rather than asserted.
"""

from __future__ import annotations

from typing import Iterable, Literal

import numpy as np

from .config import (
    DEFAULT_GATE,
    DEFAULT_GROUNDING,
    DEFAULT_WEIGHTS,
    GroundingConfig,
    PrerequisiteGate,
    ScoringWeights,
)
from .schemas import DegreeLevel, GamingReport, JobSpec, ParsedResume, SubScores

SkillMethod = Literal["coverage", "dice", "grounded"]

#: How much a "preferred" skill counts relative to a "required" one.
PREFERRED_WEIGHT = 0.5


# --------------------------------------------------------------------------
# Sub-scores
# --------------------------------------------------------------------------


def semantic_score(resume_vector: np.ndarray, jd_vector: np.ndarray) -> float:
    """Cosine similarity between the SBERT document vectors."""
    from .embedding import cosine_similarity

    return cosine_similarity(resume_vector, jd_vector)


def dice_score(candidate: Iterable[str], required: Iterable[str]) -> float:
    """Literal Sorensen-Dice overlap, as specified in the source study."""
    a, b = set(candidate), set(required)
    total = len(a) + len(b)
    if total == 0:
        return 0.0
    return 2.0 * len(a & b) / total


def coverage_score(
    candidate: Iterable[str],
    required: Iterable[str],
    preferred: Iterable[str] = (),
    preferred_weight: float = PREFERRED_WEIGHT,
) -> float:
    """Weighted fraction of the job's skills the candidate holds.

    Returns 1.0 when the JD names no skills at all: an absent requirement is
    vacuously satisfied, and scoring it as zero would drag every candidate down
    by the same amount while telling the recruiter nothing.
    """
    held = set(candidate)
    req = set(required)
    pref = set(preferred) - req

    denominator = len(req) + preferred_weight * len(pref)
    if denominator == 0:
        return 1.0

    numerator = len(req & held) + preferred_weight * len(pref & held)
    return float(min(1.0, numerator / denominator))


def skill_score(
    candidate: Iterable[str],
    required: Iterable[str],
    preferred: Iterable[str] = (),
    method: SkillMethod = "coverage",
    grounding=None,
    grounding_config: GroundingConfig = DEFAULT_GROUNDING,
) -> float:
    """Dispatch to the configured skill measure.

    ``grounded`` weights each satisfied requirement by whether the resume's own
    work history demonstrates it, and falls back to plain coverage when no
    grounding report is supplied - so the method is always safe to request.
    """
    if method == "dice":
        return dice_score(candidate, set(required) | set(preferred))
    if method == "grounded" and grounding is not None:
        from .grounding import grounded_coverage

        return grounded_coverage(
            candidate, required, preferred, report=grounding, config=grounding_config
        )
    return coverage_score(candidate, required, preferred)


def experience_score(candidate_years: float, required_years: float) -> float:
    """Ratio min(1, E_cand / E_req), per the study.

    A role with no stated minimum scores 1.0 rather than 0.0, for the same
    reason as ``coverage_score``: nothing was asked, so nothing is missing.
    Seniority beyond the bar is capped at 1.0 - being twice as experienced as
    required is not twice as good a fit, and uncapping it would let a 20-year
    veteran outrank everyone on a junior req.
    """
    if required_years <= 0:
        return 1.0
    if candidate_years <= 0:
        return 0.0
    return float(min(1.0, candidate_years / required_years))


def education_score(candidate: DegreeLevel, required: DegreeLevel) -> float:
    """Ordinal degree-tier alignment."""
    if required <= DegreeLevel.NONE:
        return 1.0
    if candidate >= required:
        return 1.0
    if candidate <= DegreeLevel.NONE:
        return 0.0
    return float(candidate) / float(required)


# --------------------------------------------------------------------------
# Combination
# --------------------------------------------------------------------------


def required_coverage(candidate: Iterable[str], required: Iterable[str]) -> float:
    """Fraction of the job's *required* skills the candidate holds.

    Preferred skills are excluded on purpose: this measures whether the hard bar
    is cleared, which is a different question from overall fit.
    """
    req = set(required)
    if not req:
        return 1.0
    return len(req & set(candidate)) / len(req)


def combine(
    semantic: float,
    skill: float,
    experience: float,
    education: float,
    penalty: float = 0.0,
    weights: ScoringWeights = DEFAULT_WEIGHTS,
    gate: float = 1.0,
    coverage: float = 1.0,
) -> SubScores:
    """Fuse the sub-scores into a total, clamped to [0, 1].

    The gate scales the weighted fit before the penalty is subtracted, so an
    integrity penalty stays a full-strength deduction rather than being softened
    for candidates who were already being damped.
    """
    weighted = (
        weights.semantic * semantic
        + weights.skill * skill
        + weights.experience * experience
        + weights.education * education
    )
    total = weighted * gate - penalty
    return SubScores(
        semantic=float(semantic),
        skill=float(skill),
        experience=float(experience),
        education=float(education),
        penalty=float(penalty),
        total=float(max(0.0, min(1.0, total))),
        gate=float(gate),
        required_coverage=float(coverage),
    )


def score_candidate(
    resume: ParsedResume,
    job: JobSpec,
    resume_vector: np.ndarray,
    jd_vector: np.ndarray,
    weights: ScoringWeights = DEFAULT_WEIGHTS,
    gaming: GamingReport | None = None,
    skill_method: SkillMethod = "coverage",
    gate: PrerequisiteGate = DEFAULT_GATE,
    grounding=None,
    grounding_config: GroundingConfig = DEFAULT_GROUNDING,
) -> SubScores:
    """Score one candidate against one job.

    ``resume.skills_expanded`` is used rather than the raw skill set so that
    ontology implication counts: a Django developer satisfies a Python
    requirement without having typed the word.
    """
    coverage = required_coverage(resume.skills_expanded, job.required_skills)
    return combine(
        semantic=semantic_score(resume_vector, jd_vector),
        skill=skill_score(
            resume.skills_expanded,
            job.required_skills,
            job.preferred_skills,
            method=skill_method,
            grounding=grounding,
            grounding_config=grounding_config,
        ),
        experience=experience_score(resume.years_experience, job.min_years),
        education=education_score(resume.degree, job.min_degree),
        penalty=gaming.penalty if gaming else 0.0,
        weights=weights,
        gate=gate.multiplier(coverage),
        coverage=coverage,
    )


def explain_contributions(
    scores: SubScores, weights: ScoringWeights = DEFAULT_WEIGHTS
) -> dict[str, float]:
    """Weighted contribution of each dimension, for the UI breakdown bars.

    These sum to ``total + penalty``, which is what makes the waterfall in the
    dashboard add up.
    """
    return {
        "semantic": weights.semantic * scores.semantic,
        "skill": weights.skill * scores.skill,
        "experience": weights.experience * scores.experience,
        "education": weights.education * scores.education,
    }
