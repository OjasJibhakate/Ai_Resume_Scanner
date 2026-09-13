"""Central configuration for TalentLens.

Deliberately dependency-light: plain frozen dataclasses rather than a settings
library, so that scoring behaviour is explicit, testable and trivially
serialisable into a benchmark result row.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / ".cache"
ONTOLOGY_PATH = DATA_DIR / "skill_ontology.yaml"

# --------------------------------------------------------------------------
# Embedding model
# --------------------------------------------------------------------------

#: Paper 4 (Kumar & Chellamani, ICCCA 2025) uses this exact bi-encoder.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
EMBEDDING_BATCH_SIZE = 32


# --------------------------------------------------------------------------
# Scoring weights
# --------------------------------------------------------------------------


class WeightError(ValueError):
    """Raised when scoring weights are not a valid convex combination."""


@dataclass(frozen=True)
class ScoringWeights:
    """Weights for the multi-factor compatibility formula.

    S_total = w_sem*S_sem + w_skill*S_skill + w_exp*S_exp + w_edu*S_edu - P

    Defaults are the study's recommended values (40/30/20/10). The weights must
    form a convex combination so that S_total lands in [0, 1] before the
    anti-gaming penalty is subtracted; this keeps scores comparable across runs
    and makes the penalty interpretable as "points deducted out of 1.0".
    """

    semantic: float = 0.40
    skill: float = 0.30
    experience: float = 0.20
    education: float = 0.10

    #: Tolerance for the sum-to-one check, to absorb float and UI-slider noise.
    TOLERANCE = 1e-6

    def __post_init__(self) -> None:
        for name in ("semantic", "skill", "experience", "education"):
            value = getattr(self, name)
            if value < 0:
                raise WeightError(f"weight {name!r} must be >= 0, got {value}")
        if abs(self.total - 1.0) > self.TOLERANCE:
            raise WeightError(
                f"weights must sum to 1.0, got {self.total:.6f} "
                f"({self.as_dict()}). Use ScoringWeights.rescaled() to normalise."
            )

    @property
    def total(self) -> float:
        return self.semantic + self.skill + self.experience + self.education

    def as_dict(self) -> dict[str, float]:
        return {
            "semantic": self.semantic,
            "skill": self.skill,
            "experience": self.experience,
            "education": self.education,
        }

    @classmethod
    def rescaled(
        cls,
        semantic: float,
        skill: float,
        experience: float,
        education: float,
    ) -> "ScoringWeights":
        """Build weights from arbitrary non-negative values, normalising to 1.0.

        This is what the Streamlit sliders use: a recruiter can drag four
        independent sliders without having to make them sum to one by hand.
        """
        raw = [semantic, skill, experience, education]
        if any(v < 0 for v in raw):
            raise WeightError(f"weights must be >= 0, got {raw}")
        total = sum(raw)
        if total <= 0:
            raise WeightError("at least one weight must be > 0")
        return cls(*(v / total for v in raw))

    def with_(self, **kwargs: float) -> "ScoringWeights":
        """Return a copy with some weights replaced, renormalised to sum to 1."""
        merged = {**self.as_dict(), **kwargs}
        return self.rescaled(**merged)


DEFAULT_WEIGHTS = ScoringWeights()


# --------------------------------------------------------------------------
# Anti-gaming detector
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AntiGamingConfig:
    """Thresholds for the keyword-stuffing detector.

    ``max_penalty`` caps how much a single resume can be docked. It is capped
    below 1.0 deliberately: the detector uses heuristics, and a false positive
    should demote a candidate rather than erase them from the shortlist. A
    recruiter still sees the flag and the evidence and can overrule it.
    """

    max_penalty: float = 0.35

    # Signal 1 — invisible text.
    #: Euclidean distance in RGB space (0-255 per channel) below which text is
    #: considered the same colour as its background, i.e. invisible.
    invisible_color_distance: float = 40.0
    #: Font sizes below this (points) are unreadable in normal viewing.
    min_readable_font_size: float = 4.0

    # Signal 2 — skill density anomaly.
    #: Z-score above which skills-per-token is anomalous versus the corpus.
    density_zscore_threshold: float = 2.5
    #: Absolute fallback when the corpus is too small for a stable z-score.
    density_absolute_threshold: float = 0.28

    # Signal 3 — repetition spike.
    #: A term repeated more than this many times per 1000 tokens is suspicious.
    repetition_per_1k_threshold: float = 12.0
    #: Terms shorter than this are ignored (stop-word-like noise).
    repetition_min_term_length: int = 3

    # Signal 4 — job-description echo.
    #: Fraction of the JD's word trigrams appearing verbatim in the resume,
    #: above which the candidate has likely pasted the posting into their CV.
    #: Genuine resumes share vocabulary with a JD but rarely long exact phrases,
    #: so this separates cleanly in practice.
    jd_echo_threshold: float = 0.25
    #: Below this many JD trigrams the ratio is too noisy to act on.
    jd_echo_min_trigrams: int = 25

    # Signal 5 — semantic incoherence.
    #: Cosine similarity between the skills section and the experience
    #: narrative, below which the claimed skills look unsupported by history.
    incoherence_threshold: float = 0.18
    #: Both sections need at least this many tokens for the check to be fair.
    incoherence_min_tokens: int = 25

    #: Weight of each signal in the combined penalty, in the order
    #: (invisible_text, skill_density, repetition, jd_echo, incoherence).
    #: Invisible text is weighted highest because it is unambiguous intent to
    #: deceive - there is no innocent reason to put white-on-white keywords in
    #: a resume. Density and repetition are weighted lowest because a genuinely
    #: broad engineer with a long tools list can trip them honestly.
    signal_weights: tuple[float, ...] = (0.35, 0.15, 0.15, 0.20, 0.15)

    def __post_init__(self) -> None:
        if len(self.signal_weights) != 5:
            raise ValueError(
                f"signal_weights must have 5 entries, got {len(self.signal_weights)}"
            )
        total = sum(self.signal_weights)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"signal_weights must sum to 1.0, got {total}")


DEFAULT_ANTIGAMING = AntiGamingConfig()


# --------------------------------------------------------------------------
# Prerequisite gate
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PrerequisiteGate:
    """Damping applied to candidates who meet few of the *required* skills.

    Motivated by a concrete failure of the purely additive formula. Experience
    and education are role-agnostic: a nurse with 11 years and a bachelor's
    degree collects the full 0.20 + 0.10 on a backend engineering req and lands
    at ~0.40 overall, which a recruiter reads as "40% match". That is not a
    borderline call, it is a wrong answer, and it comes from summing dimensions
    that carry no information about domain fit.

    This is our reading of Paper 5's directional insight: qualification
    compliance (does this person clear the hard bar?) is a different question
    from general fit, and collapsing both into one symmetric sum loses it.

    The gate multiplies the weighted score by a factor that scales with required
    -skill coverage, bottoming out at ``floor`` rather than zero so that
    ordering within the unqualified group is preserved and nobody is silently
    erased. Set ``enabled=False`` to recover the study's literal formula - the
    benchmark reports both.
    """

    enabled: bool = True
    #: Required-skill coverage at or above which no damping is applied.
    threshold: float = 0.5
    #: Lowest multiplier, so unqualified candidates are demoted, not deleted.
    floor: float = 0.25

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {self.threshold}")
        if not 0.0 <= self.floor <= 1.0:
            raise ValueError(f"floor must be in [0, 1], got {self.floor}")

    def multiplier(self, required_coverage: float) -> float:
        """Map required-skill coverage to a score multiplier in [floor, 1]."""
        if not self.enabled:
            return 1.0
        ratio = min(1.0, max(0.0, required_coverage) / self.threshold)
        return self.floor + (1.0 - self.floor) * ratio


DEFAULT_GATE = PrerequisiteGate()


# --------------------------------------------------------------------------
# Evidence grounding
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundingConfig:
    """Settings for evidence-grounded skill verification.

    ``floor`` is the ethical parameter, so it is worth stating plainly what it
    encodes. Real resumes list skills they never narrate - space is finite and
    a skills section is a legitimate summary device. Scoring an unevidenced
    claim at zero would punish ordinary resume formatting, not dishonesty. The
    floor sets what a bare claim is worth relative to a demonstrated one; at
    0.4, evidence earns a candidate 2.5x what assertion does. Setting it to 1.0
    disables grounding entirely and recovers plain coverage.
    """

    #: Credit a matched-but-unevidenced skill receives, as a fraction of the
    #: credit an evidenced one receives.
    floor: float = 0.40

    #: Cosine at or above which a narrative sentence counts as implying a skill
    #: that it never names. Below this, no semantic evidence is recorded.
    #:
    #: Calibrated, not guessed. Measured over corpus narratives, the similarity
    #: between a skill query and its best-matching sentence separates as:
    #:   skill named in the narrative (positive): mean 0.482, p10 0.388
    #:   skill absent (negative):                 mean 0.225, p95 0.378
    #: 0.40 sits just above the negative p95, so semantic evidence is only
    #: recorded where it clears essentially all unrelated pairings. The classes
    #: overlap in 0.34-0.39, and the asymmetry of the costs decides the tie:
    #: inventing evidence for a skill a candidate never demonstrated is a worse
    #: error than falling back to the listed-only floor.
    semantic_threshold: float = 0.40
    #: Cosine at which semantic evidence is considered maximally strong,
    #: set near the observed positive mean.
    semantic_ceiling: float = 0.58
    #: Ceiling on semantic grounding. Held below 1.0 so that an inferred match
    #: never counts as much as the skill being named outright.
    semantic_max: float = 0.75

    #: A bare skill token embeds poorly against full sentences; wrapping it in a
    #: natural phrase puts query and document in comparable regions of the space.
    query_template: str = "experience with {skill}"

    def __post_init__(self) -> None:
        if not 0.0 <= self.floor <= 1.0:
            raise ValueError(f"floor must be in [0, 1], got {self.floor}")
        if self.semantic_ceiling <= self.semantic_threshold:
            raise ValueError("semantic_ceiling must exceed semantic_threshold")
        if not 0.0 <= self.semantic_max <= 1.0:
            raise ValueError(f"semantic_max must be in [0, 1], got {self.semantic_max}")


DEFAULT_GROUNDING = GroundingConfig()


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------

SPACY_MODEL = "en_core_web_sm"

#: Canonical section names we segment a resume into.
SECTION_NAMES = (
    "summary",
    "experience",
    "education",
    "skills",
    "projects",
    "certifications",
    "other",
)

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "CACHE_DIR",
    "ONTOLOGY_PATH",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
    "EMBEDDING_BATCH_SIZE",
    "ScoringWeights",
    "WeightError",
    "DEFAULT_WEIGHTS",
    "AntiGamingConfig",
    "DEFAULT_ANTIGAMING",
    "PrerequisiteGate",
    "DEFAULT_GATE",
    "GroundingConfig",
    "DEFAULT_GROUNDING",
    "SPACY_MODEL",
    "SECTION_NAMES",
    "replace",
]
