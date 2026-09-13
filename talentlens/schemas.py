"""Data structures shared across the TalentLens pipeline.

Plain dataclasses rather than pydantic models: these objects are constructed
internally by our own parsers (not deserialised from untrusted input), so
runtime validation would cost more than it buys. Where an invariant genuinely
matters it is asserted explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from .grounding import GroundingReport


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TextSpan:
    """A run of text with its visual attributes, as reported by the parser.

    The visual attributes are the whole point: ``color`` and ``size`` are what
    let the anti-gaming detector spot white-on-white or 1pt keyword blocks.
    Discarding them at parse time would make that detection impossible, so
    ``parsing`` retains them even though most of the pipeline ignores them.
    """

    text: str
    page: int
    font: str = ""
    size: float = 0.0
    #: Packed 0xRRGGBB. -1 when the source format does not expose colour.
    color: int = -1
    #: Format-specific style bitfield (PyMuPDF: bold/italic/serif flags).
    flags: int = 0
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    @property
    def rgb(self) -> tuple[int, int, int] | None:
        """Unpack ``color`` into an (R, G, B) triple, or None if unknown."""
        if self.color < 0:
            return None
        return ((self.color >> 16) & 0xFF, (self.color >> 8) & 0xFF, self.color & 0xFF)


@dataclass
class ParsedDocument:
    """Raw text plus layout metadata extracted from a resume file."""

    path: str
    text: str
    source_format: str  # "pdf" | "docx" | "txt"
    spans: list[TextSpan] = field(default_factory=list)
    page_count: int = 1
    #: Dominant background colour per page as packed 0xRRGGBB. Defaults to
    #: white, which is correct for essentially every resume in the wild.
    page_backgrounds: list[int] = field(default_factory=list)

    def background_for(self, page: int) -> int:
        if 0 <= page < len(self.page_backgrounds):
            return self.page_backgrounds[page]
        return 0xFFFFFF


# --------------------------------------------------------------------------
# Education
# --------------------------------------------------------------------------


class DegreeLevel(IntEnum):
    """Ordinal education tiers.

    The study specifies BSc=1, MSc=2, PhD=3. We keep that *relative* ordering
    but add NONE and ASSOCIATE tiers, because real resumes contain both and a
    scale with no zero cannot express "no degree found". Only comparisons and
    ratios between tiers are ever used, so the absolute offsets do not matter.
    """

    NONE = 0
    ASSOCIATE = 1
    BACHELOR = 2
    MASTER = 3
    DOCTORATE = 4

    @property
    def label(self) -> str:
        return _DEGREE_LABELS[self]


_DEGREE_LABELS = {
    DegreeLevel.NONE: "No degree found",
    DegreeLevel.ASSOCIATE: "Associate / Diploma",
    DegreeLevel.BACHELOR: "Bachelors",
    DegreeLevel.MASTER: "Masters",
    DegreeLevel.DOCTORATE: "Doctorate",
}


@dataclass(frozen=True)
class ExperienceEntry:
    """One employment span parsed out of the experience section."""

    title: str = ""
    organisation: str = ""
    start_year: int | None = None
    start_month: int | None = None
    end_year: int | None = None
    end_month: int | None = None
    is_current: bool = False
    text: str = ""

    @property
    def has_dates(self) -> bool:
        return self.start_year is not None


# --------------------------------------------------------------------------
# Resume / job
# --------------------------------------------------------------------------


@dataclass
class ParsedResume:
    """A resume after parsing, segmentation and entity extraction."""

    document: ParsedDocument
    candidate_id: str
    name: str = ""
    email: str = ""
    phone: str = ""
    sections: dict[str, str] = field(default_factory=dict)
    #: Canonical skill names found verbatim in the text.
    skills: set[str] = field(default_factory=set)
    #: ``skills`` plus every ancestor implied by the ontology (django -> python).
    skills_expanded: set[str] = field(default_factory=set)
    years_experience: float = 0.0
    degree: DegreeLevel = DegreeLevel.NONE
    experience_entries: list[ExperienceEntry] = field(default_factory=list)

    @property
    def text(self) -> str:
        return self.document.text

    def section(self, name: str) -> str:
        return self.sections.get(name, "")


@dataclass
class JobSpec:
    """A job description with its extracted hard requirements."""

    title: str
    text: str
    required_skills: set[str] = field(default_factory=set)
    preferred_skills: set[str] = field(default_factory=set)
    required_skills_expanded: set[str] = field(default_factory=set)
    min_years: float = 0.0
    min_degree: DegreeLevel = DegreeLevel.NONE

    @property
    def all_skills(self) -> set[str]:
        return self.required_skills | self.preferred_skills


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SubScores:
    """The four sub-scores, the penalty, and their weighted total."""

    semantic: float
    skill: float
    experience: float
    education: float
    penalty: float
    total: float
    #: Prerequisite-gate multiplier that was applied (1.0 = ungated).
    gate: float = 1.0
    #: Fraction of the job's *required* skills the candidate holds.
    required_coverage: float = 1.0

    def as_dict(self) -> dict[str, float]:
        return {
            "semantic": self.semantic,
            "skill": self.skill,
            "experience": self.experience,
            "education": self.education,
            "penalty": self.penalty,
            "gate": self.gate,
            "required_coverage": self.required_coverage,
            "total": self.total,
        }


# --------------------------------------------------------------------------
# Anti-gaming (novel contribution #1)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GamingSignal:
    """One detector signal, with the evidence that triggered it.

    ``severity`` is in [0, 1]. Evidence is a human-readable string because the
    whole point of the module is that a recruiter can audit *why* a resume was
    penalised rather than being handed an opaque number.
    """

    name: str
    triggered: bool
    severity: float
    evidence: str


@dataclass(frozen=True)
class GamingReport:
    signals: tuple[GamingSignal, ...]
    penalty: float
    flagged: bool

    @property
    def triggered_signals(self) -> tuple[GamingSignal, ...]:
        return tuple(s for s in self.signals if s.triggered)

    def signal(self, name: str) -> GamingSignal | None:
        for s in self.signals:
            if s.name == name:
                return s
        return None


# --------------------------------------------------------------------------
# Skill gap (novel contribution #2)
# --------------------------------------------------------------------------


class GapStatus(str, Enum):
    MATCHED = "matched"
    #: Not held, but the candidate holds a close ontology relative, so the
    #: skill is plausibly reachable with little training.
    TRANSFERABLE = "transferable"
    MISSING = "missing"


@dataclass(frozen=True)
class SkillGapItem:
    skill: str
    status: GapStatus
    #: True when the JD lists this as required rather than merely preferred.
    required: bool
    #: For TRANSFERABLE items, the skill the candidate already has that bridges
    #: to this one, and the ontology path connecting them.
    via: str | None = None
    path: tuple[str, ...] = ()


@dataclass(frozen=True)
class SkillGapReport:
    items: tuple[SkillGapItem, ...]
    #: Skills the candidate has that the JD never asked for.
    surplus: frozenset[str] = frozenset()

    def _by(
        self, status: GapStatus, required: bool | None = None
    ) -> tuple[SkillGapItem, ...]:
        return tuple(
            i
            for i in self.items
            if i.status is status and (required is None or i.required == required)
        )

    @property
    def matched(self) -> tuple[SkillGapItem, ...]:
        return self._by(GapStatus.MATCHED)

    @property
    def transferable(self) -> tuple[SkillGapItem, ...]:
        return self._by(GapStatus.TRANSFERABLE)

    @property
    def missing(self) -> tuple[SkillGapItem, ...]:
        return self._by(GapStatus.MISSING)

    @property
    def missing_required(self) -> tuple[SkillGapItem, ...]:
        return self._by(GapStatus.MISSING, required=True)

    @property
    def coverage(self) -> float:
        """Fraction of requested skills the candidate already holds."""
        if not self.items:
            return 0.0
        return len(self.matched) / len(self.items)


# --------------------------------------------------------------------------
# Ranking output
# --------------------------------------------------------------------------


@dataclass
class RankedCandidate:
    """A scored, ranked candidate with everything needed to explain the rank."""

    resume: ParsedResume
    scores: SubScores
    gaming: GamingReport
    gap: SkillGapReport
    rank: int = 0
    explanation: str = ""
    #: Per-skill evidence verification. None when grounding was not computed.
    grounding: "GroundingReport | None" = None

    @property
    def candidate_id(self) -> str:
        return self.resume.candidate_id

    @property
    def display_name(self) -> str:
        return self.resume.name or self.resume.candidate_id


def summarise_skills(skills: Iterable[str], limit: int = 8) -> str:
    """Render a skill set for display, truncating politely."""
    ordered = sorted(skills)
    if len(ordered) <= limit:
        return ", ".join(ordered)
    shown = ", ".join(ordered[:limit])
    return f"{shown} (+{len(ordered) - limit} more)"
