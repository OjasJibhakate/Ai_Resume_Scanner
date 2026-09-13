"""TalentLens - explainable AI resume ranking and candidate matching.

Quick start::

    from talentlens import screen_files

    ranked, failures = screen_files(["cv1.pdf", "cv2.pdf"], job_text=jd)
    for candidate in ranked:
        print(candidate.rank, candidate.display_name, candidate.scores.total)
"""

from .config import DEFAULT_ANTIGAMING, DEFAULT_WEIGHTS, AntiGamingConfig, ScoringWeights
from .pipeline import (
    build_job,
    load_resume,
    load_resume_uploads,
    load_resumes,
    screen,
    screen_files,
)
from .ranking import (
    Bm25Ranker,
    FusionRanker,
    HybridRanker,
    SbertRanker,
    TfidfRanker,
    reciprocal_rank_fusion,
)
from .schemas import (
    DegreeLevel,
    GamingReport,
    GapStatus,
    JobSpec,
    ParsedResume,
    RankedCandidate,
    SkillGapReport,
    SubScores,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AntiGamingConfig",
    "Bm25Ranker",
    "DEFAULT_ANTIGAMING",
    "DEFAULT_WEIGHTS",
    "DegreeLevel",
    "FusionRanker",
    "GamingReport",
    "GapStatus",
    "HybridRanker",
    "JobSpec",
    "ParsedResume",
    "RankedCandidate",
    "SbertRanker",
    "ScoringWeights",
    "SkillGapReport",
    "SubScores",
    "TfidfRanker",
    "build_job",
    "load_resume",
    "load_resume_uploads",
    "load_resumes",
    "reciprocal_rank_fusion",
    "screen",
    "screen_files",
]
