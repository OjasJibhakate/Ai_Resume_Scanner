"""Rankers: the hybrid multi-factor scorer plus the baselines it is measured against.

Every ranker exposes the same ``rank(resumes, job)`` surface so the benchmark
can swap them without special cases. The three baselines correspond to the
lineage the study traces:

* ``TfidfRanker``  - Paper 1 (Surya et al. 2024): sparse lexical, very fast.
* ``Bm25Ranker``   - the probabilistic lexical baseline, with length
  normalisation that blunts naive keyword stuffing.
* ``SbertRanker``  - Paper 4 (Kumar & Chellamani 2025): dense semantic only.
* ``HybridRanker`` - this system: dense semantics fused with structured skill,
  experience and education evidence, minus the anti-gaming penalty.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import numpy as np

from . import antigaming, fairness, grounding as grounding_mod, skillgap
from .config import (
    DEFAULT_ANTIGAMING,
    DEFAULT_GATE,
    DEFAULT_GROUNDING,
    DEFAULT_WEIGHTS,
    AntiGamingConfig,
    GroundingConfig,
    PrerequisiteGate,
    ScoringWeights,
)
from .embedding import Embedder, cosine_similarity, get_embedder
from .schemas import JobSpec, ParsedResume, RankedCandidate
from .scoring import SkillMethod, score_candidate

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#.\-]*")


def _tokenise(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _normalise_scores(scores: dict[str, float]) -> dict[str, float]:
    """Scale scores to [0, 1] by their maximum.

    Order-preserving, so it never changes a ranking. It exists only so that
    unbounded scores (BM25) can be displayed and fused alongside bounded ones.
    """
    if not scores:
        return {}
    highest = max(scores.values())
    if highest <= 0:
        return {key: 0.0 for key in scores}
    return {key: value / highest for key, value in scores.items()}


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------


class BaseRanker:
    name = "base"

    def score_all(self, resumes: list[ParsedResume], job: JobSpec) -> dict[str, float]:
        raise NotImplementedError

    def rank(self, resumes: list[ParsedResume], job: JobSpec) -> list[tuple[str, float]]:
        scores = self.score_all(resumes, job)
        # Ties break on candidate id so runs are byte-for-byte reproducible.
        return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class TfidfRanker(BaseRanker):
    """Sparse TF-IDF vectors with cosine similarity (Paper 1)."""

    name = "tfidf"

    def __init__(self, ngram_range: tuple[int, int] = (1, 2)) -> None:
        self.ngram_range = ngram_range

    def score_all(self, resumes: list[ParsedResume], job: JobSpec) -> dict[str, float]:
        from sklearn.feature_extraction.text import TfidfVectorizer

        if not resumes:
            return {}
        corpus = [r.text for r in resumes] + [job.text]
        vectoriser = TfidfVectorizer(
            ngram_range=self.ngram_range, stop_words="english", sublinear_tf=True
        )
        matrix = vectoriser.fit_transform(corpus)
        jd_vector = matrix[-1]
        resume_matrix = matrix[:-1]
        similarities = (resume_matrix @ jd_vector.T).toarray().ravel()
        return {r.candidate_id: float(s) for r, s in zip(resumes, similarities)}


class Bm25Ranker(BaseRanker):
    """Okapi BM25 with the study's recommended parameters."""

    name = "bm25"

    # k1=1.2, b=0.75 matches the frozen evaluation protocol in the project
    # report (configuration C2). The source study quotes k1 in 1.2-2.0.
    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def score_all(self, resumes: list[ParsedResume], job: JobSpec) -> dict[str, float]:
        from rank_bm25 import BM25Okapi

        if not resumes:
            return {}
        corpus = [_tokenise(r.text) for r in resumes]
        # BM25 needs at least one non-empty document to build its statistics.
        if not any(corpus):
            return {r.candidate_id: 0.0 for r in resumes}
        model = BM25Okapi(corpus, k1=self.k1, b=self.b)
        raw = model.get_scores(_tokenise(job.text))
        return _normalise_scores(
            {r.candidate_id: float(s) for r, s in zip(resumes, raw)}
        )


class SbertRanker(BaseRanker):
    """Dense bi-encoder cosine similarity only (Paper 4's retrieval stage)."""

    name = "sbert"

    def __init__(self, embedder: Embedder | None = None) -> None:
        self._embedder = embedder

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    def score_all(self, resumes: list[ParsedResume], job: JobSpec) -> dict[str, float]:
        if not resumes:
            return {}
        jd_vector = self.embedder.encode_document(job.text)
        resume_vectors = self.embedder.encode_documents([r.text for r in resumes])
        return {
            r.candidate_id: cosine_similarity(v, jd_vector)
            for r, v in zip(resumes, resume_vectors)
        }


# --------------------------------------------------------------------------
# Hybrid
# --------------------------------------------------------------------------


@dataclass
class HybridRanker(BaseRanker):
    """The TalentLens scorer: dense semantics + structured evidence - penalty."""

    weights: ScoringWeights = DEFAULT_WEIGHTS
    antigaming_config: AntiGamingConfig = DEFAULT_ANTIGAMING
    skill_method: SkillMethod = "coverage"
    use_antigaming: bool = True
    gate: PrerequisiteGate = DEFAULT_GATE
    #: Embed the resume with its identifying header stripped. The name never
    #: enters the formula directly, but it is part of the document the encoder
    #: reads, so this is the mitigation the fairness audit measures against.
    anonymise: bool = False
    #: Evidence-grounding settings. Passed explicitly rather than read from the
    #: module default, because a dataclass default argument binds once at import
    #: and reassigning the global would silently have no effect.
    grounding_config: GroundingConfig = DEFAULT_GROUNDING
    embedder: Embedder | None = None
    name: str = field(default="hybrid", init=False)

    def _get_embedder(self) -> Embedder:
        if self.embedder is None:
            self.embedder = get_embedder()
        return self.embedder

    def rank_detailed(
        self, resumes: list[ParsedResume], job: JobSpec
    ) -> list[RankedCandidate]:
        """Full scoring pass returning everything needed to explain each rank."""
        if not resumes:
            return []

        embedder = self._get_embedder()
        jd_vector = embedder.encode_document(job.text)
        encode_texts = [
            fairness.anonymise_text(r) if self.anonymise else r.text for r in resumes
        ]
        resume_vectors = embedder.encode_documents(encode_texts)

        # Density is judged against this batch, so "unusually dense" means
        # unusual for this applicant pool rather than against a fixed constant.
        stats = antigaming.corpus_stats(resumes) if self.use_antigaming else None

        # Grounding is computed only for the measure that consumes it, so the
        # benchmark can attribute any difference to evidence weighting rather
        # than to an incidental change in what else got computed.
        want_grounding = self.skill_method == "grounded"

        ranked: list[RankedCandidate] = []
        for resume, vector in zip(resumes, resume_vectors):
            if self.use_antigaming:
                report = antigaming.analyse(
                    resume,
                    job=job,
                    embedder=embedder,
                    config=self.antigaming_config,
                    stats=stats,
                )
            else:
                report = antigaming.GamingReport(signals=(), penalty=0.0, flagged=False)

            report_grounding = (
                grounding_mod.ground_skills(
                    resume,
                    resume.skills_expanded,
                    embedder=embedder,
                    config=self.grounding_config,
                )
                if want_grounding
                else None
            )

            scores = score_candidate(
                resume,
                job,
                resume_vector=vector,
                jd_vector=jd_vector,
                weights=self.weights,
                gaming=report,
                skill_method=self.skill_method,
                gate=self.gate,
                grounding=report_grounding,
                grounding_config=self.grounding_config,
            )
            ranked.append(
                RankedCandidate(
                    resume=resume,
                    scores=scores,
                    gaming=report,
                    gap=skillgap.analyse(resume, job),
                    grounding=report_grounding,
                )
            )

        ranked.sort(key=lambda c: (-c.scores.total, c.candidate_id))
        for position, candidate in enumerate(ranked, start=1):
            candidate.rank = position
        return ranked

    def score_all(self, resumes: list[ParsedResume], job: JobSpec) -> dict[str, float]:
        return {c.candidate_id: c.scores.total for c in self.rank_detailed(resumes, job)}


# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------


def reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]], k: int = 60
) -> list[tuple[str, float]]:
    """Combine several ranked lists via Reciprocal Rank Fusion.

    RRF uses only rank position, never the raw score, which is exactly what we
    want when fusing a bounded cosine with an unbounded BM25 - there is no
    score calibration to get wrong. ``k=60`` is the standard constant from the
    original RRF paper and damps the influence of the very top positions.
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for position, (candidate_id, _) in enumerate(ranking, start=1):
            fused[candidate_id] = fused.get(candidate_id, 0.0) + 1.0 / (k + position)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


@dataclass
class FusionRanker(BaseRanker):
    """Reciprocal-rank fusion over several component rankers (Phase 2 of the study)."""

    rankers: list[BaseRanker]
    k: int = 60
    name: str = field(default="fusion", init=False)

    def rank(self, resumes: list[ParsedResume], job: JobSpec) -> list[tuple[str, float]]:
        return reciprocal_rank_fusion(
            [r.rank(resumes, job) for r in self.rankers], k=self.k
        )

    def score_all(self, resumes: list[ParsedResume], job: JobSpec) -> dict[str, float]:
        return dict(self.rank(resumes, job))


# --------------------------------------------------------------------------
# Timing helper
# --------------------------------------------------------------------------


@dataclass
class TimedRanking:
    method: str
    ranking: list[tuple[str, float]]
    elapsed_s: float

    @property
    def ms_per_resume(self) -> float:
        return self.elapsed_s * 1000.0 / max(1, len(self.ranking))


def timed_rank(
    ranker: BaseRanker, resumes: list[ParsedResume], job: JobSpec
) -> TimedRanking:
    """Rank while measuring wall-clock latency, for the benchmark table."""
    start = time.perf_counter()
    ranking = ranker.rank(resumes, job)
    return TimedRanking(
        method=ranker.name, ranking=ranking, elapsed_s=time.perf_counter() - start
    )
