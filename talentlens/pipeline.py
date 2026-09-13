"""End-to-end screening: files in, ranked and explained candidates out."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from .config import DEFAULT_GATE, DEFAULT_WEIGHTS, PrerequisiteGate, ScoringWeights
from .extraction import build_job_spec, build_resume
from .parsing import ParsingError, parse_bytes, parse_file
from .ranking import HybridRanker
from .schemas import DegreeLevel, JobSpec, ParsedResume, RankedCandidate
from .scoring import SkillMethod


def load_resume(path: str | Path, candidate_id: str | None = None) -> ParsedResume:
    return build_resume(parse_file(path), candidate_id=candidate_id)


def load_resumes(
    paths: Iterable[str | Path], skip_errors: bool = True
) -> tuple[list[ParsedResume], list[tuple[str, str]]]:
    """Parse and extract a batch of resumes.

    Returns ``(resumes, failures)``. Failures are reported rather than raised so
    that one corrupt PDF in a folder of eighty does not abort the whole run -
    the caller decides whether to surface them.
    """
    resumes: list[ParsedResume] = []
    failures: list[tuple[str, str]] = []
    for path in paths:
        try:
            resumes.append(load_resume(path))
        except (ParsingError, OSError) as exc:
            if not skip_errors:
                raise
            failures.append((str(path), str(exc)))
    return resumes, failures


def load_resume_uploads(
    uploads: Sequence[tuple[str, bytes]], skip_errors: bool = True
) -> tuple[list[ParsedResume], list[tuple[str, str]]]:
    """Same as :func:`load_resumes` for in-memory (filename, bytes) pairs."""
    resumes: list[ParsedResume] = []
    failures: list[tuple[str, str]] = []
    for filename, data in uploads:
        try:
            resumes.append(build_resume(parse_bytes(data, filename), candidate_id=filename))
        except (ParsingError, OSError) as exc:
            if not skip_errors:
                raise
            failures.append((filename, str(exc)))
    return resumes, failures


def build_job(
    text: str,
    title: str = "",
    min_years: float | None = None,
    min_degree: DegreeLevel | None = None,
) -> JobSpec:
    return build_job_spec(text, title=title, min_years=min_years, min_degree=min_degree)


def screen(
    resumes: list[ParsedResume],
    job: JobSpec,
    weights: ScoringWeights = DEFAULT_WEIGHTS,
    skill_method: SkillMethod = "coverage",
    use_antigaming: bool = True,
    gate: PrerequisiteGate = DEFAULT_GATE,
    explainer=None,
    explain_top_k: int = 5,
) -> list[RankedCandidate]:
    """Rank candidates and attach explanations to the top of the list.

    ``explain_top_k`` exists because of the limitation Paper 4 reports: running
    a generative model over every resume dominates latency. Ranking is cheap and
    happens for everyone; prose costs a network round trip and is only worth it
    for the shortlist a human will actually read. Pass ``explain_top_k=0`` to
    skip explanations entirely, or a large number to explain everyone.
    """
    ranker = HybridRanker(
        weights=weights,
        skill_method=skill_method,
        use_antigaming=use_antigaming,
        gate=gate,
    )
    ranked = ranker.rank_detailed(resumes, job)

    if explainer is not None and explain_top_k > 0:
        for candidate in ranked[:explain_top_k]:
            candidate.explanation = explainer.explain(candidate, job)

    return ranked


def screen_files(
    resume_paths: Iterable[str | Path],
    job_text: str,
    job_title: str = "",
    **kwargs,
) -> tuple[list[RankedCandidate], list[tuple[str, str]]]:
    """Convenience wrapper: paths and JD text straight to a ranked shortlist."""
    resumes, failures = load_resumes(resume_paths)
    job = build_job(job_text, title=job_title)
    return screen(resumes, job, **kwargs), failures
