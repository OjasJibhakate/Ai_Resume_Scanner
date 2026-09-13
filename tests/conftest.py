import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from talentlens.schemas import (  # noqa: E402
    DegreeLevel,
    JobSpec,
    ParsedDocument,
    ParsedResume,
    TextSpan,
)


@pytest.fixture(scope="session")
def ontology():
    from talentlens.ontology import get_ontology

    return get_ontology()


def make_resume(
    text: str = "",
    skills: set[str] | None = None,
    years: float = 0.0,
    degree: DegreeLevel = DegreeLevel.NONE,
    sections: dict[str, str] | None = None,
    spans: list[TextSpan] | None = None,
    candidate_id: str = "c1",
    backgrounds: list[int] | None = None,
) -> ParsedResume:
    """Build a ParsedResume directly, bypassing parsing and extraction.

    Lets the scoring and detector tests state their inputs exactly instead of
    reverse-engineering resume text that happens to extract the right things.
    """
    from talentlens.ontology import get_ontology

    document = ParsedDocument(
        path=f"{candidate_id}.txt",
        text=text,
        source_format="txt",
        spans=spans or [],
        page_backgrounds=backgrounds or [0xFFFFFF],
    )
    skills = skills or set()
    return ParsedResume(
        document=document,
        candidate_id=candidate_id,
        sections=sections or {},
        skills=set(skills),
        skills_expanded=get_ontology().expand(skills),
        years_experience=years,
        degree=degree,
    )


def make_job(
    required: set[str] | None = None,
    preferred: set[str] | None = None,
    min_years: float = 0.0,
    min_degree: DegreeLevel = DegreeLevel.NONE,
    text: str = "job description",
) -> JobSpec:
    return JobSpec(
        title="Test Role",
        text=text,
        required_skills=set(required or ()),
        preferred_skills=set(preferred or ()),
        min_years=min_years,
        min_degree=min_degree,
    )
