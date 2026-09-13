"""Structured extraction: sections, skills, experience, education, contact.

Three decisions here are worth understanding before changing anything:

1. **Ambiguous skill names.** Canonical skills like ``c``, ``r``, ``go`` and
   ``excel`` are also ordinary English words. Matching them anywhere in free
   text produces constant false positives ("go to market", "excel at"). Those
   names are therefore only honoured inside a skills-like section or in an
   obvious delimited list. Precision matters more than recall here: a spurious
   skill inflates Dice overlap and promotes the wrong candidate.

2. **Experience is computed from date ranges, not from claims.** A resume
   saying "10+ years of experience" is self-reported; overlapping employment
   spans are evidence. We merge the spans so two concurrent roles do not double
   count, and only fall back to the self-reported figure when no dates parse.

3. **Degree detection takes the maximum tier found**, since resumes list every
   qualification and the highest is the one that matters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from typing import Iterable

from .config import SPACY_MODEL
from .ontology import SkillOntology, get_ontology
from .schemas import (
    DegreeLevel,
    ExperienceEntry,
    JobSpec,
    ParsedDocument,
    ParsedResume,
)

# --------------------------------------------------------------------------
# spaCy pipeline (lazy, cached)
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _nlp():
    """Load spaCy once per process.

    The parser and lemmatiser are disabled: we need the tokeniser for phrase
    matching and the NER only for PERSON names, and dropping the rest roughly
    triples throughput on a screening batch.
    """
    import spacy

    try:
        return spacy.load(SPACY_MODEL, disable=["parser", "lemmatizer", "tagger"])
    except OSError as exc:  # pragma: no cover - depends on model install
        raise RuntimeError(
            f"spaCy model {SPACY_MODEL!r} is not installed. "
            f"Run: python -m spacy download {SPACY_MODEL}"
        ) from exc


@lru_cache(maxsize=1)
def _phrase_matcher() -> tuple[object, dict[int, str]]:
    """Build a PhraseMatcher over every ontology surface form."""
    from spacy.matcher import PhraseMatcher

    nlp = _nlp()
    ontology = get_ontology()
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

    label_to_skill: dict[int, str] = {}
    by_skill: dict[str, list[str]] = {}
    for surface, canonical in ontology.surface_forms():
        by_skill.setdefault(canonical, []).append(surface)

    for canonical, surfaces in by_skill.items():
        patterns = [nlp.make_doc(s) for s in surfaces]
        matcher.add(canonical, patterns)
        label_to_skill[nlp.vocab.strings[canonical]] = canonical

    return matcher, label_to_skill


# --------------------------------------------------------------------------
# Section segmentation
# --------------------------------------------------------------------------

#: Canonical section -> header keywords that introduce it.
_SECTION_PATTERNS: dict[str, tuple[str, ...]] = {
    "summary": (
        "summary",
        "objective",
        "profile",
        "about me",
        "professional summary",
        "career overview",
        "professional overview",
        "highlights",
        "core qualifications",
        "career focus",
    ),
    "experience": (
        "experience",
        "work experience",
        "professional experience",
        "employment",
        "employment history",
        "work history",
        "career history",
        "professional background",
        "accomplishments",
        "experience highlights",
    ),
    "education": (
        "education",
        "academic background",
        "academics",
        "qualifications",
        "education and training",
        "educational background",
    ),
    "skills": (
        "skills",
        "technical skills",
        "core skills",
        "technologies",
        "technical proficiencies",
        "competencies",
        "tech stack",
        "skill highlights",
        "areas of expertise",
        "key skills",
        "technical expertise",
    ),
    "projects": ("projects", "personal projects", "selected projects", "portfolio"),
    "certifications": (
        "certifications",
        "certificates",
        "licenses",
        "courses",
        "training",
    ),
}

#: A header line is short, mostly non-punctuation, and matches a keyword.
_MAX_HEADER_WORDS = 5

#: Real-world resume exports frequently put section headings *inline*, separated
#: from surrounding text by runs of spaces rather than newlines:
#:
#:     "HR ADMINISTRATOR       Summary     Dedicated Customer Service Manager..."
#:
#: A purely line-based segmenter finds no headings at all in those documents and
#: silently returns one undifferentiated blob. That failure is quiet and
#: catastrophic for evidence grounding: with no EXPERIENCE section to read,
#: the narrative falls back to the whole document *including the skills list*,
#: so every claimed skill grounds itself and the signal disappears. Measured on
#: 40 real IT resumes, line-only segmentation detected an experience section in
#: 0 of them and reported a grounded fraction of 1.000 for every candidate.
#:
#: So before segmenting we normalise inline headings onto their own lines.
_INLINE_HEADER_KEYWORDS = sorted(
    {kw for keywords in _SECTION_PATTERNS.values() for kw in keywords},
    key=len,
    reverse=True,
)

#: A heading is recognised inline when it is bounded by a run of whitespace (or
#: a line edge) on both sides. Requiring the run on the left and at least two
#: spaces, a newline or a colon on the right keeps ordinary prose uses of words
#: like "experience" from being mistaken for a heading.
_INLINE_HEADER_RE = re.compile(
    r"(?:(?<=\n)|(?<=\s{3})|^)\s*(" + "|".join(re.escape(k) for k in _INLINE_HEADER_KEYWORDS) + r")"
    r"(?=\s{2,}|\s*:|\n|$)",
    re.IGNORECASE,
)


def normalise_headings(text: str) -> str:
    """Put inline section headings onto their own lines.

    Idempotent for documents that already place headings on separate lines, so
    it is safe to run unconditionally.
    """
    return _INLINE_HEADER_RE.sub(lambda m: "\n" + m.group(1) + "\n", text)


def _header_section(line: str) -> str | None:
    """Return the canonical section a line introduces, if it is a header."""
    stripped = line.strip().strip(":").strip()
    if not stripped or len(stripped.split()) > _MAX_HEADER_WORDS:
        return None
    folded = re.sub(r"[^a-z ]", " ", stripped.lower())
    folded = re.sub(r"\s+", " ", folded).strip()
    if not folded:
        return None
    for section, keywords in _SECTION_PATTERNS.items():
        if folded in keywords:
            return section
    return None


def segment_sections(text: str) -> dict[str, str]:
    """Split resume text into canonical sections.

    Everything before the first recognised header lands in ``other``, which is
    where the contact block normally lives.
    """
    text = normalise_headings(text)
    sections: dict[str, list[str]] = {}
    current = "other"
    for line in text.splitlines():
        section = _header_section(line)
        if section is not None:
            current = section
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)
    return {name: "\n".join(lines).strip() for name, lines in sections.items()}


# --------------------------------------------------------------------------
# Skill extraction
# --------------------------------------------------------------------------

#: Canonical skills whose names collide with ordinary English. These are only
#: accepted from a skills-like section or a delimited list (see module docstring).
_AMBIGUOUS_SKILLS = frozenset(
    {
        "c",
        "r",
        "go",
        "rust",
        "swift",
        "scala",
        "excel",
        "spark",
        "hive",
        "flask",
        "lambda",
        "monitoring",
        "statistics",
        "communication",
        "leadership",
        "teamwork",
        "mentoring",
        "routing",
        "switching",
        "containers",
        "algorithms",
    }
)

#: Sections where a bare skill token is unambiguous enough to trust.
_SKILL_TRUSTED_SECTIONS = ("skills", "certifications", "projects")

_LIST_DELIMITERS = re.compile(r"[,|/•·;•]")


def _is_list_context(text: str, start: int, end: int, window: int = 60) -> bool:
    """True when the match sits inside a comma/bullet separated list."""
    left = text[max(0, start - window) : start]
    right = text[end : end + window]
    context = left + right
    return len(_LIST_DELIMITERS.findall(context)) >= 2


def extract_skills(
    text: str,
    sections: dict[str, str] | None = None,
    ontology: SkillOntology | None = None,
) -> set[str]:
    """Find canonical skills in ``text``.

    ``sections`` is optional but strongly recommended: without it, ambiguous
    skill names have no trusted region to appear in and are held to the
    list-context rule everywhere.
    """
    from spacy.util import filter_spans

    ontology = ontology or get_ontology()
    matcher, _ = _phrase_matcher()
    nlp = _nlp()

    trusted_text = ""
    if sections:
        trusted_text = "\n".join(
            sections.get(name, "") for name in _SKILL_TRUSTED_SECTIONS
        ).lower()

    doc = nlp.make_doc(text)
    matches = matcher(doc)
    spans = [doc[start:end] for _, start, end in matches]
    # Longest match wins, so "machine learning" is not also counted as "learning".
    spans = filter_spans(spans)

    found: set[str] = set()
    for span in spans:
        canonical = ontology.canonical(span.text)
        if canonical is None:
            continue
        if canonical in _AMBIGUOUS_SKILLS:
            surface = span.text.lower()
            in_trusted = bool(trusted_text) and surface in trusted_text
            if not in_trusted and not _is_list_context(
                text, span.start_char, span.end_char
            ):
                continue
        found.add(canonical)
    return found


# --------------------------------------------------------------------------
# Experience
# --------------------------------------------------------------------------

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

_MONTH_ALT = "|".join(_MONTHS)
_DATE_TOKEN = rf"(?:(?:{_MONTH_ALT})[a-z]*\.?\s*,?\s*\d{{4}}|\d{{1,2}}[/-]\d{{4}}|\d{{4}})"
_PRESENT = r"(?:present|current|currently|now|ongoing|to\s*date|till\s*date)"

_RANGE_RE = re.compile(
    rf"(?P<start>{_DATE_TOKEN})\s*(?:-|–|—|to|until|through)\s*(?P<end>{_DATE_TOKEN}|{_PRESENT})",
    re.IGNORECASE,
)

_SELF_REPORTED_RE = re.compile(
    r"(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:professional\s*|work\s*|industry\s*|relevant\s*)?experience",
    re.IGNORECASE,
)


def _to_months(token: str, *, default_month: int) -> int | None:
    """Convert a date token to an absolute month index (year*12 + month)."""
    token = token.strip().lower().replace(",", " ")
    token = re.sub(r"\s+", " ", token)

    if re.fullmatch(_PRESENT, token, re.IGNORECASE):
        today = date.today()
        return today.year * 12 + today.month

    match = re.fullmatch(rf"({_MONTH_ALT})[a-z]*\.?\s*(\d{{4}})", token)
    if match:
        return int(match.group(2)) * 12 + _MONTHS[match.group(1)]

    match = re.fullmatch(r"(\d{1,2})[/-](\d{4})", token)
    if match:
        month = min(max(int(match.group(1)), 1), 12)
        return int(match.group(2)) * 12 + month

    match = re.fullmatch(r"(\d{4})", token)
    if match:
        return int(match.group(1)) * 12 + default_month
    return None


def _from_months(index: int | None) -> tuple[int | None, int | None]:
    """Invert :func:`_to_months` back into ``(year, month)``.

    The month index is ``year * 12 + month`` with month in 1..12, so a naive
    ``index // 12`` pushes every December into the following year. The -1/+1
    dance keeps December where it belongs.
    """
    if index is None:
        return None, None
    return (index - 1) // 12, (index - 1) % 12 + 1


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping month intervals so concurrent roles count once."""
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def extract_experience_years(text: str) -> float:
    """Total non-overlapping years of experience implied by date ranges.

    Falls back to a self-reported "N years of experience" claim only when no
    date ranges parse, since dates are evidence and claims are not.
    """
    today_months = date.today().year * 12 + date.today().month
    intervals: list[tuple[int, int]] = []

    for match in _RANGE_RE.finditer(text):
        # A bare "2019" start means January; a bare "2022" end means December,
        # so that "2019 - 2022" reads as three full years rather than two.
        start = _to_months(match.group("start"), default_month=1)
        end = _to_months(match.group("end"), default_month=12)
        if start is None or end is None:
            continue
        end = min(end, today_months)
        if end < start:
            continue
        intervals.append((start, end))

    if intervals:
        total_months = sum(end - start + 1 for start, end in _merge_intervals(intervals))
        return round(total_months / 12.0, 2)

    claimed = _SELF_REPORTED_RE.search(text)
    if claimed:
        return float(claimed.group(1))
    return 0.0


_JOB_LINE_RE = re.compile(
    rf"^(?P<title>[^\n,|]{{3,60}}?)\s*(?:[,|@]|\bat\b|–|-)\s*(?P<org>[^\n,|]{{2,60}}?)\s*[,|]?\s*(?P<dates>{_DATE_TOKEN}\s*(?:-|–|—|to|until|through)\s*(?:{_DATE_TOKEN}|{_PRESENT}))",
    re.IGNORECASE | re.MULTILINE,
)


def extract_experience_entries(experience_text: str) -> list[ExperienceEntry]:
    """Parse individual employment rows out of the experience section.

    Best-effort: resume layouts vary wildly, so rows that do not match the
    common "Title at Company, dates" shape are simply not itemised. The
    aggregate year count in :func:`extract_experience_years` does not depend on
    this succeeding.
    """
    entries: list[ExperienceEntry] = []
    for match in _JOB_LINE_RE.finditer(experience_text):
        dates = match.group("dates")
        range_match = _RANGE_RE.search(dates)
        start_m = end_m = None
        is_current = False
        if range_match:
            start_m = _to_months(range_match.group("start"), default_month=1)
            end_token = range_match.group("end")
            is_current = bool(re.fullmatch(_PRESENT, end_token.strip(), re.IGNORECASE))
            end_m = _to_months(end_token, default_month=12)
        entries.append(
            ExperienceEntry(
                title=match.group("title").strip(),
                organisation=match.group("org").strip(),
                start_year=_from_months(start_m)[0],
                start_month=_from_months(start_m)[1],
                end_year=_from_months(end_m)[0],
                end_month=_from_months(end_m)[1],
                is_current=is_current,
                text=match.group(0).strip(),
            )
        )
    return entries


# --------------------------------------------------------------------------
# Education
# --------------------------------------------------------------------------

#: Ordered most-specific-first so that "master of science" is tested before
#: any looser pattern could claim the same text.
_DEGREE_PATTERNS: tuple[tuple[DegreeLevel, str], ...] = (
    (DegreeLevel.DOCTORATE, r"\b(ph\.?\s?d|doctorate|doctoral|d\.?phil)\b"),
    (
        DegreeLevel.MASTER,
        r"\b(m\.?\s?tech|m\.?\s?sc|msc|m\.?\s?s\b|master[''’]?s?|m\.?\s?b\.?\s?a|mba|m\.?\s?c\.?\s?a|m\.?\s?e\b)\b",
    ),
    (
        DegreeLevel.BACHELOR,
        r"\b(b\.?\s?tech|b\.?\s?sc|bsc|b\.?\s?s\b|bachelor[''’]?s?|b\.?\s?e\b|b\.?\s?c\.?\s?a|b\.?\s?com)\b",
    ),
    (DegreeLevel.ASSOCIATE, r"\b(associate[''’]?s?\s+degree|diploma|foundation\s+degree)\b"),
)


def extract_degree(text: str) -> DegreeLevel:
    """Highest degree tier mentioned anywhere in the text."""
    best = DegreeLevel.NONE
    for level, pattern in _DEGREE_PATTERNS:
        if level <= best:
            continue
        if re.search(pattern, text, re.IGNORECASE):
            best = level
    return best


# --------------------------------------------------------------------------
# Contact details
# --------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(
    r"(?:\+\d{1,3}[\s-]?)?(?:\(\d{2,4}\)[\s-]?)?\d{3,5}[\s-]?\d{3,4}[\s-]?\d{0,4}"
)


def extract_contact(text: str) -> tuple[str, str, str]:
    """Return ``(name, email, phone)``, any of which may be empty."""
    email_match = _EMAIL_RE.search(text)
    email = email_match.group(0) if email_match else ""

    phone = ""
    for line in text.splitlines()[:15]:
        # Skip lines that are mostly dates, which otherwise match the loose
        # phone pattern (a resume header often sits next to a date range).
        if _RANGE_RE.search(line):
            continue
        match = _PHONE_RE.search(line)
        if match and len(re.sub(r"\D", "", match.group(0))) >= 9:
            phone = match.group(0).strip()
            break

    def _clean_name(raw: str) -> str:
        """Keep only the first line and reject anything that is not name-shaped.

        spaCy happily returns a PERSON entity that runs across a newline into
        the contact block ("Alex Chen\\nalex@example.com"), so the entity text
        cannot be trusted verbatim.
        """
        first = raw.strip().splitlines()[0].strip(" ,|-") if raw.strip() else ""
        words = first.split()
        if not (1 < len(words) <= 4):
            return ""
        if _EMAIL_RE.search(first) or any(ch.isdigit() for ch in first):
            return ""
        return first

    name = ""
    head = "\n".join(text.splitlines()[:6])
    doc = _nlp()(head)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = _clean_name(ent.text)
            if name:
                break
    if not name:
        for line in text.splitlines()[:4]:
            candidate = line.strip()
            words = candidate.split()
            if (
                1 < len(words) <= 4
                and not _EMAIL_RE.search(candidate)
                and not any(ch.isdigit() for ch in candidate)
                and all(w[:1].isupper() for w in words if w)
            ):
                name = candidate
                break
    return name, email, phone


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def build_resume(
    document: ParsedDocument,
    candidate_id: str | None = None,
    ontology: SkillOntology | None = None,
) -> ParsedResume:
    """Run the full extraction pipeline over a parsed document."""
    ontology = ontology or get_ontology()
    text = document.text
    sections = segment_sections(text)

    skills = extract_skills(text, sections, ontology)
    name, email, phone = extract_contact(text)

    # Prefer the experience section for date maths so that education dates and
    # certification dates do not inflate the professional-experience figure.
    experience_text = sections.get("experience", "")
    years = extract_experience_years(experience_text or text)

    education_text = sections.get("education", "") or text
    degree = extract_degree(education_text)

    # Stem, not basename: a candidate id should not carry a file extension,
    # and downstream label joins key on the bare name.
    from pathlib import Path as _Path

    fallback_id = _Path(document.path).stem or "candidate"

    return ParsedResume(
        document=document,
        candidate_id=candidate_id or fallback_id,
        name=name,
        email=email,
        phone=phone,
        sections=sections,
        skills=skills,
        skills_expanded=ontology.expand(skills),
        years_experience=years,
        degree=degree,
        experience_entries=extract_experience_entries(experience_text),
    )


# --------------------------------------------------------------------------
# Job description requirements
# --------------------------------------------------------------------------

_PREFERRED_MARKERS = re.compile(
    r"\b(preferred|nice[\s-]to[\s-]have|bonus|plus|desirable|advantageous|good[\s-]to[\s-]have|optional)\b",
    re.IGNORECASE,
)
_REQUIRED_MARKERS = re.compile(
    r"\b(required|requirements|must[\s-]have|essential|minimum qualifications|you will need|qualifications)\b",
    re.IGNORECASE,
)

_MIN_YEARS_RE = re.compile(
    r"(?:at\s+least\s+|minimum\s+(?:of\s+)?|min\.?\s+)?(\d{1,2})\s*\+?\s*(?:-\s*\d{1,2}\s*)?(?:years?|yrs?)",
    re.IGNORECASE,
)


def _split_required_preferred(text: str) -> tuple[str, str]:
    """Split a JD into (required-ish, preferred-ish) regions.

    Uses the first 'preferred'-style heading as the boundary. When no such
    marker exists the whole document is treated as required, which is the safe
    default: it is better to under-promote a candidate for a missing skill than
    to quietly treat a hard requirement as optional.
    """
    lines = text.splitlines()
    boundary = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or len(stripped.split()) > 8:
            continue
        if _PREFERRED_MARKERS.search(stripped):
            boundary = index
            break
    if boundary is None:
        return text, ""
    return "\n".join(lines[:boundary]), "\n".join(lines[boundary:])


def build_job_spec(
    text: str,
    title: str = "",
    ontology: SkillOntology | None = None,
    min_years: float | None = None,
    min_degree: DegreeLevel | None = None,
) -> JobSpec:
    """Extract structured requirements from a job description.

    ``min_years`` and ``min_degree`` can be supplied explicitly to override the
    regex heuristics, which is what the Streamlit sidebar does - a recruiter who
    knows the bar should not have to phrase the JD in a particular way to get it
    read correctly.
    """
    ontology = ontology or get_ontology()
    required_text, preferred_text = _split_required_preferred(text)

    # The JD is treated as one trusted region: it is a curated document, so
    # ambiguous skill names are far less risky here than in free-form resumes.
    jd_sections = {"skills": text}
    required = extract_skills(required_text, jd_sections, ontology)
    preferred = extract_skills(preferred_text, jd_sections, ontology) if preferred_text else set()
    preferred -= required

    if min_years is None:
        match = _MIN_YEARS_RE.search(text)
        min_years = float(match.group(1)) if match else 0.0

    if min_degree is None:
        min_degree = extract_degree(text)

    if not title:
        first_line = next((l.strip() for l in text.splitlines() if l.strip()), "")
        title = first_line[:80]

    return JobSpec(
        title=title,
        text=text,
        required_skills=required,
        preferred_skills=preferred,
        required_skills_expanded=ontology.expand(required),
        min_years=float(min_years),
        min_degree=min_degree,
    )
