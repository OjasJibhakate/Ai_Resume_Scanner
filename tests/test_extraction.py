"""Extraction: sections, experience arithmetic, degrees, contact, JD requirements."""

import pytest

from talentlens.extraction import (
    build_job_spec,
    extract_contact,
    extract_degree,
    extract_experience_years,
    extract_skills,
    segment_sections,
)
from talentlens.schemas import DegreeLevel

RESUME = """Priya Raman
priya.raman@example.com | +91 98765 43210 | Bengaluru

SUMMARY
Backend engineer focused on distributed systems.

EXPERIENCE
Senior Backend Engineer at Zeta Systems, Mar 2021 - Present
  Built payment services with Django and PostgreSQL.

EDUCATION
M.Tech in Computer Science, IIT Madras, 2016 - 2018

SKILLS
Python, Django, PostgreSQL, Docker, Kubernetes, AWS, Go, Git
"""


class TestSections:
    def test_recognises_standard_headings(self):
        sections = segment_sections(RESUME)
        assert {"summary", "experience", "education", "skills"} <= set(sections)

    def test_content_lands_in_the_right_section(self):
        sections = segment_sections(RESUME)
        assert "Zeta Systems" in sections["experience"]
        assert "M.Tech" in sections["education"]

    def test_preamble_goes_to_other(self):
        sections = segment_sections(RESUME)
        assert "priya.raman@example.com" in sections["other"]

    def test_long_lines_are_not_headings(self):
        text = "This line mentions experience but is far too long to be a heading\nbody"
        assert "experience" not in segment_sections(text)


class TestExperienceYears:
    def test_single_range(self):
        # Jan 2018 to Dec 2020 inclusive is three full years.
        assert extract_experience_years("Jan 2018 - Dec 2020") == pytest.approx(3.0)

    def test_bare_years_span_full_calendar_years(self):
        assert extract_experience_years("2018 - 2020") == pytest.approx(3.0)

    def test_concurrent_roles_are_not_double_counted(self):
        """Two overlapping jobs are still one stretch of career time."""
        text = "Role A, Jan 2018 - Dec 2020\nRole B, Jan 2019 - Dec 2020"
        assert extract_experience_years(text) == pytest.approx(3.0)

    def test_sequential_roles_add_up(self):
        text = "Role A, Jan 2016 - Dec 2017\nRole B, Jan 2018 - Dec 2019"
        assert extract_experience_years(text) == pytest.approx(4.0)

    def test_present_is_resolved_to_today(self):
        assert extract_experience_years("Jan 2020 - Present") > 4.0

    def test_dates_beat_self_reported_claims(self):
        text = "20+ years of experience\nRole A, Jan 2022 - Dec 2022"
        assert extract_experience_years(text) == pytest.approx(1.0)

    def test_self_report_used_only_when_no_dates_parse(self):
        assert extract_experience_years("7 years of experience") == pytest.approx(7.0)

    def test_no_signal_returns_zero(self):
        assert extract_experience_years("no dates here at all") == 0.0

    def test_reversed_range_is_ignored(self):
        assert extract_experience_years("Jan 2020 - Dec 2015") == 0.0


class TestDegree:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("PhD in Computer Science", DegreeLevel.DOCTORATE),
            ("M.Tech in Computer Science", DegreeLevel.MASTER),
            ("MSc Physics", DegreeLevel.MASTER),
            ("MBA", DegreeLevel.MASTER),
            ("B.Tech Information Technology", DegreeLevel.BACHELOR),
            ("Bachelor of Science", DegreeLevel.BACHELOR),
            ("Diploma in IT", DegreeLevel.ASSOCIATE),
            ("Self taught developer", DegreeLevel.NONE),
        ],
    )
    def test_degree_levels(self, text, expected):
        assert extract_degree(text) == expected

    def test_highest_degree_wins(self):
        assert extract_degree("B.Tech 2016\nM.Tech 2018") == DegreeLevel.MASTER

    def test_ordering_is_meaningful(self):
        assert DegreeLevel.DOCTORATE > DegreeLevel.MASTER > DegreeLevel.BACHELOR


class TestContact:
    def test_extracts_name_email_phone(self):
        name, email, phone = extract_contact(RESUME)
        assert name == "Priya Raman"
        assert email == "priya.raman@example.com"
        assert phone

    def test_name_never_bleeds_into_the_next_line(self):
        """spaCy will happily return a PERSON spanning a newline."""
        name, _, _ = extract_contact("Alex Chen\nalex@example.com\n")
        assert name == "Alex Chen"

    def test_missing_contact_details_are_empty_not_wrong(self):
        name, email, phone = extract_contact("EXPERIENCE\nSome role somewhere")
        assert email == ""
        assert phone == ""


class TestSkills:
    def test_finds_skills_in_a_list(self):
        skills = extract_skills(RESUME, segment_sections(RESUME))
        assert {"python", "django", "postgresql", "docker", "kubernetes", "aws"} <= skills

    def test_ambiguous_name_accepted_inside_the_skills_section(self):
        skills = extract_skills(RESUME, segment_sections(RESUME))
        assert "go" in skills

    def test_ambiguous_name_rejected_in_ordinary_prose(self):
        """'go to market' must not register the Go language."""
        text = "EXPERIENCE\nI go to market meetings and help teams excel at delivery."
        assert "go" not in extract_skills(text, segment_sections(text))

    def test_longest_match_wins(self):
        text = "SKILLS\nMachine Learning, Deep Learning"
        skills = extract_skills(text, segment_sections(text))
        assert "machine-learning" in skills
        assert "deep-learning" in skills


class TestJobSpec:
    JD = """Senior Backend Engineer

Requirements:
- 5+ years of professional experience
- Strong Python and PostgreSQL
- Bachelor degree in Computer Science

Preferred:
- Kubernetes and AWS
"""

    def test_splits_required_from_preferred(self):
        job = build_job_spec(self.JD)
        assert {"python", "postgresql"} <= job.required_skills
        assert {"kubernetes", "aws"} <= job.preferred_skills

    def test_preferred_and_required_do_not_overlap(self):
        job = build_job_spec(self.JD)
        assert not (job.required_skills & job.preferred_skills)

    def test_reads_minimum_years(self):
        assert build_job_spec(self.JD).min_years == pytest.approx(5.0)

    def test_reads_minimum_degree(self):
        assert build_job_spec(self.JD).min_degree == DegreeLevel.BACHELOR

    def test_explicit_overrides_beat_the_text(self):
        job = build_job_spec(self.JD, min_years=2.0, min_degree=DegreeLevel.MASTER)
        assert job.min_years == 2.0
        assert job.min_degree == DegreeLevel.MASTER

    def test_without_a_preferred_marker_everything_is_required(self):
        """Safer to over-require than to silently downgrade a hard requirement."""
        job = build_job_spec("Requirements:\n- Python\n- Docker")
        assert job.preferred_skills == set()
        assert {"python", "docker"} <= job.required_skills
