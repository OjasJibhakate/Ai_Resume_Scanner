"""Anti-gaming detector: each signal fires on its attack and stays quiet otherwise."""

import pytest
from conftest import make_job, make_resume

from talentlens.antigaming import (
    CorpusStats,
    analyse,
    detect_invisible_text,
    detect_jd_echo,
    detect_repetition,
    detect_skill_density,
    skill_density,
)
from talentlens.config import AntiGamingConfig
from talentlens.schemas import TextSpan

CONFIG = AntiGamingConfig()

#: Long enough to clear the detector's minimum-trigram guard, which exists so a
#: two-line job description cannot produce a spurious echo ratio.
LONG_JD = (
    "We are hiring a senior backend engineer to design and operate resilient "
    "payment services across a distributed platform team with strong ownership "
    "of reliability and performance outcomes. You will build and run services "
    "in production, review designs with peers, and partner with product "
    "managers to agree quarterly delivery milestones for the wider group. "
    "The role requires deep familiarity with relational databases, container "
    "orchestration, and continuous delivery pipelines used across the company."
)


def spans(*specs) -> list[TextSpan]:
    return [
        TextSpan(text=text, page=0, size=size, color=color)
        for text, size, color in specs
    ]


class TestInvisibleText:
    def test_white_on_white_is_caught(self):
        resume = make_resume(
            spans=spans(
                ("Normal visible resume content here", 10.0, 0x000000),
                ("python docker kubernetes aws", 10.0, 0xFFFFFF),
            )
        )
        signal = detect_invisible_text(resume, CONFIG)
        assert signal.triggered
        assert signal.severity > 0
        assert "ffffff" in signal.evidence

    def test_sub_readable_font_is_caught(self):
        resume = make_resume(
            spans=spans(
                ("Normal visible resume content here", 10.0, 0x000000),
                ("python docker kubernetes aws", 1.0, 0x000000),
            )
        )
        signal = detect_invisible_text(resume, CONFIG)
        assert signal.triggered
        assert "1.0pt" in signal.evidence

    def test_ordinary_resume_is_clean(self):
        resume = make_resume(
            spans=spans(
                ("Priya Raman", 14.0, 0x000000),
                ("Backend engineer with eight years of experience", 10.0, 0x333333),
            )
        )
        assert not detect_invisible_text(resume, CONFIG).triggered

    def test_dark_grey_on_white_is_not_flagged(self):
        """Legitimate styling must not read as an attack."""
        resume = make_resume(spans=spans(("Subtle grey subheading text", 10.0, 0x444444)))
        assert not detect_invisible_text(resume, CONFIG).triggered

    def test_formats_without_span_metadata_abstain(self):
        resume = make_resume(text="plain text resume", spans=[])
        signal = detect_invisible_text(resume, CONFIG)
        assert not signal.triggered
        assert "no span metadata" in signal.evidence

    def test_respects_a_non_white_page_background(self):
        """Black text on a black page is just as hidden as white on white."""
        resume = make_resume(
            spans=spans(("hidden keywords here", 10.0, 0x000000)),
            backgrounds=[0x000000],
        )
        assert detect_invisible_text(resume, CONFIG).triggered


class TestRepetition:
    def test_hammered_keyword_is_caught(self):
        # Filler must be genuinely varied, or it becomes the most repeated term
        # and the assertion tests the wrong thing. Note the detector's tokeniser
        # drops digits, so "filler1"/"filler2" would all collapse to "filler".
        import itertools

        filler = " ".join(
            "".join(letters)
            for letters in itertools.islice(itertools.product("abcdefgh", repeat=3), 150)
        )
        body = " ".join(["kubernetes"] * 30) + " " + filler
        signal = detect_repetition(make_resume(text=body), CONFIG)
        assert signal.triggered
        assert "kubernetes" in signal.evidence

    def test_natural_prose_is_clean(self):
        body = (
            "Designed payment services and reviewed architecture proposals. "
            "Mentored engineers, improved deployment reliability, and reduced "
            "latency across the checkout path. Partnered with product managers "
            "to define quarterly roadmaps and delivery milestones for the team. "
        ) * 4
        assert not detect_repetition(make_resume(text=body), CONFIG).triggered

    def test_short_documents_abstain(self):
        """A rate per 1000 words estimated from 50 words is noise, not evidence."""
        resume = make_resume(text="docker docker docker docker docker")
        signal = detect_repetition(resume, CONFIG)
        assert not signal.triggered
        assert "too short" in signal.evidence


class TestSkillDensity:
    def test_density_uses_raw_not_expanded_skills(self):
        resume = make_resume(text=" ".join(["word"] * 100), skills={"django"})
        # django expands to include python; density must not count the ancestor.
        assert skill_density(resume) == pytest.approx(1 / 100)

    def test_dense_keyword_dump_is_caught(self):
        resume = make_resume(
            text=" ".join(["word"] * 100), skills={f"skill{i}" for i in range(40)}
        )
        assert detect_skill_density(resume, CONFIG).triggered

    def test_normal_resume_is_clean(self):
        resume = make_resume(
            text=" ".join(["word"] * 400), skills={f"skill{i}" for i in range(12)}
        )
        assert not detect_skill_density(resume, CONFIG).triggered

    def test_corpus_stats_need_enough_samples(self):
        assert not CorpusStats.from_densities([0.1, 0.2]).usable
        assert CorpusStats.from_densities([0.1 + i * 0.01 for i in range(10)]).usable


class TestJdEcho:
    def test_pasted_job_description_is_caught(self):
        jd = LONG_JD
        resume = make_resume(text="My background. " + jd)
        signal = detect_jd_echo(resume, make_job(text=jd), CONFIG)
        assert signal.triggered

    def test_shared_vocabulary_alone_does_not_trigger(self):
        jd = LONG_JD
        resume = make_resume(
            text=(
                "Backend engineer. Built payment services and improved reliability. "
                "Owned performance work on a distributed platform. Python, Docker."
            )
        )
        assert not detect_jd_echo(resume, make_job(text=jd), CONFIG).triggered

    def test_no_job_supplied_abstains(self):
        assert not detect_jd_echo(make_resume(text="x"), None, CONFIG).triggered

    def test_short_job_description_abstains(self):
        signal = detect_jd_echo(make_resume(text="x"), make_job(text="short jd"), CONFIG)
        assert not signal.triggered
        assert "too short" in signal.evidence


class TestAggregation:
    def test_clean_resume_gets_no_penalty(self):
        resume = make_resume(
            text="Backend engineer with production experience. " * 12,
            spans=spans(("Backend engineer", 10.0, 0x000000)),
            skills={"python", "docker"},
        )
        report = analyse(resume, job=None, use_semantic=False)
        assert not report.flagged
        assert report.penalty == 0.0

    def test_penalty_never_exceeds_the_cap(self):
        body = " ".join(["kubernetes"] * 200)
        resume = make_resume(
            text=body,
            spans=spans((body, 1.0, 0xFFFFFF)),
            skills={f"skill{i}" for i in range(60)},
        )
        report = analyse(resume, job=make_job(text=body), use_semantic=False)
        assert report.flagged
        assert report.penalty <= CONFIG.max_penalty + 1e-9

    def test_every_signal_is_reported_even_when_quiet(self):
        report = analyse(make_resume(text="short"), use_semantic=False)
        assert len(report.signals) == 5
        assert all(s.evidence for s in report.signals)

    def test_disabling_semantic_check_skips_the_embedder(self):
        """No model load, so this must not raise even with no network."""
        report = analyse(make_resume(text="short"), use_semantic=False)
        assert report.signal("incoherence").evidence == "semantic check disabled"


def test_signal_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        AntiGamingConfig(signal_weights=(0.5, 0.5, 0.5, 0.5, 0.5))


def test_signal_weights_must_have_five_entries():
    with pytest.raises(ValueError, match="5 entries"):
        AntiGamingConfig(signal_weights=(1.0,))
