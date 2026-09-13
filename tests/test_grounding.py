"""Evidence-grounded skill verification.

These tests pin the properties the research claim depends on, so that a future
change cannot quietly invalidate the paper's result:

* evidence comes from the work narrative, never from the skills list
* evidence flows UP the ontology (Django evidences Python) and never down
* grounded coverage is a strict refinement of plain coverage
* the floor keeps unevidenced-but-honest claims from being scored at zero
"""

import pytest
from conftest import make_resume

from talentlens.config import GroundingConfig
from talentlens.grounding import (
    ABSENT,
    EXPLICIT,
    LISTED_ONLY,
    GroundingReport,
    SkillEvidence,
    ground_skills,
    grounded_coverage,
    narrative_sentences,
)
from talentlens.scoring import coverage_score

NARRATIVE = (
    "Built payment services in Python on PostgreSQL.\n"
    "Introduced Docker to the deployment pipeline, halving release time.\n"
    "Led a team of five engineers through a full delivery cycle."
)


def resume_with(skills, narrative=NARRATIVE, skills_section="") -> object:
    return make_resume(
        text=narrative,
        skills=set(skills),
        sections={"experience": narrative, "skills": skills_section},
    )


class TestNarrativeExtraction:
    def test_sentences_come_from_the_experience_section(self):
        resume = resume_with({"python"})
        sentences = narrative_sentences(resume)
        assert any("payment services" in s for s in sentences)

    def test_skills_section_is_not_evidence(self):
        """The claim under test must never be able to support itself."""
        resume = make_resume(
            text="x",
            skills={"kubernetes"},
            sections={"experience": NARRATIVE, "skills": "Kubernetes, Kafka, Terraform"},
        )
        sentences = narrative_sentences(resume)
        assert not any("Kubernetes" in s for s in sentences)

    def test_falls_back_when_no_experience_section(self):
        """An unsegmented resume must not report every skill as unevidenced."""
        resume = make_resume(text=NARRATIVE, skills={"python"}, sections={})
        assert narrative_sentences(resume)


class TestLexicalGrounding:
    def test_named_skill_is_explicitly_grounded(self):
        report = ground_skills(resume_with({"python"}), ["python"], use_semantic=False)
        evidence = report.get("python")
        assert evidence.kind == EXPLICIT
        assert evidence.grounding == 1.0
        assert "Python" in evidence.evidence

    def test_unnarrated_skill_is_listed_only(self):
        report = ground_skills(resume_with({"kafka"}), ["kafka"], use_semantic=False)
        assert report.get("kafka").kind == LISTED_ONLY
        assert report.get("kafka").grounding == 0.0

    def test_descendant_evidences_ancestor(self):
        """A sentence about Django is evidence of Python."""
        narrative = "Built internal tools with Django and deployed them weekly."
        report = ground_skills(
            resume_with({"python"}, narrative=narrative), ["python"], use_semantic=False
        )
        assert report.get("python").kind == EXPLICIT

    def test_ancestor_does_not_evidence_descendant(self):
        """Mentioning Python is NOT evidence of Django - the asymmetry matters."""
        narrative = "Wrote a lot of Python across several services."
        report = ground_skills(
            resume_with({"django"}, narrative=narrative), ["django"], use_semantic=False
        )
        assert report.get("django").kind != EXPLICIT

    def test_alias_is_recognised(self):
        narrative = "Shipped the dashboard in ReactJS with a shared component library."
        report = ground_skills(
            resume_with({"react"}, narrative=narrative), ["react"], use_semantic=False
        )
        assert report.get("react").kind == EXPLICIT

    def test_word_boundaries_are_respected(self):
        """'go' must not be grounded by 'algorithms' or 'going'."""
        narrative = "Improved algorithms and kept the project going through delivery."
        report = ground_skills(
            resume_with({"go"}, narrative=narrative), ["go"], use_semantic=False
        )
        assert report.get("go").kind != EXPLICIT

    def test_unknown_skill_is_reported_not_crashed(self):
        report = ground_skills(
            resume_with({"quantum-basket-weaving"}),
            ["quantum-basket-weaving"],
            use_semantic=False,
        )
        assert report.get("quantum-basket-weaving").kind == LISTED_ONLY


class TestReport:
    def test_grounded_fraction(self):
        report = ground_skills(
            resume_with({"python", "docker", "kafka", "terraform"}),
            ["python", "docker", "kafka", "terraform"],
            use_semantic=False,
        )
        # python and docker are narrated; kafka and terraform are not.
        assert report.grounded_fraction == pytest.approx(0.5)

    def test_empty_claims_are_vacuously_grounded(self):
        report = ground_skills(resume_with(set()), [], use_semantic=False)
        assert report.grounded_fraction == 1.0

    def test_missing_skill_reports_absent(self):
        report = GroundingReport(evidence={})
        assert report.get("anything").kind == ABSENT
        assert report.strength("anything") == 0.0

    def test_reliability_needs_narrative(self):
        assert not GroundingReport(evidence={}, sentence_count=1).reliable
        assert GroundingReport(evidence={}, sentence_count=6).reliable


class TestGroundedCoverage:
    FULL = GroundingReport(
        evidence={
            "python": SkillEvidence("python", 1.0, EXPLICIT, "evidence"),
            "docker": SkillEvidence("docker", 1.0, EXPLICIT, "evidence"),
        },
        sentence_count=5,
    )
    NONE = GroundingReport(
        evidence={
            "python": SkillEvidence("python", 0.0, LISTED_ONLY),
            "docker": SkillEvidence("docker", 0.0, LISTED_ONLY),
        },
        sentence_count=5,
    )

    def test_fully_evidenced_scores_one(self):
        score = grounded_coverage({"python", "docker"}, {"python", "docker"}, report=self.FULL)
        assert score == pytest.approx(1.0)

    def test_unevidenced_falls_to_the_floor_not_zero(self):
        """An honest resume that lists without narrating is docked, not erased."""
        config = GroundingConfig(floor=0.4)
        score = grounded_coverage(
            {"python", "docker"}, {"python", "docker"}, report=self.NONE, config=config
        )
        assert score == pytest.approx(0.4)

    def test_evidence_is_worth_more_than_assertion(self):
        """The core claim of the method, as an inequality."""
        evidenced = grounded_coverage({"python", "docker"}, {"python", "docker"}, report=self.FULL)
        asserted = grounded_coverage({"python", "docker"}, {"python", "docker"}, report=self.NONE)
        assert evidenced > asserted

    def test_unmatched_requirement_gets_nothing(self):
        """Grounding never rescues a skill the candidate does not claim at all."""
        score = grounded_coverage({"python"}, {"python", "bgp"}, report=self.FULL)
        assert score == pytest.approx(0.5)

    def test_collapses_to_plain_coverage_at_floor_one(self):
        """Strict refinement: floor=1.0 must reproduce coverage_score exactly."""
        config = GroundingConfig(floor=1.0)
        held, required = {"python", "docker"}, {"python", "docker", "bgp"}
        grounded = grounded_coverage(held, required, report=self.NONE, config=config)
        assert grounded == pytest.approx(coverage_score(held, required))

    def test_no_report_behaves_like_plain_coverage(self):
        held, required = {"python"}, {"python", "bgp"}
        assert grounded_coverage(held, required, report=None) == pytest.approx(
            coverage_score(held, required)
        )

    def test_no_requirements_is_vacuously_satisfied(self):
        assert grounded_coverage({"python"}, set(), report=self.FULL) == 1.0

    def test_preferred_counts_half(self):
        score = grounded_coverage(
            {"python"}, required={"python"}, preferred={"docker"}, report=self.FULL
        )
        assert score == pytest.approx(1.0 / 1.5)


class TestConfig:
    def test_rejects_out_of_range_floor(self):
        with pytest.raises(ValueError, match="floor"):
            GroundingConfig(floor=1.5)

    def test_rejects_inverted_thresholds(self):
        with pytest.raises(ValueError, match="ceiling"):
            GroundingConfig(semantic_threshold=0.6, semantic_ceiling=0.3)

    def test_rejects_out_of_range_semantic_max(self):
        with pytest.raises(ValueError, match="semantic_max"):
            GroundingConfig(semantic_max=2.0)
