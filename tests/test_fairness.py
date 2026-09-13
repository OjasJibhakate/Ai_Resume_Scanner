"""Fairness auditing utilities.

These pin the properties the audit's validity rests on. Two of them exist
because the first version of the audit was wrong in exactly that way:

* ``implicit_rewrite`` must REMOVE technology names from the narrative while
  leaving the skills list intact. An earlier perturbation shortened prose while
  keeping capitalised tokens, which raised skill-term density and measured the
  opposite of the intended risk.
* ``disparate_impact`` must be flagged unreliable at small K, where one
  candidate moving one position swings the ratio by half.
"""

import pytest
from conftest import make_resume

from talentlens import fairness
from talentlens.schemas import SubScores


def experience_block(text: str) -> str:
    """Just the EXPERIENCE section, stopping at the next ALL-CAPS heading.

    Slicing to the end of the document would sweep in the SKILLS list, which
    implicit_rewrite deliberately leaves intact - the assertion would then always
    fail for the wrong reason.
    """
    lines = text.splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if l.strip().upper() == "EXPERIENCE")
    except StopIteration:
        return ""
    body = []
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if stripped and stripped.isupper() and len(stripped.split()) <= 3:
            break
        body.append(line)
    return "\n".join(body)


def scores(total=0.5, semantic=0.5, skill=0.5, experience=0.5, education=0.5):
    return SubScores(
        semantic=semantic,
        skill=skill,
        experience=experience,
        education=education,
        penalty=0.0,
        total=total,
    )


class TestNameSets:
    def test_every_group_has_names(self):
        for group in fairness.NAME_SETS:
            firsts, lasts = fairness.NAME_SETS[group]
            assert firsts and lasts

    def test_variants_are_deterministic(self):
        assert fairness.name_variants("white_male", 5) == fairness.name_variants(
            "white_male", 5
        )

    def test_variants_are_distinct_across_groups(self):
        a = set(fairness.name_variants("south_asian", 8))
        b = set(fairness.name_variants("east_asian", 8))
        assert not (a & b)


class TestNameSubstitution:
    def test_replaces_display_name(self):
        out = fairness.substitute_name("Priya Raman\nEngineer", "Priya Raman", "Jamal Jones")
        assert "Jamal Jones" in out
        assert "Priya" not in out

    def test_replaces_email_local_part(self):
        """A counterfactual that leaves the email behind is not a counterfactual."""
        out = fairness.substitute_name(
            "Priya Raman\npriya.raman@example.com", "Priya Raman", "Jamal Jones"
        )
        assert "priya.raman@example.com" not in out
        assert "jamal.jones@example.com" in out

    def test_leaves_the_rest_untouched(self):
        text = "Priya Raman\nBuilt payment services in Python."
        out = fairness.substitute_name(text, "Priya Raman", "Emily Walsh")
        assert "Built payment services in Python." in out

    def test_empty_original_is_a_no_op(self):
        assert fairness.substitute_name("body", "", "X") == "body"


class TestAnonymise:
    def test_drops_the_contact_block(self):
        resume = make_resume(
            text="x",
            sections={
                "other": "Priya Raman\npriya@example.com\n+91 98765 43210",
                "experience": "Built payment services in Python.",
            },
        )
        out = fairness.anonymise_text(resume)
        assert "Built payment services" in out
        assert "priya@example.com" not in out
        assert "Priya Raman" not in out

    def test_keeps_substance(self):
        resume = make_resume(
            text="x",
            sections={"experience": "Ran Cisco networks.", "skills": "Cisco, BGP"},
        )
        out = fairness.anonymise_text(resume)
        assert "Cisco" in out and "BGP" in out

    def test_falls_back_when_nothing_to_keep(self):
        resume = make_resume(text="only body text", sections={})
        assert fairness.anonymise_text(resume).strip()


class TestImplicitRewrite:
    def test_removes_technology_names_from_the_narrative(self):
        resume = make_resume(
            text="x",
            skills={"docker", "python"},
            sections={
                "experience": "Built services in Python and deployed with Docker.",
                "skills": "Python, Docker",
            },
        )
        out = fairness.implicit_rewrite(resume)
        body = experience_block(out)
        assert "Python" not in body
        assert "Docker" not in body

    def test_keeps_the_skills_list_intact(self):
        """The candidate must still CLAIM everything - only the evidence changes."""
        resume = make_resume(
            text="x",
            skills={"docker"},
            sections={
                "experience": "Deployed with Docker.",
                "skills": "Docker, Kubernetes",
            },
        )
        out = fairness.implicit_rewrite(resume)
        skills_block = out[out.upper().index("SKILLS") :]
        assert "Docker" in skills_block

    def test_preserves_surrounding_prose(self):
        resume = make_resume(
            text="x",
            skills={"docker"},
            sections={"experience": "Deployed the billing service with Docker."},
        )
        out = fairness.implicit_rewrite(resume)
        assert "billing service" in out

    def test_reduces_grounding(self):
        """The whole point: the same claims, less evidence."""
        from talentlens.grounding import ground_skills
        from talentlens.schemas import ParsedDocument
        from talentlens.extraction import build_resume

        original = make_resume(
            text="x",
            skills={"docker"},
            sections={
                "experience": "Deployed the billing service with Docker every week.",
                "skills": "Docker",
            },
        )
        rewritten = build_resume(
            ParsedDocument(
                path="i.txt",
                text=fairness.implicit_rewrite(original),
                source_format="txt",
            )
        )
        before = ground_skills(original, ["docker"], use_semantic=False)
        after = ground_skills(rewritten, ["docker"], use_semantic=False)
        assert before.strength("docker") > after.strength("docker")


class TestAuditReport:
    def build(self, means: dict[str, float]) -> fairness.AuditReport:
        groups = {}
        for name, value in means.items():
            result = fairness.GroupResult(name)
            for rank in range(1, 11):
                result.add(scores(total=value, semantic=value), rank)
            groups[name] = result
        return fairness.AuditReport(groups=groups, label="t")

    def test_max_gap(self):
        report = self.build({"a": 0.5, "b": 0.7})
        assert report.max_gap == pytest.approx(0.2)

    def test_no_gap_when_identical(self):
        assert self.build({"a": 0.5, "b": 0.5}).max_gap == pytest.approx(0.0)

    def test_dominant_component_is_identified(self):
        report = self.build({"a": 0.5, "b": 0.7})
        assert report.dominant_component == "semantic"

    def test_component_gaps_cover_all_terms(self):
        gaps = self.build({"a": 0.5}).component_gaps
        assert set(gaps) == {"semantic", "skill", "experience", "education"}

    def test_disparate_impact_reliability_flag(self):
        """The metric that misled the first audit run."""
        report = self.build({"a": 0.5, "b": 0.6, "c": 0.7, "d": 0.8, "e": 0.9, "f": 1.0})
        assert not report.di_is_reliable(top_k=10)  # <5 expected per group
        assert report.di_is_reliable(top_k=30)

    def test_selection_counts_are_reported(self):
        report = self.build({"a": 0.5, "b": 0.7})
        counts = report.selection_counts(5)
        assert set(counts) == {"a", "b"}
        assert all(v >= 0 for v in counts.values())

    def test_empty_report_does_not_crash(self):
        assert fairness.AuditReport(groups={}).max_gap == 0.0
