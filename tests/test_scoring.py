"""Scoring arithmetic: the sub-scores, the weights, and the prerequisite gate."""

import pytest

from talentlens.config import PrerequisiteGate, ScoringWeights, WeightError
from talentlens.schemas import DegreeLevel
from talentlens.scoring import (
    combine,
    coverage_score,
    dice_score,
    education_score,
    experience_score,
    required_coverage,
    skill_score,
)


class TestWeights:
    def test_defaults_are_the_studys_values(self):
        weights = ScoringWeights()
        assert (weights.semantic, weights.skill, weights.experience, weights.education) == (
            0.40,
            0.30,
            0.20,
            0.10,
        )

    def test_weights_must_sum_to_one(self):
        with pytest.raises(WeightError, match="sum to 1.0"):
            ScoringWeights(0.5, 0.5, 0.5, 0.5)

    def test_negative_weight_rejected(self):
        with pytest.raises(WeightError, match=">= 0"):
            ScoringWeights(-0.1, 0.4, 0.4, 0.3)

    def test_rescaled_normalises_arbitrary_slider_values(self):
        weights = ScoringWeights.rescaled(2, 2, 2, 2)
        assert weights.total == pytest.approx(1.0)
        assert weights.semantic == pytest.approx(0.25)

    def test_rescaled_rejects_all_zero(self):
        with pytest.raises(WeightError):
            ScoringWeights.rescaled(0, 0, 0, 0)


class TestDice:
    def test_identical_sets_score_one(self):
        assert dice_score({"a", "b"}, {"a", "b"}) == pytest.approx(1.0)

    def test_disjoint_sets_score_zero(self):
        assert dice_score({"a"}, {"b"}) == 0.0

    def test_empty_inputs_do_not_divide_by_zero(self):
        assert dice_score(set(), set()) == 0.0

    def test_dice_penalises_breadth(self):
        """The behaviour that motivated using coverage as the default.

        Both candidates meet every requirement; the only difference is that one
        knows extra unrelated things. Dice ranks that candidate lower.
        """
        required = {"a", "b"}
        narrow = dice_score({"a", "b"}, required)
        broad = dice_score({"a", "b", "c", "d", "e", "f"}, required)
        assert narrow == pytest.approx(1.0)
        assert broad < narrow


class TestCoverage:
    def test_full_coverage(self):
        assert coverage_score({"a", "b"}, {"a", "b"}) == pytest.approx(1.0)

    def test_breadth_is_not_penalised(self):
        """The fix: extra skills never reduce the score."""
        required = {"a", "b"}
        assert coverage_score({"a", "b"}, required) == pytest.approx(1.0)
        assert coverage_score({"a", "b", "x", "y", "z"}, required) == pytest.approx(1.0)

    def test_preferred_counts_half(self):
        # 1 of 1 required + 0 of 1 preferred -> 1.0 / 1.5
        score = coverage_score({"a"}, required={"a"}, preferred={"p"})
        assert score == pytest.approx(1.0 / 1.5)

    def test_no_requirements_is_vacuously_satisfied(self):
        assert coverage_score({"a"}, set()) == 1.0

    def test_partial(self):
        assert coverage_score({"a"}, {"a", "b"}) == pytest.approx(0.5)

    def test_skill_score_dispatches_on_method(self):
        held, required = {"a", "b", "c", "d"}, {"a", "b"}
        assert skill_score(held, required, method="coverage") == pytest.approx(1.0)
        assert skill_score(held, required, method="dice") < 1.0


class TestExperience:
    def test_meets_requirement(self):
        assert experience_score(5.0, 5.0) == pytest.approx(1.0)

    def test_exceeding_is_capped_at_one(self):
        assert experience_score(30.0, 5.0) == pytest.approx(1.0)

    def test_partial_is_the_ratio(self):
        assert experience_score(3.0, 6.0) == pytest.approx(0.5)

    def test_no_requirement_is_satisfied(self):
        assert experience_score(0.0, 0.0) == 1.0

    def test_no_experience_against_a_requirement_scores_zero(self):
        assert experience_score(0.0, 5.0) == 0.0


class TestEducation:
    def test_exact_match(self):
        assert education_score(DegreeLevel.BACHELOR, DegreeLevel.BACHELOR) == 1.0

    def test_higher_degree_is_capped_not_rewarded(self):
        assert education_score(DegreeLevel.DOCTORATE, DegreeLevel.BACHELOR) == 1.0

    def test_lower_degree_is_partial(self):
        score = education_score(DegreeLevel.BACHELOR, DegreeLevel.MASTER)
        assert 0.0 < score < 1.0
        assert score == pytest.approx(2 / 3)

    def test_no_requirement_is_satisfied(self):
        assert education_score(DegreeLevel.NONE, DegreeLevel.NONE) == 1.0

    def test_no_degree_against_a_requirement_scores_zero(self):
        assert education_score(DegreeLevel.NONE, DegreeLevel.MASTER) == 0.0


class TestCombine:
    def test_weighted_sum_is_exact(self):
        scores = combine(1.0, 1.0, 1.0, 1.0)
        assert scores.total == pytest.approx(1.0)

    def test_contributions_match_the_formula(self):
        scores = combine(0.5, 0.5, 0.5, 0.5)
        assert scores.total == pytest.approx(0.5)

    def test_penalty_is_subtracted(self):
        scores = combine(1.0, 1.0, 1.0, 1.0, penalty=0.25)
        assert scores.total == pytest.approx(0.75)

    def test_total_never_goes_negative(self):
        scores = combine(0.0, 0.0, 0.0, 0.0, penalty=0.9)
        assert scores.total == 0.0

    def test_gate_scales_the_weighted_fit(self):
        ungated = combine(1.0, 1.0, 1.0, 1.0, gate=1.0)
        gated = combine(1.0, 1.0, 1.0, 1.0, gate=0.5)
        assert gated.total == pytest.approx(ungated.total * 0.5)

    def test_gate_is_recorded_for_the_ui(self):
        scores = combine(1.0, 1.0, 1.0, 1.0, gate=0.4, coverage=0.2)
        assert scores.gate == pytest.approx(0.4)
        assert scores.required_coverage == pytest.approx(0.2)


class TestPrerequisiteGate:
    def test_full_coverage_is_not_damped(self):
        assert PrerequisiteGate().multiplier(1.0) == pytest.approx(1.0)

    def test_at_threshold_is_not_damped(self):
        gate = PrerequisiteGate(threshold=0.5)
        assert gate.multiplier(0.5) == pytest.approx(1.0)

    def test_zero_coverage_hits_the_floor_not_zero(self):
        """Unqualified candidates are demoted, never erased."""
        gate = PrerequisiteGate(floor=0.25)
        assert gate.multiplier(0.0) == pytest.approx(0.25)

    def test_is_monotonic_in_coverage(self):
        gate = PrerequisiteGate()
        values = [gate.multiplier(c / 10) for c in range(11)]
        assert values == sorted(values)

    def test_disabled_gate_is_a_no_op(self):
        gate = PrerequisiteGate(enabled=False)
        assert gate.multiplier(0.0) == 1.0

    def test_invalid_configuration_is_rejected(self):
        with pytest.raises(ValueError):
            PrerequisiteGate(threshold=0.0)
        with pytest.raises(ValueError):
            PrerequisiteGate(floor=1.5)


class TestRequiredCoverage:
    def test_counts_only_required(self):
        assert required_coverage({"a"}, {"a", "b"}) == pytest.approx(0.5)

    def test_no_requirements_is_full_coverage(self):
        assert required_coverage(set(), set()) == 1.0

    def test_the_unrelated_candidate_problem(self):
        """A domain outsider holds none of the required skills."""
        assert required_coverage({"nursing", "triage"}, {"python", "docker"}) == 0.0
