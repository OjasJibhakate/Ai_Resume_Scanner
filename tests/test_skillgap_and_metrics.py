"""Skill-gap classification, ranking fusion, and the benchmark metrics.

The metric tests check against values computed by hand rather than against
whatever the implementation currently returns, which is the only way a metric
test can catch a real error.
"""

import sys
from pathlib import Path

import pytest
from conftest import make_job, make_resume

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmark"))

from run_experiments import (  # noqa: E402
    average_precision,
    dcg,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
)

from talentlens import skillgap  # noqa: E402
from talentlens.ranking import reciprocal_rank_fusion  # noqa: E402
from talentlens.schemas import GapStatus  # noqa: E402


class TestSkillGap:
    def test_held_skill_is_matched(self):
        report = skillgap.analyse(
            make_resume(skills={"python"}), make_job(required={"python"})
        )
        assert report.items[0].status is GapStatus.MATCHED

    def test_implied_skill_is_matched(self):
        """Django implies Python, so Python is not a gap."""
        report = skillgap.analyse(
            make_resume(skills={"django"}), make_job(required={"python"})
        )
        assert report.items[0].status is GapStatus.MATCHED

    def test_sibling_skill_is_transferable_not_missing(self):
        report = skillgap.analyse(
            make_resume(skills={"django"}), make_job(required={"flask"})
        )
        item = report.items[0]
        assert item.status is GapStatus.TRANSFERABLE
        assert item.via == "django"
        assert item.path == ("django", "python", "flask")

    def test_unrelated_skill_is_missing(self):
        report = skillgap.analyse(
            make_resume(skills={"react"}), make_job(required={"bgp"})
        )
        assert report.items[0].status is GapStatus.MISSING

    def test_required_flag_is_carried_through(self):
        report = skillgap.analyse(
            make_resume(skills=set()),
            make_job(required={"python"}, preferred={"kafka"}),
        )
        by_skill = {i.skill: i for i in report.items}
        assert by_skill["python"].required is True
        assert by_skill["kafka"].required is False

    def test_surplus_reports_raw_not_implied_skills(self):
        report = skillgap.analyse(
            make_resume(skills={"django", "react"}), make_job(required={"django"})
        )
        assert "react" in report.surplus
        # python is only implied, never written down, so it is not "surplus"
        assert "python" not in report.surplus

    def test_coverage_counts_only_direct_matches(self):
        report = skillgap.analyse(
            make_resume(skills={"python"}), make_job(required={"python", "bgp"})
        )
        assert report.coverage == pytest.approx(0.5)

    def test_missing_required_is_isolated(self):
        report = skillgap.analyse(
            make_resume(skills=set()),
            make_job(required={"bgp"}, preferred={"ospf"}),
        )
        assert [i.skill for i in report.missing_required] == ["bgp"]

    def test_priorities_put_required_transferable_first(self):
        report = skillgap.analyse(
            make_resume(skills={"django"}),
            make_job(required={"flask", "bgp"}, preferred={"ospf"}),
        )
        priorities = skillgap.upskill_priorities(report)
        assert priorities[0].skill == "flask"  # required and one short hop away
        assert all(p.status is not GapStatus.MATCHED for p in priorities)

    def test_empty_job_yields_no_items(self):
        report = skillgap.analyse(make_resume(skills={"python"}), make_job())
        assert report.items == ()
        assert report.coverage == 0.0


class TestMetrics:
    def test_dcg_against_hand_computation(self):
        # 7/log2(2) + 3/log2(3) = 7 + 1.8927893
        assert dcg([3, 2], 2) == pytest.approx(8.8927893, abs=1e-6)

    def test_perfect_ranking_scores_one(self):
        assert ndcg_at_k([3, 2, 1, 0], [3, 2, 1, 0], 4) == pytest.approx(1.0)

    def test_ndcg_against_hand_computation(self):
        # ranked  [3,2,0,1] -> 7 + 1.8927893 + 0 + 0.4306766 = 9.3234659
        # ideal   [3,2,1,0] -> 7 + 1.8927893 + 0.5     + 0    = 9.3927893
        assert ndcg_at_k([3, 2, 0, 1], [3, 2, 1, 0], 4) == pytest.approx(0.992620, abs=1e-5)

    def test_worst_ranking_scores_low(self):
        assert ndcg_at_k([0, 0, 2, 3], [3, 2, 0, 0], 4) < 0.6

    def test_ndcg_with_no_relevant_documents_is_zero(self):
        assert ndcg_at_k([0, 0], [0, 0], 2) == 0.0

    def test_reciprocal_rank_finds_first_relevant(self):
        assert reciprocal_rank([0, 1, 2, 3]) == pytest.approx(1 / 3)

    def test_reciprocal_rank_ignores_marginal_relevance(self):
        """Tier 1 is marginal; MRR asks for the first genuinely relevant hit."""
        assert reciprocal_rank([1, 1, 1]) == 0.0

    def test_reciprocal_rank_top_hit(self):
        assert reciprocal_rank([3]) == pytest.approx(1.0)

    def test_average_precision_against_hand_computation(self):
        # relevant (>=2) at positions 1 and 3: (1/1 + 2/3) / 2
        assert average_precision([3, 0, 2, 0]) == pytest.approx((1 + 2 / 3) / 2)

    def test_precision_at_k(self):
        assert precision_at_k([3, 0, 2, 0, 0], 5) == pytest.approx(0.4)

    def test_precision_at_k_handles_short_lists(self):
        assert precision_at_k([3], 5) == pytest.approx(1.0)


class TestFusion:
    def test_agreed_top_result_wins(self):
        fused = reciprocal_rank_fusion(
            [[("a", 1.0), ("b", 0.5)], [("a", 9.9), ("b", 0.1)]]
        )
        assert fused[0][0] == "a"

    def test_fusion_ignores_raw_score_scale(self):
        """The point of RRF: an unbounded BM25 cannot shout down a cosine."""
        fused = reciprocal_rank_fusion(
            [[("a", 0.9), ("b", 0.8)], [("b", 1000.0), ("a", 999.0)]]
        )
        assert {c for c, _ in fused} == {"a", "b"}
        assert fused[0][1] == pytest.approx(fused[1][1])

    def test_empty_input(self):
        assert reciprocal_rank_fusion([]) == []
