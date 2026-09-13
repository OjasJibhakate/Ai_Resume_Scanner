"""Offline comparative benchmark: lexical vs dense vs hybrid ranking.

Produces the comparison the study calls for - TF-IDF vs BM25 vs SBERT vs the
hybrid multi-factor scorer across NDCG@5, NDCG@10, MRR, MAP, P@5 and latency -
plus two things the study does not ask for but that this system needs to justify
its own design:

* **Ablations.** Our two departures from the source formula (coverage instead of
  Dice, and the prerequisite gate) are measured rather than asserted. If they do
  not help, the numbers will say so.
* **Adversarial resistance.** The corpus contains keyword-stuffed resumes whose
  true relevance is low. A gameable ranker promotes them. This reports, per
  ranker, how much the stuffing paid off - which is the entire point of the
  anti-gaming detector.

Read the honesty warning in ``generate_corpus`` before quoting any of these
numbers: the corpus is synthetic and self-labelled.

Usage::

    python benchmark/run_experiments.py --corpus benchmark/corpus
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from talentlens.config import DEFAULT_ANTIGAMING, PrerequisiteGate  # noqa: E402
from talentlens.extraction import build_job_spec  # noqa: E402
from talentlens.pipeline import load_resumes  # noqa: E402
from talentlens.ranking import (  # noqa: E402
    Bm25Ranker,
    FusionRanker,
    HybridRanker,
    SbertRanker,
    TfidfRanker,
)
from talentlens.schemas import DegreeLevel  # noqa: E402

#: Graded relevance at or above which a candidate counts as "relevant" for the
#: binary metrics (MRR, MAP, P@5). Tier 2 = strong match, tier 1 = marginal.
RELEVANT_THRESHOLD = 2

DEGREE_LOOKUP = {
    "None": DegreeLevel.NONE,
    "Associate": DegreeLevel.ASSOCIATE,
    "Bachelor": DegreeLevel.BACHELOR,
    "Master": DegreeLevel.MASTER,
    "Doctorate": DegreeLevel.DOCTORATE,
}


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def dcg(relevances: list[int], k: int) -> float:
    """Discounted cumulative gain with the standard 2^rel - 1 gain function."""
    return sum(
        (2**rel - 1) / math.log2(position + 2)
        for position, rel in enumerate(relevances[:k])
    )


def ndcg_at_k(ranked_relevances: list[int], all_relevances: list[int], k: int) -> float:
    ideal = sorted(all_relevances, reverse=True)
    best = dcg(ideal, k)
    if best == 0:
        return 0.0
    return dcg(ranked_relevances, k) / best


def reciprocal_rank(ranked_relevances: list[int]) -> float:
    for position, rel in enumerate(ranked_relevances, start=1):
        if rel >= RELEVANT_THRESHOLD:
            return 1.0 / position
    return 0.0


def average_precision(ranked_relevances: list[int]) -> float:
    hits = 0
    total = 0.0
    for position, rel in enumerate(ranked_relevances, start=1):
        if rel >= RELEVANT_THRESHOLD:
            hits += 1
            total += hits / position
    relevant_count = sum(1 for r in ranked_relevances if r >= RELEVANT_THRESHOLD)
    return total / relevant_count if relevant_count else 0.0


def precision_at_k(ranked_relevances: list[int], k: int) -> float:
    top = ranked_relevances[:k]
    if not top:
        return 0.0
    return sum(1 for r in top if r >= RELEVANT_THRESHOLD) / len(top)


# --------------------------------------------------------------------------
# Adversarial resistance
# --------------------------------------------------------------------------


def stuffing_advantage(
    ranking: list[tuple[str, float]], meta_by_id: dict[str, dict], job_key: str
) -> float | None:
    """How many percentiles the stuffed resumes gained over honest peers.

    Both groups are tier-1 resumes for this job's role - identical true
    relevance - so any difference in mean rank percentile is attributable to
    the stuffing alone. Positive means the attack worked.
    """
    positions = {cid: index for index, (cid, _) in enumerate(ranking)}
    size = len(ranking)
    if size == 0:
        return None

    stuffed, honest = [], []
    for cid, meta in meta_by_id.items():
        if meta["role"] != job_key or meta["tier"] != 1 or cid not in positions:
            continue
        percentile = positions[cid] / size  # 0.0 = top of the list
        (stuffed if meta["stuffed"] else honest).append(percentile)

    if not stuffed or not honest:
        return None
    # Honest percentile minus stuffed percentile: positive = stuffed ranked higher.
    return statistics.mean(honest) - statistics.mean(stuffed)


def detector_scores(ranked_candidates, meta_by_id: dict[str, dict]) -> dict[str, float]:
    """Precision / recall / F1 of the anti-gaming detector over one job."""
    tp = fp = fn = tn = 0
    for candidate in ranked_candidates:
        truth = meta_by_id.get(candidate.candidate_id, {}).get("stuffed", False)
        predicted = candidate.gaming.flagged
        if predicted and truth:
            tp += 1
        elif predicted and not truth:
            fp += 1
        elif not predicted and truth:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def build_rankers() -> dict[str, object]:
    """The systems under test, including ablations of our own design choices."""
    no_gate = PrerequisiteGate(enabled=False)
    return {
        "tfidf": TfidfRanker(),
        "bm25": Bm25Ranker(),
        "sbert": SbertRanker(),
        "hybrid": HybridRanker(),
        "hybrid_dice": HybridRanker(skill_method="dice"),
        # The utility control for the adversarial study: a defence that wrecks
        # ranking quality on honest resumes is not a defence worth having.
        "hybrid_grounded": HybridRanker(skill_method="grounded"),
        "hybrid_no_gate": HybridRanker(gate=no_gate),
        "hybrid_no_antigaming": HybridRanker(use_antigaming=False),
        "fusion_bm25_sbert": FusionRanker(rankers=[Bm25Ranker(), SbertRanker()]),
    }


def run(corpus_dir: Path, out_dir: Path) -> dict:
    payload = json.loads((corpus_dir / "corpus.json").read_text(encoding="utf-8"))
    meta_by_id = {m["candidate_id"]: m for m in payload["resumes"]}
    labels = payload["labels"]

    resume_paths = sorted((corpus_dir / "resumes").glob("*.pdf"))
    print(f"parsing {len(resume_paths)} resumes ...")
    started = time.perf_counter()
    resumes, failures = load_resumes(resume_paths)
    print(f"  parsed {len(resumes)} in {time.perf_counter() - started:.1f}s")
    for path, error in failures:
        print(f"  FAILED {path}: {error}")

    jobs = {}
    for key, spec in payload["jobs"].items():
        jobs[key] = build_job_spec(
            spec["text"],
            title=spec["title"],
            min_years=float(spec["min_years"]),
            min_degree=DEGREE_LOOKUP[spec["min_degree"]],
        )

    # Warm the embedding cache before timing anything. Otherwise whichever
    # ranker happens to run first pays the entire encoding cost and every later
    # one reads free vectors, which would make the latency column meaningless.
    # With the cache warm for all of them, ms/resume measures ranking work only.
    from talentlens.embedding import get_embedder

    print("warming embedding cache ...")
    started = time.perf_counter()
    embedder = get_embedder()
    embedder.encode_documents([r.text for r in resumes])
    embedder.encode_documents([j.text for j in jobs.values()])
    for resume in resumes:
        skills_text = resume.section("skills")
        narrative = " ".join(
            resume.section(name) for name in ("experience", "projects", "summary")
        ).strip()
        for text in (skills_text, narrative):
            if text.strip():
                embedder.encode_document(text)
    embedder.flush()
    print(f"  warmed in {time.perf_counter() - started:.1f}s")

    rankers = build_rankers()
    rows: list[dict] = []
    detector_rows: list[dict] = []

    for name, ranker in rankers.items():
        print(f"running {name} ...")
        per_job: list[dict] = []

        for job_key, job in jobs.items():
            job_labels = labels[job_key]

            started = time.perf_counter()
            if isinstance(ranker, HybridRanker):
                detailed = ranker.rank_detailed(resumes, job)
                ranking = [(c.candidate_id, c.scores.total) for c in detailed]
            else:
                detailed = None
                ranking = ranker.rank(resumes, job)
            elapsed = time.perf_counter() - started

            ranked_relevances = [job_labels.get(cid, 0) for cid, _ in ranking]
            all_relevances = list(job_labels.values())

            per_job.append(
                {
                    "ndcg@5": ndcg_at_k(ranked_relevances, all_relevances, 5),
                    "ndcg@10": ndcg_at_k(ranked_relevances, all_relevances, 10),
                    "mrr": reciprocal_rank(ranked_relevances),
                    "map": average_precision(ranked_relevances),
                    "p@5": precision_at_k(ranked_relevances, 5),
                    "ms_per_resume": elapsed * 1000.0 / max(1, len(resumes)),
                    "stuffing_advantage": stuffing_advantage(ranking, meta_by_id, job_key),
                }
            )

            if detailed is not None and name == "hybrid":
                stats = detector_scores(detailed, meta_by_id)
                stats["job"] = job_key
                detector_rows.append(stats)

        def mean(metric: str) -> float:
            values = [row[metric] for row in per_job if row[metric] is not None]
            return statistics.mean(values) if values else 0.0

        rows.append(
            {
                "ranker": name,
                "ndcg@5": mean("ndcg@5"),
                "ndcg@10": mean("ndcg@10"),
                "mrr": mean("mrr"),
                "map": mean("map"),
                "p@5": mean("p@5"),
                "ms_per_resume": mean("ms_per_resume"),
                "stuffing_advantage": mean("stuffing_advantage"),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "results.csv", rows)
    _write_markdown(out_dir / "results.md", rows, detector_rows, len(resumes), len(jobs))
    if detector_rows:
        _write_csv(out_dir / "detector.csv", detector_rows)

    return {"rows": rows, "detector": detector_rows}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    path: Path, rows: list[dict], detector_rows: list[dict], n_resumes: int, n_jobs: int
) -> None:
    lines = [
        "# TalentLens benchmark results",
        "",
        f"Corpus: {n_resumes} synthetic resumes across {n_jobs} roles, graded relevance 0-3.",
        "",
        "## How to read these numbers",
        "",
        "> **These validate the pipeline and the metrics, not real-world accuracy.**",
        "> The corpus is synthetic and we authored both the resumes and their labels.",
        "",
        "Three specific caveats, because the headline row is misleading on its own:",
        "",
        "1. **The hybrid's near-perfect ranking is close to circular.** A resume's",
        "   relevance tier is *defined* by how many required skills, years and degree",
        "   tiers it was generated with - and those are exactly the fields the hybrid",
        "   scorer reads. It is substantially measuring the label-generating process.",
        "   Do not report NDCG@5 = 1.000 as evidence that this system ranks well.",
        "2. **The lexical baselines are unfairly disadvantaged.** All five job",
        "   descriptions are rendered from one template, so shared boilerplate",
        "   dominates their bag-of-words similarity and swamps the few discriminating",
        "   skill terms. Their true weakness is vocabulary mismatch, not this.",
        "3. **Only the adversarial rows are genuinely informative.** Detector",
        "   precision/recall and the stuffing-gain column measure something the",
        "   labels did not hand us, because a stuffed resume and its honest twin",
        "   carry identical true relevance and differ only by the attack.",
        "",
        "| Ranker | NDCG@5 | NDCG@10 | MRR | MAP | P@5 | ms/resume | Stuffing gain |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['ranker']} | {row['ndcg@5']:.3f} | {row['ndcg@10']:.3f} | "
            f"{row['mrr']:.3f} | {row['map']:.3f} | {row['p@5']:.3f} | "
            f"{row['ms_per_resume']:.1f} | {row['stuffing_advantage']:+.3f} |"
        )

    lines += [
        "",
        "Latency is measured with the embedding cache warm for every ranker, so",
        "ms/resume reflects ranking work rather than whichever method happened to",
        "encode first.",
        "",
        "**Stuffing gain** is how many rank percentiles the keyword-stuffed resumes",
        "gained over honest resumes of identical true relevance. Positive means the",
        "attack worked on that ranker; near zero or negative means it did not.",
        "",
    ]

    if detector_rows:
        precision = statistics.mean(r["precision"] for r in detector_rows)
        recall = statistics.mean(r["recall"] for r in detector_rows)
        f1 = statistics.mean(r["f1"] for r in detector_rows)
        lines += [
            "## Anti-gaming detector",
            "",
            f"- Precision: {precision:.3f}",
            f"- Recall: {recall:.3f}",
            f"- F1: {f1:.3f}",
            "",
            "| Job | TP | FP | FN | TN | Precision | Recall |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in detector_rows:
            lines.append(
                f"| {row['job']} | {row['tp']} | {row['fp']} | {row['fn']} | {row['tn']} | "
                f"{row['precision']:.3f} | {row['recall']:.3f} |"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="benchmark/corpus", type=Path)
    parser.add_argument("--out", default="benchmark/results", type=Path)
    args = parser.parse_args()

    if not (args.corpus / "corpus.json").exists():
        raise SystemExit(
            f"no corpus at {args.corpus}. Run: python benchmark/generate_corpus.py"
        )

    result = run(args.corpus, args.out)

    print()
    header = f"{'ranker':<22} {'NDCG@5':>7} {'NDCG@10':>8} {'MRR':>6} {'MAP':>6} {'P@5':>6} {'ms/CV':>7} {'stuff':>7}"
    print(header)
    print("-" * len(header))
    for row in result["rows"]:
        print(
            f"{row['ranker']:<22} {row['ndcg@5']:>7.3f} {row['ndcg@10']:>8.3f} "
            f"{row['mrr']:>6.3f} {row['map']:>6.3f} {row['p@5']:>6.3f} "
            f"{row['ms_per_resume']:>7.1f} {row['stuffing_advantage']:>+7.3f}"
        )
    if result["detector"]:
        precision = statistics.mean(r["precision"] for r in result["detector"])
        recall = statistics.mean(r["recall"] for r in result["detector"])
        f1 = statistics.mean(r["f1"] for r in result["detector"])
        print()
        print(f"anti-gaming detector: precision {precision:.3f}  recall {recall:.3f}  F1 {f1:.3f}")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
