# TalentLens benchmark results

Corpus: 215 synthetic resumes across 5 roles, graded relevance 0-3.

## How to read these numbers

> **These validate the pipeline and the metrics, not real-world accuracy.**
> The corpus is synthetic and we authored both the resumes and their labels.

Three specific caveats, because the headline row is misleading on its own:

1. **The hybrid's near-perfect ranking is close to circular.** A resume's
   relevance tier is *defined* by how many required skills, years and degree
   tiers it was generated with - and those are exactly the fields the hybrid
   scorer reads. It is substantially measuring the label-generating process.
   Do not report NDCG@5 = 1.000 as evidence that this system ranks well.
2. **The lexical baselines are unfairly disadvantaged.** All five job
   descriptions are rendered from one template, so shared boilerplate
   dominates their bag-of-words similarity and swamps the few discriminating
   skill terms. Their true weakness is vocabulary mismatch, not this.
3. **Only the adversarial rows are genuinely informative.** Detector
   precision/recall and the stuffing-gain column measure something the
   labels did not hand us, because a stuffed resume and its honest twin
   carry identical true relevance and differ only by the attack.

| Ranker | NDCG@5 | NDCG@10 | MRR | MAP | P@5 | ms/resume | Stuffing gain |
|---|---|---|---|---|---|---|---|
| tfidf | 0.112 | 0.091 | 0.030 | 0.125 | 0.000 | 0.2 | +0.122 |
| bm25 | 0.070 | 0.058 | 0.029 | 0.112 | 0.000 | 0.1 | +0.180 |
| sbert | 0.109 | 0.169 | 0.103 | 0.211 | 0.000 | 0.1 | +0.078 |
| hybrid | 0.950 | 0.869 | 1.000 | 0.711 | 0.800 | 0.8 | +0.217 |
| hybrid_dice | 0.924 | 0.921 | 0.900 | 0.870 | 0.920 | 0.8 | +0.197 |
| hybrid_grounded | 0.931 | 0.929 | 1.000 | 0.849 | 0.840 | 3.0 | +0.186 |
| hybrid_no_gate | 0.950 | 0.869 | 1.000 | 0.711 | 0.800 | 0.8 | +0.149 |
| hybrid_no_antigaming | 0.109 | 0.492 | 0.167 | 0.340 | 0.000 | 0.7 | +0.228 |
| fusion_bm25_sbert | 0.095 | 0.102 | 0.065 | 0.159 | 0.000 | 0.1 | +0.138 |

Latency is measured with the embedding cache warm for every ranker, so
ms/resume reflects ranking work rather than whichever method happened to
encode first.

**Stuffing gain** is how many rank percentiles the keyword-stuffed resumes
gained over honest resumes of identical true relevance. Positive means the
attack worked on that ranker; near zero or negative means it did not.

## Anti-gaming detector

- Precision: 0.630
- Recall: 0.227
- F1: 0.333

| Job | TP | FP | FN | TN | Precision | Recall |
|---|---|---|---|---|---|---|
| backend | 17 | 10 | 58 | 130 | 0.630 | 0.227 |
| data | 17 | 10 | 58 | 130 | 0.630 | 0.227 |
| frontend | 17 | 10 | 58 | 130 | 0.630 | 0.227 |
| devops | 17 | 10 | 58 | 130 | 0.630 | 0.227 |
| network | 17 | 10 | 58 | 130 | 0.630 | 0.227 |
