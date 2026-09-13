# Adversarial robustness study

215 resumes, 75 matched attack/control pairs, 5 jobs.

## Attack gain by system and attack type

Rank-percentile gained by the attacked resume over its identical control.
**Positive = the attack worked.** 0.000 = the manipulation bought nothing.

| System | A1 keyword block | A2 repetition | A3 hidden text | A4 LLM tailored | A5 LLM fabricated | Mean |
|---|---|---|---|---|---|---|
| tfidf | +0.115 | +0.015 | +0.262 | +0.127 | +0.079 | +0.119 |
| bm25 | +0.165 | +0.017 | +0.320 | +0.200 | +0.194 | +0.179 |
| sbert | +0.072 | -0.017 | +0.165 | +0.118 | +0.073 | +0.082 |
| hybrid_plain | +0.211 | +0.052 | +0.251 | +0.193 | +0.178 | +0.177 |
| hybrid_detector_only | +0.212 | +0.052 | +0.146 | +0.193 | +0.177 | +0.156 |
| hybrid_grounded_only | +0.117 | +0.022 | +0.162 | +0.100 | +0.158 | +0.112 |
| hybrid_full | +0.110 | +0.022 | +0.007 | +0.096 | +0.135 | +0.074 |

## Detector recall by attack type

Share of attacked resumes the five-signal detector flagged.

| Attack | Detector recall | Mean grounded fraction |
|---|---|---|
| A1 keyword block | 0.00 | 0.15 |
| A2 repetition | 0.00 | 0.23 |
| A3 hidden text | 1.00 | 0.15 |
| A4 LLM tailored | 0.07 | 0.18 |
| A5 LLM fabricated | 0.07 | 0.67 |

Honest baseline grounded fraction: **0.36**.

## Reading this

`hybrid_detector_only` is the prior-art defence: attack-specific signals
bolted onto a ranker. `hybrid_grounded_only` runs with the detector
switched off, so anything it resists, it resists structurally rather than
by recognising the attack.

The corpus is synthetic and self-labelled, so absolute ranking quality
here means little. The matched-pair design is what carries weight: a
control and its attacked twin are the same candidate, so the difference
between them is attributable to the manipulation and nothing else.

