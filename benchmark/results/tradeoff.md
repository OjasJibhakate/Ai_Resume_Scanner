# Robustness-fairness tradeoff

`floor` is the credit a claimed-but-unevidenced skill receives.
`floor = 1.0` disables grounding entirely.

- **A4 attack gain** - rank percentiles an LLM-tailored resume gains over
  its identical control. Lower is better. Measured on matched synthetic pairs.
- **Style penalty** - skill-score change when 30 real resumes are rewritten
  to describe the same work without naming the technologies. Nearer zero is fairer.

| floor | A4 attack gain | Style penalty | Original skill | Implicit skill |
|---|---|---|---|---|
| 0.0 | +0.0471 | -83.8% | 0.1995 | 0.0322 |
| 0.2 | +0.0806 | -62.8% | 0.2196 | 0.0818 |
| 0.4 | +0.0998 | -45.2% | 0.2397 | 0.1313 |
| 0.6 | +0.1237 | -30.4% | 0.2598 | 0.1809 |
| 0.8 | +0.1535 | -17.7% | 0.2799 | 0.2304 |
| 1.0 | +0.1929 | -6.7% | 0.3000 | 0.2800 |

## Reading the curve

These move in opposite directions, and that is the point: there is no
setting that is simultaneously maximally robust and maximally fair. A
deployer has to choose, and should choose explicitly.

The fairness cost is real but its *interpretation* needs care. Whether a
penalty for not naming technologies constitutes unfairness depends on
whether that writing habit correlates with protected characteristics.
This corpus carries no demographic labels and cannot answer that. What
is established here is a quantified disparate-treatment risk on a
writing-style axis - enough to require evaluation before deployment, not
enough to conclude discrimination.

