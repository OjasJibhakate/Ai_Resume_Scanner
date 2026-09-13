# Fairness audit

30 real resumes (INFORMATION-TECHNOLOGY), 6 name groups, one job description.

> **Names are a weak proxy for demographic groups.** This measures
> name-induced score disparity - the harm channel that actually exists
> for an automated screener, since the name is all it sees. It is not a
> measurement of outcomes for real demographic populations, and must not
> be reported as one.

## Name substitution

Identical resumes, only the name changed. Gap = largest difference
between any two group means.

| Audit | Max total gap | Dominant component | DI@30 |
|---|---|---|---|
| names (coverage) | 0.0028 | semantic | 1.000 |
| names (coverage, anonymised) | 0.0018 | semantic | 1.000 |
| names (grounded) | 0.0028 | semantic | 1.000 |
| names (grounded, anonymised) | 0.0018 | semantic | 1.000 |

### Per-component gaps

| Audit | semantic | skill | experience | education |
|---|---|---|---|---|
| names (coverage) | 0.0069 | 0.0000 | 0.0000 | 0.0000 |
| names (coverage, anonymised) | 0.0044 | 0.0000 | 0.0000 | 0.0000 |
| names (grounded) | 0.0069 | 0.0000 | 0.0000 | 0.0000 |
| names (grounded, anonymised) | 0.0044 | 0.0000 | 0.0000 | 0.0000 |

### Group means (total score)

| Audit | white_female | white_male | black_female | black_male | south_asian | east_asian |
|---|---|---|---|---|---|---|
| names (coverage) | 0.5465 | 0.5472 | 0.5456 | 0.5461 | 0.5483 | 0.5460 |
| names (coverage, anonymised) | 0.5424 | 0.5429 | 0.5417 | 0.5424 | 0.5434 | 0.5419 |
| names (grounded) | 0.5293 | 0.5299 | 0.5283 | 0.5288 | 0.5311 | 0.5288 |
| names (grounded, anonymised) | 0.5251 | 0.5256 | 0.5244 | 0.5251 | 0.5262 | 0.5246 |

## Writing-style counterfactual

Narratives rewritten tersely with the technologies they name preserved,
so the facts are constant and only the prose changes. This targets the
risk specific to evidence grounding: that it rewards fluent writing
rather than demonstrated competence.

| Skill measure | Original total | Implicit total | Original skill | Implicit skill | Skill delta |
|---|---|---|---|---|---|
| coverage | 0.5490 | 0.5437 | 0.3000 | 0.2800 | -0.0200 |
| grounded | 0.5309 | 0.4991 | 0.2397 | 0.1313 | -0.1084 |

A large negative skill delta under `grounded` but not under `coverage`
means evidence grounding penalises candidates who describe the same work
without naming the technology - a writing habit, not a competence gap.

