# Evidence-Grounded Relevance for Adversarially Robust Resume Screening

**Working paper — TalentLens project**
Madhur Wanjari · Vansh Mahant · Ojas Jibhakate
Symbiosis Institute of Technology, Nagpur

---

## Abstract

Automated resume screening creates an incentive to manipulate resumes, and the
manipulation has outrun the defences. Published countermeasures are *detectors*:
attack-specific signals bolted onto a ranker, looking for keyword stuffing,
repetition, or text hidden in white 1-point type. Every one of those signatures
is lexical, and none of them survives the arrival of cheap language models — a
candidate who asks an LLM to rewrite their resume for a posting produces fluent
text with no repetition, no hidden layer, no verbatim copying, and a genuinely
higher embedding similarity to the job description.

We argue the defence belongs inside the relevance function rather than beside
it. We define **evidence-grounded relevance**: a skill contributes to a
candidate's score in proportion to whether the resume's own work narrative
demonstrates it. Claims are cheap; evidence is not. Because the mechanism never
references any attack signature, it is attack-agnostic by construction.

On a 205-resume corpus with 75 matched attack/control pairs across five attack
types, a five-signal detector of the kind found in prior work significantly
reduces the payoff of **one** of five attacks. Evidence grounding, with the
detector switched off, significantly reduces **four** of five — including
LLM-tailored resumes, against which the detector has 0.07 recall and no
measurable effect (p = 0.91). Grounding costs no ranking utility: NDCG@10 rises
from 0.869 to 0.929 and MAP from 0.711 to 0.849.

We also report a negative result that bounds the contribution. When the attacker
fabricates the work narrative as well as the skill list, grounding's effect is
not significant (p = 0.079). Grounding does not make manipulation impossible; it
changes what manipulation must be. Keyword optimisation is ubiquitous, costless
and unfalsifiable. Fabricating bullet points about named employers and dates is
resume fraud: checkable by reference, and actionable by an employer. We argue
converting the former into the latter is the realistic security goal.

Finally, we report that adding structured skill matching to a dense retriever —
the standard recipe for interpretability — **doubles** adversarial vulnerability
relative to the dense retriever alone (+0.177 vs +0.082 mean attack gain).
Interpretability, implemented naively, buys an attack surface. Grounding
recovers the robustness while keeping the interpretability.

---

## 1. Threat model

An applicant wants to be ranked higher than their qualifications warrant, has
the resume and the job description, and has no access to the model. They will
not fabricate anything they consider legally risky unless it pays. Under that
model we define five attacks, each isolated so a defence can be credited for the
specific thing it handles:

| | Attack | Mechanism | Leaves a lexical artefact? |
|---|---|---|---|
| A1 | keyword block | append a dense block of JD skills | yes — density |
| A2 | repetition | repeat one high-value term | yes — term frequency |
| A3 | hidden text | paste the JD in white 1pt type | yes — span colour/size |
| A4 | **LLM tailored** | LLM rewrites SKILLS + SUMMARY to mirror the JD; work history untouched | **no** |
| A5 | **LLM fabricated** | LLM also rewrites experience bullets to narrate the new claims | **no** |

A4 and A5 are generated with a production LLM (`nvidia/nemotron-3.5-lightning-30b-a3b`).
They are not hypothetical: resume-tailoring pipelines of exactly this shape are
published and deployed (ResumeFlow, SIGIR 2024).

A1–A3 inflate claims and leave evidence untouched. A4 does the same thing
*fluently*. A5 is the only attack that manufactures evidence.

---

## 2. Positioning against prior work

| Prior work | What it does | What it leaves open |
|---|---|---|
| Surya et al., INSPECT 2024 | TF-IDF + SVM screening | no adversary modelled |
| Yao et al., ICDE 2022 | knowledge-graph person-job fit | no adversary modelled |
| Dilshan & Asanka, ICARC 2025 | KSA extraction + BERT scoring | scores claims, not evidence |
| Kumar & Chellamani, ICCCA 2025 | SBERT retrieval + LLM rationale | dense similarity is *rewarded* by A4 |
| Du et al., TPAMI 2025 | asymmetric quasi-metric matching | no adversary modelled |
| arXiv 2512.20164 | adversarial attack taxonomy for LLM screening | **detection/filtering only; no scoring function incorporating evidence verification** |
| arXiv 2108.05490 | adversarial attacks on ranking embeddings | attacks, not defences |
| arXiv 2602.18550 | validity of LLM-based screening | **does not address adversarial manipulation or verification of claimed skills** |

The gap is consistent: robustness is treated as a filter applied *before or
after* relevance. Nobody makes verification a term *inside* relevance. That is
the contribution.

---

## 3. Method

### 3.1 Grounding

Let a resume's claimed skills be `K` and its work narrative be the sentence set
`S` drawn from the EXPERIENCE and PROJECTS sections — never the skills list,
which is the claim under test. For each `k ∈ K` we compute a grounding strength
`g(k) ∈ [0,1]`:

```
g(k) = 1.00   if k, an alias of k, or any ontology DESCENDANT of k
               is named in some sentence s ∈ S          (explicit evidence)

g(k) = γ·norm(max_s cos(q(k), s))
              if that maximum clears τ                  (semantic evidence)

g(k) = 0.00   otherwise                                 (claimed, unevidenced)
```

Descendants matter because evidence flows *up* the ontology: a sentence about
shipping Django is evidence of Python. This is the mirror of the expansion used
for matching, which flows the same direction — and the asymmetry is load-bearing
in both places.

`q(k)` wraps the bare skill in a natural phrase ("experience with X"); a lone
token embeds poorly against full sentences. `γ = 0.75` caps inferred evidence
below named evidence.

**Threshold calibration.** τ is measured, not chosen. Over corpus narratives,
the similarity between a skill query and its best-matching sentence separates as:

| | n | mean | p10 | p95 |
|---|---|---|---|---|
| skill named in narrative | 8 | 0.482 | 0.388 | — |
| skill absent | 64 | 0.225 | — | 0.378 |

τ = 0.40 sits just above the negative p95. The classes overlap in 0.34–0.39, and
the asymmetry of costs settles the tie: inventing evidence for a skill the
candidate never demonstrated is a worse error than falling back to the floor.

### 3.2 Grounded coverage

Plain coverage counts a requirement as met or unmet. Grounded coverage weights
each met requirement by its evidence:

```
S_skill = Σ_{r ∈ R} [ floor + (1 − floor)·g(r) ] · 1[r ∈ K] / |R|
```

with `floor = 0.40`. This is a strict refinement: at `floor = 1.0` it collapses
exactly to plain coverage.

**The floor is an ethics parameter, not a tuning knob.** Real resumes legitimately
list skills they never narrate — space is finite and a skills section is a normal
summary device. Scoring an unevidenced claim at zero would punish ordinary resume
formatting rather than dishonesty. At 0.40, demonstrating a skill is worth 2.5×
asserting it, and asserting it is still worth something.

---

## 4. Experimental design

**Matched pairs.** Every attacked resume has an untouched control: the same
candidate, same true relevance, differing only by the manipulation. Both are
ranked in the same pool, so

```
attack_gain = percentile(control) − percentile(attacked)
```

isolates the manipulation from every property of the candidate. Positive means
the attack worked.

For A4 the evidence-preservation constraint is **enforced programmatically**,
not requested in the prompt — the model silently dropped an experience bullet in
testing, and a variant that also removes evidence would let grounding look
effective for the wrong reason. The model's rewritten summary and skills are
kept; the original EXPERIENCE and EDUCATION are spliced back verbatim.

Corpus: 205 resumes, 5 roles, 75 matched pairs (15 per attack), graded relevance
0–3. The prerequisite gate is disabled throughout the attack study, since every
attack works by claiming required skills and the gate would absorb part of the
effect and confound the comparison.

Significance: percentile bootstrap CIs (10,000 resamples) and a two-sided
sign-flip permutation test on paired differences.

---

## 5. Results

### 5.1 Attack gain by system (positive = attack succeeded)

| System | A1 kwblock | A2 repet | A3 hidden | A4 LLM-tailored | A5 fabricated | Mean |
|---|---|---|---|---|---|---|
| tfidf | +0.115 | +0.015 | +0.262 | +0.127 | +0.079 | +0.119 |
| bm25 | +0.165 | +0.017 | +0.320 | +0.200 | +0.194 | +0.179 |
| sbert | +0.072 | −0.017 | +0.165 | +0.118 | +0.073 | **+0.082** |
| hybrid_plain | +0.211 | +0.052 | +0.251 | +0.193 | +0.178 | **+0.177** |
| hybrid_detector_only | +0.212 | +0.052 | +0.146 | +0.193 | +0.177 | +0.156 |
| hybrid_grounded_only | +0.117 | +0.022 | +0.162 | +0.100 | +0.158 | +0.112 |
| hybrid_full | +0.110 | +0.022 | **+0.007** | +0.096 | +0.135 | **+0.074** |

### 5.2 Paired effect vs. the undefended multi-factor ranker

n = 15 per cell. `*` = p < 0.05.

| Defence | A1 kwblock | A2 repet | A3 hidden | A4 LLM-tailored | A5 fabricated |
|---|---|---|---|---|---|
| detector only | +0.000 (p=1.00) | +0.000 (p=1.00) | **−0.105*** | +0.001 (p=0.91) | −0.001 (p=0.87) |
| **grounding only** | **−0.094*** | **−0.029*** | **−0.089*** | **−0.093*** | −0.020 (p=0.079) |
| both | **−0.101*** | **−0.030*** | **−0.244*** | **−0.097*** | **−0.043*** |

**The detector significantly reduces one attack of five. Grounding, knowing
nothing about any attack, significantly reduces four of five.**

### 5.3 Why the detector fails

| Attack | Detector recall | Mean grounded fraction |
|---|---|---|
| A1 keyword block | 0.00 | 0.15 |
| A2 repetition | 0.00 | 0.23 |
| A3 hidden text | **1.00** | 0.15 |
| A4 LLM tailored | 0.07 | 0.18 |
| A5 LLM fabricated | 0.07 | **0.67** |
| *honest baseline* | — | *0.36* |

The five-signal detector is effectively a one-signal detector: only the physical
artefact (white 1pt text) is reliably caught. Grounded fraction separates honest
(0.36) from every claim-inflation attack (0.15–0.23) — and **inverts** on A5
(0.67), where fabricated narratives are *better* evidenced than honest ones.

### 5.4 Utility control

A defence that degrades ranking is not a defence.

| Ranker | NDCG@5 | NDCG@10 | MRR | MAP | P@5 | ms/resume |
|---|---|---|---|---|---|---|
| hybrid (coverage) | **0.950** | 0.869 | 1.000 | 0.711 | 0.800 | 0.8 |
| hybrid_grounded | 0.931 | **0.929** | 1.000 | **0.849** | **0.840** | 3.0 |
| sbert | 0.109 | 0.169 | 0.103 | 0.211 | 0.000 | 0.1 |

Grounding costs ~2% NDCG@5 and improves NDCG@10, MAP and P@5, at 3.75× latency
(3 ms/resume). Robustness here is not bought with accuracy.

---

### 5.5 External validity on real resumes

Run on 500 real resumes from the livecareer corpus (`opensporks/resumes`), and
the first result was a failure of our own pipeline rather than a finding about
grounding.

| | line-based segmentation | + inline-heading normalisation |
|---|---|---|
| experience section detected | **0 / 40 (0%)** | **40 / 40 (100%)** |
| skills section detected | — | 40 / 40 (100%) |
| reported grounded fraction | 1.000 (degenerate) | 0.530 (spread 0.17–1.00) |

Real resume exports put section headings *inline*, separated by runs of spaces
rather than newlines (`"HR ADMINISTRATOR    Summary    Dedicated..."`). A
line-based segmenter finds no headings, returns one undifferentiated blob, and
the narrative silently falls back to the whole document **including the skills
list** — so every claimed skill grounds itself and the measure reports a perfect
1.000 for everyone. The failure is silent and total, and no synthetic corpus
would ever have surfaced it.

After normalising inline headings, real resumes show a genuine spread
(0.17–1.00, mean 0.530). The real mean exceeds the synthetic baseline (0.36)
because real narratives are far richer — median 33 narrative sentences against 7
in the generator.

### 5.6 Fairness audit: name substitution

Identical real resumes, candidate name replaced with names from six groups
following the Bertrand & Mullainathan (2004) audit methodology, extended with
South Asian and East Asian sets. 30 resumes × 6 groups.

| Configuration | Max total gap | semantic | skill | experience | education | DI@30 |
|---|---|---|---|---|---|---|
| coverage | 0.0028 | **0.0069** | 0.0000 | 0.0000 | 0.0000 | 1.000 |
| coverage, anonymised | 0.0018 | **0.0044** | 0.0000 | 0.0000 | 0.0000 | 1.000 |
| grounded | 0.0028 | **0.0069** | 0.0000 | 0.0000 | 0.0000 | 1.000 |
| grounded, anonymised | 0.0018 | **0.0044** | 0.0000 | 0.0000 | 0.0000 | 1.000 |

Three things follow, and the second is the methodological point:

1. Name-induced disparity is small (0.0028 of a [0,1] score) and shows no
   adverse impact at a sample size where the four-fifths ratio is stable.
2. **The disparity localises entirely to the semantic term.** Skill, experience
   and education are exactly 0.0000 — structurally, they cannot see the name.
   This is the audit step a black-box scorer cannot perform: the gap is
   attributed to one term in the formula rather than to "the model".
3. Because it is localised, it is fixable. Stripping the identifying header
   before embedding cuts the semantic gap by 36% (0.0069 → 0.0044), and the
   audit is re-run to confirm it rather than assuming it.

The residual 0.0044 is the encoder's response to names appearing elsewhere in
the document body.

**Disparate impact is unstable at small K and we initially misread it.** At
top-10 with six groups, each group expects fewer than two selections, and a
single candidate moving one position swings the ratio from 1.00 to 0.50 — which
is what our first run reported as "below threshold". The metric is only quoted
at K where the expected count per group is ≥ 5.

### 5.7 Fairness audit: writing style — a defect in our own method

Evidence grounding reads technology names out of the work narrative. Some
candidates write *"administered the campus network and resolved escalations"*;
others write *"administered the Cisco campus network using BGP"*. Identical
work. Only the second is grounded. Naming habits track seniority, first-language
background, and access to professional CV coaching.

Real resumes were rewritten to describe the same work without naming the
technologies, leaving the skills list untouched — the candidate still *claims*
everything they claimed before.

| Skill measure | Original | Implicit | Penalty |
|---|---|---|---|
| coverage | 0.3000 | 0.2800 | **−6.7%** |
| **grounded** | 0.2397 | 0.1313 | **−45.2%** |

**Evidence grounding penalises candidates who do not name-drop by 45%, against
6.7% for plain coverage.** This is a real cost of the contribution, not a
caveat about it.

The semantic grounding path is the intended mitigation and it only partly works:
lowering τ from 0.40 to 0.22 recovers the penalty from −45.2% to −33.2%, at the
cost of admitting weaker evidence. It cannot be tuned away.

*(An earlier version of this perturbation shortened the prose while keeping
capitalised tokens. That **increased** skill-term density and made grounding
easier — it measured the opposite of the intended risk. The result is only
meaningful because the perturbation removes the names.)*

### 5.8 The robustness–fairness tradeoff

The exchange rate is the `floor` parameter: the credit a claimed-but-unevidenced
skill still receives. At `floor = 1.0` grounding is off.

| floor | A4 attack gain | Style penalty |
|---|---|---|
| 0.0 | **+0.047** | −83.8% |
| 0.2 | +0.081 | −62.8% |
| **0.4** (default) | **+0.100** | **−45.2%** |
| 0.6 | +0.124 | −30.4% |
| 0.8 | +0.154 | −17.7% |
| 1.0 | +0.193 | −6.7% |

Monotonic in both directions. At `floor = 1.0` the attack gain reproduces the
undefended multi-factor baseline (+0.193 vs +0.193 in Table 5.1), which is a
validity check on the whole ablation.

There is no setting that is simultaneously maximally robust and maximally fair.
We report the curve rather than a recommended point, because the right choice
depends on the deployment: a high-volume funnel where gaming is rife sits
leftward; a context where applicants have uneven access to CV coaching sits
rightward.

Whether the style penalty constitutes *unfairness* depends on whether naming
habits correlate with protected characteristics, and this corpus carries no
demographic labels. What is established is a quantified disparate-treatment risk
on a writing-style axis — enough to require evaluation before deployment, not
enough to conclude discrimination.

---

## 6. Limitations

1. **Synthetic corpus, self-authored labels.** Absolute ranking numbers are not
   evidence of real-world accuracy. The matched-pair design is what carries
   weight — control and attack are the same candidate — but generalisation to
   real resumes is untested.
2. **A5 is unsolved.** Fabricated narratives defeat grounding and the effect is
   not significant. We claim cost escalation, not immunity.
3. **Single generator model.** A4/A5 come from one LLM. A different model may
   produce differently-detectable text.
4. **τ calibrated on n = 72 pairs** from this corpus. Real narratives are richer
   and would need recalibration.
5. **The style penalty is a real cost, not a hypothetical one.** Grounding docks
   candidates who describe identical work without naming technologies by 45%
   (§5.7). The semantic path mitigates it only partially, and the `floor`
   parameter trades it against robustness rather than removing it.
6. **The fairness audit cannot establish discrimination.** Names are a weak
   proxy, and the real corpus carries no demographic labels, so the writing-style
   result is a disparate-*treatment* risk on a style axis. Whether that
   correlates with protected characteristics is unmeasured and is the single
   most important open question before deployment.
7. **One real corpus, one occupational slice.** The external-validity and
   fairness results use 30-40 IT resumes from one source (livecareer, US-centric).
8. **Ontology-bound.** Explicit grounding depends on a hand-built 182-skill
   taxonomy; unknown skills fall back to the weaker semantic path.

---

## 7. Claimed contributions

1. **Evidence-grounded relevance** — verification as a term inside the scoring
   function rather than a filter beside it, with a calibrated threshold procedure.
2. **Empirical demonstration that detection does not generalise** and grounding
   does, on an attack (A4) neither mechanism was designed against.
3. **The interpretability/robustness finding** — structured skill matching
   doubles adversarial vulnerability versus dense retrieval alone; grounding
   recovers it.
4. **A reproducible matched-pair adversarial protocol** for resume screening,
   including LLM-generated attacks, with generator and corpus released.
5. **A bounded negative result** locating precisely where evidence verification
   stops working (fabricated narratives, §5.2).
6. **Per-component fairness attribution** — because the scorer decomposes, a
   measured disparity is attributed to a specific term rather than to "the
   model", which is what makes it fixable. Demonstrated end to end: measure
   (0.0069 semantic gap) → localise (skill/experience/education exactly 0.0000)
   → mitigate (strip the header) → re-measure (0.0044, −36%).
7. **A published robustness-fairness tradeoff curve** (§5.8) so a deployer
   chooses a point knowingly instead of inheriting a default.
8. **A documented silent-failure mode in resume segmentation** (§5.5) that
   synthetic evaluation cannot surface: inline headings defeat line-based
   segmenters and make evidence grounding report a perfect score for everyone.

---

## 8. Reproducing

```bash
python benchmark/generate_corpus.py --pairs-per-attack 3
python benchmark/generate_adversarial.py --per-role 3   # needs NVIDIA_API_KEY
python benchmark/run_attack_study.py                    # Tables 5.1-5.3
python benchmark/run_experiments.py                     # Table 5.4

python benchmark/fetch_real_resumes.py --limit 500      # real corpus, gitignored
python benchmark/run_fairness_audit.py --limit 30       # Tables 5.6-5.7
python benchmark/run_tradeoff.py --limit 30             # Table 5.8
```

The real corpus contains documents about real people. It is gitignored, never
redistributed, and the analysis reports aggregate statistics only.

Outputs land in `benchmark/results/`.

---

## 9. Venue notes

The work is a defence paper with a bounded negative result, which suits venues
that value honest scope over headline numbers. Plausible targets: an
applied-NLP or IR workshop (adversarial IR, fairness in hiring), or a security
venue's ML track given the framing as attack-cost escalation. The fairness audit
is complete (§5.6-5.8); what remains before a full-conference submission is
relevance labels from human annotators on the real corpus, and a demographic
correlation study for the writing-style axis.

**Before submitting:** verify the five IEEE DOIs in the source study against
IEEE Xplore directly. They were taken from the project's literature-review
document and have not been independently checked.
