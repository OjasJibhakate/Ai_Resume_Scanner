# TalentLens

Explainable AI resume ranking and candidate matching.

Given a job description and a pile of resumes, TalentLens produces a ranked
shortlist where every position can be justified: which requirements the
candidate meets, which they are a short hop away from, which they genuinely
lack, and whether the resume shows signs of being written to game an automated
screener.

Built from the architecture in `AI_Resume_Ranking_Research_Study.pdf`, which
reviews five IEEE papers and converges on a hybrid design: SBERT dense retrieval
(Paper 4) fused with structured multi-factor scoring (Paper 3) over deterministic
extraction (Paper 1).

---

## Quick start

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# generate the labelled corpus (205 resumes, 5 roles, 75 matched attack pairs)
python benchmark/generate_corpus.py --pairs-per-attack 3

# run the recruiter dashboard
streamlit run app/streamlit_app.py
```

Reproduce the research results:

```bash
python benchmark/generate_adversarial.py --per-role 3   # LLM attacks; needs NVIDIA_API_KEY
python benchmark/run_attack_study.py                    # adversarial robustness
python benchmark/run_experiments.py                     # ranking utility
python benchmark/fetch_real_resumes.py --limit 500      # real corpus (gitignored)
python benchmark/run_fairness_audit.py --limit 30       # fairness
python benchmark/run_tradeoff.py --limit 30             # robustness-fairness curve
```

Copy `.env.example` to `.env` for the optional LLM features. Everything works
without a key; explanations fall back to deterministic templates.

Or from Python:

```python
from talentlens import screen_files

ranked, failures = screen_files(["cv1.pdf", "cv2.pdf"], job_text=jd_text)
for c in ranked:
    print(c.rank, c.display_name, round(c.scores.total, 3))
```

First run downloads `all-MiniLM-L6-v2` (~90 MB) and caches it. GPU is used
automatically when available.

---

## How the score works

```
S_total = (w_sem·S_sem + w_skill·S_skill + w_exp·S_exp + w_edu·S_edu) × gate − penalty
```

with default weights 0.40 / 0.30 / 0.20 / 0.10, as recommended by the study.

| Term | What it measures |
|---|---|
| `S_sem` | Cosine similarity between SBERT document vectors |
| `S_skill` | Weighted coverage of the job's skills (required full, preferred half) |
| `S_exp` | `min(1, candidate_years / required_years)` |
| `S_edu` | Ordinal degree-tier alignment |
| `gate` | Prerequisite damping for candidates missing most required skills |
| `penalty` | Anti-gaming deduction, capped at 0.35 |

### Three deliberate departures from the source study

The study's formula was implemented as specified and then corrected in three
places where it produced wrong answers. All three are switchable, and the
benchmark measures each one.

**1. Coverage instead of Sørensen–Dice.** The study specifies
`S_skill = 2|K_c ∩ K_r| / (|K_c| + |K_r|)`. Dice is symmetric, which is wrong
here: a resume lists everything a person knows, a JD lists what the role needs.
For a JD asking 10 skills, a candidate knowing 50 including all 10 scores
`2·10/(50+10) = 0.33`, while one knowing exactly those 10 scores `1.00`. Dice
ranks the broader, fully-qualified candidate *below* the narrower one for the
crime of knowing extra things. Coverage — what fraction of the requirements are
met — is the default. Pass `skill_method="dice"` for the literal formula.

**2. A prerequisite gate.** Experience and education are role-agnostic, so the
purely additive formula gives a nurse with 11 years and a bachelor's degree the
full 0.20 + 0.10 on a backend engineering req, landing them near 0.40 — which
reads as "40% match". The gate scales the weighted fit by required-skill
coverage, bottoming out at 0.25 rather than zero so unqualified candidates are
demoted rather than erased and ordering within that group is preserved. In
testing this moved that nurse from 0.396 to 0.099. This is our reading of
Paper 5's directional insight: clearing the hard bar is a different question
from general fit, and a symmetric sum loses it. Disable with
`PrerequisiteGate(enabled=False)`.

**3. Chunked embeddings.** `all-MiniLM-L6-v2` truncates at 256 tokens; real
resumes run 600–1500. Encoding one directly silently discards most of it —
usually the skills and education sections. Resumes are split into overlapping
windows, encoded, and mean-pooled, following Paper 4's recursive splitter.

---

## The research contribution

Full write-up in **[RESEARCH.md](RESEARCH.md)**.

Published defences against resume gaming are *detectors*: attack-specific
signals bolted onto a ranker, hunting keyword stuffing, repetition, or white
1-point text. Every signature is lexical, and none survives cheap language
models — a candidate who asks an LLM to rewrite their resume for a posting
produces fluent text with no artefact to find, which is *genuinely* closer to
the job description in embedding space and so gets **rewarded** by dense
retrieval.

**Evidence-grounded relevance** puts the defence inside the scoring function
instead of beside it: a skill contributes in proportion to whether the resume's
own work narrative demonstrates it. Claims are cheap, evidence is not, and
because the mechanism never references an attack signature it is attack-agnostic
by construction.

A worked example from the corpus — an LLM-tailored resume against an honest one:

| Scorer | Honest skill score | Attacker skill score | Margin |
|---|---|---|---|
| plain coverage | 0.909 | **1.000** (perfect) | 0.002 |
| **grounded** | 0.800 | **0.509** | **0.116** |

The attacker claims all seven requirements and demonstrates one. Plain coverage
gives them a perfect score; grounding halves it and widens the honest
candidate's margin 58×.

**And a bounded negative result:** fabricate the work narrative too, and
grounding's effect stops being significant (p = 0.079). Grounding does not make
manipulation impossible — it changes what manipulation has to be. Keyword
optimisation is ubiquitous, costless and unfalsifiable; fabricating bullets
about named employers and dates is resume fraud, checkable by reference.
Converting the first into the second is the realistic security goal.

## Novel contributions

### Anti-gaming detector (`talentlens/antigaming.py`)

Five orthogonal signals, each returning auditable evidence:

| Signal | Catches |
|---|---|
| `invisible_text` | Text the same colour as its background, or under 4pt |
| `skill_density` | Skills-per-token far above the rest of the applicant pool |
| `repetition` | One term hammered beyond natural frequency |
| `jd_echo` | Verbatim trigram overlap with the job description |
| `incoherence` | Claimed skills unsupported by the experience narrative (SBERT) |

`invisible_text` is the reason `parsing.py` retains span-level colour and font
metadata rather than returning a plain string — a parser that discards it makes
white-on-white keyword stuffing undetectable by construction. It carries the
highest weight because there is no innocent explanation for it.

The combined penalty is capped, and every signal reports *why* it fired, so a
recruiter can overrule a false positive rather than silently losing a candidate.

### Skill-gap analyser (`talentlens/skillgap.py`)

Standard screening is binary: you have the skill or you don't. That discards the
most actionable information available. Every requirement is classified as
**matched** (held, or implied by the ontology), **transferable** (reachable
within two ontology hops from something held, with the bridging skill and path
reported), or **missing**.

A candidate who knows Django but not Flask is not "missing Flask" in any useful
sense — they are one hop away through Python, and the report says so:

```
Transferable (1): flask via django (django -> python -> flask)
Interview focus: flask, kubernetes
```

### Skill ontology (`data/skill_ontology.yaml`)

182 skills with aliases and parent edges. Expansion is **one-directional**:
Django implies Python; Python does not imply Django. Getting that backwards
would let a generalist match every specialist requirement.

---

## Benchmark results

```bash
python benchmark/generate_corpus.py
python benchmark/run_experiments.py
```

| Ranker | NDCG@5 | NDCG@10 | MRR | MAP | P@5 | ms/CV |
|---|---|---|---|---|---|---|
| tfidf | 0.112 | 0.091 | 0.030 | 0.125 | 0.000 | 0.2 |
| bm25 | 0.070 | 0.058 | 0.029 | 0.112 | 0.000 | 0.1 |
| sbert | 0.109 | 0.169 | 0.103 | 0.211 | 0.000 | 0.1 |
| hybrid (coverage) | **0.950** | 0.869 | 1.000 | 0.711 | 0.800 | 0.8 |
| hybrid_dice | 0.924 | 0.921 | 0.900 | 0.870 | 0.920 | 0.8 |
| **hybrid_grounded** | 0.931 | **0.929** | 1.000 | **0.849** | 0.840 | 3.0 |

Evidence grounding costs ~2% NDCG@5 and improves NDCG@10, MAP and P@5 — so the
adversarial robustness below is not bought with ranking accuracy.

### Adversarial robustness (see RESEARCH.md)

Attack gain = rank percentiles the attacked resume gains over its **identical
control**. Positive means the attack worked. 75 matched pairs, `*` = p < 0.05.

| Defence vs. undefended ranker | A1 kwblock | A2 repet | A3 hidden | A4 LLM-tailored | A5 fabricated |
|---|---|---|---|---|---|
| detector (prior-art style) | +0.000 | +0.000 | **−0.105*** | +0.001 | −0.001 |
| **evidence grounding** | **−0.094*** | **−0.029*** | **−0.089*** | **−0.093*** | −0.020 |
| both | **−0.101*** | **−0.030*** | **−0.244*** | **−0.097*** | **−0.043*** |

The detector significantly reduces **one** attack of five. Grounding — with the
detector off, knowing nothing about any attack — significantly reduces **four**,
including LLM-tailored resumes the detector cannot see (recall 0.07).

**Correction to an earlier version of this README**, which reported detector
precision/recall of 1.000: that figure came from a corpus where all three
lexical attacks were applied to the *same* resume, so hidden-text detection
carried the whole score. Measured per attack, real recall is hidden_text 1.00,
keyword_block 0.00, repetition 0.00, LLM attacks 0.07.

### Validated on real resumes

500 real resumes from the livecareer corpus (`opensporks/resumes`). The first
thing that surfaced was a silent failure in our own parser: real exports put
section headings *inline* (`"HR ADMINISTRATOR    Summary    Dedicated..."`), so
line-based segmentation found **0/40** experience sections, the narrative fell
back to the whole document including the skills list, and grounding reported a
perfect 1.000 for every candidate. After normalising inline headings: **40/40**,
with a real spread of 0.17–1.00. No synthetic corpus would have caught that.

### Fairness audit

**Name substitution** (6 groups, Bertrand & Mullainathan methodology): max score
gap 0.0028, and it localises **entirely to the semantic term** — skill,
experience and education are exactly 0.0000, because structurally they cannot
see the name. Stripping the identifying header before embedding cuts the
semantic gap 36% (0.0069 → 0.0044). Measure → localise → mitigate → re-measure,
which is only possible because the scorer decomposes.

**Writing style — a defect in our own method.** Real resumes rewritten to
describe identical work *without naming the technologies* (skills list left
intact):

| Skill measure | Original | Implicit | Penalty |
|---|---|---|---|
| coverage | 0.3000 | 0.2800 | −6.7% |
| **grounded** | 0.2397 | 0.1313 | **−45.2%** |

Grounding docks candidates who don't name-drop by 45%. That's a writing habit,
not a competence gap. It is a real cost of this contribution, and the `floor`
parameter trades it against robustness rather than removing it:

| floor | A4 attack gain | Style penalty |
|---|---|---|
| 0.0 | **+0.047** | −83.8% |
| **0.4** (default) | +0.100 | −45.2% |
| 1.0 (grounding off) | +0.193 | −6.7% |

We publish the curve rather than a recommended point. Whether the style penalty
constitutes unfairness depends on whether naming habits correlate with protected
characteristics — the corpus has no demographic labels and cannot answer that.

### Read this before quoting any of the above

**The corpus is synthetic and self-labelled. These numbers validate the pipeline
and the metric implementations, not real-world accuracy.** Three specific
caveats:

1. **The hybrid's near-perfect ranking is close to circular.** A resume's
   relevance tier is *defined* by how many required skills, years, and degree
   tiers it was generated with — exactly the fields the hybrid scorer reads. It
   is substantially measuring the label-generating process. Do not report
   NDCG@5 = 1.000 as evidence that this system ranks well.
2. **The lexical baselines are unfairly disadvantaged.** All five job
   descriptions render from one template, so shared boilerplate dominates their
   bag-of-words similarity. Their real weakness is vocabulary mismatch, not this.
3. **Only the adversarial columns are genuinely informative**, because a stuffed
   resume and its honest twin carry identical true relevance and differ only by
   the attack. Those say: the detector caught every planted attack with no false
   positives, and it cut the payoff of stuffing from +0.288 to +0.151 — roughly
   halved, **but not eliminated**. Keyword stuffing still buys a small ranking
   advantage against this system. That is an open limitation, not a solved
   problem.

Getting past all of this needs real resumes with human relevance judgements.

---

## Explanations

Two backends behind one interface. The template backend is the default: it
renders the actual arithmetic and gap breakdown, always works, costs nothing,
and cannot invent a qualification.

```
Rank #1 - Priya Raman - overall 0.829

  Semantic fit    #############....... 0.640 x 0.40 = 0.256
  Skill coverage  ##################.. 0.909 x 0.30 = 0.273   6/7 requirements met
  Experience      #################### 1.000 x 0.20 = 0.200   7.6 yrs vs 5.0 required
  Education       #################### 1.000 x 0.10 = 0.100   Masters vs Bachelors required

  Matched (6): aws, docker, kafka, postgresql, python, rest-api
  Transferable (1): kubernetes via docker
  Interview focus: kubernetes
```

The LLM backends (NVIDIA NIM, Anthropic) turn the same facts into prose. They
are handed **only the extracted facts, never the raw resume** — the grounding
discipline Paper 4 uses to keep its RAG stage honest. The model phrases evidence
it is given; it cannot cite a skill outside the matched list because it never
sees one. Any backend failure falls back to the template rather than raising.

Configure in `.env` (gitignored):

```
NVIDIA_API_KEY=nvapi-...
NVIDIA_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b
```

Explanations are generated for the top N only (default 5) — the latency
mitigation Paper 4 explicitly calls for, since a generative pass over every
resume dominates runtime.

---

## Project layout

```
talentlens/
  config.py       weights, thresholds, gate  (all tunable, all validated)
  schemas.py      dataclasses shared across the pipeline
  parsing.py      PDF/DOCX/TXT -> text + span colour/size metadata
  ontology.py     skill taxonomy: aliases + one-directional inheritance
  extraction.py   sections, skills, experience dates, degrees, contact
  embedding.py    SBERT with chunking, pooling, disk cache
  scoring.py      the four sub-scores and their fusion
  grounding.py    NOVEL: evidence-grounded skill verification (the contribution)
  fairness.py     NOVEL: counterfactual audit with per-component attribution
  antigaming.py   five-signal stuffing detector (prior-art baseline)
  skillgap.py     matched / transferable / missing analyser
  ranking.py      hybrid + TF-IDF / BM25 / SBERT baselines + RRF fusion
  explain.py      template | NVIDIA | Claude, with automatic fallback
  pipeline.py     one-call screening API
app/              Streamlit recruiter dashboard
benchmark/
  generate_corpus.py      synthetic corpus + lexical attacks A1-A3
  generate_adversarial.py LLM-tailored attacks A4-A5 (needs NVIDIA_API_KEY)
  run_attack_study.py     matched-pair adversarial study
  run_experiments.py      ranking-utility benchmark
  fetch_real_resumes.py   real corpus (gitignored, never redistributed)
  run_fairness_audit.py   name + writing-style counterfactuals
  run_tradeoff.py         robustness-fairness curve
tests/            192 tests
```

---

## Tests

```bash
python -m pytest tests/ -q     # 192 passed
```

NDCG, MRR and MAP are asserted against hand-computed values rather than against
whatever the implementation currently returns, which is the only way a metric
test catches a real error.

---

## Known limitations

- **Stuffing still pays a little.** The detector halves the advantage but does
  not remove it.
- **Scanned/image PDFs** produce no text. `parsing.looks_scanned()` flags them;
  there is no OCR.
- **Experience parsing** relies on recognisable date ranges. Unusual layouts
  fall back to self-reported claims, which are weaker evidence.
- **The ontology is hand-built** (182 skills). Skills outside it still match
  verbatim but get no alias resolution or inheritance — the cold-start gap the
  study identifies.
- **English only.**
- **The writing-style penalty is measured and unresolved** (45% under grounding).
  Whether it correlates with protected characteristics is unknown — the real
  corpus carries no demographic labels. That study is the main prerequisite
  before any deployment.
- **Fairness results come from one occupational slice** (30-40 IT resumes, one
  US-centric source).

## Citations

If this work is submitted academically, verify the five papers' DOIs and
reported figures directly against IEEE Xplore first. They were taken from the
source study document and have not been independently checked here.
