"""Adversarial anti-gaming detector (novel contribution #1).

Automated screening creates an incentive to game the scorer, and the classic
attacks are well known: paste the job description into the resume in white
1-point text, repeat a keyword thirty times, or dump every technology you have
heard of into a skills block. An unguarded cosine or overlap metric rewards all
three.

Five orthogonal signals, each producing auditable evidence:

1. ``invisible_text``  - text the same colour as its background, or too small to
   read. The strongest signal by far: there is no innocent explanation.
2. ``skill_density``   - skills per token, versus the rest of the corpus.
3. ``repetition``      - one term hammered far beyond natural frequency.
4. ``jd_echo``         - verbatim trigram overlap with the job description.
5. ``incoherence``     - claimed skills that the experience narrative never
   supports, measured in SBERT space.

They are deliberately independent: a resume can trip one honestly (a genuinely
broad engineer has a long tools list), but tripping several at once is hard to
do by accident. The combined penalty is capped by
``AntiGamingConfig.max_penalty`` so a false positive demotes a candidate rather
than deleting them, and every signal reports *why* it fired so a recruiter can
overrule it.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .config import DEFAULT_ANTIGAMING, AntiGamingConfig
from .schemas import GamingReport, GamingSignal, JobSpec, ParsedResume

SIGNAL_NAMES = (
    "invisible_text",
    "skill_density",
    "repetition",
    "jd_echo",
    "incoherence",
)

_WORD_RE = re.compile(r"[a-z][a-z+#.]{1,}")

#: Small stop list, kept local so this module never has to load spaCy - the
#: detector runs on every resume and must stay cheap.
_STOPWORDS = frozenset(
    """
    the and for with from that this have has had was were are is be been being
    our your their his her its not but all any can will would should could may
    into over under more most other such than then them they you i a an of to in
    on at by as it or if we us me my he she do does did done using used use
    work working worked experience team teams project projects role roles year
    years including include includes across within also new via per etc
    """.split()
)


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _trigrams(tokens: list[str]) -> set[tuple[str, str, str]]:
    return {tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)}  # type: ignore[misc]


def _clamp(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _rgb_distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# --------------------------------------------------------------------------
# Corpus statistics
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusStats:
    """Population statistics used to judge a resume relative to its peers.

    Without these the density signal falls back to a fixed threshold, which is
    less fair: what counts as an unusually dense skills list depends on whether
    you are screening backend engineers or nurses.
    """

    density_mean: float = 0.0
    density_std: float = 0.0
    sample_size: int = 0

    @classmethod
    def from_densities(cls, densities: list[float]) -> "CorpusStats":
        n = len(densities)
        if n == 0:
            return cls()
        mean = sum(densities) / n
        variance = sum((d - mean) ** 2 for d in densities) / n
        return cls(density_mean=mean, density_std=math.sqrt(variance), sample_size=n)

    @property
    def usable(self) -> bool:
        # Below ~8 samples the standard deviation is too unstable to threshold.
        return self.sample_size >= 8 and self.density_std > 1e-9


def skill_density(resume: ParsedResume) -> float:
    """Raw skills per token.

    Uses the *raw* skill set, not the ontology-expanded one: expansion adds
    implied ancestors, which would make a specialist look like a stuffer.
    """
    total = len(_tokens(resume.text))
    if total == 0:
        return 0.0
    return len(resume.skills) / total


# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------


def detect_invisible_text(
    resume: ParsedResume, config: AntiGamingConfig = DEFAULT_ANTIGAMING
) -> GamingSignal:
    document = resume.document
    spans = [s for s in document.spans if s.text.strip()]
    total_chars = sum(len(s.text) for s in spans)

    if total_chars == 0:
        return GamingSignal(
            "invisible_text", False, 0.0, "no span metadata available for this format"
        )

    hidden_chars = 0
    colour_hits: list[str] = []
    size_hits: list[str] = []

    for span in spans:
        rgb = span.rgb
        background = document.background_for(span.page)
        bg_rgb = ((background >> 16) & 0xFF, (background >> 8) & 0xFF, background & 0xFF)

        is_hidden = False
        if rgb is not None and _rgb_distance(rgb, bg_rgb) < config.invisible_color_distance:
            is_hidden = True
            if len(colour_hits) < 3:
                colour_hits.append(
                    f"#{span.color:06x} on #{background:06x}: {span.text.strip()[:40]!r}"
                )
        if 0.0 < span.size < config.min_readable_font_size:
            is_hidden = True
            if len(size_hits) < 3:
                size_hits.append(f"{span.size:.1f}pt: {span.text.strip()[:40]!r}")

        if is_hidden:
            hidden_chars += len(span.text)

    fraction = hidden_chars / total_chars
    triggered = fraction >= 0.005
    # 5% of a resume being invisible is already a total giveaway, so severity
    # saturates there rather than scaling across the whole 0-100% range.
    severity = _clamp(fraction / 0.05) if triggered else 0.0

    if not triggered:
        evidence = "no invisible or sub-readable text detected"
    else:
        parts = [f"{hidden_chars} chars ({fraction:.1%}) not visibly rendered"]
        if colour_hits:
            parts.append("same-as-background: " + "; ".join(colour_hits))
        if size_hits:
            parts.append("sub-readable size: " + "; ".join(size_hits))
        evidence = " | ".join(parts)

    return GamingSignal("invisible_text", triggered, severity, evidence)


def detect_skill_density(
    resume: ParsedResume,
    config: AntiGamingConfig = DEFAULT_ANTIGAMING,
    stats: CorpusStats | None = None,
) -> GamingSignal:
    density = skill_density(resume)
    token_count = len(_tokens(resume.text))

    if token_count < 50:
        return GamingSignal(
            "skill_density", False, 0.0, "document too short to assess density"
        )

    if stats is not None and stats.usable:
        z = (density - stats.density_mean) / stats.density_std
        triggered = z > config.density_zscore_threshold
        severity = _clamp((z - config.density_zscore_threshold) / config.density_zscore_threshold)
        evidence = (
            f"{len(resume.skills)} skills in {token_count} tokens "
            f"(density {density:.3f}, z={z:+.2f} vs corpus mean {stats.density_mean:.3f})"
        )
    else:
        triggered = density > config.density_absolute_threshold
        severity = _clamp(
            (density - config.density_absolute_threshold) / config.density_absolute_threshold
        )
        evidence = (
            f"{len(resume.skills)} skills in {token_count} tokens "
            f"(density {density:.3f}, absolute threshold {config.density_absolute_threshold})"
        )

    return GamingSignal("skill_density", triggered, severity if triggered else 0.0, evidence)


def detect_repetition(
    resume: ParsedResume, config: AntiGamingConfig = DEFAULT_ANTIGAMING
) -> GamingSignal:
    tokens = [
        t
        for t in _tokens(resume.text)
        if len(t) >= config.repetition_min_term_length and t not in _STOPWORDS
    ]
    # A per-1000-word rate estimated from a few dozen words is noise: a genuine
    # DevOps CV that says "Docker" five times in a terse 50-word body extrapolates
    # to ~96 per 1k and looks like an attack. Require enough text for the rate to
    # mean something.
    if len(tokens) < 120:
        return GamingSignal(
            "repetition", False, 0.0, "document too short to assess repetition"
        )

    counts = Counter(tokens)
    term, count = counts.most_common(1)[0]
    per_1k = count / len(tokens) * 1000.0

    # Rate alone is not enough; require an absolute number of repeats too, so a
    # merely emphatic resume is not treated as a stuffed one.
    triggered = per_1k > config.repetition_per_1k_threshold and count >= 8
    severity = (
        _clamp((per_1k - config.repetition_per_1k_threshold) / config.repetition_per_1k_threshold)
        if triggered
        else 0.0
    )
    evidence = (
        f"{term!r} appears {count}x ({per_1k:.1f} per 1k content words; "
        f"threshold {config.repetition_per_1k_threshold})"
    )
    return GamingSignal("repetition", triggered, severity, evidence)


def detect_jd_echo(
    resume: ParsedResume,
    job: JobSpec | None,
    config: AntiGamingConfig = DEFAULT_ANTIGAMING,
) -> GamingSignal:
    if job is None:
        return GamingSignal("jd_echo", False, 0.0, "no job description supplied")

    jd_trigrams = _trigrams(_tokens(job.text))
    if len(jd_trigrams) < config.jd_echo_min_trigrams:
        return GamingSignal(
            "jd_echo", False, 0.0, "job description too short for reliable comparison"
        )

    resume_trigrams = _trigrams(_tokens(resume.text))
    shared = jd_trigrams & resume_trigrams
    ratio = len(shared) / len(jd_trigrams)

    triggered = ratio > config.jd_echo_threshold
    severity = (
        _clamp((ratio - config.jd_echo_threshold) / max(1e-9, 1.0 - config.jd_echo_threshold))
        if triggered
        else 0.0
    )

    sample = "; ".join(" ".join(t) for t in list(sorted(shared))[:3])
    evidence = (
        f"{len(shared)}/{len(jd_trigrams)} job-description trigrams appear verbatim "
        f"({ratio:.1%}, threshold {config.jd_echo_threshold:.0%})"
    )
    if triggered and sample:
        evidence += f" | e.g. {sample!r}"
    return GamingSignal("jd_echo", triggered, severity, evidence)


def detect_incoherence(
    resume: ParsedResume,
    embedder=None,
    config: AntiGamingConfig = DEFAULT_ANTIGAMING,
) -> GamingSignal:
    """Do the claimed skills match the story the experience section tells?

    A stuffed skills block is semantically adrift from the narrative around it.
    This is the one signal that needs the embedder, and it is nearly free
    because the model is already resident for the semantic score.
    """
    skills_text = resume.section("skills")
    narrative = " ".join(
        resume.section(name) for name in ("experience", "projects", "summary")
    ).strip()

    if (
        len(_tokens(skills_text)) < config.incoherence_min_tokens
        or len(_tokens(narrative)) < config.incoherence_min_tokens
    ):
        return GamingSignal(
            "incoherence", False, 0.0, "skills or experience section too short to compare"
        )

    if embedder is None:
        from .embedding import get_embedder

        embedder = get_embedder()

    from .embedding import cosine_similarity

    similarity = cosine_similarity(
        embedder.encode_document(skills_text), embedder.encode_document(narrative)
    )
    triggered = similarity < config.incoherence_threshold
    severity = (
        _clamp((config.incoherence_threshold - similarity) / config.incoherence_threshold)
        if triggered
        else 0.0
    )
    evidence = (
        f"skills-vs-experience similarity {similarity:.3f} "
        f"(threshold {config.incoherence_threshold}); "
        + ("claimed skills are not reflected in the work history" if triggered else "consistent")
    )
    return GamingSignal("incoherence", triggered, severity, evidence)


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def analyse(
    resume: ParsedResume,
    job: JobSpec | None = None,
    embedder=None,
    config: AntiGamingConfig = DEFAULT_ANTIGAMING,
    stats: CorpusStats | None = None,
    use_semantic: bool = True,
) -> GamingReport:
    """Run every signal and combine them into a capped penalty.

    ``use_semantic=False`` skips the embedding-based coherence check, which is
    what the lexical-only benchmark baselines want.
    """
    signals = [
        detect_invisible_text(resume, config),
        detect_skill_density(resume, config, stats),
        detect_repetition(resume, config),
        detect_jd_echo(resume, job, config),
    ]
    if use_semantic:
        signals.append(detect_incoherence(resume, embedder, config))
    else:
        signals.append(
            GamingSignal("incoherence", False, 0.0, "semantic check disabled")
        )

    weighted = sum(
        weight * signal.severity
        for weight, signal in zip(config.signal_weights, signals)
    )
    penalty = float(min(config.max_penalty, config.max_penalty * weighted))

    return GamingReport(
        signals=tuple(signals),
        penalty=penalty,
        flagged=any(s.triggered for s in signals),
    )


def corpus_stats(resumes: list[ParsedResume]) -> CorpusStats:
    """Build density statistics across a batch, for peer-relative judgement."""
    return CorpusStats.from_densities([skill_density(r) for r in resumes])
