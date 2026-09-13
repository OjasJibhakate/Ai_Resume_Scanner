"""Evidence-grounded skill verification.

Every screening system reviewed in the source study scores what a resume
*claims*. This module scores what it *demonstrates*.

The distinction is not cosmetic. A skills section is a list of assertions, cheap
to write and impossible to falsify from the document alone. An experience
narrative is a set of specific claims about work actually performed. When a
candidate writes::

    SKILLS
    Python, Kubernetes, Kafka, Terraform, Go, Rust, Scala

    EXPERIENCE
    Backend Engineer, 2021-Present
      Built payment services in Python on PostgreSQL.

only one of those seven skills is *evidenced*. The rest are unsupported
assertions. Existing scorers cannot tell the difference - set overlap counts all
seven, and a dense encoder sees a document that mentions all seven.

This matters more now than when the reviewed papers were written. Their implicit
threat model is lexical: white text, repeated keywords, a pasted job
description. All of those leave artefacts our lexical detectors catch. But a
candidate who asks a language model to "rewrite my resume for this job" produces
a document with no artefacts at all - fluent, well-formed, no repetition, no
verbatim copying - and it is *genuinely* closer to the job description in
embedding space. Dense retrieval rewards it. The only thing that does not move
is whether the work history actually supports the new claims.

Grounding is therefore doing three jobs at once:

* **Integrity.** Fabricated skills are ungrounded by construction.
* **Scoring.** Requirements satisfied by evidence outrank requirements
  satisfied by assertion.
* **Explanation.** Each matched skill can cite the sentence that justifies it,
  which is a far stronger answer to "why did this candidate rank here?" than a
  cosine value.

Grounding is deliberately **soft**. Real resumes legitimately list skills they
never narrate - there is only so much room - so an unevidenced claim earns
partial credit (``GroundingConfig.floor``) rather than zero. The claim is that
evidence should count for *more*, not that listing should count for nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .config import DEFAULT_GROUNDING, GroundingConfig
from .ontology import SkillOntology, get_ontology
from .schemas import ParsedResume

#: Sections that constitute evidence. The skills section is excluded on
#: purpose: it is the claim under test, not support for it.
EVIDENCE_SECTIONS = ("experience", "projects")

#: Sections searched only when the above are empty, so that a resume with an
#: unconventional layout still gets assessed rather than scoring zero.
FALLBACK_SECTIONS = ("summary", "certifications", "other")

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+|\s*[•·]\s*")
_MIN_SENTENCE_CHARS = 12

EXPLICIT = "explicit"
SEMANTIC = "semantic"
LISTED_ONLY = "listed_only"
ABSENT = "absent"


@dataclass(frozen=True)
class SkillEvidence:
    """Where, and how strongly, a claimed skill is supported."""

    skill: str
    #: Evidence strength in [0, 1]. 1.0 = named in the work narrative.
    grounding: float
    #: One of EXPLICIT / SEMANTIC / LISTED_ONLY / ABSENT.
    kind: str
    #: The narrative sentence that supports the claim, if any.
    evidence: str = ""
    similarity: float = 0.0

    @property
    def is_grounded(self) -> bool:
        return self.kind in (EXPLICIT, SEMANTIC)

    def describe(self) -> str:
        if self.kind == EXPLICIT:
            return f"{self.skill}: demonstrated - {self.evidence[:90]!r}"
        if self.kind == SEMANTIC:
            return (
                f"{self.skill}: implied (sim {self.similarity:.2f}) - "
                f"{self.evidence[:90]!r}"
            )
        if self.kind == LISTED_ONLY:
            return f"{self.skill}: listed but not evidenced in the work history"
        return f"{self.skill}: not present"


@dataclass(frozen=True)
class GroundingReport:
    evidence: dict[str, SkillEvidence]
    #: Number of narrative sentences available as evidence. When this is very
    #: small the report is unreliable and callers should treat it as abstaining.
    sentence_count: int = 0

    def get(self, skill: str) -> SkillEvidence:
        return self.evidence.get(
            skill, SkillEvidence(skill=skill, grounding=0.0, kind=ABSENT)
        )

    def strength(self, skill: str) -> float:
        return self.get(skill).grounding

    @property
    def grounded(self) -> tuple[SkillEvidence, ...]:
        return tuple(e for e in self.evidence.values() if e.is_grounded)

    @property
    def ungrounded(self) -> tuple[SkillEvidence, ...]:
        return tuple(e for e in self.evidence.values() if not e.is_grounded)

    @property
    def grounded_fraction(self) -> float:
        """Share of claimed skills with narrative support.

        This is the headline integrity statistic: a resume whose skills section
        has drifted away from its work history scores low here.
        """
        if not self.evidence:
            return 1.0
        return len(self.grounded) / len(self.evidence)

    @property
    def reliable(self) -> bool:
        return self.sentence_count >= 3


# --------------------------------------------------------------------------
# Narrative extraction
# --------------------------------------------------------------------------


def narrative_sentences(resume: ParsedResume) -> list[str]:
    """Sentences from the parts of a resume that describe work performed."""
    text = "\n".join(resume.section(name) for name in EVIDENCE_SECTIONS).strip()
    if not text:
        text = "\n".join(resume.section(name) for name in FALLBACK_SECTIONS).strip()
    if not text:
        # Unsegmented document: fall back to the whole thing rather than
        # reporting every skill as unevidenced, which would be an artefact of
        # parsing rather than a property of the candidate.
        text = resume.text

    sentences = []
    for chunk in _SENTENCE_SPLIT.split(text):
        cleaned = chunk.strip(" \t-–—*")
        if len(cleaned) >= _MIN_SENTENCE_CHARS:
            sentences.append(cleaned)
    return sentences


def _surface_forms(skill: str, ontology: SkillOntology) -> set[str]:
    """Every spelling that counts as naming ``skill``, including descendants.

    Descendants are included because evidence flows upward: a sentence about
    shipping Django is evidence of Python.
    """
    forms: set[str] = set()
    for name in {skill} | set(ontology.descendants(skill)):
        node = ontology.nodes.get(name)
        if node is None:
            forms.add(name.replace("-", " "))
            continue
        forms.add(node.name)
        forms.add(node.name.replace("-", " "))
        forms.update(node.aliases)
    return {f for f in forms if f}


def _mentions(sentence_lower: str, forms: Iterable[str]) -> str | None:
    for form in forms:
        # Word-boundary match so "go" does not fire inside "algorithm" and
        # "r" does not fire inside every word containing the letter.
        if re.search(rf"(?<![a-z0-9]){re.escape(form)}(?![a-z0-9])", sentence_lower):
            return form
    return None


# --------------------------------------------------------------------------
# Grounding
# --------------------------------------------------------------------------


def ground_skills(
    resume: ParsedResume,
    skills: Iterable[str],
    ontology: SkillOntology | None = None,
    embedder=None,
    config: GroundingConfig = DEFAULT_GROUNDING,
    use_semantic: bool = True,
) -> GroundingReport:
    """Verify each claimed skill against the resume's own work narrative.

    Two passes. The lexical pass asks whether the skill (or something that
    implies it) is *named* in the narrative - unambiguous, cheap, and the
    strongest form of evidence. The semantic pass catches the case where the
    work is described without naming the technology ("containerised the
    services and orchestrated rollouts" evidencing Kubernetes), which is common
    in well-written resumes and is exactly where a bag-of-words check fails.
    """
    ontology = ontology or get_ontology()
    claimed = sorted({s for s in skills if s})
    sentences = narrative_sentences(resume)

    if not claimed:
        return GroundingReport(evidence={}, sentence_count=len(sentences))

    lowered = [s.lower() for s in sentences]
    evidence: dict[str, SkillEvidence] = {}
    unresolved: list[str] = []

    # -- lexical pass -----------------------------------------------------
    for skill in claimed:
        forms = _surface_forms(skill, ontology)
        hit_sentence = ""
        for sentence, lower in zip(sentences, lowered):
            if _mentions(lower, forms):
                hit_sentence = sentence
                break
        if hit_sentence:
            evidence[skill] = SkillEvidence(
                skill=skill,
                grounding=1.0,
                kind=EXPLICIT,
                evidence=hit_sentence,
                similarity=1.0,
            )
        else:
            unresolved.append(skill)

    # -- semantic pass ----------------------------------------------------
    if unresolved and use_semantic and sentences:
        if embedder is None:
            from .embedding import get_embedder

            embedder = get_embedder()

        from .embedding import cosine_matrix

        sentence_vectors = embedder.encode(sentences)
        # A bare token embeds poorly; a short natural phrase sits much closer to
        # the sentence manifold and makes the similarities comparable.
        queries = [config.query_template.format(skill=s.replace("-", " ")) for s in unresolved]
        skill_vectors = embedder.encode(queries)

        similarities = cosine_matrix(skill_vectors, sentence_vectors)
        for row, skill in enumerate(unresolved):
            best_index = int(similarities[row].argmax())
            best = float(similarities[row][best_index])
            if best >= config.semantic_threshold:
                span = max(1e-9, config.semantic_ceiling - config.semantic_threshold)
                scaled = (best - config.semantic_threshold) / span
                evidence[skill] = SkillEvidence(
                    skill=skill,
                    grounding=float(
                        min(config.semantic_max, config.semantic_max * max(0.0, min(1.0, scaled)))
                    ),
                    kind=SEMANTIC,
                    evidence=sentences[best_index],
                    similarity=best,
                )
            else:
                evidence[skill] = SkillEvidence(
                    skill=skill, grounding=0.0, kind=LISTED_ONLY, similarity=best
                )
    else:
        for skill in unresolved:
            evidence[skill] = SkillEvidence(skill=skill, grounding=0.0, kind=LISTED_ONLY)

    return GroundingReport(evidence=evidence, sentence_count=len(sentences))


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def grounded_coverage(
    held: Iterable[str],
    required: Iterable[str],
    preferred: Iterable[str] = (),
    report: GroundingReport | None = None,
    config: GroundingConfig = DEFAULT_GROUNDING,
    preferred_weight: float = 0.5,
) -> float:
    """Coverage weighted by how well each satisfied requirement is evidenced.

    Each matched requirement contributes ``floor + (1 - floor) * grounding``
    instead of a flat 1.0. With ``floor = 0.4``, merely listing a skill is worth
    40% of demonstrating it. Unmatched requirements contribute nothing, exactly
    as in plain coverage, so this is a strict refinement: set ``floor = 1.0`` and
    it collapses back to :func:`talentlens.scoring.coverage_score`.
    """
    held_set = set(held)
    req = set(required)
    pref = set(preferred) - req

    denominator = len(req) + preferred_weight * len(pref)
    if denominator == 0:
        return 1.0

    def credit(skill: str) -> float:
        if skill not in held_set:
            return 0.0
        if report is None:
            return 1.0
        strength = report.strength(skill)
        return config.floor + (1.0 - config.floor) * strength

    numerator = sum(credit(s) for s in req)
    numerator += preferred_weight * sum(credit(s) for s in pref)
    return float(min(1.0, numerator / denominator))


def evidence_for_requirements(
    report: GroundingReport, required: Iterable[str], held: Iterable[str]
) -> list[SkillEvidence]:
    """The evidence rows a recruiter actually wants to read, most-supported first."""
    held_set = set(held)
    rows = [report.get(s) for s in sorted(set(required)) if s in held_set]
    return sorted(rows, key=lambda e: (-e.grounding, e.skill))
