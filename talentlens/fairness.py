"""Counterfactual fairness auditing for the scoring pipeline.

Most fairness audits of hiring systems treat the scorer as a black box: perturb
an input, watch the final number move, report a disparity. That tells you a
system is unfair without telling you *where* the unfairness enters, which makes
it very hard to fix.

A multi-factor scorer is decomposable by construction, and this module exploits
that. Every audit reports the disparity **per component** - semantic, skill,
experience, education - so a measured gap can be attributed to a specific term
in the formula and then targeted. That is the argument for interpretable scoring
stated as an engineering property rather than a slogan: you cannot repair a
component you cannot isolate.

Two families of perturbation:

**Name substitution.** Replace the candidate's name and nothing else. Any score
movement is caused by the name alone. Our pipeline never *scores* the name, but
it does embed the whole document, so the name reaches the semantic term. The
audit measures whether that matters, and :func:`anonymise_text` provides the
mitigation to re-measure against.

**Writing-style perturbation.** Rewrite the work narrative more tersely, or with
simpler grammar, holding the *facts* constant. This targets a risk specific to
evidence grounding: it rewards resumes that describe work in detail, and
narrative richness is not evenly distributed. It correlates with writing
confidence, first-language background, and access to professional CV coaching.
If grounding penalises terse or non-native phrasing that reports identical
experience, that is a fairness defect in the method itself.

## On names as demographic proxies

The name sets below follow the established resume-audit methodology introduced
by Bertrand & Mullainathan (2004), *Are Emily and Greg More Employable than
Lakisha and Jamal?*, which uses names statistically associated with perceived
race and gender.

The limitation is real and must be stated wherever results are reported: **a
name is a weak proxy.** Individuals of every background carry every kind of
name, association strength varies by region and generation, and the groupings
are coarse. What this audit measures is *name-induced score disparity* - which
is exactly the harm channel that matters for an automated screener, since the
name is all the system ever sees - not the fairness of outcomes for actual
demographic groups. Do not report it as the latter.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .schemas import ParsedResume, SubScores

# --------------------------------------------------------------------------
# Name sets
# --------------------------------------------------------------------------

#: Group -> (first names, last names). The first four groups follow Bertrand &
#: Mullainathan (2004). The South Asian and East Asian sets are added because
#: they are a large share of the applicant pool this system would actually see,
#: and because omitting them would make the audit silently US-centric.
NAME_SETS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "white_female": (
        ("Allison", "Anne", "Emily", "Jill", "Laurie", "Kristen", "Meredith", "Sarah"),
        ("Baker", "Kelly", "McCarthy", "Murphy", "Murray", "Ryan", "Sullivan", "Walsh"),
    ),
    "white_male": (
        ("Brad", "Brendan", "Geoffrey", "Greg", "Brett", "Jay", "Matthew", "Todd"),
        ("Baker", "Kelly", "McCarthy", "Murphy", "Murray", "Ryan", "Sullivan", "Walsh"),
    ),
    "black_female": (
        ("Aisha", "Ebony", "Keisha", "Kenya", "Lakisha", "Latonya", "Latoya", "Tanisha"),
        ("Jackson", "Jones", "Robinson", "Washington", "Williams", "Banks", "Charles", "Booker"),
    ),
    "black_male": (
        ("Darnell", "Hakim", "Jamal", "Jermaine", "Kareem", "Leroy", "Rasheed", "Tyrone"),
        ("Jackson", "Jones", "Robinson", "Washington", "Williams", "Banks", "Charles", "Booker"),
    ),
    "south_asian": (
        ("Priya", "Rahul", "Ananya", "Vikram", "Arjun", "Meera", "Aditya", "Kavya"),
        ("Sharma", "Patel", "Iyer", "Reddy", "Nair", "Desai", "Joshi", "Menon"),
    ),
    "east_asian": (
        ("Wei", "Mei", "Hiroshi", "Yuki", "Jin", "Ling", "Haruto", "Soo-jin"),
        ("Chen", "Wang", "Tanaka", "Kim", "Nguyen", "Liu", "Park", "Sato"),
    ),
}


def name_variants(group: str, count: int = 8) -> list[str]:
    """Deterministic full names for a group, pairing first and last names."""
    firsts, lasts = NAME_SETS[group]
    return [f"{firsts[i % len(firsts)]} {lasts[i % len(lasts)]}" for i in range(count)]


# --------------------------------------------------------------------------
# Perturbations
# --------------------------------------------------------------------------

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?:\+\d{1,3}[\s-]?)?(?:\(\d{2,4}\)[\s-]?)?\d{3,5}[\s-]?\d{3,4}[\s-]?\d{0,4}")


def substitute_name(text: str, original: str, replacement: str) -> str:
    """Replace a candidate's name throughout, including the email local part.

    Swapping only the display name would leave "priya.raman@example.com" in
    place, and the embedder reads that too - the counterfactual has to be
    complete or the measured effect is diluted.
    """
    result = text
    if original:
        result = re.sub(re.escape(original), replacement, result, flags=re.IGNORECASE)
        for part in original.split():
            if len(part) > 2:
                result = re.sub(
                    rf"(?<![A-Za-z]){re.escape(part)}(?![A-Za-z])",
                    replacement.split()[0],
                    result,
                )
        handle = original.lower().replace(" ", ".")
        result = result.replace(handle, replacement.lower().replace(" ", "."))
    return result


def anonymise_text(resume: ParsedResume) -> str:
    """Resume text with the identifying header removed.

    The candidate block - name, email, phone - sits above the first section
    heading and lands in the ``other`` bucket. Dropping it and redacting any
    stragglers leaves the substance untouched: nothing in the scoring formula
    needs to know who the applicant is.

    This is the mitigation the name audit measures against.
    """
    keep = [
        body
        for name, body in resume.sections.items()
        if name != "other" and body.strip()
    ]
    text = "\n\n".join(keep).strip() or resume.text
    if resume.name:
        text = re.sub(re.escape(resume.name), "", text, flags=re.IGNORECASE)
    text = _EMAIL.sub("", text)
    return _PHONE.sub("", text)


def terse_rewrite(text: str, keep_ratio: float = 0.5) -> str:
    """Shorten each narrative line while preserving the terms it names.

    Simulates a candidate who writes compactly but still name-drops their
    tools. Note this *raises* skill-term density, so it is a control rather
    than the real risk - see :func:`implicit_rewrite`.
    """
    out = []
    for line in text.splitlines():
        words = line.split()
        if len(words) <= 6:
            out.append(line)
            continue
        keep = max(5, int(len(words) * keep_ratio))
        head = words[:keep]
        tail_terms = [
            w for w in words[keep:] if w[:1].isupper() or any(c in w for c in "+#./")
        ]
        out.append(" ".join(head + tail_terms))
    return "\n".join(out)


def implicit_rewrite(resume: ParsedResume, ontology=None) -> str:
    """Remove technology names from the *narrative*, leaving the skills list intact.

    This is the perturbation that actually tests evidence grounding's fairness
    risk, and it took a failed first attempt to see why.

    The obvious test - shorten the prose - is the wrong one: trimming filler
    while keeping capitalised tokens *increases* skill-term density and makes
    grounding easier, so it measures nothing. The real risk is different. Some
    candidates write "Administered the campus network and resolved escalations"
    where others write "Administered the Cisco campus network using BGP".
    Identical work, and the first writer is not less qualified - they simply do
    not name-drop. Resume-writing conventions vary by culture, seniority, and
    whether someone has had professional CV coaching.

    Evidence grounding reads exactly those names. If the first candidate is
    scored lower, grounding is rewarding a writing habit rather than
    demonstrated competence, and that is a defect in the method.

    Only the narrative sections are stripped; the skills list is untouched, so
    the candidate still *claims* everything they claimed before.
    """
    from .grounding import EVIDENCE_SECTIONS, _surface_forms
    from .ontology import get_ontology

    ontology = ontology or get_ontology()

    forms: set[str] = set()
    for skill in resume.skills:
        forms |= _surface_forms(skill, ontology)
    # Longest first, so "network security" is removed before "network".
    ordered = sorted((f for f in forms if len(f) > 2), key=len, reverse=True)

    rebuilt = []
    for name, body in resume.sections.items():
        if name in EVIDENCE_SECTIONS and body.strip():
            stripped = body
            for form in ordered:
                stripped = re.sub(
                    rf"(?<![A-Za-z0-9]){re.escape(form)}(?![A-Za-z0-9])",
                    "the system",
                    stripped,
                    flags=re.IGNORECASE,
                )
            rebuilt.append(f"{name.upper()}\n{stripped}")
        elif body.strip():
            rebuilt.append(f"{name.upper()}\n{body}")
    return "\n\n".join(rebuilt)


# --------------------------------------------------------------------------
# Audit results
# --------------------------------------------------------------------------


@dataclass
class GroupResult:
    group: str
    totals: list[float] = field(default_factory=list)
    components: dict[str, list[float]] = field(default_factory=dict)
    ranks: list[int] = field(default_factory=list)

    def add(self, scores: SubScores, rank: int) -> None:
        self.totals.append(scores.total)
        self.ranks.append(rank)
        for key in ("semantic", "skill", "experience", "education"):
            self.components.setdefault(key, []).append(getattr(scores, key))

    @property
    def mean_total(self) -> float:
        return statistics.mean(self.totals) if self.totals else float("nan")

    def mean_component(self, key: str) -> float:
        values = self.components.get(key, [])
        return statistics.mean(values) if values else float("nan")


@dataclass
class AuditReport:
    """Disparity across groups, decomposed by scoring component."""

    groups: dict[str, GroupResult]
    label: str = ""

    @property
    def group_means(self) -> dict[str, float]:
        return {name: g.mean_total for name, g in self.groups.items()}

    @property
    def max_gap(self) -> float:
        """Largest difference between any two group means on the total score."""
        means = [m for m in self.group_means.values() if m == m]
        return max(means) - min(means) if len(means) > 1 else 0.0

    def component_gap(self, key: str) -> float:
        means = [g.mean_component(key) for g in self.groups.values()]
        means = [m for m in means if m == m]
        return max(means) - min(means) if len(means) > 1 else 0.0

    @property
    def component_gaps(self) -> dict[str, float]:
        return {
            key: self.component_gap(key)
            for key in ("semantic", "skill", "experience", "education")
        }

    @property
    def dominant_component(self) -> str:
        """Which term carries most of the disparity - the actionable output."""
        gaps = self.component_gaps
        return max(gaps, key=lambda k: gaps[k]) if gaps else ""

    def selection_counts(self, top_k: int) -> dict[str, int]:
        return {
            name: sum(1 for r in group.ranks if r <= top_k)
            for name, group in self.groups.items()
        }

    def disparate_impact(self, top_k: int) -> float:
        """Ratio of the worst group's top-K selection rate to the best group's.

        The US EEOC "four-fifths rule" treats a ratio below 0.80 as evidence of
        adverse impact. It is a screening heuristic, not a legal verdict.

        **This statistic is extremely noisy at small K.** With six groups and a
        top-10 cut, each group expects fewer than two selections, so a single
        candidate moving one position swings the ratio from 1.00 to 0.50. Read
        it together with :meth:`selection_counts` and :meth:`di_is_reliable`,
        and do not report a ratio whose expected count per group is below ~5.
        """
        rates = []
        for group in self.groups.values():
            if not group.ranks:
                continue
            rates.append(sum(1 for r in group.ranks if r <= top_k) / len(group.ranks))
        rates = [r for r in rates if r == r]
        if not rates or max(rates) == 0:
            return float("nan")
        return min(rates) / max(rates)

    def di_is_reliable(self, top_k: int, min_expected: int = 5) -> bool:
        """Whether top_k is large enough for the ratio to mean anything."""
        active = [g for g in self.groups.values() if g.ranks]
        if not active:
            return False
        return top_k / len(active) >= min_expected

    def summary(self, top_k: int = 5) -> str:
        lines = [f"{self.label}: max total-score gap {self.max_gap:.4f}"]
        for key, gap in sorted(self.component_gaps.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {key:<12} gap {gap:.4f}")
        ratio = self.disparate_impact(top_k)
        if ratio == ratio:
            verdict = "below 0.80 threshold" if ratio < 0.80 else "within 0.80 threshold"
            lines.append(f"    disparate impact @{top_k}: {ratio:.3f} ({verdict})")
        return "\n".join(lines)
