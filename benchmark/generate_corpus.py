"""Generate a labelled synthetic resume corpus for offline benchmarking.

Why synthetic: computing NDCG and MRR requires graded relevance labels, and no
public resume dataset ships per-JD relevance judgements. Generating the corpus
means the labels are known by construction.

**This is a validation harness, not evidence of real-world accuracy.** We author
both the resumes and their labels, so a good score here demonstrates that the
pipeline and the metrics work - nothing more. Any claim about real screening
performance needs real resumes with human relevance judgements. The README
repeats this warning; please do not quote benchmark numbers without it.

Resumes are written as real PDFs through PyMuPDF rather than plain text, so the
benchmark exercises the production parsing path - including the span-level
colour and font metadata that the anti-gaming detector depends on. The
adversarial subset contains genuine white-on-white and 1pt text, so signal 1 is
tested against the real thing rather than a mock.

Usage::

    python benchmark/generate_corpus.py --out benchmark/corpus --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz  # type: ignore[no-redef]


# --------------------------------------------------------------------------
# Role definitions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Role:
    key: str
    title: str
    required: tuple[str, ...]
    preferred: tuple[str, ...]
    min_years: int
    min_degree: str
    duties: tuple[str, ...]
    employers: tuple[str, ...]


ROLES: tuple[Role, ...] = (
    Role(
        key="backend",
        title="Senior Backend Engineer",
        required=("Python", "REST APIs", "PostgreSQL", "Docker"),
        preferred=("AWS", "Kubernetes", "Kafka"),
        min_years=5,
        min_degree="Bachelor",
        duties=(
            "Designed and shipped {skill} services handling {n}k requests per second",
            "Owned the {skill} data layer and cut p99 latency by {n}%",
            "Built idempotent payment flows backed by {skill}",
            "Migrated a monolith to {skill}-based microservices",
            "Introduced {skill} to the deployment pipeline, halving release time",
        ),
        employers=("Zeta Systems", "Nimbus Labs", "Corewave", "Farley Pay", "Octane Retail"),
    ),
    Role(
        key="data",
        title="Data Scientist",
        required=("Python", "Machine Learning", "pandas", "SQL"),
        preferred=("PyTorch", "NLP", "Spark"),
        min_years=3,
        min_degree="Master",
        duties=(
            "Built {skill} models that lifted conversion by {n}%",
            "Ran {skill} experiments across {n} million user sessions",
            "Productionised a churn model using {skill}",
            "Developed {skill} feature pipelines over warehouse data",
            "Presented {skill} findings to product and leadership",
        ),
        employers=("Helio Analytics", "Brightside Health", "Kestrel Media", "Vantage Retail"),
    ),
    Role(
        key="frontend",
        title="Frontend Developer",
        required=("JavaScript", "React", "CSS", "HTML"),
        preferred=("TypeScript", "Next.js", "Redux"),
        min_years=3,
        min_degree="Bachelor",
        duties=(
            "Rebuilt the checkout flow in {skill}, lifting completion by {n}%",
            "Established a {skill} component library used by {n} teams",
            "Cut bundle size by {n}% through {skill} code splitting",
            "Implemented accessible, responsive layouts with {skill}",
            "Drove the migration from legacy templates to {skill}",
        ),
        employers=("Lantern Studio", "Pagecraft", "Moss Commerce", "Indigo Travel"),
    ),
    Role(
        key="devops",
        title="DevOps Engineer",
        required=("Docker", "Kubernetes", "Terraform", "AWS"),
        preferred=("CI/CD", "Ansible", "Prometheus"),
        min_years=4,
        min_degree="Bachelor",
        duties=(
            "Ran {n}-node {skill} clusters across three regions",
            "Codified all infrastructure in {skill}, eliminating manual provisioning",
            "Cut deployment time by {n}% by rebuilding the {skill} pipeline",
            "Instrumented services with {skill} and defined the on-call SLOs",
            "Led the migration of {n} services to {skill}",
        ),
        employers=("Northwind Cloud", "Arclight Systems", "Beacon Freight", "Pallas Bank"),
    ),
    Role(
        key="network",
        title="Network Engineer",
        required=("Networking", "TCP/IP", "Routing", "Firewall"),
        preferred=("BGP", "Cisco", "VPN"),
        min_years=4,
        min_degree="Bachelor",
        duties=(
            "Operated {skill} across {n} branch sites",
            "Designed the {skill} topology for a {n}-floor campus network",
            "Hardened perimeter {skill} rules and cut incidents by {n}%",
            "Ran {skill} peering with three upstream providers",
            "Led the {skill} refresh covering {n} switches",
        ),
        employers=("Halcyon Telecom", "Ridgeline ISP", "Summit Logistics", "Cobalt Health"),
    ),
)

ROLE_BY_KEY = {role.key: role for role in ROLES}

#: Every resume gets at least this many experience bullets, so narrative length
#: does not correlate with candidate strength.
_MIN_BULLETS = 5

#: Accomplishment bullets that name no technology. Real resumes are full of
#: these, and they are what stops "amount of narrative" from being a proxy for
#: "amount of evidence" in the grounding experiment.
GENERIC_DUTIES: list[str] = [
    "Led a team of {n} engineers through a full delivery cycle",
    "Reduced production incidents by {n}% through better release discipline",
    "Mentored {n} junior engineers and ran the weekly design review",
    "Partnered with product managers to scope and ship {n} quarterly initiatives",
    "Cut onboarding time for new hires from weeks to {n} days",
    "Improved test coverage across the core service by {n}%",
    "Ran the on-call rotation and authored the incident response runbook",
    "Presented quarterly delivery metrics to {n} stakeholder groups",
]

#: Roles that share enough real skill overlap that a strong candidate in one is
#: a partially relevant candidate for the other. ``ADJACENCY[from][to] = grade``
#: is the relevance a tier-3 resume of role ``from`` earns against role ``to``.
#: Asymmetric on purpose - a backend engineer is a plausible devops hire more
#: readily than a network engineer is a plausible data scientist.
ADJACENCY: dict[str, dict[str, int]] = {
    "backend": {"devops": 1, "data": 1, "frontend": 1},
    "devops": {"backend": 1, "network": 1},
    "data": {"backend": 1},
    "frontend": {"backend": 1},
    "network": {"devops": 1},
}

FIRST_NAMES = (
    "Priya Aditya Nikhil Sneha Rahul Ananya Vikram Meera Arjun Kavya Rohan Ishita "
    "Daniel Laura Marcus Elena Tobias Nadia Owen Freya Samuel Chloe Ethan Maya "
    "Hassan Leila Kofi Amara Diego Sofia Yuki Haruto Mei Ravi Tara Nabil"
).split()

LAST_NAMES = (
    "Raman Sharma Iyer Nair Kapoor Desai Menon Bose Chatterjee Reddy Bhatt Joshi "
    "Whitfield Okafor Lindqvist Moreau Alvarez Kowalski Hayashi Petrov Mbeki Silva "
    "Nakamura Castillo Ferreira Novak Haddad Bergman Tan Osei"
)
LAST_NAMES = LAST_NAMES.split()

DEGREE_TEXT = {
    "None": None,
    "Associate": "Diploma in Information Technology",
    "Bachelor": "B.Tech in Computer Science",
    "Master": "M.Tech in Computer Science",
    "Doctorate": "PhD in Computer Science",
}

DEGREE_ORDER = ("None", "Associate", "Bachelor", "Master", "Doctorate")

UNIVERSITIES = (
    "National Institute of Technology",
    "State Technical University",
    "Riverside Institute of Technology",
    "Capital University",
    "Northgate University",
)


# --------------------------------------------------------------------------
# Relevance tiers
# --------------------------------------------------------------------------
#
# Tier is the graded relevance label (0-3) a resume carries for the job of its
# own role. Against every *other* role's job the label is 0.

# ``narrated_frac`` is the share of a candidate's role-relevant skills that the
# experience section actually demonstrates. Stronger candidates have more to say
# about the work they did, so they narrate more of what they claim - but nobody
# narrates everything, which is what makes the grounding signal non-trivial.
TIER_PLAN = {
    3: {"n": 4, "required_frac": 1.00, "preferred": 2, "years_delta": (2, 5),
        "degree_delta": 0, "narrated_frac": 0.80},
    2: {"n": 5, "required_frac": 0.75, "preferred": 1, "years_delta": (0, 1),
        "degree_delta": 0, "narrated_frac": 0.65},
    1: {"n": 5, "required_frac": 0.35, "preferred": 0, "years_delta": (-2, -1),
        "degree_delta": -1, "narrated_frac": 0.50},
}


@dataclass
class ResumeMeta:
    candidate_id: str
    role: str
    tier: int
    name: str
    years: float
    degree: str
    skills: list[str]
    #: Skills the experience section actually demonstrates.
    narrated_skills: list[str]
    #: Skills claimed in the skills list but never narrated.
    listed_only_skills: list[str]
    stuffed: bool
    #: Attack label: "none", or one of the adversarial variants.
    attack: str
    stuffing_techniques: list[str]
    #: For an attacked resume, the id of its untouched matched-pair control.
    control_id: str = ""


def _shift_degree(base: str, delta: int) -> str:
    index = DEGREE_ORDER.index(base)
    return DEGREE_ORDER[max(0, min(len(DEGREE_ORDER) - 1, index + delta))]


def build_resume_text(
    role: Role, tier: int, rng: random.Random, index: int
) -> tuple[str, ResumeMeta]:
    plan = TIER_PLAN[tier]
    name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
    handle = name.lower().replace(" ", ".")

    required = list(role.required)
    rng.shuffle(required)
    keep = max(1, round(len(required) * plan["required_frac"]))
    chosen_required = required[:keep]
    chosen_preferred = list(rng.sample(list(role.preferred), plan["preferred"]))

    # Tier-1 candidates carry some off-role skills, which is what makes them
    # genuinely marginal rather than merely thin.
    filler: list[str] = []
    if tier == 1:
        other = rng.choice([r for r in ROLES if r.key != role.key])
        filler = list(rng.sample(list(other.required), 2))

    # Split claimed skills into those the narrative will actually demonstrate
    # and those merely listed. Real resumes do both: there is only so much room
    # in an experience section, so a genuine candidate lists generic tooling
    # (Git, Linux, Agile) without ever narrating it.
    #
    # This ratio is what keeps the grounding experiment honest. If honest
    # resumes narrated everything they list, grounding would separate them from
    # attacks perfectly - but only because the corpus was built that way. Here
    # honest resumes carry a realistic fraction of unevidenced claims, and the
    # attacks are distinguished by *adding claims without adding evidence*,
    # which is a true property of the attack rather than an artefact.
    demonstrable = chosen_required + chosen_preferred
    rng.shuffle(demonstrable)
    narrated_count = max(1, round(len(demonstrable) * plan["narrated_frac"]))
    narrated = demonstrable[:narrated_count]
    listed_only = demonstrable[narrated_count:]

    skills = chosen_required + chosen_preferred + filler + ["Git", "Linux", "Agile"]

    years = max(0.5, role.min_years + rng.uniform(*plan["years_delta"]))
    degree = _shift_degree(role.min_degree, plan["degree_delta"])

    current_year = 2026
    start_year = int(current_year - years)
    mid_year = start_year + max(1, int(years / 2))

    # One bullet per narrated skill, so the narrative genuinely evidences them,
    # PLUS generic accomplishment bullets that name no technology at all.
    #
    # The generic bullets matter for validity. If bullet count tracked narrated
    # skills exactly, a weak candidate would end up with a one-line work history
    # and grounding would separate honest from attacked resumes partly on
    # narrative *length*. Real resumes describe outcomes, teamwork and delivery
    # without naming a tool, so every candidate gets a substantive narrative and
    # the only thing that varies is how much of it evidences a claimed skill.
    duty_cycle = list(role.duties)
    rng.shuffle(duty_cycle)
    bullets = [
        duty_cycle[i % len(duty_cycle)].format(
            skill=skill, n=rng.choice([2, 3, 5, 8, 12, 20, 30, 40])
        )
        for i, skill in enumerate(narrated)
    ]
    generic_needed = max(0, _MIN_BULLETS - len(bullets))
    for template in rng.sample(GENERIC_DUTIES, min(generic_needed, len(GENERIC_DUTIES))):
        bullets.append(template.format(n=rng.choice([2, 3, 4, 5, 6, 8, 10, 12])))
    rng.shuffle(bullets)

    employer_a, employer_b = rng.sample(list(role.employers), 2)
    degree_line = DEGREE_TEXT[degree]

    lines = [
        name,
        f"{handle}@example.com | +1 555 {rng.randint(100, 999)} {rng.randint(1000, 9999)}",
        "",
        "SUMMARY",
        f"{role.title} with {years:.0f} years building production systems.",
        "",
        "EXPERIENCE",
        f"{role.title} at {employer_a}, Jan {mid_year} - Present",
    ]
    split = max(1, len(bullets) // 2 + len(bullets) % 2)
    lines += [f"  {b}" for b in bullets[:split]]
    lines += [
        "",
        f"{role.title} at {employer_b}, Feb {start_year} - Dec {mid_year - 1}",
    ]
    lines += [f"  {b}" for b in bullets[split:]]
    lines += ["", "EDUCATION"]
    if degree_line:
        lines.append(f"{degree_line}, {rng.choice(UNIVERSITIES)}, {start_year - 4} - {start_year}")
    else:
        lines.append(f"Self-taught, various online courses, {start_year - 2} - {start_year}")
    lines += ["", "SKILLS", ", ".join(skills)]

    meta = ResumeMeta(
        candidate_id=f"{role.key}_{tier}_{index:02d}",
        role=role.key,
        tier=tier,
        name=name,
        years=round(years, 1),
        degree=degree,
        skills=skills,
        narrated_skills=list(narrated),
        listed_only_skills=list(listed_only) + filler + ["Git", "Linux", "Agile"],
        stuffed=False,
        attack="none",
        stuffing_techniques=[],
        control_id="",
    )
    return "\n".join(lines), meta


# --------------------------------------------------------------------------
# Adversarial variants
# --------------------------------------------------------------------------


#: Attack taxonomy. Each variant isolates ONE manipulation technique so that a
#: defence can be credited or blamed for the specific thing it handles. The
#: earlier version of this generator applied all three lexical attacks to the
#: same resume, which made it impossible to tell which signal was doing the work.
#:
#: A1-A3 are lexical/visual and leave artefacts in the document. A4-A5 are
#: generated by a language model (see generate_adversarial.py) and leave none -
#: they are the modern threat the reviewed literature does not address.
ATTACKS = ("keyword_block", "hidden_text", "repetition")


def attack_keyword_block(
    base_text: str, role: Role, job_text: str, rng: random.Random
) -> tuple[str, str]:
    """A1 - append a dense block of job-relevant keywords.

    The claims are added to the skills list; the work history is untouched. This
    is the attack evidence-grounding is designed to make worthless.
    """
    keywords = list(role.required) + list(role.preferred)
    block = ", ".join(keywords * 3)
    return base_text + "\n\nADDITIONAL SKILLS\n" + block, ""


def attack_hidden_text(
    base_text: str, role: Role, job_text: str, rng: random.Random
) -> tuple[str, str]:
    """A2 - paste the job description in white 1pt text, invisible to a human."""
    return base_text, job_text


def attack_repetition(
    base_text: str, role: Role, job_text: str, rng: random.Random
) -> tuple[str, str]:
    """A3 - hammer a single high-value keyword far beyond natural frequency."""
    term = rng.choice(list(role.required))
    padding = " ".join([term] * 25)
    return base_text + "\n\nCORE COMPETENCIES\n" + padding, ""


ATTACK_FUNCTIONS = {
    "keyword_block": attack_keyword_block,
    "hidden_text": attack_hidden_text,
    "repetition": attack_repetition,
}


def make_attack(
    base_text: str,
    meta: ResumeMeta,
    role: Role,
    job_text: str,
    attack: str,
    rng: random.Random,
) -> tuple[str, str, ResumeMeta]:
    """Apply one named attack, returning ``(visible, hidden, meta)``."""
    visible, hidden = ATTACK_FUNCTIONS[attack](base_text, role, job_text, rng)
    meta.stuffed = True
    meta.attack = attack
    meta.stuffing_techniques = [attack]
    meta.candidate_id = f"{meta.candidate_id}_{attack}"
    return visible, hidden, meta


# --------------------------------------------------------------------------
# PDF writing
# --------------------------------------------------------------------------

_PAGE_WIDTH, _PAGE_HEIGHT = 595, 842  # A4 points
_MARGIN = 54
_LINE_HEIGHT = 13


def write_pdf(path: Path, visible_text: str, hidden_text: str = "") -> None:
    """Render text to a PDF, optionally with genuinely invisible extra text."""
    document = fitz.open()
    page = document.new_page(width=_PAGE_WIDTH, height=_PAGE_HEIGHT)
    y = _MARGIN

    for line in visible_text.splitlines():
        if y > _PAGE_HEIGHT - _MARGIN:
            page = document.new_page(width=_PAGE_WIDTH, height=_PAGE_HEIGHT)
            y = _MARGIN
        # Section headings are the all-caps lines; render them slightly larger.
        is_heading = line.isupper() and len(line.split()) <= 3 and line.strip()
        page.insert_text(
            (_MARGIN, y),
            line,
            fontsize=10.5 if is_heading else 9.5,
            fontname="hebo" if is_heading else "helv",
            color=(0, 0, 0),
        )
        y += _LINE_HEIGHT

    if hidden_text:
        # White on white at 1pt: invisible to a human, fully indexed by an ATS.
        # This is the attack the detector's first signal exists to catch.
        hidden_y = _PAGE_HEIGHT - _MARGIN
        for line in hidden_text.splitlines():
            if not line.strip():
                continue
            page.insert_text(
                (_MARGIN, hidden_y),
                line,
                fontsize=1.0,
                fontname="helv",
                color=(1, 1, 1),
            )
            hidden_y -= 1.5
            if hidden_y < _MARGIN:
                break

    path.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(path))
    document.close()


# --------------------------------------------------------------------------
# Job descriptions
# --------------------------------------------------------------------------


def build_job_text(role: Role) -> str:
    return "\n".join(
        [
            role.title,
            "",
            f"We are hiring a {role.title} to join a product engineering team.",
            "",
            "Requirements:",
            f"- {role.min_years}+ years of professional experience",
        ]
        + [f"- Strong hands-on {skill}" for skill in role.required]
        + [f"- {role.min_degree} degree in Computer Science or equivalent"]
        + [
            "",
            "Preferred:",
        ]
        + [f"- Experience with {skill}" for skill in role.preferred]
        + [
            "",
            "You will design, build and operate production systems, review code,",
            "and work closely with product partners to ship reliable software.",
        ]
    )


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def generate(out_dir: Path, seed: int = 7, stuffed_per_role: int = 0,
             pairs_per_attack: int = 3) -> dict:
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    resume_dir = out_dir / "resumes"

    jobs: dict[str, dict] = {}
    for role in ROLES:
        jobs[role.key] = {
            "job_id": role.key,
            "title": role.title,
            "text": build_job_text(role),
            "min_years": role.min_years,
            "min_degree": role.min_degree,
            "required": list(role.required),
            "preferred": list(role.preferred),
        }

    metas: list[ResumeMeta] = []

    for role in ROLES:
        job_text = jobs[role.key]["text"]
        for tier, plan in TIER_PLAN.items():
            for index in range(plan["n"]):
                text, meta = build_resume_text(role, tier, rng, index)
                write_pdf(resume_dir / f"{meta.candidate_id}.pdf", text)
                metas.append(meta)

        # Adversarial variants are built from tier-1 resumes: the whole point is
        # a weak candidate trying to buy their way up the ranking. One resume per
        # attack type, all derived from the SAME base resume, so any ranking
        # difference between them is attributable to the attack alone.
        variants = [
            (attack, repeat)
            for repeat in range(pairs_per_attack)
            for attack in ATTACKS
        ]
        for index, (attack, repeat) in enumerate(variants):
            text, meta = build_resume_text(role, 1, rng, 90 + index)

            # Emit the untouched resume alongside the attacked one. This is a
            # matched-pair design: the control and the attack are the same
            # candidate with the same true relevance, differing only by the
            # manipulation, so any ranking difference between them isolates the
            # effect of the attack rather than of the candidate.
            control = replace(meta, candidate_id=f"{meta.candidate_id}_control")
            write_pdf(resume_dir / f"{control.candidate_id}.pdf", text)
            metas.append(control)

            visible, hidden, meta = make_attack(text, meta, role, job_text, attack, rng)
            meta.control_id = control.candidate_id
            write_pdf(resume_dir / f"{meta.candidate_id}.pdf", visible, hidden)
            metas.append(meta)

    # Labels: a resume is graded by its tier against its own role's job.
    #
    # Against a *different* role it is not automatically irrelevant. Adjacent
    # roles share real skills - a strong backend engineer is a plausible
    # devops hire - so a tier-3 resume scores 1 against an adjacent role. This
    # matters: without it every job has a clean, linearly separable set of
    # relevant candidates and any competent ranker scores a perfect NDCG,
    # which measures nothing. The adjacency labels are what force the ranking
    # to discriminate among genuinely similar candidates.
    #
    # Stuffed resumes keep their true, low tier - that is precisely what makes
    # them a trap for a gameable ranker.
    labels: dict[str, dict[str, int]] = {}
    for job_key in jobs:
        row: dict[str, int] = {}
        for meta in metas:
            if meta.role == job_key:
                row[meta.candidate_id] = meta.tier
            elif meta.tier == 3 and job_key in ADJACENCY.get(meta.role, {}):
                row[meta.candidate_id] = ADJACENCY[meta.role][job_key]
            else:
                row[meta.candidate_id] = 0
        labels[job_key] = row

    payload = {
        "seed": seed,
        "jobs": jobs,
        "labels": labels,
        "resumes": [asdict(m) for m in metas],
    }
    (out_dir / "corpus.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="benchmark/corpus", type=Path)
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--stuffed-per-role", default=len(ATTACKS), type=int)
    parser.add_argument(
        "--pairs-per-attack", default=3, type=int,
        help="matched attack/control pairs per attack type per role")
    args = parser.parse_args()

    payload = generate(
        args.out, seed=args.seed, stuffed_per_role=args.stuffed_per_role,
        pairs_per_attack=args.pairs_per_attack,
    )
    resumes = payload["resumes"]
    stuffed = sum(1 for r in resumes if r["stuffed"])
    print(f"corpus written to {args.out}")
    print(f"  jobs      : {len(payload['jobs'])}")
    print(f"  resumes   : {len(resumes)} ({stuffed} adversarial)")
    by_tier: dict[int, int] = {}
    for r in resumes:
        by_tier[r["tier"]] = by_tier.get(r["tier"], 0) + 1
    print(f"  by tier   : {dict(sorted(by_tier.items(), reverse=True))}")


if __name__ == "__main__":
    main()
