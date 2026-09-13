"""TalentLens recruiter dashboard.

Run with::

    streamlit run app/streamlit_app.py

Design note: parsing and embedding are the expensive steps, so uploaded resumes
are parsed once and held in session state. Moving a weight slider then re-ranks
from cached vectors, which is what makes the weights feel like a live control
rather than a form you submit.
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from talentlens.config import PrerequisiteGate, ScoringWeights  # noqa: E402
from talentlens.explain import available_backends, get_explainer  # noqa: E402
from talentlens.pipeline import build_job, load_resume_uploads, screen  # noqa: E402
from talentlens.schemas import DegreeLevel, GapStatus  # noqa: E402
from talentlens.skillgap import upskill_priorities  # noqa: E402

st.set_page_config(page_title="TalentLens", page_icon="🔎", layout="wide")

SAMPLE_JD = """Senior Backend Engineer

We are hiring a backend engineer to build and operate payment services.

Requirements:
- 5+ years of professional experience
- Strong Python and REST API design
- PostgreSQL and Docker in production
- Bachelor degree in Computer Science or equivalent

Preferred:
- Kubernetes and AWS
- Experience with Kafka
"""


def _upload_key(files) -> str:
    digest = hashlib.sha256()
    for f in files:
        digest.update(f.name.encode())
        digest.update(str(f.size).encode())
    return digest.hexdigest()


@st.cache_resource(show_spinner=False)
def _warm_models() -> str:
    from talentlens.embedding import get_embedder

    embedder = get_embedder()
    embedder.encode(["warmup"])
    return embedder.device


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

st.sidebar.title("TalentLens")
st.sidebar.caption("Explainable multi-factor resume ranking")

st.sidebar.subheader("Scoring weights")
st.sidebar.caption("Sliders are normalised to sum to 1.0")
w_sem = st.sidebar.slider("Semantic fit", 0.0, 1.0, 0.40, 0.05)
w_skill = st.sidebar.slider("Skill coverage", 0.0, 1.0, 0.30, 0.05)
w_exp = st.sidebar.slider("Experience", 0.0, 1.0, 0.20, 0.05)
w_edu = st.sidebar.slider("Education", 0.0, 1.0, 0.10, 0.05)

if w_sem + w_skill + w_exp + w_edu <= 0:
    st.sidebar.error("At least one weight must be above zero.")
    st.stop()

weights = ScoringWeights.rescaled(w_sem, w_skill, w_exp, w_edu)
st.sidebar.caption(
    " / ".join(f"{k[:3]} {v:.2f}" for k, v in weights.as_dict().items())
)

st.sidebar.subheader("Requirements override")
st.sidebar.caption("Leave at -1 to read them from the job description")
min_years_input = st.sidebar.number_input("Minimum years", -1.0, 40.0, -1.0, 0.5)
degree_choice = st.sidebar.selectbox(
    "Minimum degree", ["(from job description)"] + [d.label for d in DegreeLevel]
)

st.sidebar.subheader("Scoring behaviour")
skill_method = st.sidebar.radio(
    "Skill measure",
    ["coverage", "dice"],
    help=(
        "coverage = fraction of the job's skills the candidate holds (default). "
        "dice = the literal Sorensen-Dice formula from the source study, which "
        "penalises candidates for knowing extra things."
    ),
)
use_antigaming = st.sidebar.checkbox("Anti-gaming detector", value=True)
use_gate = st.sidebar.checkbox(
    "Prerequisite gate",
    value=True,
    help=(
        "Damps candidates who meet few of the required skills, so an unrelated "
        "but experienced applicant cannot coast to a mid-range score on "
        "experience and education alone."
    ),
)

st.sidebar.subheader("Explanations")
backends = available_backends()
backend_labels = {
    "template": "Template (offline, instant)",
    "nvidia": "NVIDIA NIM (LLM prose)",
    "claude": "Claude (LLM prose)",
}
usable = [name for name, ok in backends.items() if ok]
backend = st.sidebar.selectbox(
    "Backend",
    usable,
    format_func=lambda n: backend_labels.get(n, n),
    index=usable.index("template") if "template" in usable else 0,
)
unavailable = [n for n, ok in backends.items() if not ok]
if unavailable:
    st.sidebar.caption(f"Unavailable (no API key): {', '.join(unavailable)}")
explain_top_k = st.sidebar.number_input("Explain top N", 0, 50, 5, 1)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

st.title("Candidate screening")

left, right = st.columns([3, 2])
with left:
    job_text = st.text_area("Job description", value=SAMPLE_JD, height=280)
with right:
    uploads = st.file_uploader(
        "Resumes",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        help="PDF, DOCX or plain text. Nothing is uploaded anywhere - parsing is local.",
    )
    st.caption(f"{len(uploads or [])} file(s) selected")
    run = st.button("Rank candidates", type="primary", use_container_width=True)

if run:
    if not uploads:
        st.warning("Add at least one resume first.")
        st.stop()
    if not job_text.strip():
        st.warning("A job description is required.")
        st.stop()

    with st.spinner("Loading models ..."):
        device = _warm_models()

    key = _upload_key(uploads)
    if st.session_state.get("upload_key") != key:
        with st.spinner(f"Parsing {len(uploads)} resume(s) ..."):
            resumes, failures = load_resume_uploads(
                [(f.name, f.getvalue()) for f in uploads]
            )
        st.session_state["upload_key"] = key
        st.session_state["resumes"] = resumes
        st.session_state["failures"] = failures
    st.session_state["job_text"] = job_text
    st.session_state["ran"] = True
    st.session_state["device"] = device

if st.session_state.get("ran"):
    resumes = st.session_state.get("resumes", [])
    failures = st.session_state.get("failures", [])

    for name, error in failures:
        st.error(f"Could not read {name}: {error}")

    if not resumes:
        st.stop()

    min_degree = (
        None
        if degree_choice.startswith("(")
        else next(d for d in DegreeLevel if d.label == degree_choice)
    )
    job = build_job(
        st.session_state["job_text"],
        min_years=None if min_years_input < 0 else min_years_input,
        min_degree=min_degree,
    )

    explainer = get_explainer(backend)
    gate = PrerequisiteGate(enabled=use_gate)

    started = time.perf_counter()
    with st.spinner("Scoring ..."):
        ranked = screen(
            resumes,
            job,
            weights=weights,
            skill_method=skill_method,
            use_antigaming=use_antigaming,
            gate=gate,
            explainer=explainer,
            explain_top_k=int(explain_top_k),
        )
    elapsed = time.perf_counter() - started

    top = st.columns(5)
    top[0].metric("Candidates", len(ranked))
    top[1].metric("Flagged", sum(1 for c in ranked if c.gaming.flagged))
    top[2].metric("Requirements", len(job.required_skills))
    top[3].metric("Device", st.session_state.get("device", "cpu"))
    top[4].metric("Elapsed", f"{elapsed:.1f}s")

    with st.expander("What the system read from the job description"):
        jd_left, jd_right = st.columns(2)
        jd_left.write(f"**Required:** {', '.join(sorted(job.required_skills)) or '-'}")
        jd_left.write(f"**Preferred:** {', '.join(sorted(job.preferred_skills)) or '-'}")
        jd_right.write(f"**Minimum experience:** {job.min_years:.1f} years")
        jd_right.write(f"**Minimum degree:** {job.min_degree.label}")

    st.subheader("Ranking")
    frame = pd.DataFrame(
        [
            {
                "#": c.rank,
                "Candidate": c.display_name,
                "Score": c.scores.total,
                "Semantic": c.scores.semantic,
                "Skills": c.scores.skill,
                "Experience": c.scores.experience,
                "Education": c.scores.education,
                "Req. covered": c.scores.required_coverage,
                "Penalty": c.scores.penalty,
                "Flagged": "YES" if c.gaming.flagged else "",
                "Years": c.resume.years_experience,
                "Degree": c.resume.degree.label,
            }
            for c in ranked
        ]
    )
    st.dataframe(
        frame,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Score", min_value=0.0, max_value=1.0, format="%.3f"
            ),
            "Semantic": st.column_config.NumberColumn(format="%.3f"),
            "Skills": st.column_config.NumberColumn(format="%.3f"),
            "Experience": st.column_config.NumberColumn(format="%.3f"),
            "Education": st.column_config.NumberColumn(format="%.3f"),
            "Req. covered": st.column_config.NumberColumn(format="%.2f"),
            "Penalty": st.column_config.NumberColumn(format="%.3f"),
        },
    )

    st.download_button(
        "Download ranking (CSV)",
        frame.to_csv(index=False).encode(),
        file_name="talentlens_ranking.csv",
        mime="text/csv",
    )

    st.subheader("Candidate detail")
    for candidate in ranked:
        flag = " [FLAGGED]" if candidate.gaming.flagged else ""
        header = (
            f"#{candidate.rank}  {candidate.display_name}  -  "
            f"{candidate.scores.total:.3f}{flag}"
        )
        with st.expander(header, expanded=candidate.rank <= 3):
            detail_left, detail_right = st.columns([2, 3])

            with detail_left:
                st.markdown("**Score breakdown**")
                for label, value, weight in (
                    ("Semantic fit", candidate.scores.semantic, weights.semantic),
                    ("Skill coverage", candidate.scores.skill, weights.skill),
                    ("Experience", candidate.scores.experience, weights.experience),
                    ("Education", candidate.scores.education, weights.education),
                ):
                    st.caption(f"{label} - {value:.3f} x {weight:.2f} = {value * weight:.3f}")
                    st.progress(min(1.0, max(0.0, value)))

                if candidate.scores.gate < 1.0:
                    st.warning(
                        f"Prerequisite gate x{candidate.scores.gate:.2f} - only "
                        f"{candidate.scores.required_coverage:.0%} of required skills held."
                    )
                if candidate.scores.penalty > 0:
                    st.error(f"Anti-gaming penalty -{candidate.scores.penalty:.3f}")

                if candidate.resume.email:
                    st.caption(f"Contact: {candidate.resume.email}")

            with detail_right:
                gap = candidate.gap
                st.markdown("**Skill gap**")
                if gap.matched:
                    st.success("Matched: " + ", ".join(i.skill for i in gap.matched))
                if gap.transferable:
                    st.info(
                        "Transferable: "
                        + "; ".join(f"{i.skill} (via {i.via})" for i in gap.transferable)
                    )
                if gap.missing:
                    st.warning(
                        "Missing: "
                        + ", ".join(
                            f"{i.skill}{'*' if i.required else ''}" for i in gap.missing
                        )
                        + "   (* = required)"
                    )
                priorities = upskill_priorities(gap, limit=3)
                if priorities:
                    st.caption(
                        "Interview focus: " + ", ".join(i.skill for i in priorities)
                    )

            if candidate.gaming.flagged:
                st.markdown("**Integrity signals**")
                for signal in candidate.gaming.triggered_signals:
                    st.markdown(f"- `{signal.name}` - {signal.evidence}")

            if candidate.explanation:
                st.markdown("**Explanation**")
                if backend == "template":
                    st.code(candidate.explanation, language=None)
                else:
                    st.write(candidate.explanation)
            elif explain_top_k and candidate.rank > explain_top_k:
                st.caption(f"Explanation generated for the top {int(explain_top_k)} only.")
else:
    st.info(
        "Paste a job description, add resumes, and press **Rank candidates**. "
        "Sample resumes are available in `benchmark/corpus/resumes/` after running "
        "`python benchmark/generate_corpus.py`."
    )
