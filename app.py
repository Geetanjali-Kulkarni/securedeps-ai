"""SecureDeps AI dashboard.   Run:  streamlit run app.py

This file only shows things. The work is done in securedeps/pipeline.py and securedeps/chat.py,
which are covered by tests.
"""
import os

import pandas as pd
import streamlit as st

from securedeps.chat import answer_question, chat_docs
from securedeps.explain import risk_text
from securedeps.llm import get_llm
from securedeps.pipeline import TIERS, ScanResult, explain_for, fix_rows, run_scan
from securedeps.rag import get_embedder
from securedeps.repo import RepoError

st.set_page_config(page_title="SecureDeps AI", page_icon="🛡️", layout="wide")


@st.cache_resource
def embedder():
    return get_embedder()


def show_sources(sources):
    if not sources:
        return
    st.markdown("**Sources**")
    for cid, c in sources.items():
        st.markdown(f"- [{cid}] [{c.title}]({c.url})" if c.url else f"- [{cid}] {c.title}")


def show_risk(e):
    """The risk line in a coloured box: red for HIGH, amber for ELEVATED, neutral otherwise (never green)."""
    box = {"high": st.error, "elevated": st.warning}.get(e.upgrade_risk, st.info)
    box(f"**Upgrade risk.** {risk_text(e)} (confidence: {e.confidence})")


# ------------------------------------------------------------------ sidebar
llm = get_llm()
with st.sidebar:
    st.title("🛡️ SecureDeps AI")
    st.caption("Paste a public GitHub repository. Get a ranked 'fix these first' list with cited explanations.")
    url = st.text_input("GitHub repository URL", placeholder="https://github.com/OWASP/NodeGoat")
    go = st.button("Scan", type="primary")
    st.divider()
    st.caption("AI model: " + (f"{type(llm).__name__} ({llm.model})" if llm else
                               "none (set GEMINI_API_KEY). Explanations use the deterministic text."))
    st.caption("GitHub token: " + ("set" if os.environ.get("GITHUB_TOKEN") else
                                   "not set (60 requests/hour; set GITHUB_TOKEN for more)"))

if go and url.strip():
    st.session_state.pop("chat", None)
    st.session_state.pop("explained", None)
    with st.spinner("Reading the repository and checking OSV, CISA KEV and EPSS..."):
        try:
            st.session_state["result"] = run_scan(url.strip(), os.environ.get("GITHUB_TOKEN"))
        except RepoError as e:
            st.session_state.pop("result", None)
            st.error(str(e))

result: ScanResult = st.session_state.get("result")
if result is None:
    st.header("Which vulnerable dependency should you fix first?")
    st.write("Scanners list hundreds of problems and call most of them 'critical'. SecureDeps AI ranks them "
             "by whether attackers are actually exploiting them (CISA KEV), how likely exploitation is "
             "(EPSS) and how severe they are (CVSS), then explains each fix using the advisories and "
             "release notes, with a citation for every claim.")
    st.stop()

# ---------------------------------------------------------------- dashboard
st.header(result.full_name)
st.caption(f"Scan date {result.scan_date}. KEV and EPSS change daily, so re-scan before acting on old results.")
c = st.columns(6)
c[0].metric("Dependencies", result.dependency_count)
c[1].metric("Distinct issues", len(result.issues))
for col, tier in zip(c[2:], TIERS):
    col.metric(tier, result.tier_counts[tier])
st.caption("Priority tiers come from exploit data and severity. 'Low' means no known exploitation and a "
           "moderate score; it does not mean 'ignore'.")
for w in result.warnings:
    st.warning(w)
with st.expander(f"Files scanned ({len(result.files)})"):
    for f in result.files:
        st.write("- " + f)

tab_fix, tab_ask = st.tabs(["Fix these first", "Ask about this scan"])

with tab_fix:
    if not result.plan:
        st.success("No known vulnerabilities found in the dependencies we could read exactly.")
        st.stop()
    st.dataframe(pd.DataFrame(fix_rows(result)).rename(columns={"#": "Rank", "Fixes": "Issues fixed"}),
                 hide_index=True)
    cmp = result.baseline(10)
    if cmp["n"]:
        st.info(f"A severity-only ranking's top {cmp['n']} would come from {cmp['tie_size']} issues tied at "
                f"CVSS >= {cmp['cutoff']}, so it cannot tell them apart. Of our top {cmp['n']}, "
                f"{len(cmp['promoted'])} are lower severity but ranked higher because of exploit data.")

    st.subheader("Explain a fix")
    labels = [f"{n}. {a.name} {a.installed} -> {a.target or 'no fix known'} [{a.tier}]"
              for n, a in enumerate(result.plan[:25], 1)]
    pick = st.selectbox("Choose a fix", range(len(labels)), format_func=lambda i: labels[i])
    action = result.plan[pick]
    with st.expander("Why this score?", expanded=False):
        for r in action.reasons:
            st.write("- " + r)
    if st.button("Explain this fix"):
        with st.spinner("Fetching release notes and writing a grounded explanation..."):
            st.session_state["explained"] = (pick, *explain_for(result, action, llm, embedder(),
                                                                token=os.environ.get("GITHUB_TOKEN")))
    done = st.session_state.get("explained")
    if done and done[0] == pick:
        e, notes = done[1], done[2]
        st.caption("Written by AI using only the sources listed below; code checked every citation, and the "
                   "fix itself comes from the scanner, not the AI." if e.used_llm else
                   "Summary built directly from the sources below (no AI text was used).")
        st.markdown(f"**What is wrong.** {e.what_is_wrong}")
        st.markdown(f"**Fix.** {e.fix}")
        st.markdown(f"**Compatibility.** {e.compatibility}")
        show_risk(e)
        if e.breaking_lines:
            with st.expander("Which release-note lines triggered the risk label?"):
                st.caption("A keyword check picked these lines. It can be wrong, so read the notes yourself.")
                for line, cid in e.breaking_lines:
                    st.write(f"- {line} [{cid}]")
        show_sources(e.sources)
        for n in notes:
            st.caption("Note: " + n)

with tab_ask:
    st.caption("Answers use only the scan results and the advisories / release notes you have opened "
               "(use 'Explain this fix' first to load release notes).")
    history = st.session_state.setdefault("chat", [])
    for role, text, sources in history:
        with st.chat_message(role):
            st.markdown(text)
            if sources:
                show_sources(sources)
    q = st.chat_input("e.g. What should I fix first and why?")
    if q:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Searching the sources..."):
                a = answer_question(q, chat_docs(result), llm, embedder())
            st.markdown(a.text)
            show_sources(a.sources)
            for w in a.warnings:
                st.caption("Note: " + w)
        history += [("user", q, None), ("assistant", a.text, a.sources)]
