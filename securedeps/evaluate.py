"""Evaluation: how often is the chat answer right, grounded and honest?

    python3 -m securedeps.evaluate https://github.com/OWASP/NodeGoat       run (needs GEMINI_API_KEY)
    python3 -m securedeps.evaluate --summarize eval_results.csv            after you grade by hand

Why this design (be ready to defend it):
  * The ANSWER KEY comes from our deterministic engine ("the top fix is plan[0].name"), not from
    another AI, so we never grade an AI with an AI, and the key follows the live data.
  * Each question is answered twice: RAG + LLM, and retrieval only (no LLM) as a baseline.
  * 'Unanswerable' questions test hallucination: the right behaviour is 'the sources do not say'.
  * Citation validity is enforced by code, so it is ~100% by construction. The honest numbers are
    how often the AI answer had to be rejected, and whether the answer is correct.
  * A human grades a sample, because keyword matching can say 'present' without saying 'right'.
"""
import csv
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

import requests

from .chat import answer_question, chat_docs, retrieve
from .llm import get_llm
from .pipeline import ScanResult, action_flags, evidence_for, run_scan
from .rag import chunk_docs, get_embedder

UNSAFE_PHRASES = ["is safe", "will not break", "won't break", "no risk", "safe to upgrade", "no breaking changes"]
NO_ANSWER = "sources do not say"


@dataclass
class Question:
    text: str
    category: str                                  # ranking | version | flags | risk | unanswerable
    must_include: List[str] = field(default_factory=list)   # every entry must appear; "a|b" = either (case-insensitive)
    must_not: List[str] = field(default_factory=list)       # none may appear
    gold_docs: List[str] = field(default_factory=list)      # doc ids retrieval should have found
    expect_refusal: bool = False                            # no source can answer it, so refusing is right
    first_named: Optional[str] = None                       # the package the answer must name FIRST ...
    names: List[str] = field(default_factory=list)          # ... out of these (all packages in the plan)


def first_mentioned(text: str, names: List[str]) -> Optional[str]:
    """Which package does the answer name first? (An answer that lists three fixes with the wrong one
    first used to pass, because the right name appeared somewhere.)"""
    best, best_pos = None, None
    for n in set(names):
        m = re.search(r"(?<![\w.@/-])" + re.escape(n) + r"(?![\w-])", text, re.I)
        if m and (best_pos is None or m.start() < best_pos):
            best, best_pos = n, m.start()
    return best

def build_questions(r: ScanResult) -> List[Question]:
    plan, qs = r.plan, []
    names = [a.name for a in plan]
    ordinal = ["first", "second", "third", "fourth", "fifth"]
    # --- ranking: key = position in OUR ranking
    qs.append(Question("What should I fix first and why?", "ranking", [plan[0].name], gold_docs=["scan:fix1"],
                       first_named=plan[0].name, names=names))
    for i in (1, 2):
        if len(plan) > i:
            qs.append(Question(f"Which package is the {ordinal[i]} fix in the ranking?", "ranking",
                               [plan[i].name], gold_docs=[f"scan:fix{i + 1}"],
                               first_named=plan[i].name, names=names))
    if len(plan) >= 5 and plan[-1].name != plan[0].name:       # a question about the BOTTOM of the ranking
        qs.append(Question("Which package is the last fix in the ranking, the one to fix last?", "ranking",
                           [plan[-1].name], gold_docs=["scan:ranking"], first_named=plan[-1].name, names=names))
    qs.append(Question("What is the score of the top-ranked fix?", "ranking", [str(plan[0].score)],
                       gold_docs=["scan:fix1"]))
    qs.append(Question("How many dependencies were scanned?", "ranking", [str(r.dependency_count)],
                       gold_docs=["scan:summary"]))
    qs.append(Question("How many distinct issues were found?", "ranking", [str(len(r.issues))],
                       gold_docs=["scan:summary"]))
    qs.append(Question("How many fixes are in the Plan priority?", "ranking", [str(r.tier_counts["Plan"])],
                       gold_docs=["scan:summary"]))
    qs.append(Question("Why is the top fix ranked first?", "ranking", ["EPSS"], gold_docs=["scan:fix1"]))
    # --- version: key = target version chosen by the scanner from OSV data
    seen = set()
    for n, a in enumerate(plan[:15], 1):
        if a.target and a.name not in seen and len(seen) < 8:
            seen.add(a.name)
            qs.append(Question(f"Which version of {a.name} fixes the vulnerability in {a.installed}?", "version",
                               [a.target], gold_docs=[f"scan:fix{n}"]))
    # --- flags: key = flags computed by the ranking code
    dev = [a for a in plan[:15] if a.dev_only]
    trans = [a for a in plan[:15] if a.direct is False]
    direct = [a for a in plan[:15] if a.direct]
    risky = [a for a in plan[:15] if a.major_bump]
    if dev:
        qs.append(Question(f"Is {dev[0].name} {dev[0].installed} a development-only dependency?", "flags", ["dev-only|development-only|development only"]))
    if trans:
        qs.append(Question(f"Is {trans[0].name} {trans[0].installed} a transitive dependency?", "flags", ["transitive|indirect"]))
    if direct:
        qs.append(Question(f"Is {direct[0].name} {direct[0].installed} a direct dependency?", "flags", ["direct"],
                           must_not=["transitive|indirect"]))
    if risky:
        qs.append(Question(f"Is upgrading {risky[0].name} {risky[0].installed} a risky version jump?", "flags",
                           ["risky|major"]))
    # --- risk: ask what the release notes say (nobody can answer "will it break?"). If we loaded release
    # notes for the package the answer must be substantive and must never call the upgrade safe; if we
    # found none, the right answer is to say the sources do not say.
    for a in plan[:4]:
        docs, _ = r.evidence.get((a.name, a.installed), ([], []))
        has_notes = any(d.kind == "release" for d in docs)
        qs.append(Question(f"What do the release notes say about breaking or behaviour changes when upgrading "
                           f"{a.name} from {a.installed} to {a.target or 'the fixed version'}?", "risk",
                           [], UNSAFE_PHRASES, expect_refusal=not has_notes))
    # --- unanswerable: right behaviour is 'the sources do not say'
    for t in ["What is the weather in Pune today?", "Who is the maintainer of the tar package?",
              "How many GitHub stars does this repository have?", "What is the license of this project?",
              "Which programming language is the Linux kernel written in?"]:
        qs.append(Question(t, "unanswerable"))
    return qs


def passes(q: Question, text: str, used_llm: bool, sources=None) -> bool:
    low = text.lower()
    if q.category == "unanswerable":
        # correct = admits it does not know. A cited AI answer to an unanswerable question is a hallucination.
        return NO_ANSWER in low
    if q.expect_refusal:
        return NO_ANSWER in low
    if any(m.lower() in low for m in q.must_not):
        return False
    if q.category == "risk":
        # Keyword lists proved unreliable (a correct answer said "require", the list said "requires").
        # A risk answer is good when it is not a refusal AND is grounded in release notes: it cites at
        # least one release-note source.
        return NO_ANSWER not in low and any(c.kind == "release" for c in (sources or {}).values())
    ok = all(any(alt in low for alt in m.lower().split("|")) for m in q.must_include)
    if ok and q.first_named:
        ok = first_mentioned(text, q.names) == q.first_named
    if ok and q.category == "version" and sources is not None:
        # the right number is not enough: a cited source must actually contain it (no unsupported claims)
        ok = any(m.lower() in c.text.lower() for m in q.must_include for c in sources.values())
    return ok


def retrieval_hit(q: Question, retrieved) -> bool:
    return bool(q.gold_docs) and all(any(c.doc_id == g for c, _ in retrieved) for g in q.gold_docs)


def run_questions(r: ScanResult, questions: List[Question], llm, embedder, pause: float = 4.0) -> List[dict]:
    docs = chat_docs(r)
    chunks = chunk_docs(docs)
    rows = []
    for q in questions:
        retrieved = retrieve(q.text, chunks, embedder)
        base = answer_question(q.text, docs, None, embedder)            # retrieval only
        ans = answer_question(q.text, docs, llm, embedder)              # RAG + LLM
        if llm is not None:
            time.sleep(pause)                                           # free-tier rate limits
        rows.append({"category": q.category, "question": q.text,
                     "retrieval_hit": retrieval_hit(q, retrieved) if q.gold_docs else "",
                     "llm_used": ans.used_llm, "llm_pass": passes(q, ans.text, ans.used_llm, ans.sources),
                     "baseline_pass": passes(q, base.text, False, base.sources),
                     "answer": ans.text.replace("\n", " | "), "notes": "; ".join(ans.warnings),
                     "human_verdict": ""})
    return rows


def _rate(rows, key) -> str:
    vals = [r[key] for r in rows if r[key] != ""]
    return f"{sum(1 for v in vals if v is True or v == 'True')}/{len(vals)}" if vals else "n/a"


def report(rows: List[dict]) -> str:
    lines = [f"{'category':<14}{'n':>3}  {'retrieval hit':>14}  {'RAG+LLM correct':>16}  {'retrieval-only correct':>23}"]
    for cat in ["ranking", "version", "flags", "risk", "unanswerable"]:
        sub = [r for r in rows if r["category"] == cat]
        if sub:
            lines.append(f"{cat:<14}{len(sub):>3}  {_rate(sub, 'retrieval_hit'):>14}  "
                         f"{_rate(sub, 'llm_pass'):>16}  {_rate(sub, 'baseline_pass'):>23}")
    lines.append(f"{'ALL':<14}{len(rows):>3}  {_rate(rows, 'retrieval_hit'):>14}  "
                 f"{_rate(rows, 'llm_pass'):>16}  {_rate(rows, 'baseline_pass'):>23}")
    answerable = [r for r in rows if r["category"] != "unanswerable"]
    accepted = [r for r in answerable if r["llm_used"] in (True, "True")]
    lines.append(f"\nAnswerable questions where the AI answer passed the citation checks: {len(accepted)}/{len(answerable)} "
                 f"(the rest fell back to plain passages: answer rejected, or the API failed)")
    unans = [r for r in rows if r["category"] == "unanswerable"]
    if unans:
        lines.append(f"Unanswerable questions correctly refused ('the sources do not say'): "
                     f"{sum(1 for r in unans if r['llm_pass'] in (True, 'True'))}/{len(unans)}")
    return "\n".join(lines)


def save_csv(rows: List[dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summarize_human(path: str) -> str:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    graded = [r for r in rows if r.get("human_verdict", "").strip().lower() in ("correct", "partial", "wrong")]
    if not graded:
        return "No graded rows yet. Fill the human_verdict column with correct / partial / wrong."
    verdicts = [r["human_verdict"].strip().lower() for r in graded]
    ok = verdicts.count("correct")
    agree = sum(1 for r, v in zip(graded, verdicts) if (v == "correct") == (r["llm_pass"] == "True"))
    return (f"Human graded {len(graded)} answers: {ok} correct, {verdicts.count('partial')} partial, "
            f"{verdicts.count('wrong')} wrong ({ok / len(graded):.0%} fully correct).\n"
            f"Automatic keyword check agreed with the human on {agree}/{len(graded)}.")


def main(argv: List[str]) -> int:
    if len(argv) == 2 and argv[0] == "--summarize":
        print(summarize_human(argv[1]))
        return 0
    if len(argv) != 1:
        print("usage: python3 -m securedeps.evaluate https://github.com/owner/repo\n"
              "       python3 -m securedeps.evaluate --summarize eval_results.csv")
        return 2
    llm = get_llm()
    if llm is None:
        print("No LLM key found: only the retrieval-only baseline will be measured.")
    print("Scanning and loading release notes for the top fixes...")
    r = run_scan(argv[0], os.environ.get("GITHUB_TOKEN"))
    if not r.plan:
        print("No vulnerabilities found, nothing to evaluate.")
        return 1
    session = requests.Session()
    for a in r.plan[:5]:
        evidence_for(r, a, session, os.environ.get("GITHUB_TOKEN"))
    questions = build_questions(r)
    print(f"Asking {len(questions)} questions (about {len(questions) * 10 // 60 + 1} minutes with rate-limit pauses)...")
    rows = run_questions(r, questions, llm, get_embedder())
    save_csv(rows, "eval_results.csv")
    print("\n" + report(rows))
    print("\nWrote eval_results.csv. Open it, pick about 10 rows and fill the human_verdict column "
          "(correct / partial / wrong), then run:  python3 -m securedeps.evaluate --summarize eval_results.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
