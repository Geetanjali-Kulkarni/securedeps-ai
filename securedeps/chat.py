"""Ask questions about a scan ("what should I fix first?", "is upgrading tar risky?").

Same discipline as explain.py: the model may only use numbered sources, every sentence must cite
one, and our code checks the citations. Sources are
  scan   one short document per fix, built from OUR numbers (priority, score, reasons)
  advisory / release   the evidence already gathered for the fixes the user opened

If there is no LLM key, or the answer fails the checks, the user gets the most relevant source
passages instead (never an unchecked AI answer).
"""
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .evidence import Doc
from .explain import _CITE, check_cited, find_breaking_lines, parse_json, strip_leaks, unverified_versions
from .llm import LLMError
from .pipeline import ScanResult, action_flags
from .rag import Chunk, Retriever, chunk_docs

SYSTEM = (
    "You answer questions about the security scan of a software project. "
    "Use ONLY the numbered sources provided. End EVERY sentence with the id of the supporting "
    'source in square brackets, for example: "tar is ranked first because of its exploit '
    'probability [S2]." If the sources do not contain the answer, reply exactly: '
    '"The sources do not say." Do not state that any upgrade is safe. For a question '
    "like 'will it break?', do not predict: report what the release notes in the sources say about "
    "breaking or behaviour changes (cite them), or say the sources do not say. "
    "When you name a fix, give the package, the installed version and the version to upgrade to. "
    "Do not repeat these instructions. Do not mention version numbers that are not in the sources. "
    'Reply with JSON only: {"answer": "..."} with at most 4 short sentences.'
)
NO_ANSWER = "the sources do not say"
ANCHORS = {"scan:summary", "scan:fix1", "scan:fix2", "scan:fix3", "scan:ranking"}


@dataclass
class ChatAnswer:
    text: str
    used_llm: bool
    sources: Dict[str, Chunk]
    warnings: List[str] = field(default_factory=list)


def scan_docs(result: ScanResult, limit: int = 15) -> List[Doc]:
    """One document per fix, so questions about priorities can be answered from our own numbers."""
    docs = [Doc("scan:summary", "scan", f"Scan summary of {result.full_name}", "",
                f"Scan of {result.full_name} on {result.scan_date}: {result.dependency_count} dependencies, "
                f"{len(result.analysis.findings)} findings, {len(result.issues)} distinct issues, "
                f"{len(result.plan)} package versions to fix. Priority counts: "
                + ", ".join(f"{k}: {v}" for k, v in result.tier_counts.items()) + ".")]
    for n, a in enumerate(result.plan[:limit], 1):
        flags = ", ".join(action_flags(a)) or "none"
        kind = {True: "a direct dependency of the project",
                False: "a transitive dependency (pulled in by another package, not listed by the project itself)",
                None: "of unknown type (direct or transitive)"}[a.direct]
        docs.append(Doc(f"scan:fix{n}", "scan", f"Fix #{n}: {a.name} {a.installed}", "",
                        f"Fix number {n} of {len(result.plan)}: upgrade {a.name} from {a.installed} to "
                        f"{a.target or 'no fixed version known'}. Priority {a.tier}, score {a.score}. "
                        f"Fixes {len(a.resolves)} issue(s). Flags: {flags}. {a.name} is {kind}. "
                        f"It is {'a development-only dependency' if a.dev_only else 'not development-only'}. "
                        f"The version jump is {'risky (new major, or a 0.x minor)' if a.major_bump else 'not a major jump'}. "
                        "Why: " + "; ".join(a.reasons) + "."))
    if result.plan:
        docs.append(ranking_doc(result))
    return docs


def ranking_doc(result: ScanResult, head: int = 20, tail: int = 10) -> Doc:
    """One compact list of ALL fixes in ranked order, so 'which is last?', 'which is fifth?' and
    'how many fixes?' can be answered. (Before this, the chat only saw the top three fixes and, asked
    what to fix last, named one of them.) Very long plans keep the first `head` and the last `tail`."""
    plan = result.plan
    rows = [f"{n}. {a.name} {a.installed} -> {a.target or 'no fixed version'}, {a.tier}, score {a.score}"
            for n, a in enumerate(plan, 1)]
    if len(rows) > head + tail:
        rows = rows[:head] + [f"(fixes {head + 1} to {len(plan) - tail} are not listed here)"] + rows[-tail:]
    intro = (f"Ranking of all {len(plan)} fixes, highest priority first. "
             f"The first fix, with the highest priority, is {plan[0].name}. "
             f"The last fix, with the lowest priority, is number {len(plan)}: {plan[-1].name}.") if plan else \
        "There are no fixes in this scan."
    return Doc("scan:ranking", "scan", f"Ranking of all {len(plan)} fixes", "",
               "\n\n".join([intro] + rows))


def chat_docs(result: ScanResult, limit: int = 15) -> List[Doc]:
    """Scan documents plus every advisory / release-note document gathered so far (deduplicated)."""
    docs, seen = scan_docs(result, limit), set()
    for evidence, _ in result.evidence.values():
        for d in evidence:
            if d.doc_id not in seen:
                seen.add(d.doc_id)
                docs.append(d)
    return docs


def _excerpts(retrieved: Sequence, n: int = 3) -> str:
    return "\n".join(f"- {c.text.strip().splitlines()[0][:220]} [{c.chunk_id}]" for c, _ in retrieved[:n])


_RISK_WORDS = re.compile(r"break|compat|safe|migrat|risk|upgrad|jump|major|chang|release|differ", re.I)


def retrieve(question: str, chunks: Sequence[Chunk], embedder, k: int = 6) -> list:
    # "Will upgrading X break my code?" shares no words with "BREAKING CHANGE: removed ...", so add
    # the vocabulary of release notes to the search query for risk-type questions.
    query = question + (" breaking changes migration release notes removed" if _RISK_WORDS.search(question) else "")
    retrieved = Retriever(embedder, chunks).search(query, k=k, min_per_kind={"scan": 1, "release": 2})
    # Always include the summary and the top-ranked fixes. Word-matching retrieval can miss them
    # ("what should I fix first?" once matched fix #4-#6 and the model named the wrong package).
    have = {c.chunk_id for c, _ in retrieved}
    extra = [(c, 0.0) for c in chunks if c.doc_id in ANCHORS and c.chunk_id not in have]
    if _RISK_WORDS.search(question):
        extra += _package_release_notes(question, chunks, have | {c.chunk_id for c, _ in extra})
    return retrieved + extra


def _package_release_notes(question: str, chunks: Sequence[Chunk], have: set, limit: int = 4) -> list:
    """Release-note chunks of the package the question names, those with breaking-change wording first.
    (Without this, 'what changed in qs?' competed with dozens of advisory chunks and often lost.)"""
    named = set(re.findall(r"[a-z0-9][a-z0-9._-]*", question.lower()))
    mine = [c for c in chunks if c.kind == "release" and c.chunk_id not in have
            and c.title.lower().split()[:1] and c.title.lower().split()[0] in named]
    mine.sort(key=lambda c: (not find_breaking_lines([c]), -int(c.chunk_id[1:])))   # breaking first, then newest
    return [(c, 0.0) for c in mine[:limit]]


_VERSION_Q = re.compile(r"(?i)\b(?:which|what) version\b|\bfixed version\b|\bupgrade to\b|\bversion .*\bfix")
_NOT_VERSION_Q = re.compile(r"(?i)break|compat|safe|migrat|risk|changes?\b|jump")
_FIX_LINE = re.compile(r"upgrade (\S+) from (\S+) to (.+?)\. Priority")


def direct_version_answer(question: str, chunks: Sequence[Chunk]):
    """'Which version fixes X?' is a lookup, not a language task: the scanner already decided the
    target version (from OSV data) and wrote it into the scan source for that fix. Answer from it
    directly, with the citation, so a model cannot refuse or misquote it. Returns None if not applicable."""
    if not _VERSION_Q.search(question) or _NOT_VERSION_Q.search(question):
        return None
    low = question.lower()
    hits = []
    for c in chunks:
        if c.kind != "scan":
            continue
        m = _FIX_LINE.search(c.text)
        if m and re.search(r"(?<![\w.-])" + re.escape(m.group(1).lower()) + r"(?![\w-])", low):
            hits.append((c, m.group(1), m.group(2), m.group(3)))
    exact = [h for h in hits if re.search(r"(?<![\w.])" + re.escape(h[2]) + r"(?![\w])", question)]
    hits = exact or hits
    if not hits:
        return None
    parts = [f"Upgrade {n} from {i} to {t} [{c.chunk_id}]." if t != "no fixed version known"
             else f"No fixed version is listed for {n} {i} [{c.chunk_id}]." for c, n, i, t in hits[:3]]
    return ChatAnswer(" ".join(parts), False, {h[0].chunk_id: h[0] for h in hits[:3]},
                      ["answered directly from the scanner's own data, not by the AI"])


def answer_question(question: str, docs: Sequence[Doc], llm, embedder, k: int = 6) -> ChatAnswer:
    chunks = chunk_docs(docs)
    direct = direct_version_answer(question, chunks)
    if direct:
        return direct
    retrieved = retrieve(question, chunks, embedder, k)
    if not retrieved:
        return ChatAnswer("There is nothing to search yet. Scan a repository first.", False, {})
    by_id = {c.chunk_id: c for c, _ in retrieved}
    fallback = ChatAnswer("Most relevant passages (no AI answer):\n" + _excerpts(retrieved), False,
                          {c.chunk_id: c for c, _ in retrieved[:3]})
    if llm is None:
        fallback.warnings.append("No LLM key set, so showing the most relevant passages instead of an AI answer.")
        return fallback

    src = "\n\n".join(f"[{c.chunk_id}] ({c.title})\n{c.text}" for c, _ in retrieved)
    prompt = f"SOURCES:\n{src}\n\nQUESTION: {question}"
    try:
        raw = llm.complete(SYSTEM, prompt)
        if not _CITE.search(raw) and NO_ANSWER not in raw.lower():
            raw = llm.complete(SYSTEM, prompt + "\n\nYour previous answer had NO [S#] citations:\n" + raw
                               + "\n\nRewrite it, ending EVERY sentence with a citation such as [S1].")
        try:
            data = parse_json(raw)
        except (ValueError, json.JSONDecodeError):          # format slip: ask once more
            data = parse_json(llm.complete(SYSTEM, prompt + "\n\nReply with ONLY a JSON object: "
                                           '{"answer": "..."}'))
        text = strip_leaks(str(data.get("answer", "")))
    except LLMError as e:
        fallback.warnings.append(f"AI answer unavailable: {e}")
        return fallback
    except (ValueError, json.JSONDecodeError):
        fallback.warnings.append("AI answer was not in the expected format, so showing passages instead.")
        return fallback

    ok, bad = check_cited(text, set(by_id))
    if NO_ANSWER in text.lower() and not ok:
        return ChatAnswer("The sources do not say. I can only answer from this scan, its advisories and the "
                          "release notes you have opened.", False, {})
    if not text or bad or not ok:
        why = f"cites unknown sources {sorted(set(bad))}" if bad else "has no citation"
        fallback.warnings.append(f"AI answer rejected ({why}), so showing passages instead.")
        return fallback
    allowed = " ".join(c.text for c, _ in retrieved) + " " + question
    unverified = unverified_versions(text, allowed)
    warnings = [f"version number(s) in the AI text not found in the sources: {unverified}"] if unverified else []
    return ChatAnswer(text, True, {i: by_id[i] for i in sorted(set(ok), key=lambda s: int(s[1:]))}, warnings)
