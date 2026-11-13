"""Grounded explanations: the LLM may only use retrieved sources, and we check that it did.

Division of labour (this is the key design decision):
  DETERMINISTIC code decides   the upgrade-risk label, the confidence, which release-note lines
                               look like breaking changes. These must not depend on an LLM's mood.
  The LLM only writes         plain-English prose, with a [S#] citation after each claim.
  Our code then VERIFIES      every citation points to a real retrieved chunk, and every version
                               number in the prose appears in the sources. A failed check means
                               the AI text is discarded and the deterministic text is shown.

We never say an upgrade is "safe": release notes cannot prove that. The strongest label is
"no_warnings_found", which says only that nothing alarming was found in what we could read.
"""
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .evidence import Doc
from .llm import LLMError
from .rag import Chunk, Retriever, chunk_docs

RISK_LABELS = {
    "unknown": "UNKNOWN: no release notes could be read, so compatibility is not assessed",
    "no_warnings_found": "NO WARNING KEYWORDS FOUND in the release notes we could read (a keyword check only; "
                         "read the compatibility notes and the notes themselves before upgrading)",
    "elevated": "ELEVATED: release notes mention possible breaking changes",
    "elevated_jump": "ELEVATED: risky version jump (new major, or a 0.x minor); no breaking-change wording "
                     "found in the release notes we could read, which does not rule it out",
    "high": "HIGH: risky version jump (new major, or a 0.x minor) AND release notes mention possible breaking changes",
}
# A release-note line counts as a breaking-change warning when it uses STRONG wording, or when it
# says something was removed/renamed AND mentions an API-like thing, without being about docs/tests.
# (An earlier version matched any "remove", and flagged "remove dead link to benchmarks".)
_STRONG = re.compile(r"(?i)(breaking[ -]?changes?|\bBREAKING\b|backwards?[- ]incompatible|\bincompatible\b|"
                     r"drop(?:ped|s|ping)? (?:support|node|python)|"
                     r"no longer (?:supports?|works?|available|exported?|returns?|accepts?|allows?|default|bundled)|"
                     r"(?:now )?requires? (?:node|python)\b|minimum (?:node|python))")
_WEAK = re.compile(r"(?i)\b(?:removed?|removes|deleted?|renamed?)\b")
_API_WORD = re.compile(r"(?i)\b(?:deprecated|api|methods?|functions?|options?|arguments?|param(?:eter)?s?|"
                       r"exports?|exported|default|class|module|command|flags?|propert(?:y|ies)|support|"
                       r"helpers?|behaviou?r)\b")
_NOT_API = re.compile(r"(?i)\b(?:docs?|documentation|readme|typos?|links?|comments?|changelog|badges?|"
                      r"lint(?:ing)?|ci|tests?|benchmarks?|examples?)\b")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")


def is_breaking_line(line: str) -> bool:
    if _STRONG.search(line):
        return True
    return bool(_WEAK.search(line) and _API_WORD.search(line) and not _NOT_API.search(line))


_CITE = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")      # [S3] and also grouped [S3, S7]
_VERSION_IN_TEXT = re.compile(r"\b\d+\.\d+(?:\.\d+)*\b")

SYSTEM = (
    "You help developers understand a security-related dependency upgrade. "
    "Use ONLY the numbered sources provided. Put the id of the supporting source after every "
    "claim, like [S2]. If the sources do not say something, write 'not stated in the sources' "
    "instead of guessing. Do not state that any upgrade is safe or will not break anything. "
    "Do not mention version numbers that are not in the sources or facts. "
    'Reply with JSON only: {"what_is_wrong": "...", "compatibility_notes": "..."}. '
    "Each value is 1 to 3 short sentences, and EVERY sentence must end with a citation in square "
    'brackets, for example: "An attacker can crash the server with a crafted query string [S1]. '
    'A later release changed how arrays are parsed [S3]."'
)


@dataclass
class Explanation:
    name: str
    installed: str
    target: Optional[str]
    what_is_wrong: str
    fix: str
    compatibility: str
    breaking_lines: List[Tuple[str, str]]          # (line from release notes, chunk id)
    upgrade_risk: str
    confidence: str                                # "low" or "medium" (never "high")
    sources: Dict[str, Chunk]                      # chunks actually cited, by id
    used_llm: bool
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------- helpers
def find_breaking_lines(chunks: Sequence[Chunk], limit: int = 6) -> List[Tuple[str, str]]:
    out = []
    for c in chunks:
        if c.kind != "release":
            continue
        for line in c.text.splitlines():
            clean = _MD_LINK.sub(r"\1", line).replace("`", "").replace("**", "").strip(" \t-*#>")
            clean = re.sub(r"\s+[0-9a-f]{7,40}$", "", clean)       # trailing commit hash, e.g. "... node 4+ c499f76"
            if len(clean) >= 12 and not clean.lower().startswith(("release ", "changelog entry")) \
                    and is_breaking_line(clean):
                out.append((clean[:220], c.chunk_id))
    return out[:limit]


def risk_and_confidence(has_release_notes: bool, breaking: list, major_bump: bool) -> Tuple[str, str]:
    if not has_release_notes:
        return "unknown", "low"
    if breaking:
        return ("high" if major_bump else "elevated"), "medium"
    return ("elevated" if major_bump else "no_warnings_found"), "medium"


def risk_text(e: "Explanation") -> str:
    """The sentence to show. 'elevated' has two causes and the wording must say which one."""
    if e.upgrade_risk == "elevated" and not e.breaking_lines:
        return RISK_LABELS["elevated_jump"]
    return RISK_LABELS[e.upgrade_risk]


_LEAK = re.compile(r"(?i)never claim an upgrade|do not state that any upgrade|nobody can promise|\[developer\]|\[system\]|use only the numbered|"
                   r"reply with json|do not mention version numbers that are not")


def strip_leaks(text: str) -> str:
    """Small models sometimes copy our instructions into their answer. Drop those sentences."""
    parts = re.split(r"(?<=[.!?\]])\s+", text.strip())
    return " ".join(p for p in parts if not _LEAK.search(p)).strip()


def drop_trailing_not_stated(text: str) -> str:
    """'Calls now return false [S6]. Not stated in the sources.' -> keep the real statement. The model
    often tacks the admission onto an answer that already says something."""
    parts = re.split(r"(?<=[.!?\]])\s+", text.strip())
    kept = [p for p in parts if not p.lower().startswith("not stated")]
    return " ".join(kept) if kept else text.strip()


def parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)         # JSON wrapped in extra words: take the object
        if not m:
            raise
        data = json.loads(m.group(0))
    if not isinstance(data, dict):
        raise ValueError("not an object")
    return data


def check_cited(text: str, valid_ids: set) -> Tuple[List[str], List[str]]:
    found = [i for group in _CITE.findall(text) for i in re.findall(r"S\d+", group)]
    return [i for i in found if i in valid_ids], [i for i in found if i not in valid_ids]


def unverified_versions(text: str, allowed_text: str) -> List[str]:
    return sorted({v for v in _VERSION_IN_TEXT.findall(text) if v not in allowed_text})


def build_prompt(facts: str, retrieved: Sequence[Tuple[Chunk, float]]) -> str:
    src = "\n\n".join(f"[{c.chunk_id}] ({c.title})\n{c.text}" for c, _ in retrieved)
    return (f"FACTS (computed by our scanner, reliable):\n{facts}\n\nSOURCES:\n{src}\n\n"
            "Write: what_is_wrong (what the vulnerabilities allow) and "
            "compatibility_notes (anything in the sources about breaking changes or migration). "
            "Do not write the fix; the scanner already decided the target version.")


# ------------------------------------------------------------- main entry
def explain_action(name: str, installed: str, target: Optional[str], major_bump: bool,
                   docs: Sequence[Doc], facts: str, llm, embedder, k: int = 6) -> Explanation:
    chunks = chunk_docs(docs)
    by_id = {c.chunk_id: c for c in chunks}
    warnings: List[str] = []
    breaking = find_breaking_lines(chunks)
    risk, confidence = risk_and_confidence(any(c.kind == "release" for c in chunks), breaking, major_bump)

    # deterministic text: always available, used when there is no LLM or its answer fails checks
    adv = [c for c in chunks if c.kind == "advisory"]
    first_per_doc = {c.doc_id: c for c in reversed(adv)}.values()
    firsts = sorted(first_per_doc, key=lambda c: int(c.chunk_id[1:]))[:3]
    d_wrong = " ".join(f"{c.title.replace('Advisory ', '')}: {_headline(c)} [{c.chunk_id}]"
                       for c in firsts) or "No advisory text was available."
    d_fix = (f"Upgrade {name} from {installed} to {target}." if target
             else "No fixed version is listed for these vulnerabilities.")
    d_compat = (" ".join(f"\"{t}\" [{cid}]" for t, cid in breaking[:3]) if breaking
                else "No breaking-change wording found in the release notes we could read."
                if risk != "unknown" else "No release notes were found, so compatibility could not be checked.")
    wrong, fix, compat, used_llm, cited_ids = d_wrong, d_fix, d_compat, False, set(re.findall(r"S\d+", d_wrong))

    if llm is not None and chunks:
        retrieved = Retriever(embedder, chunks).search(
            f"{name} vulnerability fix upgrade {installed} to {target} breaking changes migration",
            k=k, min_per_kind={"advisory": 2, "release": 2})
        valid = {c.chunk_id for c, _ in retrieved}
        allowed = facts + " " + " ".join(c.text for c, _ in retrieved) + f" {installed} {target or ''}"
        raw = ""
        try:
            prompt = build_prompt(facts, retrieved)
            raw = llm.complete(SYSTEM, prompt)
            if not _CITE.search(raw):
                # Small models often forget the citations. Ask once more, showing what was missing.
                raw = llm.complete(SYSTEM, prompt + "\n\nYour previous answer had NO [S#] citations:\n" + raw
                                   + "\n\nRewrite it. Keep only what the sources support and end EVERY "
                                     "sentence with a citation such as [S1], using only the ids above.")
            data = parse_json(raw)
            w, n = (strip_leaks(str(data.get(x, ""))) for x in ("what_is_wrong", "compatibility_notes"))
            ok_w, bad_w = check_cited(w, valid)
            ok_n, bad_n = check_cited(n, valid)
            if not w:
                raise ValueError("missing fields")
            if bad_w or bad_n:
                raise ValueError(f"cites sources that were not provided: {sorted(set(bad_w + bad_n))}")
            if not ok_w:
                raise ValueError("what_is_wrong has no citation")
            if n and not ok_n and "not stated" not in n.lower():
                n = ""                                   # uncited compatibility claim: drop it
                warnings.append("dropped an uncited compatibility statement from the AI answer")
            n = drop_trailing_not_stated(n)
            if breaking and re.sub(r"\[[^\]]*\]", "", n).strip(" .").lower().startswith("not stated"):
                n = ""          # our keyword check found breaking-style lines; show them instead of "not stated"
            bad_versions = unverified_versions(" ".join((w, n)), allowed)
            if bad_versions:
                warnings.append(f"version number(s) in the AI text not found in the sources: {bad_versions}")
            # `fix` stays the scanner's own sentence: the target version is decided by code, never by the LLM
            wrong, used_llm = w, True
            compat = n or d_compat
            cited_ids = set(ok_w + ok_n)
        except LLMError as e:
            warnings.append(f"AI explanation unavailable: {e}; showing the deterministic explanation")
        except (ValueError, json.JSONDecodeError) as e:
            warnings.append(f"AI answer rejected ({e}); showing the deterministic explanation")
            if os.environ.get("SECUREDEPS_DEBUG"):
                warnings.append("raw AI answer: " + raw[:700].replace("\n", " "))
    elif llm is not None:
        warnings.append("no source documents, so no AI explanation was attempted")

    cited_ids |= set(re.findall(r"S\d+", compat))      # includes ids inside deterministic text
    return Explanation(name, installed, target, wrong, fix, compat, breaking, risk, confidence,
                       {i: by_id[i] for i in sorted(cited_ids, key=lambda s: int(s[1:])) if i in by_id},
                       used_llm, warnings)


def _headline(c: Chunk) -> str:
    """First meaningful sentence of an advisory chunk (skip the id line)."""
    lines = [l.strip() for l in c.text.splitlines() if l.strip() and not l.startswith(("GHSA-", "PYSEC-", "CVE-"))]
    return (lines[0] if lines else c.text)[:170]
