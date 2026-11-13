"""Collect the documents the LLM is allowed to use ("evidence").

Two kinds:
  advisory  the OSV record for each vulnerability (summary + details). We already downloaded
            these in stage 2, so no extra network call.
  release   GitHub release notes for the versions between the installed one and the fixed one.
            Found by: package name -> npm/PyPI registry -> source repo URL -> GitHub releases.

Everything here is best-effort: a missing repo, a rate limit or a project without GitHub
releases gives fewer documents and a warning, never a crash. The explanation step then says
plainly that no release notes were found.
"""
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import requests

from .intel import _newer
from .models import NPM, PYPI
from .repo import API, _headers

MAX_RELEASES = 12
MAX_DOC_CHARS = 8000


@dataclass(frozen=True)
class Doc:
    doc_id: str      # e.g. "osv:GHSA-xxxx" or "release:v4.17.21"
    kind: str        # "advisory" or "release"
    title: str
    url: str
    text: str


def _is_boundary(ver: str) -> bool:
    """A release that starts a new major version (4.0.0) or a new 0.x minor (0.5.0). These are the
    releases where breaking changes live, so the cap below must never throw them away."""
    major, minor, patch = ([int(x) for x in re.findall(r"\d+", ver)[:3]] + [0, 0, 0])[:3]
    return patch == 0 and (minor == 0 or major == 0)


def cap_docs(found: List[Tuple[str, Doc]], limit: int = MAX_RELEASES) -> Tuple[List[Tuple[str, Doc]], Optional[str]]:
    """found is sorted oldest -> newest. Keep at most `limit`: every major / 0.x-minor release first,
    then fill with the newest ones. (Keeping only the newest 12 hid the 7.0.0 notes of tar.)"""
    if len(found) <= limit:
        return found, None
    keep = {i for i, (v, _) in enumerate(found) if _is_boundary(v)}
    if len(keep) > limit:
        keep = set(sorted(keep)[-limit:])
    for i in range(len(found) - 1, -1, -1):
        if len(keep) >= limit:
            break
        keep.add(i)
    return [found[i] for i in sorted(keep)], \
        f"{len(found)} entries between versions; using {limit} (all major and 0.x-minor releases first, then the newest)"


# ------------------------------------------------------------------ advisories
def advisory_docs(records: Dict[str, dict], vuln_ids: List[str], limit: int = 10) -> List[Doc]:
    docs = []
    for vid in vuln_ids[:limit]:
        rec = records.get(vid)
        if not rec:
            continue
        text = "\n\n".join(p.strip() for p in (rec.get("summary"), rec.get("details")) if p and p.strip())
        aliases = ", ".join(a for a in rec.get("aliases", []) if a.startswith("CVE-"))
        if aliases:
            text = f"{vid} ({aliases})\n\n{text}"
        if text:
            docs.append(Doc(f"osv:{vid}", "advisory", f"Advisory {vid}",
                            f"https://osv.dev/vulnerability/{vid}", text[:MAX_DOC_CHARS]))
    return docs


# ----------------------------------------------------------- package -> repo
def github_slug(url: str) -> Optional[Tuple[str, str]]:
    m = re.search(r"github\.com[:/]+([\w.-]+)/([\w.-]+)", url or "") \
        or re.match(r"^(?:github:)?([\w.-]+)/([\w.-]+)$", (url or "").strip())   # npm shorthand
    if not m:
        return None
    return m.group(1), re.sub(r"\.git$", "", m.group(2))


def find_repo(ecosystem: str, name: str, session) -> Optional[Tuple[str, str]]:
    """Ask the package registry where the source code lives."""
    try:
        if ecosystem == NPM:
            r = session.get(f"https://registry.npmjs.org/{quote(name, safe='@')}/latest", timeout=20)
            if not r.ok:
                return None
            repo = r.json().get("repository")
            return github_slug(repo.get("url", "") if isinstance(repo, dict) else repo or "")
        if ecosystem == PYPI:
            r = session.get(f"https://pypi.org/pypi/{quote(name)}/json", timeout=20)
            if not r.ok:
                return None
            info = r.json().get("info", {})
            urls = list((info.get("project_urls") or {}).values()) + [info.get("home_page") or ""]
            for u in urls:
                slug = github_slug(u)
                if slug:
                    return slug
    except (requests.RequestException, ValueError):
        return None
    return None


# --------------------------------------------------------------- release notes
_PREFIX = re.compile(r"^(?:v|release[-_]?|version[-_]?|rel[-_]?)", re.I)
_VERSION = re.compile(r"(\d+(?:\.\d+){1,3}(?:[-+][0-9A-Za-z.]+)?)")


def tag_version(tag: str, package: str) -> Optional[str]:
    """'v4.17.21' / 'lodash@4.17.21' / 'release-1.2.3' -> the version; other tags -> None.
    Tags that belong to a different package in a monorepo ('@babel/core@7.0.0' when we
    want 'babel-traverse') are rejected."""
    rest = _PREFIX.sub("", tag.strip())
    if not rest[:1].isdigit() and package.lower() not in tag.lower():
        return None
    m = _VERSION.search(tag if package.lower() in tag.lower() else rest)
    return m.group(1) if m else None


def release_docs(ecosystem: str, name: str, installed: str, target: str, repo: Tuple[str, str],
                 token: Optional[str], session) -> Tuple[List[Doc], List[str]]:
    """Release notes for versions in (installed, target]."""
    owner, rname = repo
    found, warnings = [], []
    for page in (1, 2, 3):
        try:
            r = session.get(f"{API}/repos/{owner}/{rname}/releases", headers=_headers(token),
                            params={"per_page": 100, "page": page}, timeout=30)
        except requests.RequestException as e:
            return [], [f"GitHub releases unreachable ({type(e).__name__})"]
        if not r.ok:
            return [], [f"GitHub releases unavailable (HTTP {r.status_code}); set GITHUB_TOKEN if rate limited"]
        batch = r.json()
        for rel in batch:
            if rel.get("draft") or rel.get("prerelease"):
                continue
            ver = tag_version(rel.get("tag_name", ""), name)
            body = (rel.get("body") or "").strip()
            if not ver or not body:
                continue
            if _newer(ecosystem, ver, installed) and not _newer(ecosystem, ver, target):
                found.append((ver, Doc(f"release:{rel['tag_name']}", "release",
                                       f"{name} {rel['tag_name']} release notes",
                                       rel.get("html_url") or f"https://github.com/{owner}/{rname}/releases",
                                       f"Release {rel['tag_name']} of {name}:\n{body}"[:MAX_DOC_CHARS])))
        if len(batch) < 100:
            break
    found.sort(key=lambda t: [int(x) for x in re.findall(r"\d+", t[0])[:4]])
    found, cap_note = cap_docs(found)
    if cap_note:
        warnings.append(cap_note)
    if not found:
        warnings.append(f"no GitHub release notes found for {name} {installed} -> {target}")
    return [d for _, d in found], warnings


_CHANGELOG_FILE = re.compile(r"^(?:changelog|changes|history|news)(?:\.md|\.markdown|\.txt)?$", re.I)
_MD_HEADING = re.compile(r"^\s{0,3}#{1,4}\s*(.*)$")
_VERSION_AT_START = re.compile(r"^[\s\[\*_v]*(\d+\.\d+(?:\.\d+)?(?:-[0-9A-Za-z.]+)?)")


def split_changelog(text: str) -> List[Tuple[str, str]]:
    """Cut a CHANGELOG file into (version, section text). Understands '## [1.2.3] - date',
    '## v1.2.3', and the older underlined style  '1.2.3 / 2015-01-01' + '=====' ."""
    lines = text.splitlines()
    heads = []
    for i, line in enumerate(lines):
        m = _MD_HEADING.match(line)
        if m:
            cand = m.group(1)
        elif line.strip() and i + 1 < len(lines) and re.fullmatch(r"\s*(?:=+|-{3,})\s*", lines[i + 1]):
            cand = line
        else:
            continue
        v = _VERSION_AT_START.match(cand)
        if v:
            heads.append((i, v.group(1)))
    out = []
    for n, (i, v) in enumerate(heads):
        end = heads[n + 1][0] if n + 1 < len(heads) else len(lines)
        out.append((v, "\n".join(lines[i:end]).strip()))
    return out


def changelog_docs(ecosystem: str, name: str, installed: str, target: str, repo: Tuple[str, str],
                   token: Optional[str], session) -> Tuple[List[Doc], List[str]]:
    """Fallback for projects (such as qs) that keep a CHANGELOG file instead of GitHub Releases."""
    owner, rname = repo
    try:
        r = session.get(f"{API}/repos/{owner}/{rname}/contents", headers=_headers(token), timeout=30)
        if not r.ok:
            return [], [f"changelog lookup unavailable (HTTP {r.status_code})"]
        listing = r.json()
        if not isinstance(listing, list):          # GitHub sent an error object, not a file list
            return [], ["changelog lookup failed"]
        files = [e["name"] for e in listing if isinstance(e, dict) and e.get("type") == "file"
                 and _CHANGELOG_FILE.match(e.get("name", ""))]
        if not files:
            return [], []
        fname = files[0]
        r = session.get(f"{API}/repos/{owner}/{rname}/contents/{fname}",
                        headers=_headers(token, raw=True), timeout=30)
        if not r.ok:
            return [], [f"could not read {fname} (HTTP {r.status_code})"]
        text = r.text
    except (requests.RequestException, ValueError, TypeError, KeyError, AttributeError):
        return [], ["changelog lookup failed"]
    url = f"https://github.com/{owner}/{rname}/blob/HEAD/{fname}"
    found = [(v, Doc(f"changelog:{v}", "release", f"{name} {v} ({fname})", url,
                     f"Changelog entry for {name} {v}:\n{body}"[:MAX_DOC_CHARS]))
             for v, body in split_changelog(text)
             if _newer(ecosystem, v, installed) and not _newer(ecosystem, v, target)]
    found.sort(key=lambda t: [int(x) for x in re.findall(r"\d+", t[0])[:4]])
    warnings = []
    found, cap_note = cap_docs(found)
    if cap_note:
        warnings.append(cap_note)
    return [d for _, d in found], warnings


def gather_evidence(ecosystem: str, name: str, installed: str, target: Optional[str],
                    records: Dict[str, dict], vuln_ids: List[str], token: Optional[str] = None,
                    session=None) -> Tuple[List[Doc], List[str]]:
    s = session or requests.Session()
    docs = advisory_docs(records, vuln_ids)
    warnings: List[str] = []
    if not target:
        warnings.append("no fixed version known, so no release notes to compare")
        return docs, warnings
    repo = find_repo(ecosystem, name, s)
    if not repo:
        warnings.append(f"could not find the GitHub repository of {name}")
        return docs, warnings
    rel, w = release_docs(ecosystem, name, installed, target, repo, token, s)
    if not rel:                                  # no GitHub Releases: try a CHANGELOG file instead
        cl, w2 = changelog_docs(ecosystem, name, installed, target, repo, token, s)
        if cl:
            return docs + cl, warnings + w2
        w = w + w2
    return docs + rel, warnings + w
