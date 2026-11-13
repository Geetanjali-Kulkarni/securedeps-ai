"""Read a public GitHub repo and turn its dependency files into Dependency objects.

Flow:  URL -> (owner, repo) -> default branch -> list every file in the repo (one API call)
       -> keep the dependency files we understand -> download them -> parse -> resolve.
"""
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import requests

from .models import Dependency
from .parsers import parser_for

API = "https://api.github.com"
SKIP_DIRS = ("node_modules/", "vendor/", "venv/", ".venv/", "site-packages/", "dist/", "build/",
             "test/", "tests/", "fixtures/", "examples/", "docs/")
MAX_FILES = 40            # safety limit for huge monorepos
MAX_FILE_BYTES = 5_000_000


class RepoError(Exception):
    """A problem we can explain to the user in one sentence."""


def parse_github_url(url: str) -> Tuple[str, str]:
    m = re.match(r"^(?:https?://)?(?:www\.)?github\.com/([^/\s]+)/([^/\s#?]+)", url.strip())
    if not m:
        raise RepoError("That doesn't look like a GitHub repository link (github.com/owner/repo).")
    return m.group(1), re.sub(r"\.git$", "", m.group(2))


def _headers(token: Optional[str], raw: bool = False) -> dict:
    h = {"Accept": "application/vnd.github.raw" if raw else "application/vnd.github+json",
         "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "securedeps-ai"}
    token = token or os.environ.get("GITHUB_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _get(session, url, token, raw=False, **kw):
    r = session.get(url, headers=_headers(token, raw), timeout=30, **kw)
    if r.status_code == 404:
        raise RepoError("Repository or file not found. Is it public, and is the link right?")
    if r.status_code in (403, 429) and r.headers.get("X-RateLimit-Remaining") == "0":
        raise RepoError("GitHub rate limit reached. Set a GITHUB_TOKEN (free) to raise the limit "
                        "from 60 to 5,000 requests per hour.")
    if r.status_code == 401:
        raise RepoError("GitHub rejected the token. Check GITHUB_TOKEN.")
    if not r.ok:
        raise RepoError(f"GitHub returned HTTP {r.status_code} for {url.replace(API, '')}.")
    return r


def list_dependency_files(owner: str, repo: str, token: Optional[str] = None,
                          session: Optional[requests.Session] = None) -> Tuple[str, List[str]]:
    """Return (default_branch, [paths of files we can parse])."""
    s = session or requests.Session()
    info = _get(s, f"{API}/repos/{owner}/{repo}", token).json()
    branch = info["default_branch"]
    tree = _get(s, f"{API}/repos/{owner}/{repo}/git/trees/{branch}", token,
                params={"recursive": "1"}).json()
    paths = []
    for item in tree.get("tree", []):
        p = item["path"]
        if item["type"] != "blob" or any(("/" + p).find("/" + d) != -1 for d in SKIP_DIRS):
            continue
        if parser_for(p) and item.get("size", 0) <= MAX_FILE_BYTES:
            paths.append(p)
    return branch, sorted(paths)[:MAX_FILES]


def fetch_file(owner: str, repo: str, path: str, branch: str, token: Optional[str] = None,
               session: Optional[requests.Session] = None) -> str:
    s = session or requests.Session()
    url = f"{API}/repos/{owner}/{repo}/contents/{path}"
    return _get(s, url, token, raw=True, params={"ref": branch}).text


def resolve(files: Dict[str, str]) -> Tuple[List[Dependency], List[str]]:
    """files: {repo path: file text}. Returns (dependencies, warnings).

    Rule: inside one folder, per ecosystem, a LOCKFILE wins over a manifest, because a
    lockfile says what is really installed. Manifests are used only when no lockfile exists,
    and then ranges (unknown versions) are kept but marked unscannable.
    """
    groups = defaultdict(lambda: {"lock": [], "manifest": []})
    warnings: List[str] = []
    for path, text in files.items():
        found = parser_for(path)
        if not found:
            continue
        parser, ecosystem, is_lock = found
        folder = path.rsplit("/", 1)[0] if "/" in path else ""
        try:
            deps = parser(text, path)
        except Exception as e:   # one broken file must not kill the whole scan
            warnings.append(f"Could not parse {path}: {type(e).__name__}: {e}")
            continue
        groups[(folder, ecosystem)]["lock" if is_lock else "manifest"].extend(deps)

    out: List[Dependency] = []
    for (folder, ecosystem), g in groups.items():
        if g["lock"]:
            out.extend(g["lock"])
            if g["manifest"]:
                warnings.append(f"{folder or '/'}: using the {ecosystem} lockfile; manifest "
                                f"entries ignored ({len(g['manifest'])}).")
        else:
            out.extend(g["manifest"])
    # de-duplicate the same package+version found in several files
    seen, unique = set(), []
    for d in out:
        key = (d.ecosystem, d.name, d.version, d.source_file if d.version is None else "")
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique, warnings


def scan_repo_dependencies(url: str, token: Optional[str] = None) -> dict:
    """The one function the rest of the app calls."""
    owner, repo = parse_github_url(url)
    s = requests.Session()
    branch, paths = list_dependency_files(owner, repo, token, s)
    if not paths:
        raise RepoError("No supported dependency files found (requirements*.txt, package.json, "
                        "package-lock.json, Pipfile.lock, poetry.lock). Not supported yet: projects that only "
                        "use pyproject.toml or uv.lock (common in newer Python projects), Java (Maven/Gradle), "
                        "Go and Rust. If this repository uses one of those, it cannot be scanned yet.")
    files = {p: fetch_file(owner, repo, p, branch, token, s) for p in paths}
    deps, warnings = resolve(files)
    return {"owner": owner, "repo": repo, "branch": branch, "files": paths,
            "dependencies": deps, "warnings": warnings}


if __name__ == "__main__":      # quick manual check: python -m securedeps.repo <github url>
    import sys
    res = scan_repo_dependencies(sys.argv[1])
    deps = res["dependencies"]
    print(f"{res['owner']}/{res['repo']} @ {res['branch']}: {len(res['files'])} dependency file(s)")
    for f in res["files"]:
        print("  file:", f)
    scannable = [d for d in deps if d.scannable]
    print(f"{len(deps)} dependencies, {len(scannable)} with exact versions, "
          f"{len(deps) - len(scannable)} unresolved (ranges)")
    for w in res["warnings"]:
        print("  warning:", w)
    for d in deps[:15]:
        print(f"  {d.ecosystem:5} {d.name:30} {d.version or '(range: ' + d.spec + ')':20} {d.confidence}")
