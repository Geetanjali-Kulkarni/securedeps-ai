"""Scan a sample of public GitHub repositories and report what the ranking finds.

    python3 -m securedeps.sample_scan collect            pick ~120 candidate repos -> sample_repos.txt
    python3 -m securedeps.sample_scan scan               scan them (resumable)    -> sample_results.csv
    python3 -m securedeps.sample_scan report             numbers + methodology    -> printed

Needs GITHUB_TOKEN (the search and file APIs allow only 60 requests/hour without one).
No LLM is used: this measures the deterministic scanner only.

METHOD (write this in the README exactly as it is, including the limits):
  * Candidates: GitHub repository search, one query per language (JavaScript, TypeScript, Python),
    stars 300..3000, not archived, not a fork, pushed within the last 180 days. Up to 300 per
    language are collected, shuffled with a fixed seed (reproducible), and 40 per language kept.
    This is a sample of ACTIVE, MID-SIZED open-source projects, not of all software.
  * A repo is 'usable' only if it has a lockfile or exact pins, so versions are known.
    Repos with only version ranges are reported as 'no_exact_versions' and excluded from rates.
  * 'Actively exploited' means: a CVE of the issue is in CISA's KEV list on the scan date.
  * 'Critical' means CVSS v3 base score >= 9.0 (our own calculation from the vector).
  * KEV and EPSS change daily, so results are a snapshot: the scan date is recorded.
"""
import csv
import os
import random
import statistics
import sys
import time
from datetime import date, timedelta
from typing import Dict, List, Optional

import requests

from .intel import IntelError, analyze
from .ranking import build_issues, fix_plan
from .repo import API, RepoError, _headers, scan_repo_dependencies

LANGUAGES = ["JavaScript", "TypeScript", "Python"]
STARS = "300..3000"
POOL_PER_LANGUAGE = 300
KEEP_PER_LANGUAGE = 40
SEED = 42
FIELDS = ["repo", "language", "status", "error", "deps", "scanned", "findings", "issues", "packages_to_fix",
          "fix_now", "fix_soon", "plan", "low", "kev_issues", "critical_issues", "critical_deprioritised",
          "scan_date"]


# --------------------------------------------------------------- 1. collect
def search_repos(language: str, token: Optional[str], session, today: Optional[date] = None) -> List[str]:
    since = ((today or date.today()) - timedelta(days=180)).isoformat()
    q = f"language:{language} stars:{STARS} archived:false fork:false pushed:>={since}"
    found = []
    for page in range(1, POOL_PER_LANGUAGE // 100 + 1):
        r = session.get(f"{API}/search/repositories", headers=_headers(token), timeout=30,
                        params={"q": q, "sort": "stars", "order": "desc", "per_page": 100, "page": page})
        if not r.ok:
            raise RepoError(f"GitHub search failed (HTTP {r.status_code}). Set GITHUB_TOKEN and try again.")
        items = r.json().get("items", [])
        found += [it["full_name"] for it in items]
        if len(items) < 100:
            break
        time.sleep(2)                                   # the search API allows 30 requests/minute
    return found


def collect(token: Optional[str], session=None, keep: int = KEEP_PER_LANGUAGE) -> List[tuple]:
    s = session or requests.Session()
    rng = random.Random(SEED)
    chosen = []
    for lang in LANGUAGES:
        pool = search_repos(lang, token, s)
        rng.shuffle(pool)
        chosen += [(name, lang) for name in pool[:keep]]
    return chosen


# ------------------------------------------------------------------ 2. scan
def scan_one(name: str, language: str, token: Optional[str]) -> Dict[str, object]:
    row: Dict[str, object] = {f: "" for f in FIELDS}
    row.update(repo=name, language=language, scan_date=str(date.today()))
    try:
        repo = scan_repo_dependencies(f"https://github.com/{name}", token)
        deps = repo["dependencies"]
        row["deps"] = len(deps)
        res = analyze(deps)
    except (RepoError, IntelError, requests.RequestException) as e:
        row.update(status="error", error=str(e)[:200])
        return row
    row["scanned"] = res.scanned
    if res.scanned == 0:
        row["status"] = "no_exact_versions"
        return row
    issues, plan = build_issues(res.findings), fix_plan(res.findings)
    tiers = {t: sum(1 for a in plan if a.tier == t) for t in ("Fix now", "Fix soon", "Plan", "Low")}
    critical = [i for i in issues if i.cvss_value is not None and i.cvss_value >= 9.0]
    row.update(status="ok", findings=len(res.findings), issues=len(issues), packages_to_fix=len(plan),
               fix_now=tiers["Fix now"], fix_soon=tiers["Fix soon"], plan=tiers["Plan"], low=tiers["Low"],
               kev_issues=sum(1 for i in issues if i.in_kev), critical_issues=len(critical),
               critical_deprioritised=sum(1 for i in critical if i.tier in ("Plan", "Low")))
    return row


def done_repos(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {r["repo"] for r in csv.DictReader(f)}


def scan_all(repos: List[tuple], out_path: str, token: Optional[str], pause: float = 1.0, scanner=scan_one) -> int:
    """Append one CSV row per repo as soon as it is scanned, so an interruption loses nothing."""
    skip = done_repos(out_path)
    new_file = not os.path.exists(out_path)
    n = 0
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for i, (name, lang) in enumerate(repos, 1):
            if name in skip:
                continue
            row = scanner(name, lang, token)
            w.writerow(row)
            f.flush()
            n += 1
            print(f"[{i}/{len(repos)}] {name}: {row['status']}" + (f" ({row['error']})" if row["error"] else ""))
            time.sleep(pause)
    return n


# ---------------------------------------------------------------- 3. report
def _int(v) -> int:
    return int(v) if str(v).strip() else 0


def report(path: str) -> str:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return "No results yet."
    ok = [r for r in rows if r["status"] == "ok"]
    n_err = sum(1 for r in rows if r["status"] == "error")
    n_range = sum(1 for r in rows if r["status"] == "no_exact_versions")
    dates = sorted({r["scan_date"] for r in rows})
    lines = [f"Sample scan, scan date {', '.join(dates)}",
             f"Repositories attempted: {len(rows)}  |  usable (exact versions known): {len(ok)}  |  "
             f"no exact versions: {n_range}  |  errors: {n_err}"]
    if not ok:
        return "\n".join(lines + ["No usable repositories, so no rates can be reported."])
    pct = lambda k: f"{k}/{len(ok)} ({k / len(ok):.0%})"
    with_vuln = [r for r in ok if _int(r["issues"]) > 0]
    lines += [f"\nOf {len(ok)} usable repositories:",
              f"  with at least one known vulnerability:                     {pct(len(with_vuln))}",
              f"  with a vulnerability whose CVE is in CISA KEV (exploited): {pct(sum(1 for r in ok if _int(r['kev_issues']) > 0))}",
              f"  with at least one 'Fix now' or 'Fix soon' package:         "
              f"{pct(sum(1 for r in ok if _int(r['fix_now']) + _int(r['fix_soon']) > 0))}",
              f"  median distinct issues per repository:                     {statistics.median(_int(r['issues']) for r in ok):g}"]
    crit = sum(_int(r["critical_issues"]) for r in ok)
    depri = sum(_int(r["critical_deprioritised"]) for r in ok)
    if crit:
        lines.append(f"\nIssues with CVSS >= 9.0 (a severity-only scanner calls these 'critical'): {crit}")
        lines.append(f"  of those, ranked only 'Plan' or 'Low' by exploit data and context:  {depri} ({depri / crit:.0%})")
    by_lang: Dict[str, List[dict]] = {}
    for r in ok:
        by_lang.setdefault(r["language"], []).append(r)
    lines.append("\nBy language (usable repos / with a KEV-listed vulnerability):")
    for lang, rs in sorted(by_lang.items()):
        lines.append(f"  {lang:<11}{len(rs):>3} / {sum(1 for r in rs if _int(r['kev_issues']) > 0)}")
    lines.append("\nLimits: active, mid-sized open-source projects only; a snapshot (KEV and EPSS change daily); "
                 "exact versions required; results describe this sample, not all software.")
    return "\n".join(lines)


# --------------------------------------------------------------------- CLI
def main(argv: List[str]) -> int:
    token = os.environ.get("GITHUB_TOKEN")
    cmd = argv[0] if argv else ""
    if cmd == "collect":
        if not token:
            print("Set GITHUB_TOKEN first (GitHub allows only 60 requests/hour without one).")
            return 1
        try:
            repos = collect(token)
        except RepoError as e:
            print("Error:", e)
            return 1
        with open("sample_repos.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(f"{n}\t{lang}" for n, lang in repos) + "\n")
        print(f"Wrote {len(repos)} candidate repositories to sample_repos.txt")
        return 0
    if cmd == "scan":
        if not token:
            print("Set GITHUB_TOKEN first (about 10 API requests per repository are needed).")
            return 1
        if not os.path.exists("sample_repos.txt"):
            print("Run 'collect' first.")
            return 1
        repos = [tuple(line.split("\t")) for line in open("sample_repos.txt", encoding="utf-8").read().splitlines() if line]
        scan_all(repos, "sample_results.csv", token)
        print("\n" + report("sample_results.csv"))
        return 0
    if cmd == "report":
        print(report(argv[1] if len(argv) > 1 else "sample_results.csv"))
        return 0
    print("usage: python3 -m securedeps.sample_scan collect | scan | report")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
