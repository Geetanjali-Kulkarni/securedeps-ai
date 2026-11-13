"""Command line scan:  python3 -m securedeps.scan <github url> [--all] [--explain] [--top=N]

Stage 1 (find dependencies) -> stage 2 (look up vulnerabilities) -> stage 3 (rank them)
-> stage 4 (--explain: evidence + RAG + grounded explanation).
"""
import sys
import time
from datetime import date

import requests

from .evidence import gather_evidence
from .explain import explain_action, risk_text
from .intel import analyze
from .llm import get_llm
from .rag import get_embedder
from .ranking import build_issues, compare_with_baseline, fix_plan
from .repo import RepoError, scan_repo_dependencies


def _kev(v):
    return {True: "YES", False: "no", None: "unknown"}[v]


def print_explanations(plan, issues, res, count: int) -> None:
    llm = get_llm()
    print(f"\n=== EXPLANATIONS FOR THE TOP {min(count, len(plan))} FIXES ===")
    if llm is None:
        print("No LLM key found (set GEMINI_API_KEY or ANTHROPIC_API_KEY): using the deterministic "
              "explanation built from the advisories and release notes.")
    else:
        print(f"LLM: {type(llm).__name__} ({llm.model}). Every AI sentence must cite a source; "
              "answers that fail the checks are replaced.")
    print("Loading the embedding model (the first run downloads it)...")
    embedder = get_embedder()
    print(f"Embedder: {embedder.name}")
    session = requests.Session()
    for n, a in enumerate(plan[:count], 1):
        if llm is not None and n > 1:
            time.sleep(5)          # space out requests: free tiers allow only a few per minute
        ids = [i.vuln_id for i in issues
               if i.name == a.name and a.installed in i.versions and i.vuln_id in a.resolves] or a.resolves
        docs, ev_warn = gather_evidence("npm" if _is_npm(a) else "PyPI", a.name, a.installed, a.target,
                                        res.records, ids, session=session)
        facts = (f"Package: {a.name} {a.installed} -> {a.target or 'no fixed version known'}. "
                 f"Priority: {a.tier} (score {a.score}). "
                 f"Upgrade is a risky version jump (new major version, or a 0.x minor): {'yes' if a.major_bump else 'no'}. "
                 f"Score breakdown: " + "; ".join(a.reasons))
        e = explain_action(a.name, a.installed, a.target, a.major_bump, docs, facts, llm, embedder)
        print(f"\n{n}. {a.name} {a.installed} -> {a.target or '(no fix known)'}   [{'AI text' if e.used_llm else 'deterministic text'}]")
        print(f"   What is wrong: {e.what_is_wrong}")
        print(f"   Fix:           {e.fix}")
        print(f"   Compatibility: {e.compatibility}")
        print(f"   Upgrade risk:  {risk_text(e)} (confidence: {e.confidence})")
        for cid, c in e.sources.items():
            print(f"   [{cid}] {c.title}  {c.url}")
        for w in ev_warn + e.warnings:
            print(f"   note: {w}")


def _is_npm(a) -> bool:
    return a.ecosystem == "npm"


def main(url: str, show_all: bool = False, explain: int = 0) -> int:
    try:
        repo = scan_repo_dependencies(url)
    except RepoError as e:
        print("Error:", e)
        return 1
    deps = repo["dependencies"]
    print(f"{repo['owner']}/{repo['repo']}: {len(repo['files'])} dependency file(s), "
          f"{len(deps)} dependencies   (scan date {date.today()})")
    for w in repo["warnings"]:
        print("  note:", w)
    res = analyze(deps)
    print(f"Checked {res.scanned} exact package versions against OSV; "
          f"{res.unscannable} skipped (version unknown).")
    for w in res.warnings:
        print("  warning:", w)

    issues = build_issues(res.findings)
    plan = fix_plan(res.findings)
    tiers = {}
    for a in plan:
        tiers[a.tier] = tiers.get(a.tier, 0) + 1
    print(f"\n{len(res.findings)} findings = {len(issues)} distinct issues across {len(plan)} "
          f"package versions.  Fix list by priority: "
          + ", ".join(f"{k}: {v}" for k, v in sorted(tiers.items(), key=lambda t: ["Fix now", "Fix soon", "Plan", "Low"].index(t[0]))))

    limit = len(plan) if show_all else 10
    print(f"\n=== FIX THESE FIRST (top {min(limit, len(plan))} of {len(plan)}) ===")
    for n, a in enumerate(plan[:limit], 1):
        action = (f"upgrade to {a.target}" if a.target else "no fixed version known")
        flags = []
        if a.major_bump:
            flags.append("RISKY version jump (new major, or a 0.x minor): check release notes for breaking changes")
        if a.dev_only:
            flags.append("dev-only")
        if a.direct is False:
            flags.append("transitive: update the package that depends on it, or use npm 'overrides'")
        elif a.direct:
            flags.append("direct dependency")
        if a.unfixed:
            flags.append(f"{len(a.unfixed)} issue(s) have no known fix")
        print(f"\n{n}. [{a.tier}] {a.name} {a.installed}  ->  {action}   (score {a.score})")
        print(f"   fixes {len(a.resolves)} issue(s): {', '.join(a.resolves[:4])}"
              + (" ..." if len(a.resolves) > 4 else ""))
        if flags:
            print("   note: " + "; ".join(flags))
        for r in a.reasons:
            print("   - " + r)

    cmp = compare_with_baseline(issues, 10)
    print("\n=== COMPARED WITH A SEVERITY-ONLY (CVSS) RANKING ===")
    if cmp["n"]:
        print(f"A severity-only ranking's top {cmp['n']} would come from {cmp['tie_size']} issues tied at "
              f"CVSS >= {cmp['cutoff']} (it cannot tell them apart).")
        print(f"Our top {cmp['n']}: {cmp['overlap']} are in that group, "
              f"{len(cmp['promoted'])} are LOWER severity but ranked higher by exploit data and context.")
        for i in cmp["promoted"][:5]:
            print(f"  promoted: {i.name} {i.vuln_id} (severity {i.cvss_value}, score {i.score})")
        if cmp["pushed_down"]:
            print(f"Pushed down: {len(cmp['pushed_down'])} of the tied high-severity issues are ranked "
                  f"outside our top {cmp['n']}. Lowest scored:")
            for i in cmp["pushed_down"][:5]:
                why = [r.split(":")[0] for r in i.reasons if "Development" in r or "EPSS" in r]
                print(f"  {i.name} {i.vuln_id} (severity {i.cvss_value}, our score {i.score}) <- {'; '.join(why)}")
    if explain:
        print_explanations(plan, issues, res, explain)
    if not show_all and len(plan) > limit:
        print(f"\n(run again with --all to list all {len(plan)})")
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        print("usage: python3 -m securedeps.scan https://github.com/owner/repo [--all] [--explain] [--top=N]")
        sys.exit(2)
    top = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--top=")), 3)
    sys.exit(main(args[0], "--all" in sys.argv, top if "--explain" in sys.argv else 0))
