"""The whole scan as plain functions, so the dashboard (and tests) can call them.

    run_scan(url)                     repo -> dependencies -> vulnerabilities -> ranked fix plan
    evidence_for(result, action)      advisories + release notes for one fix (cached)
    explain_for(result, action, ...)  grounded, cited explanation for one fix

The Streamlit file contains no logic of its own; everything it shows comes from here.
"""
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

import requests

from .evidence import Doc, gather_evidence
from .explain import Explanation, explain_action
from .intel import Analysis, analyze
from .ranking import FixAction, Issue, build_issues, compare_with_baseline, fix_plan
from .repo import scan_repo_dependencies

TIERS = ["Fix now", "Fix soon", "Plan", "Low"]


@dataclass
class ScanResult:
    owner: str
    repo: str
    files: List[str]
    dependency_count: int
    notes: List[str]                     # warnings from reading the repo
    analysis: Analysis
    issues: List[Issue]
    plan: List[FixAction]
    scan_date: str
    evidence: Dict[Tuple[str, str], Tuple[List[Doc], List[str]]] = field(default_factory=dict)

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.repo}"

    @property
    def tier_counts(self) -> Dict[str, int]:
        counts = {t: 0 for t in TIERS}
        for a in self.plan:
            counts[a.tier] += 1
        return counts

    @property
    def warnings(self) -> List[str]:
        return list(self.notes) + list(self.analysis.warnings)

    def baseline(self, n: int = 10) -> dict:
        return compare_with_baseline(self.issues, n)


def run_scan(url: str, token: Optional[str] = None) -> ScanResult:
    """Raises RepoError (readable message) when the repository cannot be read."""
    repo = scan_repo_dependencies(url, token)
    deps = repo["dependencies"]
    analysis = analyze(deps)
    return ScanResult(repo["owner"], repo["repo"], list(repo["files"]), len(deps), list(repo["warnings"]),
                      analysis, build_issues(analysis.findings), fix_plan(analysis.findings),
                      str(date.today()))


def action_flags(a: FixAction) -> List[str]:
    flags = []
    if a.major_bump:
        flags.append("risky version jump")
    if a.dev_only:
        flags.append("dev-only")
    if a.direct is False:
        flags.append("transitive")
    elif a.direct:
        flags.append("direct")
    if a.unfixed:
        flags.append(f"{len(a.unfixed)} unfixed")
    return flags


def fix_rows(result: ScanResult, limit: int = 25) -> List[dict]:
    """Rows for the table on the dashboard."""
    return [{"#": n, "Priority": a.tier, "Package": a.name,
             "Upgrade": f"{a.installed} -> {a.target or 'no fix known'}",
             "Score": a.score, "Fixes": len(a.resolves), "Notes": ", ".join(action_flags(a))}
            for n, a in enumerate(result.plan[:limit], 1)]


def facts_for(a: FixAction) -> str:
    return (f"Package: {a.name} {a.installed} -> {a.target or 'no fixed version known'}. "
            f"Priority: {a.tier} (score {a.score}). "
            f"Upgrade is a risky version jump (new major version, or a 0.x minor): {'yes' if a.major_bump else 'no'}. "
            f"Score breakdown: " + "; ".join(a.reasons))


def evidence_for(result: ScanResult, a: FixAction, session=None, token: Optional[str] = None
                 ) -> Tuple[List[Doc], List[str]]:
    key = (a.name, a.installed)
    if key not in result.evidence:
        ids = [i.vuln_id for i in result.issues
               if i.name == a.name and a.installed in i.versions and i.vuln_id in a.resolves] or a.resolves
        result.evidence[key] = gather_evidence(a.ecosystem, a.name, a.installed, a.target,
                                               result.analysis.records, ids, token, session or requests.Session())
    return result.evidence[key]


def explain_for(result: ScanResult, a: FixAction, llm, embedder, session=None,
                token: Optional[str] = None) -> Tuple[Explanation, List[str]]:
    docs, ev_warn = evidence_for(result, a, session, token)
    e = explain_action(a.name, a.installed, a.target, a.major_bump, docs, facts_for(a), llm, embedder)
    return e, ev_warn + e.warnings
