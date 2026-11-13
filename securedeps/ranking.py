"""Explainable risk ranking: turn a long list of findings into 'fix these first'.

The score is plain arithmetic on three public signals, so every number can be explained:

    KEV     listed by CISA as exploited in real attacks ........ +50
    EPSS    chance of exploitation in 30 days (counts fully at 50%+) up to +30
    CVSS    technical severity (0..10) ......................... up to +20
    then x0.5 if the package is a development-only dependency.

Why these weights? KEV is the only signal that says "attackers are doing this today", so it
outweighs everything else and always puts a finding in the top tier. EPSS predicts what
attackers will do next, CVSS says how bad it would be. The weights are a documented judgement,
not a law of nature: change WEIGHTS and re-run to see how the ordering moves.

Missing data is never treated as 'safe'. If KEV/EPSS can't be checked (no CVE id, or the
service was down) that part adds 0 and the reason says so.
"""
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .intel import version_key
from .models import Finding

WEIGHTS = {"kev": 50.0, "epss": 30.0, "cvss": 20.0}
EPSS_SATURATION = 0.5       # an exploitation probability of 50% or more earns the full EPSS points
DEV_FACTOR = 0.5
LABEL_TO_CVSS = {"CRITICAL": 9.5, "HIGH": 7.5, "MEDIUM": 5.0, "LOW": 2.0}   # when no v3 score
FIX_SOON, PLAN = 35.0, 20.0                                                 # tier thresholds
TIER_ORDER = {"Fix now": 0, "Fix soon": 1, "Plan": 2, "Low": 3}


@dataclass
class Scored:
    score: float
    tier: str
    reasons: List[str] = field(default_factory=list)


def severity_value(f: Finding) -> Optional[float]:
    v = f.vuln
    return v.cvss if v.cvss is not None else LABEL_TO_CVSS.get(v.severity or "")


def tier_for(score: float, in_kev) -> str:
    if in_kev:
        return "Fix now"
    return "Fix soon" if score >= FIX_SOON else "Plan" if score >= PLAN else "Low"


def score_finding(f: Finding) -> Scored:
    pts, why = 0.0, []
    if f.in_kev:
        pts += WEIGHTS["kev"]
        why.append(f"Listed in CISA KEV (exploited in real attacks): +{WEIGHTS['kev']:.0f}")
    elif f.in_kev is None:
        why.append("Exploitation status unknown (no CVE id, or KEV unavailable): +0")
    if f.epss is not None:
        p = WEIGHTS["epss"] * min(f.epss / EPSS_SATURATION, 1.0)
        pts += p
        pct = f", higher than {f.epss_percentile:.0%} of CVEs" if f.epss_percentile is not None else ""
        why.append(f"EPSS {f.epss:.3f} ({f.epss:.1%} chance of exploitation in 30 days{pct}): +{p:.1f}")
    else:
        why.append("Exploit probability unknown (no EPSS data): +0")
    sev = severity_value(f)
    if sev is not None:
        p = WEIGHTS["cvss"] * sev / 10
        pts += p
        src = f"CVSS {f.vuln.cvss}" if f.vuln.cvss is not None else f"{f.vuln.severity} (no CVSS v3 score)"
        why.append(f"{src}: +{p:.1f}")
    else:
        why.append("No severity data: +0")
    if f.dependency.dev:
        pts *= DEV_FACTOR
        why.append(f"Development-only dependency: score x{DEV_FACTOR}")
    pts = round(pts, 1)
    return Scored(pts, tier_for(pts, f.in_kev), why)


# ------------------------------------------------------------------ issues
@dataclass
class Issue:
    """One vulnerability in one package, merged across every installed version of it."""
    ecosystem: str
    name: str
    vuln_id: str
    cves: tuple
    summary: str
    versions: Tuple[str, ...]
    fixed: Dict[str, Optional[str]]       # installed version -> fixed version (or None)
    score: float
    tier: str
    reasons: List[str]
    in_kev: Optional[bool]
    cvss_value: Optional[float]
    dev_only: bool


def build_issues(findings: List[Finding]) -> List[Issue]:
    groups: Dict[tuple, List[Finding]] = {}
    for f in findings:
        groups.setdefault((f.dependency.ecosystem, f.dependency.name, f.vuln.id), []).append(f)
    issues = []
    for (eco, name, vid), fs in groups.items():
        scored = [(score_finding(f), f) for f in fs]
        best, bf = max(scored, key=lambda t: t[0].score)
        issues.append(Issue(eco, name, vid, bf.vuln.cves, bf.vuln.summary,
                            tuple(sorted({f.dependency.version for f in fs})),
                            {f.dependency.version: f.vuln.fixed_version for f in fs},
                            best.score, best.tier, best.reasons, bf.in_kev, severity_value(bf),
                            all(f.dependency.dev for f in fs)))
    issues.sort(key=lambda i: (TIER_ORDER[i.tier], -i.score, i.name, i.vuln_id))
    return issues


# ---------------------------------------------------------------- fix plan
@dataclass
class FixAction:
    ecosystem: str
    name: str
    installed: str
    target: Optional[str]                 # version that fixes everything fixable, or None
    resolves: List[str]                   # vulnerability ids fixed by upgrading to target
    unfixed: List[str]                    # vulnerability ids with no known fix
    tier: str
    score: float
    major_bump: bool
    dev_only: bool
    reasons: List[str]                    # why the top vulnerability scored what it did
    direct: Optional[bool] = None         # True = you depend on it directly, False = pulled in by
                                          # another package, None = unknown


def _major(v: str) -> Optional[int]:
    m = re.match(r"v?(\d+)", v or "")
    return int(m.group(1)) if m else None


def breaking_capable_bump(installed: str, target: Optional[str]) -> bool:
    """Semver: a new MAJOR version may break things. For 0.x versions the same is true of a new
    MINOR version (0.2.x -> 0.5.x), because 0.x makes no stability promise."""
    if not target:
        return False
    a, b = (re.findall(r"\d+", v)[:2] for v in (installed, target))
    if not a or not b:
        return False
    if a[0] != b[0]:
        return True
    return a[0] == "0" and len(a) > 1 and len(b) > 1 and a[1] != b[1]


def fix_plan(findings: List[Finding]) -> List[FixAction]:
    """One action per installed package version: 'upgrade X from A to B'. B is the highest
    fixed version needed, so one upgrade closes every vulnerability that has a fix."""
    by_pkg: Dict[tuple, List[Finding]] = {}
    for f in findings:
        d = f.dependency
        by_pkg.setdefault((d.ecosystem, d.name, d.version), []).append(f)
    actions = []
    for (eco, name, ver), fs in by_pkg.items():
        scored = [(score_finding(f), f) for f in fs]
        best = max(scored, key=lambda t: t[0].score)[0]
        fixed = [f.vuln.fixed_version for f in fs if f.vuln.fixed_version]
        target = max(fixed, key=lambda v: version_key(eco, v)) if fixed else None
        any_kev = any(f.in_kev for f in fs)
        score = best.score
        actions.append(FixAction(
            eco, name, ver, target,
            sorted({f.vuln.id for f in fs if f.vuln.fixed_version}),
            sorted({f.vuln.id for f in fs if not f.vuln.fixed_version}),
            "Fix now" if any_kev else tier_for(score, False), score,
            breaking_capable_bump(ver, target),
            all(f.dependency.dev for f in fs), best.reasons,
            True if any(f.dependency.direct for f in fs) else
            False if all(f.dependency.direct is False for f in fs) else None))
    actions.sort(key=lambda a: (TIER_ORDER[a.tier], -a.score, a.name, a.installed))
    return actions


# ----------------------------------------------------- compare with CVSS-only
def cvss_only_order(issues: List[Issue]) -> List[Issue]:
    """The 'normal scanner' ordering: highest severity first, nothing else."""
    return sorted(issues, key=lambda i: (-(i.cvss_value if i.cvss_value is not None else -1),
                                         i.name, i.vuln_id))


def compare_with_baseline(issues: List[Issue], n: int = 10) -> dict:
    """Compare our top-n with what a severity-only (CVSS) ranking can tell us.

    Many issues share the same CVSS score (dozens of 9.8s), and a severity-only ranking cannot
    order them, so 'its top 10' is really 'any 10 of the tied group'. Comparing against an
    arbitrary 10 would be misleading. Instead we use the cutoff severity of its n-th item:
      tie_group   every issue a severity-only ranking might put in its top n
      promoted    issues in OUR top n that score below that cutoff (severity-only buries them)
      pushed_down tie-group issues that are NOT in our top n (we rank lower than severity ties)
    """
    sev = lambda i: i.cvss_value if i.cvss_value is not None else -1.0
    ours = issues[:n]
    order = cvss_only_order(issues)
    if not order:
        return {"n": 0, "cutoff": None, "tie_size": 0, "overlap": 0, "promoted": [], "pushed_down": []}
    cutoff = sev(order[min(n, len(order)) - 1])
    tie_group = [i for i in issues if sev(i) >= cutoff]
    ours_keys = {(i.ecosystem, i.name, i.vuln_id) for i in ours}
    return {"n": len(ours), "cutoff": cutoff, "tie_size": len(tie_group),
            "overlap": sum(1 for i in ours if sev(i) >= cutoff),
            "promoted": [i for i in ours if sev(i) < cutoff],
            "pushed_down": sorted((i for i in tie_group if (i.ecosystem, i.name, i.vuln_id) not in ours_keys),
                                  key=lambda i: i.score)}
