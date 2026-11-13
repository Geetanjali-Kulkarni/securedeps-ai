from securedeps.models import HIGH, NPM, PYPI, Dependency, Finding, Vulnerability
from securedeps.ranking import (build_issues, compare_with_baseline, fix_plan, score_finding,
                                tier_for)


def finding(name="pkg", version="1.0.0", vid="V1", cves=("CVE-1",), cvss=7.5, sev="HIGH",
            fixed="1.0.1", kev=False, epss=0.1, pct=0.5, dev=False, eco=NPM):
    d = Dependency(name, eco, version, HIGH, "lock", dev=dev)
    v = Vulnerability(vid, tuple(cves), f"summary {vid}", sev, cvss, fixed)
    return Finding(d, v, kev, epss, pct)


def test_score_arithmetic_is_exactly_the_documented_formula():
    s = score_finding(finding(kev=True, epss=0.74, pct=0.98, cvss=7.5))
    assert s.score == 95.0                       # 50 + 30 (EPSS 0.74 is capped at full points) + 15
    assert s.tier == "Fix now"
    assert any("KEV" in r for r in s.reasons) and len(s.reasons) == 3


def test_non_kev_score_and_tiers():
    assert score_finding(finding(epss=0.5, pct=1.0, cvss=10.0)).score == 50.0    # max without KEV
    assert tier_for(35.0, False) == "Fix soon" and tier_for(34.9, False) == "Plan"
    assert tier_for(20.0, False) == "Plan" and tier_for(19.9, False) == "Low"
    assert tier_for(0.0, True) == "Fix now"       # KEV always wins, whatever the score


def test_dev_dependency_halves_score():
    prod, dev = score_finding(finding()), score_finding(finding(dev=True))
    assert dev.score == round(prod.score * 0.5, 1) and any("Development" in r for r in dev.reasons)


def test_missing_data_adds_nothing_and_says_so():
    s = score_finding(finding(kev=None, epss=None, pct=None, cvss=None, sev=None, cves=()))
    assert s.score == 0.0 and s.tier == "Low"
    assert sum("unknown" in r.lower() or "no severity" in r.lower() or "no epss" in r.lower()
               for r in s.reasons) == 3


def test_label_used_when_no_cvss_score():
    s = score_finding(finding(cvss=None, sev="CRITICAL", epss=None, pct=None))
    assert s.score == 19.0 and any("no CVSS v3" in r for r in s.reasons)    # 20 * 9.5/10


def test_epss_uses_probability_and_saturates_at_half():
    pts = lambda e: score_finding(finding(epss=e, pct=0.99, cvss=None, sev=None)).score
    assert pts(0.07) == 4.2 and pts(0.25) == 15.0 and pts(0.5) == 30.0 and pts(0.9) == 30.0   # percentile ignored


# ------------------------------------------------------------------ issues
def test_issues_merge_versions_of_same_package_and_vuln():
    fs = [finding("lodash", "4.17.20", fixed="4.17.21"), finding("lodash", "4.17.11", fixed="4.17.21"),
          finding("lodash", "4.17.11", vid="V2", fixed="4.17.19")]
    issues = build_issues(fs)
    assert len(issues) == 2
    v1 = [i for i in issues if i.vuln_id == "V1"][0]
    assert v1.versions == ("4.17.11", "4.17.20") and v1.fixed["4.17.20"] == "4.17.21"


def test_issues_sorted_kev_first_then_score():
    fs = [finding("a", epss=0.9, pct=0.99, cvss=10.0), finding("b", kev=True, epss=0.01, pct=0.1, cvss=2.0),
          finding("c", epss=0.01, pct=0.1, cvss=2.0)]
    assert [i.name for i in build_issues(fs)] == ["b", "a", "c"]


# ---------------------------------------------------------------- fix plan
def test_fix_plan_one_upgrade_covers_all_fixable_vulns():
    fs = [finding("lodash", "4.17.11", vid="A", fixed="4.17.19"),
          finding("lodash", "4.17.11", vid="B", fixed="4.17.21"),
          finding("lodash", "4.17.11", vid="C", fixed=None)]
    [a] = fix_plan(fs)
    assert a.target == "4.17.21" and a.resolves == ["A", "B"] and a.unfixed == ["C"]
    assert not a.major_bump


def test_fix_plan_uses_numeric_version_order_and_flags_major_jump():
    fs = [finding("tar", "2.2.1", vid="A", fixed="3.2.2"), finding("tar", "2.2.1", vid="B", fixed="4.4.14"),
          finding("tar", "2.2.1", vid="C", fixed="4.4.9")]
    [a] = fix_plan(fs)
    assert a.target == "4.4.14" and a.major_bump                     # 4.4.14 > 4.4.9 (numeric)


def test_fix_plan_no_fix_available():
    [a] = fix_plan([finding("old", "1.0.0", fixed=None)])
    assert a.target is None and a.unfixed == ["V1"] and not a.major_bump


def test_fix_plan_ordering_and_separate_versions():
    fs = [finding("a", "1.0.0", cvss=3.0, epss=0.01, pct=0.1),
          finding("b", "2.0.0", kev=True, cvss=3.0, epss=0.01, pct=0.1),
          finding("a", "0.9.0", cvss=9.0, epss=0.3, pct=0.9)]
    plan = fix_plan(fs)
    assert [(x.name, x.installed) for x in plan] == [("b", "2.0.0"), ("a", "0.9.0"), ("a", "1.0.0")]
    assert plan[0].tier == "Fix now"


def test_pypi_versions_compare_with_pep440():
    [a] = fix_plan([finding("requests", "2.9.0", vid="A", fixed="2.20.0", eco=PYPI),
                    finding("requests", "2.9.0", vid="B", fixed="2.9.1", eco=PYPI)])
    assert a.target == "2.20.0"


# ---------------------------------------------------- baseline comparison
def test_severity_only_baseline_buries_exploited_issue():
    fs = [finding("big-but-quiet", cvss=9.8, epss=0.001, pct=0.05),          # severe, rarely attacked
          finding("small-but-exploited", cvss=5.0, kev=True, epss=0.9, pct=0.99),
          finding("mid", cvss=7.0, epss=0.2, pct=0.7)]
    issues = build_issues(fs)
    cmp = compare_with_baseline(issues, n=1)
    assert issues[0].name == "small-but-exploited"
    assert cmp["cutoff"] == 9.8 and cmp["tie_size"] == 1 and cmp["overlap"] == 0
    assert [i.name for i in cmp["promoted"]] == ["small-but-exploited"]
    assert [i.name for i in cmp["pushed_down"]] == ["big-but-quiet"]


def test_baseline_ties_are_not_compared_arbitrarily():
    fs = [finding("aaa", vid="A", cvss=9.8, epss=0.001, pct=0.05, dev=True),   # tied 9.8s: a severity-only
          finding("zzz", vid="Z", cvss=9.8, epss=0.4, pct=0.95),               # ranking cannot order them
          finding("low", vid="L", cvss=2.0, epss=0.0, pct=0.0)]
    cmp = compare_with_baseline(build_issues(fs), n=1)
    assert cmp["tie_size"] == 2 and cmp["overlap"] == 1 and cmp["promoted"] == []
    assert [i.name for i in cmp["pushed_down"]] == ["aaa"]                      # the dev-only one


def test_direct_vs_transitive_flag_on_fix_plan():
    def mk(direct):
        f = finding("p")
        return Finding(Dependency("p", NPM, "1.0.0", HIGH, "lock", direct=direct), f.vuln, f.in_kev, f.epss, f.epss_percentile)
    assert fix_plan([mk(True)])[0].direct is True
    assert fix_plan([mk(False)])[0].direct is False
    assert fix_plan([mk(None)])[0].direct is None


def test_empty_input():
    assert build_issues([]) == [] and fix_plan([]) == []
    assert compare_with_baseline([])["n"] == 0 and compare_with_baseline([])["pushed_down"] == []
