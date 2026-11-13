from securedeps import pipeline
from securedeps.evidence import Doc
from securedeps.intel import Analysis
from securedeps.models import HIGH, NPM, Dependency, Finding, Vulnerability
from securedeps.rag import HashingEmbedder

DEP = Dependency("lodash", NPM, "4.17.11", HIGH, "package-lock.json", direct=False)
DEV = Dependency("handlebars", NPM, "4.0.0", HIGH, "package-lock.json", direct=True, dev=True)
F1 = Finding(DEP, Vulnerability("GHSA-1", ("CVE-2020-1",), "Prototype pollution", "HIGH", 7.2, "5.0.0"), False, 0.2, 0.9)
F2 = Finding(DEV, Vulnerability("GHSA-2", (), "RCE", "CRITICAL", 9.8, "4.0.1"), None, None)
DOCS = [Doc("osv:GHSA-1", "advisory", "Advisory GHSA-1", "https://osv.dev/vulnerability/GHSA-1",
            "GHSA-1\n\nPrototype pollution in lodash. Fixed in 5.0.0."),
        Doc("release:v5.0.0", "release", "lodash v5.0.0", "https://gh/r", "Release v5.0.0:\n- BREAKING CHANGE: removed pluck")]


def fake_scan(monkeypatch, gathered=None):
    monkeypatch.setattr(pipeline, "scan_repo_dependencies",
                        lambda url, token=None: {"owner": "o", "repo": "r", "files": ["package-lock.json"],
                                                 "dependencies": [DEP, DEV], "warnings": ["w1"]})
    monkeypatch.setattr(pipeline, "analyze", lambda deps: Analysis(findings=[F1, F2], records={}, scanned=2,
                                                                  warnings=["w2"]))
    calls = gathered if gathered is not None else []
    monkeypatch.setattr(pipeline, "gather_evidence", lambda *a, **k: (calls.append(a) or (DOCS, ["ev warn"])))
    return calls


def test_run_scan_builds_everything_the_dashboard_needs(monkeypatch):
    fake_scan(monkeypatch)
    r = pipeline.run_scan("https://github.com/o/r")
    assert r.full_name == "o/r" and r.dependency_count == 2 and len(r.plan) == 2
    assert sum(r.tier_counts.values()) == 2 and set(r.tier_counts) == set(pipeline.TIERS)
    assert r.warnings == ["w1", "w2"]
    assert r.baseline(10)["n"] == 2


def test_fix_rows_are_plain_data_for_the_table(monkeypatch):
    fake_scan(monkeypatch)
    rows = pipeline.fix_rows(pipeline.run_scan("u"), limit=1)
    assert len(rows) == 1 and rows[0]["#"] == 1
    assert set(rows[0]) == {"#", "Priority", "Package", "Upgrade", "Score", "Fixes", "Notes"}
    assert "->" in rows[0]["Upgrade"]


def test_action_flags_describe_the_risky_cases(monkeypatch):
    fake_scan(monkeypatch)
    plan = {a.name: a for a in pipeline.run_scan("u").plan}
    assert "risky version jump" in pipeline.action_flags(plan["lodash"]) and "transitive" in pipeline.action_flags(plan["lodash"])
    assert "dev-only" in pipeline.action_flags(plan["handlebars"]) and "direct" in pipeline.action_flags(plan["handlebars"])


def test_evidence_is_fetched_once_per_fix(monkeypatch):
    calls = fake_scan(monkeypatch)
    r = pipeline.run_scan("u")
    a = r.plan[0]
    first = pipeline.evidence_for(r, a)
    second = pipeline.evidence_for(r, a)
    assert first == second == (DOCS, ["ev warn"]) and len(calls) == 1


def test_explain_for_without_llm_gives_deterministic_text_and_all_warnings(monkeypatch):
    fake_scan(monkeypatch)
    r = pipeline.run_scan("u")
    lodash = next(a for a in r.plan if a.name == "lodash")
    e, notes = pipeline.explain_for(r, lodash, None, HashingEmbedder())
    assert not e.used_llm and "5.0.0" in e.fix and "ev warn" in notes
    assert e.upgrade_risk == "high"          # major bump + breaking-change wording


def test_facts_contain_the_scanners_numbers(monkeypatch):
    fake_scan(monkeypatch)
    a = pipeline.run_scan("u").plan[0]
    f = pipeline.facts_for(a)
    assert a.name in f and str(a.score) in f and "risky version jump" in f
