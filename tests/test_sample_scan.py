import csv
from datetime import date

import pytest

from securedeps import sample_scan as ss
from securedeps.intel import Analysis
from securedeps.models import HIGH, NPM, Dependency, Finding, Vulnerability
from securedeps.repo import RepoError

DEP = Dependency("lodash", NPM, "4.17.11", HIGH, "package-lock.json", direct=True)
CRIT = Finding(DEP, Vulnerability("GHSA-1", ("CVE-1",), "x", "CRITICAL", 9.8, "4.17.21"), False, 0.001, 0.1)
KEV = Finding(Dependency("pkg", NPM, "1.0.0", HIGH, "package-lock.json"),
              Vulnerability("GHSA-2", ("CVE-2",), "y", "HIGH", 7.5, "1.0.1"), True, 0.9, 0.99)


class Resp:
    def __init__(self, items, ok=True):
        self.items, self.ok, self.status_code = items, ok, 200 if ok else 403

    def json(self):
        return {"items": self.items}


class FakeSession:
    def __init__(self, per_language=3):
        self.queries, self.n = [], per_language

    def get(self, url, headers=None, params=None, timeout=None):
        self.queries.append(params["q"])
        lang = params["q"].split()[0].split(":")[1]
        return Resp([{"full_name": f"{lang.lower()}/repo{i}"} for i in range(self.n)])


def test_search_query_documents_the_selection_rules():
    s = FakeSession()
    ss.search_repos("Python", "tok", s, today=date(2026, 11, 5))
    q = s.queries[0]
    assert "language:Python" in q and "stars:300..3000" in q and "archived:false" in q and "fork:false" in q
    assert "pushed:>=2026-05-09" in q                      # 180 days before the scan date


def test_collect_is_reproducible_and_keeps_n_per_language(monkeypatch):
    monkeypatch.setattr(ss, "POOL_PER_LANGUAGE", 100)
    a = ss.collect("t", FakeSession(per_language=50), keep=5)
    b = ss.collect("t", FakeSession(per_language=50), keep=5)
    assert a == b and len(a) == 15 and {lang for _, lang in a} == set(ss.LANGUAGES)


def test_search_failure_gives_readable_error():
    class Bad:
        def get(self, *a, **k):
            return Resp([], ok=False)
    with pytest.raises(RepoError, match="GITHUB_TOKEN"):
        ss.search_repos("Python", None, Bad())


def fake_pipeline(monkeypatch, findings, scanned=2, deps=2):
    monkeypatch.setattr(ss, "scan_repo_dependencies", lambda url, token=None: {"dependencies": [DEP] * deps})
    monkeypatch.setattr(ss, "analyze", lambda d: Analysis(findings=findings, scanned=scanned))


def test_scan_one_counts_tiers_kev_and_deprioritised_criticals(monkeypatch):
    fake_pipeline(monkeypatch, [CRIT, KEV])
    row = ss.scan_one("o/r", "JavaScript", "t")
    assert row["status"] == "ok" and row["issues"] == 2 and row["kev_issues"] == 1 and row["fix_now"] == 1
    assert row["critical_issues"] == 1 and row["critical_deprioritised"] == 1      # CVSS 9.8 but tiny EPSS, not KEV
    assert set(row) == set(ss.FIELDS)


def test_repo_without_exact_versions_and_errors_are_recorded_not_raised(monkeypatch):
    fake_pipeline(monkeypatch, [], scanned=0)
    assert ss.scan_one("o/r", "Python", "t")["status"] == "no_exact_versions"
    monkeypatch.setattr(ss, "scan_repo_dependencies", lambda url, token=None: (_ for _ in ()).throw(RepoError("rate limit")))
    row = ss.scan_one("o/r", "Python", "t")
    assert row["status"] == "error" and "rate limit" in row["error"]


def test_scan_all_is_resumable(monkeypatch, tmp_path):
    out = str(tmp_path / "r.csv")
    calls = []

    def fake(name, lang, token):
        calls.append(name)
        row = {f: "" for f in ss.FIELDS}
        row.update(repo=name, language=lang, status="ok", issues=1, kev_issues=0)
        return row
    repos = [("a/a", "Python"), ("b/b", "Python")]
    assert ss.scan_all(repos, out, "t", pause=0, scanner=fake) == 2
    assert ss.scan_all(repos + [("c/c", "Python")], out, "t", pause=0, scanner=fake) == 1     # only the new one
    assert calls == ["a/a", "b/b", "c/c"]
    assert len(list(csv.DictReader(open(out, encoding="utf-8")))) == 3


def test_report_numbers(tmp_path):
    def row(**kw):
        r = {f: "" for f in ss.FIELDS}
        r.update(status="ok", language="Python", scan_date="2026-11-05", issues=0, kev_issues=0, fix_now=0,
                 fix_soon=0, critical_issues=0, critical_deprioritised=0)
        r.update(kw)
        return r
    rows = [row(repo="a", issues=4, kev_issues=1, fix_now=1, critical_issues=2, critical_deprioritised=1),
            row(repo="b", issues=2, fix_soon=1, critical_issues=2, critical_deprioritised=2),
            row(repo="c"), row(repo="d", status="no_exact_versions"), row(repo="e", status="error", error="x")]
    p = str(tmp_path / "r.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ss.FIELDS)
        w.writeheader()
        w.writerows(rows)
    text = ss.report(p)
    assert "attempted: 5" in text and "usable (exact versions known): 3" in text and "errors: 1" in text
    assert "1/3 (33%)" in text                              # one of three has a KEV-listed vulnerability
    assert "2/3 (67%)" in text                              # a and b have at least one vulnerability / fix soon-or-now
    assert "3 (75%)" in text                                # 3 of 4 critical issues deprioritised
    assert "Limits:" in text


def test_cli_requires_token_and_usage(monkeypatch, capsys):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert ss.main(["collect"]) == 1 and "GITHUB_TOKEN" in capsys.readouterr().out
    assert ss.main([]) == 2 and "usage" in capsys.readouterr().out
