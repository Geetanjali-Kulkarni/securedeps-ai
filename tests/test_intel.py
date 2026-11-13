"""Tests the OSV / KEV / EPSS logic with a fake internet. All records below are MADE-UP
fixtures in OSV's response shape; they test our logic, not real vulnerability facts."""
import pytest
import requests

from securedeps import intel
from securedeps.intel import (IntelError, analyze, fixed_version_for, osv_query, parse_vuln,
                              version_key)
from securedeps.models import HIGH, NPM, PYPI, Dependency


def dep(name, version, eco=PYPI):
    return Dependency(name, eco, version, HIGH if version else "low", "lock")


class Resp:
    def __init__(self, status=200, body=None):
        self.status_code, self._body = status, body

    def json(self):
        return self._body


class FakeNet:
    """Pretends to be OSV + CISA + FIRST. `routes` can be overridden per test."""
    def __init__(self, osv_hits, records, kev=(), epss=None, kev_status=200, epss_status=200):
        self.osv_hits, self.records, self.kev = osv_hits, records, list(kev)
        self.epss, self.kev_status, self.epss_status = epss or {}, kev_status, epss_status
        self.calls = []

    def post(self, url, json=None, timeout=None):
        self.calls.append(("post", url))
        assert url == intel.OSV_BATCH
        results = []
        for q in json["queries"]:
            key = (q["package"]["ecosystem"], q["package"]["name"], q["version"])
            ids = self.osv_hits.get(key, [])
            results.append({"vulns": [{"id": i} for i in ids]} if ids else {})
        return Resp(body={"results": results})

    def get(self, url, params=None, timeout=None):
        self.calls.append(("get", url))
        if url.startswith("https://api.osv.dev/v1/vulns/"):
            return Resp(body=self.records[url.rsplit("/", 1)[1]])
        if url == intel.KEV_URL:
            return Resp(self.kev_status, {"vulnerabilities": [{"cveID": c} for c in self.kev]})
        if url == intel.EPSS_URL:
            if self.epss_status != 200:
                return Resp(self.epss_status)
            wanted = params["cve"].split(",")
            return Resp(body={"data": [{"cve": c, "epss": str(self.epss[c][0]),
                                        "percentile": str(self.epss[c][1])}
                                       for c in wanted if c in self.epss]})
        raise AssertionError(url)

    def count(self, kind, url_part):
        return sum(1 for k, u in self.calls if k == kind and url_part in u)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(intel.time, "sleep", lambda s: None)


def record(vid, aliases=(), fixed=("2.20.0",), name="requests", eco="PyPI", **extra):
    r = {"id": vid, "aliases": list(aliases), "summary": f"Problem in {name}",
         "severity": [{"type": "CVSS_V3", "score": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N"}],
         "affected": [{"package": {"name": name, "ecosystem": eco},
                       "ranges": [{"type": "ECOSYSTEM",
                                   "events": [{"introduced": "0"}] + [{"fixed": f} for f in fixed]}]}]}
    r.update(extra)
    return r


# ----------------------------------------------------------- version logic
def test_version_ordering():
    k = lambda e, v: version_key(e, v)
    assert k(PYPI, "2.20.0") > k(PYPI, "2.19.0") > k(PYPI, "2.9.0")        # numeric, not alphabetical
    assert k(PYPI, "1.0") > k(PYPI, "1.0rc1")
    assert k(NPM, "4.17.21") > k(NPM, "4.17.15")
    assert k(NPM, "1.0.0") > k(NPM, "1.0.0-beta.1")


def test_fixed_version_picks_nearest_newer_fix():
    rec = record("X", fixed=("2.20.0", "3.1.0", "1.5.0"))
    assert fixed_version_for(rec, dep("requests", "2.19.0")) == "2.20.0"   # same branch first
    assert fixed_version_for(rec, dep("requests", "3.0.0")) == "3.1.0"
    assert fixed_version_for(rec, dep("requests", "9.0.0")) is None        # nothing newer exists


def test_fixed_version_ignores_other_packages_and_normalizes_pypi_names():
    rec = record("X", name="Django_REST.framework")
    assert fixed_version_for(rec, dep("django-rest-framework", "1.0")) == "2.20.0"
    assert fixed_version_for(rec, dep("other", "1.0")) is None


# ------------------------------------------------------------- parse_vuln
def test_parse_vuln_collects_cve_from_aliases_and_computes_score():
    v = parse_vuln(record("GHSA-1", aliases=["CVE-2020-0001", "PYSEC-1"]), dep("requests", "2.19.0"))
    assert v.cves == ("CVE-2020-0001",) and v.cvss == 7.5 and v.severity == "HIGH"
    assert v.fixed_version == "2.20.0" and "requests" in v.summary


def test_parse_vuln_cve_as_primary_id_and_no_cve_case():
    assert parse_vuln(record("CVE-2021-1"), dep("requests", "1")).cves == ("CVE-2021-1",)
    assert parse_vuln(record("GHSA-2"), dep("requests", "1")).cves == ()


def test_parse_vuln_label_fallbacks_and_withdrawn():
    r = record("GHSA-3", severity=[], database_specific={"severity": "MODERATE"})
    v = parse_vuln(r, dep("requests", "1"))
    assert v.severity == "MEDIUM" and v.cvss is None
    assert parse_vuln(record("GHSA-4", withdrawn="2021-01-01"), dep("requests", "1")) is None


# -------------------------------------------------------------- OSV batching
def test_osv_query_skips_unscannable_and_dedupes():
    net = FakeNet({(PYPI, "requests", "2.19.0"): ["GHSA-1"]}, {})
    out = osv_query([dep("requests", "2.19.0"), dep("requests", "2.19.0"), dep("flask", None)], net)
    assert out == {(PYPI, "requests", "2.19.0"): ["GHSA-1"]}
    assert net.count("post", "querybatch") == 1


def test_osv_pagination():
    class Paged(FakeNet):
        def post(self, url, json=None, timeout=None):
            q = json["queries"][0]
            if "page_token" in q:
                return Resp(body={"results": [{"vulns": [{"id": "B"}]}]})
            return Resp(body={"results": [{"vulns": [{"id": "A"}], "next_page_token": "t"}]})
    out = osv_query([dep("x", "1")], Paged({}, {}))
    assert out[(PYPI, "x", "1")] == ["A", "B"]


# ---------------------------------------------------------- full pipeline
def make_net(**kw):
    records = {"GHSA-1": record("GHSA-1", aliases=["CVE-2018-0001"]),
               "GHSA-2": record("GHSA-2", name="lodash", eco="npm", fixed=("4.17.21",)),  # no CVE
               "GHSA-3": record("GHSA-3", aliases=["CVE-2019-0002"], name="requests")}
    hits = {(PYPI, "requests", "2.19.0"): ["GHSA-1", "GHSA-3"],
            (NPM, "lodash", "4.17.15"): ["GHSA-2"],
            (NPM, "lodash", "4.17.0"): ["GHSA-2"]}                      # shares one record
    args = dict(osv_hits=hits, records=records, kev=["CVE-2018-0001"],
                epss={"CVE-2018-0001": (0.74, 0.98), "CVE-2019-0002": (0.01, 0.30)})
    args.update(kw)
    return FakeNet(**args)


DEPS = [dep("requests", "2.19.0"), dep("lodash", "4.17.15", NPM), dep("lodash", "4.17.0", NPM),
        dep("flask", "1.0"), dep("numpy", None)]


def test_analyze_end_to_end():
    net = make_net()
    res = analyze(DEPS, net)
    assert (res.scanned, res.unscannable, res.warnings) == (4, 1, [])
    assert len(res.findings) == 4
    by_id = {(f.dependency.name, f.vuln.id): f for f in res.findings if f.dependency.version != "4.17.0"}
    f1 = by_id[("requests", "GHSA-1")]
    assert f1.in_kev is True and f1.epss == 0.74 and f1.epss_percentile == 0.98 and not f1.no_cve
    f3 = by_id[("requests", "GHSA-3")]
    assert f3.in_kev is False and f3.epss == 0.01
    f2 = by_id[("lodash", "GHSA-2")]
    assert f2.no_cve and f2.in_kev is None and f2.epss is None         # unknown, not 'no'
    assert net.count("get", "/vulns/GHSA-2") == 1                      # shared record fetched once


def test_clean_project_makes_no_kev_or_epss_calls():
    net = make_net()
    res = analyze([dep("flask", "1.0")], net)
    assert res.findings == [] and net.count("get", "cisa.gov") == 0 and net.count("get", "first.org") == 0


def test_kev_outage_degrades_gracefully():
    res = analyze(DEPS, make_net(kev_status=404))
    assert any("KEV" in w for w in res.warnings)
    assert len(res.findings) == 4 and all(f.in_kev is None for f in res.findings)
    assert [f.epss for f in res.findings if f.vuln.id == "GHSA-1"] == [0.74]   # EPSS still works


def test_epss_outage_degrades_gracefully():
    res = analyze(DEPS, make_net(epss_status=404))
    assert any("EPSS" in w for w in res.warnings)
    assert all(f.epss is None for f in res.findings)
    assert [f.in_kev for f in res.findings if f.vuln.id == "GHSA-1"] == [True]


# --------------------------------------------------------------- retries
def test_retry_then_success_and_final_failure():
    class Flaky:
        def __init__(self, statuses): self.statuses, self.n = statuses, 0
        def get(self, url, timeout=None, **kw):
            s = self.statuses[min(self.n, len(self.statuses) - 1)]; self.n += 1
            return Resp(s, {"ok": True})
    assert intel._request(Flaky([500, 429, 200]), "get", "u").json() == {"ok": True}
    with pytest.raises(IntelError):
        intel._request(Flaky([500]), "get", "u")
    with pytest.raises(IntelError):                       # 404 is not retried
        f = Flaky([404]); intel._request(f, "get", "u")


def test_connection_error_is_wrapped():
    class Down:
        def get(self, url, timeout=None, **kw): raise requests.ConnectionError("no internet")
    with pytest.raises(IntelError, match="ConnectionError"):
        intel._request(Down(), "get", "u")
