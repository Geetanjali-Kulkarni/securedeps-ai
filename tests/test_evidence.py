import pytest
import requests

from securedeps import evidence
from securedeps.evidence import (advisory_docs, find_repo, gather_evidence, github_slug,
                                 release_docs, tag_version)


class Resp:
    def __init__(self, status=200, body=None):
        self.status_code, self._body, self.ok = status, body, status < 400

    def json(self):
        return self._body


@pytest.mark.parametrize("url,expected", [
    ("git+https://github.com/lodash/lodash.git", ("lodash", "lodash")),
    ("https://github.com/psf/requests", ("psf", "requests")),
    ("git@github.com:expressjs/express.git", ("expressjs", "express")),
    ("https://github.com/a/b/tree/main/pkg", ("a", "b")),
    ("owner/repo", ("owner", "repo")), ("github:owner/repo", ("owner", "repo")),
    ("https://gitlab.com/a/b", None), ("", None), (None, None)])
def test_github_slug(url, expected):
    assert github_slug(url) == expected


@pytest.mark.parametrize("tag,pkg,expected", [
    ("v4.17.21", "lodash", "4.17.21"), ("lodash@4.17.21", "lodash", "4.17.21"),
    ("release-1.2.3", "x", "1.2.3"), ("2.0.0", "x", "2.0.0"), ("v2.0.0-beta.1", "x", "2.0.0-beta.1"),
    ("vue-2.0.0", "vue", "2.0.0"),
    ("@babel/core@7.0.0", "babel-traverse", None), ("nightly", "x", None), ("docs-update", "x", None)])
def test_tag_version(tag, pkg, expected):
    assert tag_version(tag, pkg) == expected


def test_advisory_docs():
    recs = {"A": {"id": "A", "aliases": ["CVE-2020-1", "OTHER"], "summary": "Bad thing", "details": "Details here"},
            "B": {"id": "B", "summary": "", "details": ""}}
    docs = advisory_docs(recs, ["A", "B", "MISSING"])
    assert len(docs) == 1 and docs[0].doc_id == "osv:A" and docs[0].kind == "advisory"
    assert "CVE-2020-1" in docs[0].text and "Details here" in docs[0].text
    assert docs[0].url == "https://osv.dev/vulnerability/A"
    assert len(advisory_docs({str(i): {"summary": "s"} for i in range(30)}, [str(i) for i in range(30)])) == 10


class Net:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append((url, params))
        for key, val in self.routes.items():
            if key in url:
                if isinstance(val, Exception):
                    raise val
                return val if isinstance(val, Resp) else Resp(body=val)
        return Resp(404)


def test_find_repo_npm_forms_and_scoped_names():
    n = Net({"registry.npmjs.org": {"repository": {"type": "git", "url": "git+https://github.com/lodash/lodash.git"}}})
    assert find_repo("npm", "lodash", n) == ("lodash", "lodash")
    n2 = Net({"registry.npmjs.org": {"repository": "github:owner/repo"}})   # string form
    assert find_repo("npm", "x", n2) == ("owner", "repo")                    # npm "github:" shorthand
    n3 = Net({"registry.npmjs.org": {"repository": "https://github.com/owner/repo"}})
    assert find_repo("npm", "@scope/pkg", n3) == ("owner", "repo")
    assert "@scope%2Fpkg" in n3.calls[0][0]


def test_find_repo_pypi_and_failures():
    n = Net({"pypi.org": {"info": {"project_urls": {"Docs": "https://docs.x", "Source": "https://github.com/psf/requests"}}}})
    assert find_repo("PyPI", "requests", n) == ("psf", "requests")
    assert find_repo("PyPI", "x", Net({"pypi.org": {"info": {"home_page": "https://example.com"}}})) is None
    assert find_repo("npm", "x", Net({})) is None                                   # 404
    assert find_repo("npm", "x", Net({"registry": requests.ConnectionError()})) is None


def rel(tag, body="notes", **kw):
    return {"tag_name": tag, "body": body, "html_url": f"https://gh/{tag}", **kw}


def test_release_docs_selects_versions_between_installed_and_target():
    releases = [rel("v4.17.22"), rel("v4.17.21"), rel("v4.17.20"), rel("v4.17.19"), rel("v4.17.18"),
                rel("v4.17.21-beta", prerelease=True), rel("v4.17.20.1", draft=True),
                rel("nightly"), rel("lodash-docs"), rel("v4.17.20", body="   ")]
    net = Net({"/releases": releases})
    docs, warnings = release_docs("npm", "lodash", "4.17.19", "4.17.21", ("lodash", "lodash"), None, net)
    assert [d.doc_id for d in docs] == ["release:v4.17.20", "release:v4.17.21"]       # oldest first, range only
    assert all(d.kind == "release" and "lodash" in d.text for d in docs) and warnings == []


def test_release_docs_caps_and_warns():
    releases = [rel(f"v1.{i}.0") for i in range(1, 31)]
    docs, warnings = release_docs("npm", "p", "1.0.0", "1.30.0", ("a", "b"), None, Net({"/releases": releases}))
    assert len(docs) == 12 and docs[-1].doc_id == "release:v1.30.0" and any("using 12" in w for w in warnings)


def test_release_docs_failures_never_raise():
    docs, w = release_docs("npm", "p", "1.0.0", "2.0.0", ("a", "b"), None, Net({"/releases": Resp(403)}))
    assert docs == [] and "HTTP 403" in w[0]
    docs, w = release_docs("npm", "p", "1.0.0", "2.0.0", ("a", "b"), None, Net({"/releases": requests.Timeout()}))
    assert docs == [] and "unreachable" in w[0]
    docs, w = release_docs("npm", "p", "1.0.0", "2.0.0", ("a", "b"), None, Net({"/releases": []}))
    assert docs == [] and "no GitHub release notes" in w[0]


def test_gather_evidence_paths():
    recs = {"A": {"summary": "bad", "details": "d"}}
    docs, w = gather_evidence("npm", "p", "1.0.0", None, recs, ["A"], session=Net({}))
    assert len(docs) == 1 and "no fixed version" in w[0]
    docs, w = gather_evidence("npm", "p", "1.0.0", "2.0.0", recs, ["A"], session=Net({}))
    assert len(docs) == 1 and "could not find the GitHub repository" in w[0]
    net = Net({"registry.npmjs.org": {"repository": "https://github.com/o/r"}, "/releases": [rel("v2.0.0")]})
    docs, w = gather_evidence("npm", "p", "1.0.0", "2.0.0", recs, ["A"], session=net)
    assert [d.kind for d in docs] == ["advisory", "release"] and w == []
