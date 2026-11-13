"""Tests for: tighter breaking-change detection, CHANGELOG fallback, 0.x version-jump rule."""
import pytest
import requests

from securedeps import evidence
from securedeps.evidence import changelog_docs, gather_evidence, split_changelog
from securedeps.explain import find_breaking_lines, is_breaking_line
from securedeps.models import HIGH, NPM, Dependency, Finding, Vulnerability
from securedeps.rag import Chunk
from securedeps.ranking import breaking_capable_bump, fix_plan


# ------------------------------------------------------- breaking-change lines
@pytest.mark.parametrize("line", [
    "BREAKING CHANGE: removed the deprecated pluck helper",
    "Breaking changes: the default export is now a named export",
    "Drop support for Node 6",
    "dropped support for python 2",
    "qs no longer supports IE8",
    "Now requires Node 12 or newer",
    "Minimum Node version is now 14",
    "Removed the deprecated `parse` method",
    "renamed the `format` option to `style`",
    "This release is backwards-incompatible with 1.x"])
def test_real_breaking_wording_is_detected(line):
    assert is_breaking_line(line)


@pytest.mark.parametrize("line", [
    "remove dead link to benchmarks (#313)",                  # the real false positive from NodeGoat/tar
    "removed unused tests",
    "Fix typo in README",
    "update docs links",
    "remove old example from the documentation",
    "improved performance of merge",
    "bump dev dependencies",
    "Add support for Node 20"])
def test_harmless_lines_are_not_flagged(line):
    assert not is_breaking_line(line)


def test_find_breaking_lines_on_the_exact_tar_line_and_markdown_noise():
    tar = ("Release v6.1.12 of tar:\n- [`f129929`](https://github.com/npm/node-tar/commit/f1299) "
           "[#313](https://github.com/npm/node-tar/pull/313) remove dead link to benchmarks (#313) (@yetzt)")
    real = "Release v7.0.0 of tar:\n- **BREAKING:** drops support for Node 14"
    chunks = [Chunk("S1", "r1", "release", "t", "u", tar), Chunk("S2", "r2", "release", "t", "u", real),
              Chunk("S3", "a1", "advisory", "t", "u", "BREAKING advisory text")]      # advisories ignored
    found = find_breaking_lines(chunks)
    assert [cid for _, cid in found] == ["S2"] and "drops support" in found[0][0]
    assert "](" not in found[0][0] and "**" not in found[0][0].replace("BREAKING:", "")


# ------------------------------------------------------ version-jump rule
@pytest.mark.parametrize("a,b,expected", [
    ("4.4.8", "7.5.21", True), ("1.9.1", "1.13.8", False), ("2.0.0", "2.9.9", False),
    ("0.0.10", "0.2.4", True), ("0.2.4", "0.2.9", False), ("0.9.0", "1.0.0", True),
    ("0.0.8", "0.0.9", False), ("1.2", "1.3", False),
    ("1.0.0", None, False), ("weird", "1.0.0", False), ("1.0.0", "latest", False)])
def test_breaking_capable_bump(a, b, expected):
    assert breaking_capable_bump(a, b) is expected


def test_fix_plan_flags_zero_x_minor_jump():
    f = Finding(Dependency("minimist", NPM, "0.0.10", HIGH, "lock"),
                Vulnerability("V", ("CVE-1",), "s", "HIGH", 9.8, "0.2.4"), False, 0.04, 0.9)
    assert fix_plan([f])[0].major_bump is True


# ------------------------------------------------------------ changelog parsing
CHANGELOG = """# Changelog

## Unreleased
- something upcoming

## [6.5.3] - 2020-01-02
- **[Fix]** `parse`: ignore `__proto__` keys

## v6.5.2
### Fixed
- bug fix

6.5.1 / 2019-05-05
==================
- old style entry

## 6.5.0
- removed the deprecated `decode` option
"""


def test_split_changelog_styles_and_boundaries():
    secs = dict(split_changelog(CHANGELOG))
    assert list(secs) == ["6.5.3", "6.5.2", "6.5.1", "6.5.0"]                   # "Unreleased" skipped
    assert "__proto__" in secs["6.5.3"] and "bug fix" in secs["6.5.2"] and "Fixed" in secs["6.5.2"]
    assert "old style entry" in secs["6.5.1"] and "bug fix" not in secs["6.5.1"]
    assert "something upcoming" not in "".join(secs.values())
    assert split_changelog("") == [] and split_changelog("just prose\nno versions") == []


class Resp:
    def __init__(self, status=200, body=None, text=""):
        self.status_code, self._body, self.text, self.ok = status, body, text, status < 400

    def json(self):
        return self._body


class RepoNet:
    """Fake GitHub for one repo: file listing, raw files, releases, plus npm registry."""
    def __init__(self, listing, files=None, releases=None, list_status=200, file_status=200):
        self.listing, self.files, self.releases = listing, files or {}, releases or []
        self.list_status, self.file_status, self.calls = list_status, file_status, []

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append((url, headers))
        if "registry.npmjs.org" in url:
            return Resp(body={"repository": "https://github.com/o/r"})
        if url.endswith("/releases"):
            return Resp(body=self.releases)
        if url.endswith("/contents") or url.endswith("/contents/"):
            return Resp(self.list_status, self.listing)
        if "/contents/" in url:
            if self.file_status != 200:
                return Resp(self.file_status)
            return Resp(text=self.files[url.rsplit("/", 1)[1]])
        raise AssertionError(url)

    def count(self, part):
        return sum(1 for u, _ in self.calls if part in u)


LISTING = [{"name": "README.md", "type": "file"}, {"name": "CHANGELOG.md", "type": "file"},
           {"name": "src", "type": "dir"}]


def test_changelog_docs_selects_range_and_marks_as_release_kind():
    net = RepoNet(LISTING, {"CHANGELOG.md": CHANGELOG})
    docs, w = changelog_docs("npm", "qs", "6.5.1", "6.5.3", ("o", "r"), None, net)
    assert [d.doc_id for d in docs] == ["changelog:6.5.2", "changelog:6.5.3"] and w == []
    assert all(d.kind == "release" and d.url.endswith("/blob/HEAD/CHANGELOG.md") for d in docs)
    raw_call = [h for u, h in net.calls if u.endswith("CHANGELOG.md")][0]
    assert raw_call["Accept"] == "application/vnd.github.raw"


def test_changelog_docs_failure_modes_never_raise():
    assert changelog_docs("npm", "q", "1.0.0", "2.0.0", ("o", "r"), None, RepoNet([{"name": "README.md", "type": "file"}])) == ([], [])
    docs, w = changelog_docs("npm", "q", "1.0.0", "2.0.0", ("o", "r"), None, RepoNet(LISTING, list_status=403))
    assert docs == [] and "HTTP 403" in w[0]
    docs, w = changelog_docs("npm", "q", "1.0.0", "2.0.0", ("o", "r"), None, RepoNet(LISTING, file_status=500))
    assert docs == [] and "HTTP 500" in w[0]

    class Boom(RepoNet):
        def get(self, *a, **k):
            raise requests.ConnectionError()
    assert changelog_docs("npm", "q", "1.0.0", "2.0.0", ("o", "r"), None, Boom(LISTING))[1] == ["changelog lookup failed"]
    assert changelog_docs("npm", "q", "1.0.0", "2.0.0", ("o", "r"), None, RepoNet({"message": "x"}))[0] == []  # not a list


def test_gather_evidence_uses_changelog_only_when_no_releases():
    recs = {"A": {"summary": "bad", "details": "d"}}
    net = RepoNet(LISTING, {"CHANGELOG.md": CHANGELOG}, releases=[])
    docs, w = gather_evidence("npm", "qs", "6.5.1", "6.5.3", recs, ["A"], session=net)
    assert [d.doc_id for d in docs] == ["osv:A", "changelog:6.5.2", "changelog:6.5.3"]

    with_rel = RepoNet(LISTING, {"CHANGELOG.md": CHANGELOG},
                       releases=[{"tag_name": "v6.5.2", "body": "rel notes", "html_url": "u"}])
    docs, w = gather_evidence("npm", "qs", "6.5.1", "6.5.3", recs, ["A"], session=with_rel)
    assert [d.doc_id for d in docs] == ["osv:A", "release:v6.5.2"] and with_rel.count("/contents") == 0


# ------------------------------------------------ stage 4.2: capping keeps major releases
from securedeps.evidence import Doc, _is_boundary, cap_docs
from securedeps.explain import RISK_LABELS, Explanation, risk_text


@pytest.mark.parametrize("v,expected", [("7.0.0", True), ("1.0.0", True), ("0.5.0", True), ("0.0.0", True),
                                        ("6.5.0", False), ("6.5.3", False), ("0.5.1", False), ("1.2.0", False),
                                        ("7.0.0-rc.1", True), ("2", True)])
def test_is_boundary(v, expected):
    assert _is_boundary(v) is expected


def _docs(versions):
    return [(v, Doc(f"r:{v}", "release", v, "u", "t")) for v in versions]


def test_cap_keeps_old_major_release_instead_of_only_the_newest():
    versions = ["5.0.0"] + [f"6.1.{i}" for i in range(1, 31)] + ["7.0.0"] + [f"7.1.{i}" for i in range(1, 11)]
    kept, note = cap_docs(_docs(versions), limit=12)
    kept_versions = [v for v, _ in kept]
    assert len(kept) == 12 and "5.0.0" in kept_versions and "7.0.0" in kept_versions
    assert kept_versions[-1] == "7.1.10" and kept_versions == sorted(kept_versions, key=lambda v: [int(x) for x in v.split(".")])
    assert "using 12" in note and "major" in note


def test_cap_no_change_when_under_limit_and_many_majors_keep_newest_majors():
    small = _docs(["1.0.0", "1.0.1"])
    assert cap_docs(small, 12) == (small, None)
    many = _docs([f"{i}.0.0" for i in range(1, 21)])
    kept, _ = cap_docs(many, 5)
    assert [v for v, _ in kept] == ["16.0.0", "17.0.0", "18.0.0", "19.0.0", "20.0.0"]


# ------------------------------------------------ stage 4.2: honest risk wording
def _exp(risk, breaking):
    return Explanation("p", "1.0.0", "2.0.0", "w", "f", "c", breaking, risk, "medium", {}, False)


def test_risk_text_distinguishes_jump_only_from_breaking_notes():
    assert risk_text(_exp("elevated", [("BREAKING: x", "S1")])) == RISK_LABELS["elevated"]
    jump = risk_text(_exp("elevated", []))
    assert "version jump" in jump and "does not rule it out" in jump and "mention possible breaking" not in jump
    assert risk_text(_exp("high", [("x", "S1")])) == RISK_LABELS["high"]
    assert risk_text(_exp("unknown", [])) == RISK_LABELS["unknown"]
    assert all("safe" not in v.lower().replace("not a guarantee", "") for v in RISK_LABELS.values())
