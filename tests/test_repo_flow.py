"""Tests the GitHub reading flow with a fake GitHub, so no internet is needed."""
import json

import pytest

from securedeps.repo import API, RepoError, scan_repo_dependencies


class FakeResp:
    def __init__(self, status=200, body=None, text="", headers=None):
        self.status_code, self._body, self.text = status, body, text
        self.headers = headers or {}
        self.ok = status < 400

    def json(self):
        return self._body


class FakeGitHub:
    """Pretends to be the GitHub API for one repo."""
    def __init__(self, files, status=200, headers=None):
        self.files, self.status, self.headers_out, self.calls = files, status, headers or {}, []

    def get(self, url, headers=None, timeout=None, params=None):
        self.calls.append(url)
        if self.status != 200:
            return FakeResp(self.status, headers=self.headers_out)
        if url.endswith("/repos/o/r"):
            return FakeResp(body={"default_branch": "trunk"})
        if "/git/trees/trunk" in url:
            assert params == {"recursive": "1"}
            return FakeResp(body={"tree": [{"path": p, "type": "blob", "size": len(t)}
                                           for p, t in self.files.items()] +
                                          [{"path": "src", "type": "tree"}]})
        if "/contents/" in url:
            assert params == {"ref": "trunk"} and headers["Accept"] == "application/vnd.github.raw"
            return FakeResp(text=self.files[url.split("/contents/")[1]])
        raise AssertionError("unexpected url " + url)


@pytest.fixture
def patched(monkeypatch):
    def install(fake):
        monkeypatch.setattr("requests.Session", lambda: fake)
        return fake
    return install


def test_full_flow_skips_junk_and_prefers_lockfile(patched):
    lock = {"lockfileVersion": 3, "packages": {"": {"dependencies": {"express": "4.17.1"}},
                                               "node_modules/express": {"version": "4.17.1"}}}
    fake = patched(FakeGitHub({
        "package.json": json.dumps({"dependencies": {"express": "^4.17.1"}}),
        "package-lock.json": json.dumps(lock),
        "requirements.txt": "flask==1.0\n",
        "node_modules/left-pad/package.json": json.dumps({"dependencies": {"x": "1.0.0"}}),
        "tests/fixtures/requirements.txt": "evil==9.9.9\n",
        "README.md": "hi",
    }))
    res = scan_repo_dependencies("https://github.com/o/r")
    assert res["branch"] == "trunk"
    assert res["files"] == ["package-lock.json", "package.json", "requirements.txt"]
    pairs = {(d.name, d.version) for d in res["dependencies"]}
    assert pairs == {("express", "4.17.1"), ("flask", "1.0")}       # junk paths never fetched
    assert not any("left-pad" in c or "fixtures" in c for c in fake.calls)


def test_no_supported_files(patched):
    patched(FakeGitHub({"README.md": "hi", "main.c": "int main(){}"}))
    with pytest.raises(RepoError, match="No supported dependency files"):
        scan_repo_dependencies("https://github.com/o/r")


def test_not_found_message(patched):
    patched(FakeGitHub({}, status=404))
    with pytest.raises(RepoError, match="not found"):
        scan_repo_dependencies("https://github.com/o/r")


def test_rate_limit_message(patched):
    patched(FakeGitHub({}, status=403, headers={"X-RateLimit-Remaining": "0"}))
    with pytest.raises(RepoError, match="GITHUB_TOKEN"):
        scan_repo_dependencies("https://github.com/o/r")


def test_other_http_errors_become_readable(patched):
    patched(FakeGitHub({}, status=500))
    with pytest.raises(RepoError, match="HTTP 500"):
        scan_repo_dependencies("https://github.com/o/r")


def test_no_files_message_names_what_is_not_supported_yet(monkeypatch):
    import pytest
    from securedeps import repo as repo_mod
    monkeypatch.setattr(repo_mod, "list_dependency_files", lambda *a, **k: ("main", []))
    with pytest.raises(repo_mod.RepoError) as e:
        repo_mod.scan_repo_dependencies("https://github.com/o/r")
    assert "pyproject.toml" in str(e.value) and "uv.lock" in str(e.value) and "cannot be scanned yet" in str(e.value)
