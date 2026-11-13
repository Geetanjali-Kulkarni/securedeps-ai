"""End-to-end test of `scan --explain` with every network piece replaced by a fake."""
import json

from securedeps import scan
from securedeps.evidence import Doc
from securedeps.intel import Analysis
from securedeps.models import HIGH, NPM, Dependency, Finding, Vulnerability
from securedeps.rag import HashingEmbedder

DEP = Dependency("lodash", NPM, "4.17.11", HIGH, "package-lock.json", direct=False)
FINDING = Finding(DEP, Vulnerability("GHSA-1", ("CVE-2020-1",), "Prototype pollution in lodash", "HIGH", 7.2, "4.17.21"),
                  False, 0.2, 0.9)
DOCS = [Doc("osv:GHSA-1", "advisory", "Advisory GHSA-1", "https://osv.dev/vulnerability/GHSA-1",
            "GHSA-1\n\nPrototype pollution in lodash\n\nFixed in 4.17.21."),
        Doc("release:v4.17.21", "release", "lodash v4.17.21", "https://gh/r",
            "Release v4.17.21 of lodash:\n- BREAKING CHANGE: removed pluck helper")]


class FakeLLM:
    model = "fake-model"

    def __init__(self, answer):
        self.answer = answer

    def complete(self, system, user):
        return json.dumps(self.answer)


def run(monkeypatch, capsys, llm, argv_explain=3):
    monkeypatch.setattr(scan, "scan_repo_dependencies",
                        lambda url: {"owner": "o", "repo": "r", "files": ["package-lock.json"],
                                     "dependencies": [DEP], "warnings": []})
    monkeypatch.setattr(scan, "analyze", lambda deps: Analysis(findings=[FINDING], records={}, scanned=1))
    monkeypatch.setattr(scan, "gather_evidence", lambda *a, **k: (DOCS, ["example evidence warning"]))
    monkeypatch.setattr(scan, "get_llm", lambda: llm)
    monkeypatch.setattr(scan, "get_embedder", lambda: HashingEmbedder())
    assert scan.main("https://github.com/o/r", explain=argv_explain) == 0
    return capsys.readouterr().out


def test_explain_without_llm_key(monkeypatch, capsys):
    out = run(monkeypatch, capsys, None)
    assert "No LLM key found" in out and "deterministic text" in out
    assert "Upgrade risk:  ELEVATED" in out and "[S2]" in out and "https://gh/r" in out
    assert "note: example evidence warning" in out


def test_explain_with_good_llm(monkeypatch, capsys):
    good = {"what_is_wrong": "Attackers can pollute prototypes [S1].", 
            "compatibility_notes": "The release removes the pluck helper [S2]."}
    out = run(monkeypatch, capsys, FakeLLM(good))
    assert "AI text" in out and "LLM: FakeLLM (fake-model)" in out
    assert "Attackers can pollute prototypes [S1]." in out


def test_explain_with_bad_llm_answer_is_replaced(monkeypatch, capsys):
    out = run(monkeypatch, capsys, FakeLLM({"what_is_wrong": "Bad [S99].", "compatibility_notes": ""}))
    assert "deterministic text" in out and "AI answer rejected" in out and "Bad [S99]" not in out


def test_no_explain_flag_prints_no_explanations(monkeypatch, capsys):
    assert "EXPLANATIONS" not in run(monkeypatch, capsys, None, argv_explain=0)
