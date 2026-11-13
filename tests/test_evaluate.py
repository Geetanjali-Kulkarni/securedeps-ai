import csv

from securedeps import evaluate, pipeline
from securedeps.rag import Chunk, HashingEmbedder
from tests.test_chat import SeqLLM
from tests.test_pipeline import fake_scan


def scanned(monkeypatch):
    fake_scan(monkeypatch)
    r = pipeline.run_scan("u")
    for a in r.plan:
        pipeline.evidence_for(r, a)
    return r


def test_answer_key_comes_from_the_engine_not_from_an_ai(monkeypatch):
    r = scanned(monkeypatch)
    qs = evaluate.build_questions(r)
    first = next(q for q in qs if q.text.startswith("What should I fix first"))
    assert first.must_include == [r.plan[0].name]
    ver = [q for q in qs if q.category == "version"]
    assert ver and all(q.must_include[0] in {a.target for a in r.plan} for q in ver)
    assert {q.category for q in qs} >= {"ranking", "version", "flags", "risk", "unanswerable"}
    assert sum(q.category == "unanswerable" for q in qs) == 5


def test_passes_rules():
    Q = evaluate.Question
    assert evaluate.passes(Q("q", "ranking", ["tar"]), "Fix TAR first [S1]", True)
    assert not evaluate.passes(Q("q", "ranking", ["tar"]), "Fix qs first", True)
    assert evaluate.passes(Q("q", "flags", ["dev-only|development-only"]), "It is a development-only package", True)
    assert not evaluate.passes(Q("q", "risk", must_not=evaluate.UNSAFE_PHRASES), "It is safe to upgrade.", True)
    assert not evaluate.passes(Q("q", "risk", [], evaluate.UNSAFE_PHRASES), "The sources do not say.", False)
    rel = {"S2": Chunk("S2", "release:v1", "release", "x v1", "", "t")}
    adv = {"S1": Chunk("S1", "osv:1", "advisory", "Advisory 1", "", "t")}
    risk = Q("q", "risk", [], evaluate.UNSAFE_PHRASES)
    assert evaluate.passes(risk, "Releases now require Node 14 [S2].", True, rel)      # 'require' once failed a keyword list
    assert not evaluate.passes(risk, "Something changed [S1].", True, adv)             # not grounded in release notes
    assert not evaluate.passes(risk, "Something changed.", True, {})
    assert not evaluate.passes(risk, "It is safe to upgrade [S2].", True, rel)
    assert evaluate.passes(Q("q", "unanswerable"), "The sources do not say.", False)
    assert not evaluate.passes(Q("q", "unanswerable"), "It is sunny [S1].", True)      # a hallucination
    assert not evaluate.passes(Q("q", "unanswerable"), "Most relevant passages (no AI answer): ...", False)


def test_run_questions_without_llm_measures_only_the_baseline(monkeypatch):
    r = scanned(monkeypatch)
    qs = evaluate.build_questions(r)
    rows = evaluate.run_questions(r, qs, None, HashingEmbedder(), pause=0)
    assert len(rows) == len(qs) and not any(x["llm_used"] for x in rows)
    top = next(x for x in rows if x["question"].startswith("What should I fix first"))
    assert top["retrieval_hit"] is True and top["baseline_pass"] is True      # anchors guarantee the hit
    assert all(x["baseline_pass"] is False for x in rows if x["category"] == "unanswerable")


def test_run_questions_with_llm_and_report(monkeypatch):
    r = scanned(monkeypatch)
    qs = evaluate.build_questions(r)[:2]
    llm = SeqLLM({"answer": f"{r.plan[0].name} is first [S1]."}, {"answer": "The sources do not say."},
                 {"answer": f"{r.plan[1].name} is second [S1]."}, {"answer": "The sources do not say."})
    rows = evaluate.run_questions(r, qs, llm, HashingEmbedder(), pause=0)
    assert rows[0]["llm_used"] is True and rows[0]["llm_pass"] is True
    text = evaluate.report(rows)
    assert "ranking" in text and "ALL" in text and "passed the citation checks" in text and "correctly refused" not in text


def test_csv_roundtrip_and_human_summary(monkeypatch, tmp_path):
    r = scanned(monkeypatch)
    rows = evaluate.run_questions(r, evaluate.build_questions(r)[:3], None, HashingEmbedder(), pause=0)
    path = str(tmp_path / "eval.csv")
    evaluate.save_csv(rows, path)
    assert "No graded rows" in evaluate.summarize_human(path)
    data = list(csv.DictReader(open(path, encoding="utf-8")))
    data[0]["human_verdict"], data[1]["human_verdict"] = "correct", "wrong"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)
    out = evaluate.summarize_human(path)
    assert "graded 2 answers: 1 correct" in out and "agreed" in out


def test_cli_usage_and_summarize(monkeypatch, capsys, tmp_path):
    assert evaluate.main([]) == 2 and "usage" in capsys.readouterr().out
    p = tmp_path / "x.csv"
    p.write_text("category,human_verdict,llm_pass\nranking,correct,True\n")
    assert evaluate.main(["--summarize", str(p)]) == 0 and "1 correct" in capsys.readouterr().out


def test_report_counts_honest_refusals_separately():
    rows = [{"category": "unanswerable", "retrieval_hit": "", "llm_used": False, "llm_pass": True, "baseline_pass": False},
            {"category": "ranking", "retrieval_hit": True, "llm_used": True, "llm_pass": True, "baseline_pass": True}]
    text = evaluate.report(rows)
    assert "1/1 " in text.split("passed the citation checks")[0].splitlines()[-1] or "checks: 1/1" in text
    assert "correctly refused ('the sources do not say'): 1/1" in text


def test_risk_question_expects_refusal_only_when_no_release_notes_were_loaded(monkeypatch):
    r = scanned(monkeypatch)
    risk = [q for q in evaluate.build_questions(r) if q.category == "risk"]
    assert risk and all(not q.expect_refusal for q in risk)     # fake docs include release notes
    r.evidence = {k: ([d for d in docs if d.kind != "release"], w) for k, (docs, w) in r.evidence.items()}
    risk = [q for q in evaluate.build_questions(r) if q.category == "risk"]
    assert all(q.expect_refusal for q in risk)
    assert evaluate.passes(risk[0], "The sources do not say.", False)
    assert not evaluate.passes(risk[0], "It is fine [S1].", True)


def test_version_answer_needs_a_cited_source_that_contains_the_version():
    from securedeps.rag import Chunk
    q = evaluate.Question("Which version of x fixes 1.0?", "version", ["1.2.3"])
    good = {"S1": Chunk("S1", "scan:fix1", "scan", "t", "", "upgrade x from 1.0 to 1.2.3.")}
    bad = {"S2": Chunk("S2", "osv:1", "advisory", "t", "", "unrelated text")}
    assert evaluate.passes(q, "Use 1.2.3 [S1].", True, good)
    assert not evaluate.passes(q, "Use 1.2.3 [S2].", True, bad)
    assert evaluate.passes(q, "Use 1.2.3.", True)          # no sources given: old behaviour


# ------------------------------------------- v1.2: stricter 'first/last' grading
def test_first_mentioned_finds_the_earliest_package_not_just_any_mention():
    names = ["black", "pytest", "flask"]
    assert evaluate.first_mentioned("Fix pytest first, then black.", names) == "pytest"
    assert evaluate.first_mentioned("nothing here", names) is None
    assert evaluate.first_mentioned("flask-login is unrelated", ["flask"]) is None      # whole names only


def test_wrong_package_first_fails_even_if_the_right_one_is_mentioned():
    q = evaluate.Question("What should I fix first?", "ranking", ["black"], first_named="black",
                          names=["black", "pytest"])
    assert evaluate.passes(q, "Fix black first [S2].", True)
    assert not evaluate.passes(q, "Fix pytest first, black is third [S2].", True)


def test_last_fix_question_is_built_for_longer_plans(monkeypatch):
    from dataclasses import replace
    fake_scan(monkeypatch)
    r = pipeline.run_scan("u")
    base = r.plan[0]
    r.plan = [replace(base, name=f"p{i}", score=50 - i) for i in range(6)]
    qs = {q.text: q for q in evaluate.build_questions(r)}
    last = [q for q in qs.values() if "last fix" in q.text][0]
    assert last.must_include == ["p5"] and last.first_named == "p5" and last.gold_docs == ["scan:ranking"]
    r.plan = r.plan[:3]
    assert not any("last fix" in q.text for q in evaluate.build_questions(r))        # too short: skipped
