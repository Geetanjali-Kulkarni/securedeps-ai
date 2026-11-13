import json

from securedeps import chat, pipeline
from securedeps.llm import LLMError
from securedeps.rag import HashingEmbedder
from tests.test_pipeline import DOCS, fake_scan


class SeqLLM:
    def __init__(self, *answers, error=None):
        self.answers, self.prompts, self.error = list(answers), [], error

    def complete(self, system, user):
        self.prompts.append(user)
        if self.error:
            raise self.error
        a = self.answers.pop(0)
        return a if isinstance(a, str) else json.dumps(a)


def result(monkeypatch):
    fake_scan(monkeypatch)
    r = pipeline.run_scan("u")
    return r


def ask(docs, llm, q="Which fix is first and why?"):
    return chat.answer_question(q, docs, llm, HashingEmbedder())


def test_scan_docs_hold_our_numbers_and_summary(monkeypatch):
    r = result(monkeypatch)
    docs = chat.scan_docs(r)
    assert docs[0].doc_id == "scan:summary" and "o/r" in docs[0].text and "2 dependencies" in docs[0].text
    assert any("lodash" in d.text and "score" in d.text for d in docs)
    assert all(d.kind == "scan" for d in docs)


def test_chat_docs_add_gathered_evidence_without_duplicates(monkeypatch):
    r = result(monkeypatch)
    for a in r.plan:
        pipeline.evidence_for(r, a)              # both fixes receive the same DOCS
    ids = [d.doc_id for d in chat.chat_docs(r)]
    assert ids.count("osv:GHSA-1") == 1 and "release:v5.0.0" in ids


def test_without_llm_returns_passages_not_a_made_up_answer(monkeypatch):
    docs = chat.chat_docs(result(monkeypatch))
    a = ask(docs, None)
    assert not a.used_llm and "no AI answer" in a.text and a.sources and any("No LLM key" in w for w in a.warnings)


def test_good_cited_answer_is_accepted_and_sources_returned(monkeypatch):
    docs = chat.chat_docs(result(monkeypatch))
    llm = SeqLLM({"answer": "Fix lodash first [S1]."})
    a = ask(docs, llm)
    assert a.used_llm and a.text == "Fix lodash first [S1]." and "S1" in a.sources and a.warnings == []


def test_uncited_answer_is_retried_once(monkeypatch):
    docs = chat.chat_docs(result(monkeypatch))
    llm = SeqLLM({"answer": "Fix lodash first."}, {"answer": "Fix lodash first [S1]."})
    a = ask(docs, llm)
    assert a.used_llm and len(llm.prompts) == 2 and "NO [S#] citations" in llm.prompts[1]


def test_still_uncited_falls_back_to_passages(monkeypatch):
    docs = chat.chat_docs(result(monkeypatch))
    a = ask(docs, SeqLLM({"answer": "Fix lodash first."}, {"answer": "Fix lodash first."}))
    assert not a.used_llm and any("no citation" in w for w in a.warnings)


def test_invented_source_is_rejected(monkeypatch):
    docs = chat.chat_docs(result(monkeypatch))
    a = ask(docs, SeqLLM({"answer": "It is critical [S99]."}))
    assert not a.used_llm and any("S99" in w for w in a.warnings)


def test_llm_error_and_bad_json_fall_back(monkeypatch):
    docs = chat.chat_docs(result(monkeypatch))
    a = ask(docs, SeqLLM(error=LLMError("rate limit")))
    assert not a.used_llm and any("rate limit" in w for w in a.warnings)
    a = ask(docs, SeqLLM("not json at all [S1]", "still not json"))
    assert not a.used_llm and any("expected format" in w for w in a.warnings)


def test_model_may_admit_the_sources_do_not_say(monkeypatch):
    docs = chat.chat_docs(result(monkeypatch))
    llm = SeqLLM({"answer": "The sources do not say."})
    a = ask(docs, llm, "What is the weather?")
    assert not a.used_llm and "sources do not say" in a.text.lower() and len(llm.prompts) == 1


def test_invented_version_number_is_flagged(monkeypatch):
    docs = chat.chat_docs(result(monkeypatch))
    a = ask(docs, SeqLLM({"answer": "Upgrade to 9.9.9 [S1]."}))
    assert a.used_llm and any("9.9.9" in w for w in a.warnings)


def test_empty_docs_message():
    assert "nothing to search" in chat.answer_question("x", [], None, HashingEmbedder()).text


def test_top_fixes_are_always_in_the_prompt_even_if_retrieval_misses_them(monkeypatch):
    from securedeps.evidence import Doc
    docs = [Doc("scan:summary", "scan", "Scan summary", "", "Scan of o/r: 5 package versions."),
            Doc("scan:fix1", "scan", "Fix #1: alpha 1.0.0", "", "Fix number 1 of 5: upgrade alpha from 1.0.0 to 1.0.1."),
            *[Doc(f"scan:fix{n}", "scan", f"Fix #{n}: minimist {n}.0.0", "",
                  f"Fix number {n}: minimist first first first priority why score fixes.") for n in (2, 3, 4, 5)]]
    llm = SeqLLM({"answer": "Alpha is first [S2]."})
    chat.answer_question("what should i fix first and why?", docs, llm, HashingEmbedder(), k=2)
    assert "Fix number 1 of 5" in llm.prompts[0] and "Scan of o/r" in llm.prompts[0]


def test_risk_questions_pull_in_release_notes_even_when_scan_text_matches_better():
    from securedeps.evidence import Doc
    from securedeps.rag import chunk_docs
    docs = [Doc(f"scan:fix{n}", "scan", f"Fix #{n}: alpha", "", "Will upgrading alpha break my code? alpha upgrading break code.")
            for n in range(1, 7)]
    docs.append(Doc("release:v2", "release", "alpha v2", "", "Release v2.0.0 of alpha:\n- BREAKING CHANGE: removed pluck"))
    got = chat.retrieve("Will upgrading alpha break my code?", chunk_docs(docs), HashingEmbedder(), k=3)
    assert any(c.kind == "release" for c, _ in got)


def test_scan_docs_state_dependency_kind_in_plain_words(monkeypatch):
    r = result(monkeypatch)
    text = " ".join(d.text for d in chat.scan_docs(r))
    assert "transitive dependency (pulled in by another package" in text
    assert "a development-only dependency" in text and "a direct dependency of the project" in text


def test_leaked_instructions_are_removed_from_the_answer(monkeypatch):
    docs = chat.chat_docs(result(monkeypatch))
    leaky = {"answer": "Fix lodash first [S1]. Never claim an upgrade is safe or will not break anything. [Developer]"}
    a = ask(docs, SeqLLM(leaky))
    assert a.used_llm and a.text == "Fix lodash first [S1]." and "Never claim" not in a.text


def test_json_wrapped_in_extra_words_is_still_parsed(monkeypatch):
    docs = chat.chat_docs(result(monkeypatch))
    a = ask(docs, SeqLLM('Sure! Here you go: {"answer": "Fix lodash first [S1]."} Hope that helps.'))
    assert a.used_llm and a.text == "Fix lodash first [S1]."


def test_unparseable_format_gets_one_retry(monkeypatch):
    docs = chat.chat_docs(result(monkeypatch))
    llm = SeqLLM("this is not json [S1]", {"answer": "Fix lodash first [S1]."})
    a = ask(docs, llm)
    assert a.used_llm and len(llm.prompts) == 2 and "ONLY a JSON object" in llm.prompts[1]


def test_refusal_message_is_generic_not_a_misleading_hint(monkeypatch):
    docs = chat.chat_docs(result(monkeypatch))
    a = ask(docs, SeqLLM({"answer": "The sources do not say."}), "What is the weather in Pune?")
    assert "sources do not say" in a.text.lower() and "dashboard" not in a.text.lower()


def test_risk_question_gets_the_named_packages_release_notes_breaking_ones_first():
    from securedeps.evidence import Doc
    from securedeps.rag import chunk_docs
    docs = [Doc("scan:summary", "scan", "Scan summary", "", "Scan of o/r.")]
    docs += [Doc(f"osv:{n}", "advisory", f"Advisory {n}", "", f"qs upgrade breaking changes advisory text {n} " * 8) for n in range(8)]
    docs += [Doc("release:a", "release", "qs 6.1.0 release notes", "", "Release 6.1.0 of qs:\n- fixed a typo in the readme file"),
             Doc("release:b", "release", "qs 7.0.0 release notes", "", "Release 7.0.0 of qs:\n- BREAKING CHANGE: removed the old parse option"),
             Doc("release:c", "release", "other 1.0.0 release notes", "", "Release 1.0.0 of other:\n- BREAKING CHANGE: removed everything")]
    got = chat.retrieve("What do the release notes say about breaking changes when upgrading qs from 6.0.0 to 7.0.0?",
                        chunk_docs(docs), HashingEmbedder(), k=3)
    ids = [c.doc_id for c, _ in got]
    assert "release:b" in ids and "release:a" in ids          # the package's own notes are present


def test_paraphrased_instruction_is_also_removed(monkeypatch):
    docs = chat.chat_docs(result(monkeypatch))
    a = ask(docs, SeqLLM({"answer": "Fix lodash first [S1]. Nobody can promise an upgrade is safe [S1]."}))
    assert a.used_llm and a.text == "Fix lodash first [S1]."


# ------------------------------------------- version questions are answered from scan data
def test_version_question_is_answered_directly_from_the_scan_without_the_llm(monkeypatch):
    r = result(monkeypatch)
    docs = chat.chat_docs(r)
    a0 = r.plan[0]
    llm = SeqLLM(error=AssertionError("LLM must not be called"))
    a = ask(docs, llm, f"Which version of {a0.name} fixes the vulnerability in {a0.installed}?")
    assert not a.used_llm and f"Upgrade {a0.name} from {a0.installed} to {a0.target}" in a.text
    assert "[S" in a.text and a.sources and llm.prompts == []


def test_version_question_picks_the_installed_version_named(monkeypatch):
    r = result(monkeypatch)
    docs = chat.chat_docs(r)
    a1 = r.plan[1]
    a = ask(docs, None, f"What version of {a1.name} should I upgrade to from {a1.installed}?")
    assert f"to {a1.target}" in a.text and a1.name in a.text


def test_risk_question_is_not_hijacked_by_the_version_shortcut(monkeypatch):
    r = result(monkeypatch)
    docs = chat.chat_docs(r)
    a0 = r.plan[0]
    llm = SeqLLM({"answer": "The release notes mention a removal [S1]."})
    a = ask(docs, llm, f"Which version of {a0.name} fixes it, and will the upgrade break anything?")
    assert llm.prompts, "risk-type question must still go to the model"


def test_unknown_package_version_question_goes_the_normal_route(monkeypatch):
    docs = chat.chat_docs(result(monkeypatch))
    llm = SeqLLM({"answer": "The sources do not say."})
    a = ask(docs, llm, "Which version of notapackage fixes the vulnerability in 1.0.0?")
    assert llm.prompts and "do not say" in a.text.lower()


# ------------------------------------------- v1.2: the chat can see the whole ranking
def _big_result(monkeypatch, n=40):
    from dataclasses import replace
    r = result(monkeypatch)
    base = r.plan[0]
    r.plan = [replace(base, name=f"pkg{i}", score=round(50 - i * 0.5, 1)) for i in range(n)]
    return r


def test_ranking_doc_lists_every_fix_and_names_the_last_one(monkeypatch):
    r = _big_result(monkeypatch, n=8)
    d = chat.ranking_doc(r)
    assert d.doc_id == "scan:ranking" and d.kind == "scan"
    assert "1. pkg0" in d.text and "8. pkg7" in d.text
    assert "last fix, with the lowest priority, is number 8: pkg7" in d.text


def test_ranking_doc_of_a_long_plan_keeps_the_top_and_the_bottom(monkeypatch):
    r = _big_result(monkeypatch, n=40)
    t = chat.ranking_doc(r).text
    assert "1. pkg0" in t and "20. pkg19" in t and "40. pkg39" in t and "31. pkg30" in t
    assert "21. pkg20" not in t and "fixes 21 to 30 are not listed" in t


def test_last_fix_question_always_gets_the_bottom_of_the_ranking(monkeypatch):
    from securedeps.rag import chunk_docs
    r = _big_result(monkeypatch, n=40)
    got = chat.retrieve("what should i fix at the last?", chunk_docs(chat.chat_docs(r)), HashingEmbedder())
    assert any("pkg39" in c.text for c, _ in got)          # used to see only the top three fixes


def test_ranking_doc_comes_after_the_per_fix_sources_so_their_numbers_do_not_move(monkeypatch):
    ids = [d.doc_id for d in chat.scan_docs(_big_result(monkeypatch, n=5))]
    assert ids[0] == "scan:summary" and ids[1] == "scan:fix1" and ids[-1] == "scan:ranking"


def test_what_changed_questions_pull_in_release_notes():
    from securedeps.evidence import Doc
    from securedeps.rag import chunk_docs
    docs = [Doc(f"scan:fix{n}", "scan", f"Fix #{n}: alpha", "", "What changed in alpha? alpha changed from to.")
            for n in range(1, 7)]
    docs.append(Doc("release:v2", "release", "alpha v2 release notes", "", "Release v2.0.0 of alpha:\n- returns false now"))
    got = chat.retrieve("What changes happened in alpha from 1.0.0 to 2.0.0?", chunk_docs(docs), HashingEmbedder(), k=3)
    assert any(c.kind == "release" for c, _ in got)
