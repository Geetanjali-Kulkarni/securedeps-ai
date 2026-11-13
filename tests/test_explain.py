import json

import pytest

from securedeps.evidence import Doc
from securedeps.explain import (Explanation, build_prompt, check_cited, explain_action,
                                find_breaking_lines, parse_json, risk_and_confidence,
                                unverified_versions)
from securedeps.llm import LLMError
from securedeps.rag import HashingEmbedder, chunk_docs

ADV = Doc("osv:GHSA-1", "advisory", "Advisory GHSA-1", "https://osv.dev/vulnerability/GHSA-1",
          "GHSA-1 (CVE-2020-1)\n\nPrototype pollution in lodash\n\nlodash before 4.17.21 lets attackers modify "
          "object prototypes through merge. Fixed in 4.17.21.")
REL_BREAKING = Doc("release:v4.17.21", "release", "lodash v4.17.21 release notes", "https://gh/v4.17.21",
                   "Release v4.17.21 of lodash:\n- BREAKING CHANGE: removed the deprecated pluck helper\n- fixed a bug")
REL_CLEAN = Doc("release:v4.17.21", "release", "lodash v4.17.21 release notes", "https://gh/v4.17.21",
                "Release v4.17.21 of lodash:\n- fixed a typo in the docs\n- improved performance of merge")
FACTS = "Package: lodash 4.17.11 -> 4.17.21. Priority: Plan (score 27.2). CVSS 7.2."


class FakeLLM:
    def __init__(self, answer=None, error=None):
        self.answer, self.error, self.prompts = answer, error, []

    def complete(self, system, user):
        self.prompts.append((system, user))
        if self.error:
            raise self.error
        return self.answer if isinstance(self.answer, str) else json.dumps(self.answer)


GOOD = {"what_is_wrong": "Attackers can modify object prototypes through merge [S1].",
        "compatibility_notes": "The release removes the deprecated pluck helper [S2]."}


def run(docs, llm=None, major=False):
    return explain_action("lodash", "4.17.11", "4.17.21", major, docs, FACTS, llm, HashingEmbedder())


# ------------------------------------------------- deterministic pieces
def test_breaking_line_extraction():
    chunks = chunk_docs([ADV, REL_BREAKING])
    lines = find_breaking_lines(chunks)
    assert len(lines) == 1 and "removed the deprecated pluck helper" in lines[0][0] and lines[0][1] == "S2"
    assert find_breaking_lines(chunk_docs([REL_CLEAN])) == []
    assert find_breaking_lines(chunk_docs([ADV])) == []          # advisories are not release notes


@pytest.mark.parametrize("has_rel,breaking,major,expected", [
    (False, [], False, ("unknown", "low")), (False, [("x", "S1")], True, ("unknown", "low")),
    (True, [], False, ("no_warnings_found", "medium")), (True, [], True, ("elevated", "medium")),
    (True, [("x", "S1")], False, ("elevated", "medium")), (True, [("x", "S1")], True, ("high", "medium"))])
def test_risk_table_and_never_safe(has_rel, breaking, major, expected):
    assert risk_and_confidence(has_rel, breaking, major) == expected
    assert "safe" not in expected[0]


def test_citation_and_version_checks():
    assert check_cited("a [S1] b [S3] c [S9]", {"S1", "S3"}) == (["S1", "S3"], ["S9"])
    assert check_cited("no citations", {"S1"}) == ([], [])
    assert unverified_versions("use 4.17.21, not 9.9.9 or 1.2", "fixed in 4.17.21 and 1.2") == ["9.9.9"]
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(ValueError):
        parse_json("[1, 2]")


# -------------------------------------------------------- no LLM at all
def test_without_llm_gives_cited_deterministic_explanation():
    e = run([ADV, REL_BREAKING])
    assert isinstance(e, Explanation) and not e.used_llm and e.warnings == []
    assert "Prototype pollution in lodash" in e.what_is_wrong and "[S1]" in e.what_is_wrong
    assert e.fix == "Upgrade lodash from 4.17.11 to 4.17.21."
    assert e.upgrade_risk == "elevated" and e.confidence == "medium"
    assert "pluck" in e.compatibility and "[S2]" in e.compatibility
    assert set(e.sources) == {"S1", "S2"} and e.sources["S2"].url == "https://gh/v4.17.21"


def test_risk_levels_end_to_end():
    assert run([ADV]).upgrade_risk == "unknown" and run([ADV]).confidence == "low"
    assert run([ADV, REL_CLEAN]).upgrade_risk == "no_warnings_found"
    assert run([ADV, REL_CLEAN], major=True).upgrade_risk == "elevated"
    assert run([ADV, REL_BREAKING], major=True).upgrade_risk == "high"
    assert "No release notes" in run([ADV]).compatibility


def test_no_target_version():
    e = explain_action("lodash", "1.0.0", None, False, [ADV], FACTS, None, HashingEmbedder())
    assert "No fixed version" in e.fix


# ----------------------------------------------------- with an LLM
def test_good_llm_answer_is_accepted():
    llm = FakeLLM(GOOD)
    e = run([ADV, REL_BREAKING], llm)
    assert e.used_llm and e.warnings == []
    assert e.what_is_wrong == GOOD["what_is_wrong"] and e.fix == "Upgrade lodash from 4.17.11 to 4.17.21."
    assert set(e.sources) == {"S1", "S2"}
    system, user = llm.prompts[0]
    assert "ONLY the numbered sources" in system and "do not state that any upgrade is safe" in system.lower()
    assert "FACTS" in user and "[S1]" in user and "[S2]" in user and "lodash 4.17.11" in user


def test_llm_cannot_change_the_risk_label():
    answer = {**GOOD, "compatibility_notes": "There are no breaking changes at all [S2]."}
    e = run([ADV, REL_BREAKING], FakeLLM(answer))
    assert e.used_llm and e.upgrade_risk == "elevated"                  # decided by our code, not the AI
    assert e.breaking_lines and "pluck" in e.breaking_lines[0][0]


@pytest.mark.parametrize("bad,fragment", [
    ({**GOOD, "compatibility_notes": "Changed things [S9]."}, "not provided"),   # invented source
    ({**GOOD, "what_is_wrong": "It allows code execution."}, "no citation"),  # claim without a citation
    ({**GOOD, "what_is_wrong": ""}, "missing"),
    ("this is not json", "rejected"),
    ('["a list"]', "rejected")])
def test_ungrounded_answers_are_rejected_and_replaced(bad, fragment):
    e = run([ADV, REL_BREAKING], FakeLLM(bad))
    assert not e.used_llm and any(fragment in w for w in e.warnings)
    assert e.fix == "Upgrade lodash from 4.17.11 to 4.17.21."           # deterministic text shown instead


def test_llm_errors_fall_back():
    e = run([ADV, REL_BREAKING], FakeLLM(error=LLMError("LLM rate limit reached")))
    assert not e.used_llm and "rate limit" in e.warnings[0] and "[S1]" in e.what_is_wrong


def test_hallucinated_version_is_flagged_but_answer_kept():
    e = run([ADV, REL_BREAKING], FakeLLM({**GOOD, "compatibility_notes": "Needs 9.9.9 [S2]."}))
    assert e.used_llm and any("9.9.9" in w for w in e.warnings)


def test_json_in_code_fence_is_accepted():
    assert run([ADV, REL_BREAKING], FakeLLM("```json\n" + json.dumps(GOOD) + "\n```")).used_llm


def test_uncited_compatibility_is_dropped_unless_it_admits_ignorance():
    e = run([ADV, REL_BREAKING], FakeLLM({**GOOD, "compatibility_notes": "Probably fine."}))
    assert e.used_llm and "Probably fine" not in e.compatibility and any("uncited" in w for w in e.warnings)
    e = run([ADV, REL_CLEAN], FakeLLM({**GOOD, "compatibility_notes": "Not stated in the sources."}))
    assert "Not stated" in e.compatibility and not e.warnings        # nothing flagged: admitting ignorance is fine


def test_ai_saying_not_stated_cannot_contradict_a_breaking_label():
    # our keyword check found a breaking-style line; "ELEVATED/HIGH ... breaking changes" next to
    # "Not stated" confused users, so the flagged line itself is shown instead
    e = run([ADV, REL_BREAKING], FakeLLM({**GOOD, "compatibility_notes": "Not stated in the sources [S2]."}))
    assert e.upgrade_risk == "elevated" and "Not stated" not in e.compatibility
    assert "removed the deprecated pluck helper" in e.compatibility and "[S2]" in e.compatibility
    assert "S2" in e.sources and e.used_llm


def test_llm_with_no_documents_is_not_called():
    llm = FakeLLM(GOOD)
    e = run([], llm)
    assert llm.prompts == [] and not e.used_llm and any("no source documents" in w for w in e.warnings)


def test_prompt_contains_only_given_sources():
    chunks = chunk_docs([ADV, REL_CLEAN])
    p = build_prompt("FACT LINE", [(chunks[1], 0.9)])
    assert "[S2]" in p and "[S1]" not in p and "FACT LINE" in p


def test_fix_line_always_comes_from_the_scanner_not_the_llm():
    e = run([ADV, REL_BREAKING], FakeLLM({**GOOD, "fix": "Upgrade lodash to 9.9.9 [S1]."}))
    assert e.used_llm and e.fix == "Upgrade lodash from 4.17.11 to 4.17.21." and e.warnings == []


def test_grouped_citations_are_understood_and_checked():
    from securedeps.explain import check_cited
    assert check_cited("A [S1, S2] and B [S3].", {"S1", "S2", "S3"}) == (["S1", "S2", "S3"], [])
    assert check_cited("A [S1, S9].", {"S1"}) == (["S1"], ["S9"])
    e = run([ADV, REL_BREAKING], FakeLLM({**GOOD, "what_is_wrong": "Prototype pollution [S1, S2]."}))
    assert e.used_llm and set(e.sources) >= {"S1", "S2"}


class SeqLLM:
    def __init__(self, *answers):
        self.answers, self.prompts = list(answers), []

    def complete(self, system, user):
        self.prompts.append(user)
        a = self.answers.pop(0)
        return a if isinstance(a, str) else json.dumps(a)


def test_missing_citations_trigger_one_retry_that_can_succeed():
    uncited = {"what_is_wrong": "Attackers can modify prototypes.", "compatibility_notes": "Removes pluck."}
    llm = SeqLLM(uncited, GOOD)
    e = run([ADV, REL_BREAKING], llm)
    assert e.used_llm and len(llm.prompts) == 2 and "NO [S#] citations" in llm.prompts[1]


def test_still_uncited_after_retry_falls_back_to_deterministic_text():
    uncited = {"what_is_wrong": "Attackers can modify prototypes.", "compatibility_notes": ""}
    llm = SeqLLM(uncited, uncited)
    e = run([ADV, REL_BREAKING], llm)
    assert not e.used_llm and len(llm.prompts) == 2 and any("no citation" in w for w in e.warnings)


def test_cited_answer_does_not_trigger_a_retry():
    llm = SeqLLM(GOOD)
    run([ADV, REL_BREAKING], llm)
    assert len(llm.prompts) == 1


def test_trailing_not_stated_is_dropped_after_a_real_statement_but_kept_alone():
    from securedeps.explain import drop_trailing_not_stated as d
    assert d("Calls now return false [S2]. Not stated in the sources.") == "Calls now return false [S2]."
    assert d("Not stated in the sources [S1].") == "Not stated in the sources [S1]."
    assert d("One [S1]. Two [S2].") == "One [S1]. Two [S2]."


def test_quoted_release_note_lines_do_not_end_with_a_commit_hash():
    doc = Doc("release:v1", "release", "x v1 release notes", "",
              "Release v1 of x:\n- [Fix] Buffer.from and Buffer.alloc require node 4+ c499f76")
    e = run([ADV, doc])
    assert e.breaking_lines and e.breaking_lines[0][0].endswith("require node 4+")
