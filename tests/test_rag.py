import math

from securedeps.evidence import Doc
from securedeps.rag import Chunk, HashingEmbedder, Retriever, chunk_docs, chunk_text, cosine


def test_chunk_text_basics():
    assert chunk_text("") == [] and chunk_text("   \n\n ") == []
    assert chunk_text("short text") == ["short text"]
    paras = "\n\n".join(f"paragraph {i} " + "x" * 200 for i in range(8))
    chunks = chunk_text(paras, max_chars=700)
    assert len(chunks) > 1 and all(len(c) <= 700 for c in chunks)
    assert "".join(chunks).count("paragraph") == 8                 # nothing lost


def test_long_paragraph_is_cut_with_overlap():
    text = "".join(f"{i:04d}" for i in range(600))                 # 2400 unique-ish characters
    chunks = chunk_text(text, max_chars=700, overlap=100)
    assert len(chunks) >= 3 and all(len(c) <= 700 for c in chunks)
    assert chunks[0][-100:] == chunks[1][:100]                     # boundary text appears in both


def test_chunk_docs_assigns_stable_ids_and_keeps_source_info():
    docs = [Doc("osv:A", "advisory", "Advisory A", "u1", "alpha text"),
            Doc("release:v2", "release", "pkg v2", "u2", "beta text")]
    chunks = chunk_docs(docs)
    assert [(c.chunk_id, c.doc_id, c.kind, c.url) for c in chunks] == \
        [("S1", "osv:A", "advisory", "u1"), ("S2", "release:v2", "release", "u2")]


def test_hashing_embedder_is_normalized_deterministic_and_meaningful():
    e = HashingEmbedder()
    a, b, c = e.embed(["prototype pollution in lodash merge function",
                       "lodash merge allows prototype pollution attack",
                       "release notes: dropped support for node 6"])
    assert math.isclose(cosine(a, a), 1.0, rel_tol=1e-9)
    assert cosine(a, b) > cosine(a, c) and cosine(a, c) < 0.3
    assert e.embed(["same words"]) == e.embed(["same words"])
    assert cosine(e.embed([""])[0], a) == 0.0                      # empty text: no crash


def make_chunks():
    texts = [("advisory", "prototype pollution in lodash merge allows attackers to modify objects"),
             ("advisory", "command injection in lodash template function via crafted input"),
             ("advisory", "regular expression denial of service in lodash trim functions"),
             ("release", "dropped support for node 6 and removed the deprecated pluck helper")]
    return [Chunk(f"S{i+1}", f"d{i}", k, f"t{i}", "u", t) for i, (k, t) in enumerate(texts)]


def test_retriever_ranks_relevant_chunk_first():
    r = Retriever(HashingEmbedder(), make_chunks())
    top = r.search("prototype pollution", k=2)
    assert top[0][0].chunk_id == "S1" and top[0][1] > top[1][1]


def test_min_per_kind_forces_release_notes_in():
    r = Retriever(HashingEmbedder(), make_chunks())
    plain = r.search("lodash attackers input", k=3)
    assert all(c.kind == "advisory" for c, _ in plain)
    forced = r.search("lodash attackers input", k=3, min_per_kind={"release": 1})
    assert len(forced) == 3 and any(c.kind == "release" for c, _ in forced)


def test_empty_retriever():
    assert Retriever(HashingEmbedder(), []).search("anything") == []
