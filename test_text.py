from odyssey_anchor.retrieval.bm25 import BM25
from odyssey_anchor.retrieval.text import Lexicon, norm_key, stem, tokens


def test_stem_plurals_but_not_latin():
    assert stem("seizures") == "seizure"
    assert stem("anomalies") == "anomaly"
    assert stem("corpus") == "corpus"
    assert stem("ptosis") == "ptosis"


def test_tokens_case_insensitive_and_stopwords():
    assert tokens("The Seizures of the Patient") == ["seizure", "patient"]


def test_lexicon_longest_match_and_reorder():
    lex = Lexicon()
    lex.add("Hypoplasia of the corpus callosum", "HP:0002079")
    lex.add("Corpus callosum", "X")
    found = lex.find("MRI showed corpus callosum hypoplasia today")
    assert found[0]["concepts"] == ["HP:0002079"] and found[0]["match"] == "reordered"
    found = lex.find("Hypoplasia of the Corpus Callosum")
    assert found[0]["concepts"] == ["HP:0002079"] and found[0]["match"] == "exact"


def test_norm_key_drops_articles():
    assert norm_key("Abnormality of the eye") == ("abnormality", "eye")


def test_bm25_new_documents_score_like_fitted():
    docs = [["seizure", "hypotonia"], ["ectopia", "lentis", "myopia"], ["seizure", "seizure", "ataxia"]]
    bm = BM25().fit(docs)
    q = ["seizure", "ataxia"]
    assert abs(bm.score(q)[2] - bm.score_new(q, [docs[2]])[0]) < 1e-5
    assert bm.score(q).argmax() == 2
