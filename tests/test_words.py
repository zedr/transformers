import pytest

from words import words_rxp, OrderedSet, MarkovChain, WeightedList


def test_weighted_list():
    seq = WeightedList()
    assert list(seq) == []
    seq.expand()
    seq.inc(index=0)
    assert list(seq) == [1.0]
    seq.expand()
    assert list(seq) == [1.0, 0.0]
    seq.inc(index=1)
    assert list(seq) == [0.5, 0.5]
    seq2 = WeightedList()
    seq2.expand(size=5)
    assert list(seq2) == [0, 0, 0, 0, 0]


def test_words_rxp():
    txt = "show me my files"
    assert ["show", "me", "my", "files"] == words_rxp.findall(txt)


def test_ordered_set():
    s = OrderedSet(str, initial=("one",))
    s.add("two")
    s.add("two")
    s.add("three")
    assert list(s) == ["one", "two", "three"]
    assert "one" in s
    assert s.index("one") == 0
    assert s[0] == "one"


def test_basic_markov_chain():
    chain = MarkovChain()
    assert chain.tokens == ()
    assert chain.as_matrix() == []
    chain.add_text("one")
    assert chain.as_matrix() == [[0]]
    chain.add_text("two")
    assert chain.as_matrix() == [[0, 0], [0, 0]]


def test_parse_sentence_into_markov_chain():
    chain = MarkovChain().add_text("show me my photos please")
    assert chain.tokens == ("show", "me", "my", "photos", "please")


def test_parse_sentence_and_apply_weights():
    chain = MarkovChain().add_text("show me my photos please")
    assert chain.as_matrix() == [
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
    ]
    chain.add_text("show me my documents please")
    assert chain.as_matrix() == [
        [0, 1.0, 0, 0, 0, 0],
        [0, 0, 1.0, 0, 0, 0],
        [0, 0, 0, 0.5, 0, 0.5],
        [0, 0, 0, 0, 1.0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1.0, 0],
    ]


def test_render_chain_as_dot():
    chain = MarkovChain()
    chain.add_text("I give Lila a toy boat")
    chain.add_text("I give Sami a diving mask")
    result = chain.render_as_dot()
    expected = """digraph G {
    rankdir="LR";
    I -> give [label="1.0"];
    give -> Lila [label="0.5"];
    give -> Sami [label="0.5"];
    Lila -> a [label="1.0"];
    a -> toy [label="0.5"];
    a -> diving [label="0.5"];
    toy -> boat [label="1.0"];
    Sami -> a [label="1.0"];
    diving -> mask [label="1.0"];
}"""
    assert result.strip() == expected.strip()


def test_get_one_hot_encoding_for_word():
    chain = MarkovChain()
    chain.add_text("I give Lila a toy boat")
    assert list(chain.encode("Lila")) == [0, 0, 1, 0, 0, 0]


def test_get_distribution():
    chain = MarkovChain()
    chain.add_text("I give Lila a toy boat")
    chain.add_text("I give Sami a diving mask")
    assert chain.get_distribution("give") == [
        0.0,
        0.0,
        0.5,
        0.0,
        0.0,
        0.0,
        0.5,
        0.0,
        0.0,
    ]
    with pytest.raises(ValueError):
        chain.get_distribution("foo")


def test_get_transitions():
    chain = MarkovChain()
    chain.add_text("I give Lila a toy boat")
    chain.add_text("I give Sami a diving mask")
    assert chain.get_transitions("give") == {"Lila": 0.5, "Sami": 0.5}
    chain.add_text("I give Sami another diving mask")
    chain.add_text("I give Sami another toy boat")
    assert chain.get_transitions("give") == {"Lila": 0.25, "Sami": 0.75}
