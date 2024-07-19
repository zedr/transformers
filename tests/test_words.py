import textwrap

from words import words_rxp, add_one_and_rebalance, OrderedSet, MarkovChain


def test_add_and_rebalance():
    seq = [0.0, 0.0, 0.0, 0.0]
    add_one_and_rebalance(seq, 0)
    assert seq == [1.0, 0.0, 0.0, 0.0]
    add_one_and_rebalance(seq, 0)
    assert seq == [1.0, 0.0, 0.0, 0.0]
    add_one_and_rebalance(seq, 1)
    assert seq == [0.5, 0.5, 0.0, 0.0]


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
    assert chain._matrix == []
    chain.add_text("one")
    assert chain._matrix == [[0]]
    chain.add_text("two")
    assert chain._matrix == [[0, 0], [0, 0]]


def test_parse_sentence_into_markov_chain():
    chain = MarkovChain().add_text("show me my photos please")
    assert chain.tokens == ("show", "me", "my", "photos", "please")


def test_parse_sentence_and_apply_weights():
    chain = MarkovChain().add_text("show me my photos please")
    assert chain._matrix == [
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
    ]
    chain.add_text("show me my documents please")
    assert chain._matrix == [
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
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
        Lila -> a [label="0.5"];
        Sami -> a [label="0.5"];
        a -> toy [label="0.5"];
        a -> diving [label="0.5"];
        toy -> boat [label="1.0"];
    }"""
    assert textwrap.dedent(result).strip() == expected.strip()
