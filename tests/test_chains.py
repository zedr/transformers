from chains import MarkovChain


def test_basic_markov_chain_first_order():
    chain = MarkovChain()
    chain.add_text("Check whether the battery ran down please.")
    chain.add_text("Check whether the program ran please.")
    assert sorted(chain._vocabulary) == [
        "battery",
        "check",
        "down",
        "please",
        "program",
        "ran",
        "the",
        "whether",
    ]
    assert chain.predict("Check if the battery ran") == {
        "down": 0.5,
        "please": 0.5,
    }


def test_basic_markov_chain_second_order():
    chain = MarkovChain(order=2)
    chain.add_text("Check whether the battery ran down please.")
    chain.add_text("Check whether the program ran please.")
    assert sorted(chain._tuple_vocabulary) == [
        ("battery", "ran"),
        ("check", "whether"),
        ("program", "ran"),
        ("ran", "down"),
        ("the", "battery"),
        ("the", "program"),
        ("whether", "the"),
    ]
    assert chain.predict("Check if the battery ran") == {"down": 1.0}


def test_render_markov_chain_as_table():
    chain = MarkovChain(order=1)
    chain.add_text("Show me my directories please.")
    chain.add_text("Show me my files please.")
    chain.add_text("Show me my photos please.")
    rendered_table = chain.render_as_csv()
    assert len(chain._vocabulary) == chain._matrix.size.cols
    assert (
        rendered_table.readline().rstrip()
        == ";show;me;my;directories;please;files;photos"
    )
    assert (
        rendered_table.readline().rstrip()
        == "show;0.0;1.0;0.0;0.0;0.0;0.0;0.0"
    )
    assert (
        rendered_table.readline().rstrip() == "me;0.0;0.0;1.0;0.0;0.0;0.0;0.0"
    )


def test_render_markov_chain_as_dot_graph():
    chain = MarkovChain(order=1)
    chain.add_text("Show me my directories please.")
    chain.add_text("Show me my files please.")
    chain.add_text("Show me my photos please.")
    assert chain.render_as_dot() == (
        "digraph G {\n"
        '    rankdir="LR"\n'
        '    show -> me [label="1.0"]\n'
        '    me -> my [label="1.0"]\n'
        '    my -> directories [label="0.33"]\n'
        '    my -> files [label="0.33"]\n'
        '    my -> photos [label="0.33"]\n'
        '    directories -> please [label="1.0"]\n'
        '    files -> please [label="1.0"]\n'
        '    photos -> please [label="1.0"]\n'
        "}"
    )


def test_render_markov_chain_second_order_as_dot_graph():
    chain = MarkovChain(order=2)
    chain.add_text("Show me my directories please.")
    chain.add_text("Show me my files please.")
    chain.add_text("Show me my photos please.")
    assert chain.render_as_dot() == (
        "digraph G {\n"
        '    rankdir="LR"\n'
        '    show,me -> my [label="1.0"]\n'
        '    me,my -> directories [label="0.33"]\n'
        '    me,my -> files [label="0.33"]\n'
        '    me,my -> photos [label="0.33"]\n'
        '    my,directories -> please [label="1.0"]\n'
        '    my,files -> please [label="1.0"]\n'
        '    my,photos -> please [label="1.0"]\n'
        "}"
    )
