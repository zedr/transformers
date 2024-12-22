from matrices import get_size, dot_product, mat_mul, take_columns
from chains import WeightedList, WeightedMatrix, words_rxp, OrderedSet


def test_get_size():
    """Can get the size of a regular matrix"""
    assert get_size([]) == (0, 0)
    assert get_size([[1, 2, 3]]) == (1, 3)
    assert get_size([[1], [2], [3]]) == (3, 1)


def test_dot_product_simple():
    v1 = [0, 1, 1, 2]
    v2 = [1, 0, 1, 2]
    assert dot_product(v1, v2) == 5.0


def test_dot_product_self():
    v1 = [0, 1, 0, 0]
    assert dot_product(v1, v1) == 1.0


def test_dot_product_other():
    v1 = [0, 1, 0, 0]
    v2 = [0, 0, 1, 0]
    assert dot_product(v1, v2) == 0.0


def test_columns_generator():
    mat = [[0.2, 0.9], [0.7, 0.0], [0.8, 0.3]]
    result = list(take_columns(mat))
    assert [[0.2, 0.7, 0.8], [0.9, 0.0, 0.3]] == result


def test_mat_mul():
    m1 = [[0, 0, 1, 0]]
    m2 = [[0.2], [0.7], [0.8], [0.1]]
    assert mat_mul(m1, m2) == [[0.8]]


def test_mat_mul_2():
    m1 = [[1, 0, 0, 0], [0, 0, 1, 0]]
    m2 = [[0.2], [0.7], [0.8], [0.1]]
    assert mat_mul(m1, m2) == [[0.2], [0.8]]


def test_mat_mul_3():
    m1 = [[0, 0, 1, 0]]
    m2 = [[0.2, 0.9], [0.7, 0.0], [0.8, 0.3], [0.1, 0.4]]
    assert mat_mul(m1, m2) == [[0.8, 0.3]]


def test_mat_mul_4():
    m1 = [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
    m2 = [[0.2, 0.9], [0.7, 0.0], [0.8, 0.3], [0.1, 0.4]]
    assert mat_mul(m1, m2) == [[0.2, 0.9], [0.1, 0.4], [0.8, 0.3]]


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


def test_matrix():
    # At least a row and a col needed for values
    matrix = WeightedMatrix()
    assert list(matrix.as_lists()) == []
    matrix.grow(n_rows=1)
    assert list(matrix.as_lists()) == [[]]
    matrix.grow(n_cols=1)
    assert list(matrix.as_lists()) == [[0.0]]

    matrix = WeightedMatrix()
    matrix.grow(n_rows=1)
    assert list(matrix.as_lists()) == [[]]
    matrix.grow(n_cols=1)
    assert list(matrix.as_lists()) == [[0.0]]

    matrix = WeightedMatrix()
    matrix.grow(n_rows=1, n_cols=1)
    assert list(matrix.as_lists()) == [[0.0]]

    matrix = WeightedMatrix()
    matrix.grow(n_rows=1)
    matrix.grow(n_cols=1)
    matrix.grow(n_cols=2)
    assert list(matrix.as_lists()) == [[0.0, 0.0, 0.0]]

    matrix = WeightedMatrix()
    matrix.grow(n_cols=1)
    matrix.grow(n_cols=2)
    matrix.grow(n_rows=3)
    assert list(matrix.as_lists()) == [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]


def test_words_rxp():
    txt = "show me my files"
    assert ["show", "me", "my", "files"] == words_rxp.findall(txt)


def test_ordered_set():
    s = OrderedSet(initial=("one",))
    s.add("two")
    s.add("two")
    s.add("three")
    assert list(s) == ["one", "two", "three"]
    assert "one" in s
    assert s.index("one") == 0
    assert s[0] == "one"
