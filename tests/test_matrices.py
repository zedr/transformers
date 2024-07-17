from matrices import dot_product


def test_dot_product():
    result = dot_product(
        [0, 1, 1, 2],
        [1, 0, 1, 2]
    )
    assert list(result) == [0, 0, 1, 4]


def test_dot_product_self_sum():
    vec = [0, 1, 0, 0]
    assert sum(dot_product(vec, vec)) == 1


def test_dot_product_other_sum():
    vec1 = [0, 1, 0, 0]
    vec2 = [0, 0, 1, 0]
    assert sum(dot_product(vec1, vec2)) == 0


def test_dot_product_sum_float():
    vec1 = [0, 0, 1, 0]
    vec2 = [.2, .7, .8, .1]
    assert sum(dot_product(vec1, vec2)) == .8

def test_dot_product_row_and_columns():
    vec1 = [[0], [0], [1], [0]]
    vec2 = []
