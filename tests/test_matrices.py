from matrices import get_size, dot_product, mat_mul, take_columns


def test_get_size():
    """Can get the size of a regular matrix"""
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
