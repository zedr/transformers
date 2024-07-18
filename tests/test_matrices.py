from matrices import get_size, dot_product, mat_mul, columns


def test_get_size():
    """Can get the size of a regular matrix"""
    assert get_size([[1, 2, 3]]) == (1, 3)
    assert get_size([[1],
                     [2],
                     [3]]) == (3, 1)


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
    mat = [[.2, .9],
          [.7, .0],
          [.8, .3]]
    result = list(columns(mat))
    assert [[.2, .7, .8], [.9, .0, .3]] == result



def test_mat_mul():
    m1 = [[0, 0, 1, 0]]
    m2 = [[.2],
          [.7],
          [.8],
          [.1]]
    assert list(mat_mul(m1, m2)) == [[.8]]


def test_mat_mul_2a():
    m1 = [[1, 0, 0, 0],
          [0, 0, 1, 0]]
    m2 = [[.2],
          [.7],
          [.8],
          [.1]]
    assert list(mat_mul(m1, m2)) == [[.2], [.8]]


def test_mat_mul_2b():
    m1 = [[0, 0, 1, 0]]
    m2 = [[.2, .9],
          [.7, .0],
          [.8, .3],
          [.1, .4]]
    assert list(mat_mul(m1, m2)) == [[.8, .3]]
