from typing import Sequence, Any, NamedTuple, TypeVar, Generator


class Size(NamedTuple):
    rows: int
    cols: int


def get_size(matrix: Sequence[Sequence[Any]]) -> Size:
    """Get the size of a matrix"""
    rows = len(matrix)
    lengths = [len(row) for row in matrix]
    if all(True for ln in lengths if ln == (cols := lengths[0])):
        return Size(rows, cols)
    else:
        raise ValueError("Jagged matrices are not supported")


T = TypeVar("T", float, int)


def dot_product(
    v1: Sequence[int | float],
    v2: Sequence[int | float],
) -> float:
    """Dot product for two vectors"""
    acc = 0.0
    for item1, item2 in zip(v1, v2):
        acc += item1 * item2
    return acc


def take_columns(matrix: Sequence[Sequence[T]]) -> Generator[list[T], None, None]:
    n_cols = get_size(matrix).cols
    for idx in range(n_cols):
        yield [row[idx] for row in matrix]


def mat_mul(
    m1: Sequence[Sequence[int | float]],
    m2: Sequence[Sequence[int | float]],
) -> list[list[float]]:
    rows = []
    for v1 in m1:
        cols = []
        for v2 in take_columns(m2):
            cols.append(dot_product(v1, v2))
        rows.append(cols)

    return rows
