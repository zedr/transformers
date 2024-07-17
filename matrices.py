from typing import Iterable, Generator, Union

IntOrFloat = Union[int, float]


def dot_product(
        a: Iterable[int | float],
        b: Iterable[int | float]
) -> Generator[float, None, None]:
    """Dot product for two matrices given as iterables"""
    for num1, num2 in zip(a, b):
        yield num1 * num2
