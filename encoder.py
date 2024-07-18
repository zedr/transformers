from typing import Iterable, Generator
from functools import cached_property


def _make_uniq(words: Iterable[str]) -> list[str]:
    seq = []
    for word in words:
        if word not in seq:
            seq.append(word)
        else:
            raise ValueError(f"Repeated word: {word}")
    return seq


class OneHotEncoding:
    def __init__(self, size: int, words_ns: dict[int, str]):
        if max_ := max(words_ns) > size:
            raise ValueError(f"Out of bounds: {size} < {max_}")
        if min(words_ns) < 0:
            raise ValueError("Words cannot have a negative encoding")

        self._size = size
        self._words_ns = words_ns

    def __len__(self) -> int:
        return len(self._words_ns)

    def __iter__(self):
        return iter(self.as_seq())

    def as_seq(self) -> Generator[int, None, None]:
        for idx in range(self._size):
            yield 1 if self._words_ns.get(idx) else 0


class OneHotEncoder:
    """One hot encoder using a list of given words"""

    def __init__(self, words: Iterable[str]):
        self._words = _make_uniq(words)

    @cached_property
    def length(self):
        return len(self._words)

    def get_index(self, word: str) -> int:
        return self._words.index(word)

    def get_word(self, idx: int) -> str:
        return self._words[idx]

    def encode(self, *words: str) -> Generator[OneHotEncoding, None, None]:
        for word in words:
            yield OneHotEncoding(self.length, {self.get_index(word): word})

    @classmethod
    def from_file(cls, path: str) -> "OneHotEncoder":
        with open(path) as fd:
            return cls(line.strip() for line in fd)
