import re
import math
from typing import TypeVar, Literal, Sequence, MutableSequence
from collections import OrderedDict

T = TypeVar("T", str, bytes)

LiteralTrue = Literal[True]

words_rxp = re.compile(r"(\w+)+")


def add_one_and_rebalance(seq: MutableSequence[float], idx: int) -> None:
    """Add an element to a list and re-balance the ratio across it"""
    seq[idx] = math.nan
    count = sum(1 for item in seq if item)
    new_value = 1.0 if count == 0 else 1 / count
    for num in range(len(seq)):
        if seq[num]:
            seq[num] = new_value


class OrderedSet:
    def __init__(self, T, initial: Sequence[T] = ()):
        self.__ordered_dict_key_type = T
        self._ns: dict[T, LiteralTrue] = OrderedDict()
        for item in initial:
            self.add(item)

    def add(self, item: T) -> None:
        """Add an item to the set"""
        self._ns[item] = True

    def index(self, item: T) -> int:
        return tuple(self._ns).index(item)

    def __contains__(self, item: T) -> bool:
        return item in self._ns

    def __iter__(self):
        return iter(self._ns)

    def __len__(self):
        return len(self._ns)

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"{self.__ordered_dict_key_type}, "
            f"initial={tuple(self)})"
        )


class MarkovChain:
    """A Python representation of a Markov Chain"""

    def __init__(self):
        self._words = OrderedSet(str)
        self._matrix = []

    @property
    def tokens(self) -> tuple[str]:
        return tuple(self._words)

    def _add_word(self, word: str) -> None:
        """Grow the namespace of the chain by adding a single word to it"""
        self._words.add(word)
        for row in self._matrix:
            row += [0]
        self._matrix.append([0] * len(self._words))

    def _update(self, words: Sequence[T]) -> None:
        """Update this chain with the given sequence of words"""
        for num, word in enumerate(words):
            idx_word = self._words.index(word)
            try:
                next_word = words[idx_word + 1]
            except IndexError:
                pass
            else:
                idx_next_word = self._words.index(next_word)
                row_word = self._matrix[idx_word]
                add_one_and_rebalance(row_word, idx_next_word)

    def add_text(self, text: str) -> "MarkovChain":
        """Parse a sentence and add it to this chain"""
        words = words_rxp.findall(text)
        for word in words:
            self._add_word(word)
        self._update(words)
        return self