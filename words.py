import os
import re
from typing import TypeVar, Literal, Sequence, Optional, Iterator
from collections import OrderedDict

from encoder import OneHotEncoding
from matrices import mat_mul

T = TypeVar("T", str, bytes)

LiteralTrue = Literal[True]

words_rxp = re.compile(r"(\w+)+")


class WeightedList:
    def __init__(self, initial: Optional[Sequence[int]] = None):
        self._seq = [] if initial is None else list(initial)

    def expand(self, size: int = 1) -> "WeightedList":
        self._seq += [0] * size
        return self

    def inc(self, index: int) -> "WeightedList":
        self._seq[index] += 1
        return self

    def __iter__(self) -> Iterator[float]:
        tot_weight = sum(self._seq)
        return iter(val / tot_weight if tot_weight else 0.0 for val in self._seq)

    def __repr__(self):
        return f"{type(self).__name__}(initial={self._seq})"


class OrderedSet:
    def __init__(self, T, initial: Sequence[T] = ()):
        self.__ordered_dict_key_type = T
        self._ns: dict[T, LiteralTrue] = OrderedDict()
        for item in initial:
            self.add(item)

    def add(self, item: T) -> bool:
        """Add an item to the set

        If the item was already present, return False
        If the item wasn't present, return True
        """
        if item in self._ns:
            return False
        else:
            self._ns[item] = True
            return True

    def index(self, item: T) -> int:
        return tuple(self._ns).index(item)

    def __getitem__(self, item) -> str:
        return tuple(self._ns)[item]

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
        self._matrix: list[WeightedList] = []

    def as_matrix(self) -> list[list[float]]:
        return [list(row) for row in self._matrix]

    @property
    def tokens(self) -> tuple[str]:
        return tuple(self._words)

    def _add_word(self, word: str) -> None:
        """Grow the namespace of the chain by adding a single word to it"""
        added = self._words.add(word)
        if added:
            for row in self._matrix:
                row.expand()
            self._matrix.append(WeightedList().expand(size=len(self._words)))

    def _update(self, words: Sequence[T]) -> None:
        """Update this chain with the given sequence of words"""
        for num, word in enumerate(words):
            idx_word = self._words.index(word)
            try:
                next_word = words[num + 1]
            except IndexError:
                pass
            else:
                idx_next_word = self._words.index(next_word)
                row_word = self._matrix[idx_word]
                row_word.inc(idx_next_word)

    def add_text(self, text: str) -> "MarkovChain":
        """Parse a sentence and add it to this chain"""
        words = words_rxp.findall(text)
        for word in words:
            self._add_word(word)
        self._update(words)
        return self

    def render_as_dot(self) -> str:
        """Render this Markov chain as a DOT graph"""
        lines = ["digraph G {", '    rankdir="LR";']
        for idx in range(len(self._matrix)):
            word = self._words[idx]
            row = self._matrix[idx]
            for rdx, value in enumerate(row):
                if value:
                    next_word = self._words[rdx]
                    lines.append(f'    {word} -> {next_word} [label="{value}"];')

        lines += "}"
        return os.linesep.join(lines)

    def encode(self, word: T) -> OneHotEncoding:
        """Get the one hot eoding of the given word in this namespace"""
        try:
            idx = self.tokens.index(word)
        except ValueError:
            raise ValueError(f"Word not in namespace: '{word}'")
        else:
            return OneHotEncoding(len(self.tokens), {idx: str(word)})

    def get_distribution(self, word: T) -> list[float]:
        """Get the one hot encoded probability distribution for a word"""
        encoded_word = list(self.encode(word))
        result = mat_mul([encoded_word], self.as_matrix())
        return result[0]

    def get_transitions(self, word: T) -> dict[str, float]:
        """Get all the probable transitions for a word"""
        dist = self.get_distribution(word)
        return {key: val for key, val in zip(self.tokens, dist) if val}
