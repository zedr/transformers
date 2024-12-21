import os
import re
from typing import Literal, Sequence, Iterator, Generator, Hashable
from collections import OrderedDict

from encoder import OneHotEncoding
from matrices import mat_mul, get_size, Size

LiteralTrue = Literal[True]

words_rxp = re.compile(r"(\w+)+")


class WeightedList:
    def __init__(self):
        self._seq = []

    def expand(self, size: int = 1) -> "WeightedList":
        self._seq += [0] * size
        return self

    def inc(self, index: int) -> "WeightedList":
        self._seq[index] += 1
        return self

    def __iter__(self) -> Iterator[float]:
        tot_weight = sum(self._seq)
        return iter(
            val / tot_weight if tot_weight else 0.0 for val in self._seq
        )


class WeightedMatrix:
    def __init__(self):
        self._rows: list[WeightedList] = []

    def __iter__(self):
        return iter(self._rows)

    def __getitem__(self, item) -> WeightedList:
        return self._rows[item]

    def grow(self, /, n_cols=0, n_rows=0) -> None:
        for _ in range(n_rows):
            self._rows.append(WeightedList().expand(self.size.cols))
        if n_cols and not self._rows:
            raise ValueError("Cannot grow a matrix without rows")
        for row in self._rows:
            row.expand(n_cols)

    def as_lists(self):
        return [list(row) for row in self]

    @property
    def size(self) -> Size:
        return get_size(self.as_lists())


class OrderedSet[T: Hashable]:
    def __init__(self, initial: Sequence[T] = ()):
        self._ns = OrderedDict()
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

    def __getitem__(self, idx) -> str:
        return tuple(self._ns)[idx]

    def __contains__(self, item: T) -> bool:
        return item in self._ns

    def __iter__(self):
        return iter(self._ns)

    def __len__(self):
        return len(self._ns)

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"initial={tuple(self)}"
            f")"
        )


class MarkovChain[T: Hashable]:
    """A Python representation of a Markov Chain"""

    def __init__(self, order: int = 1):
        self._tokens: OrderedSet[T] = OrderedSet()
        self._matrix = WeightedMatrix()
        self._order = order

    def as_lists(self) -> list[list[float]]:
        return self._matrix.as_lists()

    @property
    def tokens(self) -> tuple[T, ...]:
        return tuple(self._tokens)

    def _add_token(self, token: T) -> None:
        """Grow the namespace of the chain by adding a single token to it"""
        added = self._tokens.add(token)
        if added:
            self._matrix.grow(n_cols=1, n_rows=1)

    def _update(self, tokens: Sequence[T]) -> None:
        """Update this chain with the given sequence of words"""
        for num, token in enumerate(tokens):
            idx_token = self._tokens.index(token)
            try:
                next_token = tokens[num + 1]
            except IndexError:
                pass
            else:
                idx_next_token = self._tokens.index(next_token)
                row_token = self._matrix[idx_token]
                row_token.inc(idx_next_token)

    def _get_words(self, text: str) -> Generator[str, None, None]:
        """Get a sequence of words from the given text"""
        return (word for word in words_rxp.findall(text))

    def add_text(self, text: str) -> "MarkovChain":
        """Parse a sentence and add it to this chain"""
        words = list(self._get_words(text))
        for word in words:
            self._add_token(word)
        self._update(words)
        return self

    def render_as_dot(self) -> str:
        """Render this Markov chain as a DOT graph"""
        lines = ["digraph G {", '    rankdir="LR";']
        for idx in range(self._matrix.size.rows):
            word = self._tokens[idx]
            row = self._matrix[idx]
            for rdx, value in enumerate(row):
                if value:
                    next_token = self._tokens[rdx]
                    lines.append(
                        f'    {word} -> {next_token} [label="{value}"];'
                    )
        lines += "}"
        return os.linesep.join(lines)

    def encode(self, token: T) -> OneHotEncoding:
        """Get the one hot encoding of the given token in this namespace"""
        try:
            idx = self.tokens.index(token)
        except ValueError:
            raise ValueError(f"Token not in namespace: '{token}'")
        else:
            return OneHotEncoding(len(self.tokens), {idx: str(token)})

    def get_distribution(self, token: T) -> list[float]:
        """Get the one hot encoded probability distribution for a token"""
        encoded_token = list(self.encode(token))
        result = mat_mul([encoded_token], self._matrix.as_lists())
        return result[0]

    def get_transitions(self, token: T) -> dict[str, float]:
        """Get all the probable transitions for a token"""
        dist = self.get_distribution(token)
        return {key: val for key, val in zip(self.tokens, dist) if val}

    def predict(self, text: str) -> dict[str, float]:
        words = words_rxp.findall(text)
        return self.get_transitions(words[-1])
