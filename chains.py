import os
import re
import csv
from io import StringIO
from typing import Literal, Sequence, Iterator, Hashable
from collections import OrderedDict

from matrices import Size

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
        self.__n_cols: int = 0

    def __iter__(self):
        return iter(self._rows)

    def __getitem__(self, item) -> WeightedList:
        return self._rows[item]

    def grow(self, /, n_cols=0, n_rows=0) -> None:
        self.__n_cols += n_cols
        for row in self._rows:
            row.expand(n_cols)
        for _ in range(n_rows):
            self._rows.append(WeightedList().expand(self.__n_cols))

    def as_lists(self):
        return [list(row) for row in self]

    @property
    def size(self) -> Size:
        return Size(len(self._rows), self.__n_cols)


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
        try:
            return tuple(self._ns).index(item)
        except ValueError:
            raise IndexError(f"{item}")

    def __getitem__(self, idx) -> str:
        return tuple(self._ns)[idx]

    def __contains__(self, item: T) -> bool:
        return item in self._ns

    def __iter__(self):
        return iter(self._ns)

    def __len__(self):
        return len(self._ns)

    def __repr__(self):
        return f"{type(self).__name__}(" f"initial={tuple(self)}" f")"


class MarkovChain[T: Hashable]:
    """A Python representation of a Markov Chain"""

    def __init__(self, order: int = 1):
        self._vocabulary: OrderedSet[str] = OrderedSet()
        self._tuple_vocabulary: OrderedSet[tuple[T, ...]] = OrderedSet()
        self._matrix: WeightedMatrix = WeightedMatrix()
        self._order = order

    def add_text(self, text: str) -> None:
        """Add text to the vocabulary"""
        words = [word.lower() for word in words_rxp.findall(text)]
        for idx, word in enumerate(words):
            if self._vocabulary.add(word):
                self._matrix.grow(n_cols=1)
            tup = tuple(words[idx - self._order : idx])
            if len(tup) == self._order:
                if self._tuple_vocabulary.add(tup):
                    self._matrix.grow(n_rows=1)
                self._matrix[self._tuple_vocabulary.index(tup)].inc(
                    self._vocabulary.index(word)
                )

    def render_as_csv(self) -> StringIO:
        fd = StringIO()
        writer = csv.writer(fd, delimiter=";")
        writer.writerow([""] + list(self._vocabulary))
        for idx, tup in enumerate(self._tuple_vocabulary):
            writer.writerow([",".join(tup)] + list(self._matrix[idx]))
        fd.seek(0)
        return fd

    def render_as_dot(self) -> str:
        lines = [
            "digraph G {",
            '    rankdir="LR"',
        ]
        for idx, row in enumerate(self._matrix):
            tup = self._tuple_vocabulary[idx]
            tup_s = ",".join(tup)
            for rdx, weight in enumerate(row):
                if weight > 0.0:
                    word = self._vocabulary[rdx]
                    lines.append(
                        f'    {tup_s} -> {word} [label="{weight:.2}"]'
                    )
        lines += "}"
        return os.linesep.join(lines)

    def predict(self, text: str) -> dict[str, float]:
        """Predict the next word from the given text"""
        prediction = {}
        words = words_rxp.findall(text)
        if words:
            length = len(words)
            tup = tuple(words[length - self._order: length])
            try:
                row = self._matrix.as_lists()[self._tuple_vocabulary.index(tup)]
            except IndexError:
                pass
            else:

                for word, probability in zip(self._vocabulary, row):
                    if probability > 0.0:
                        prediction[word] = probability
        return prediction



