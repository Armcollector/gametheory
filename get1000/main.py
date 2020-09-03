#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 16:02:48 2020

@author: tommy
"""


import random
import time
import itertools
import copy
import tabulate
import random
import sys

sys.setrecursionlimit(2 ** 30)


class Get:

    EMPTY_ENTRY_CHAR = "x"

    @classmethod
    def get42(cls):
        return cls(shape=(2, 2), target=42, digits=(1, 2, 3, 4))

    @classmethod
    def get1000(cls):
        return cls(shape=(3, 3), target=1000, digits=(1, 2, 3, 4, 5, 6))

    def __init__(self, shape=(3, 3), target=1000, digits=(1, 2, 3, 4, 5, 6)):
        """Create a new (immutable) GET instance."""
        nrows, ncols = shape
        self.target = target
        self.digits = list(set(digits))
        self.shape = nrows, ncols
        self.board = [[None for _ in range(ncols)] for _ in range(nrows)]
        assert len(self.board) == nrows
        assert len(self.board[0]) == ncols

    def copy(self):
        new = type(self)(shape=self.shape, target=self.target, digits=self.digits)
        new.board = copy.deepcopy(self.board)
        return new

    def set_item(self, index, value):
        """Set an item and return a copy."""
        new = self.copy()
        i, j = index
        assert value in self.digits
        assert new.board[i][j] is None
        new.board[i][j] = value
        return new

    def num_free_positions(self):
        return len(list(self.free_positions()))

    def free_positions(self):
        """Yield all free positions as tuples (i, j), canonical or not."""
        for i, row in enumerate(self.board):
            for j, entry in enumerate(row):
                if entry is None:
                    yield (i, j)

    def canonical_free_positions(self):
        """Yields tuple of indices (i, j) of canonical free positions.

        A free position is canonical if it's the first free entry in a column."""
        rows, cols = self.shape

        for j in range(cols):
            column_entries = [self.board[i][j] for i in range(rows)]

            for i, entry in enumerate(column_entries):
                if entry is None:
                    yield (i, j)
                    break  # Go to the next column

    def __getitem__(self, index):
        i, j = index
        return self.board[i][j]

    def __repr__(self):
        to_print = []
        for row in self.board:
            print_row = []
            for entry in row:
                new_entry = self.EMPTY_ENTRY_CHAR if entry is None else entry
                print_row.append(new_entry)
            to_print.append(print_row)
        return tabulate.tabulate(to_print)

    def score(self):
        """Return the (signed) distance from the target value."""
        if list(self.free_positions()):
            return float("inf")
        else:
            row_values = (int("".join(str(e) for e in r)) for r in self.board)
            return sum(row_values) - self.target


if __name__ == "__main__":

    # =============================================================================
    # ============ NATURE IS FIXED, TRY EVERY POSSIBLE STRATEGY ===================
    # =============================================================================

    def search_nature_fixed(get, nature: list, level=0, verbose=True):
        """Given a fixed sequence from nature, execute all possible strategies."""
        if verbose:
            print("  " * level, f"level = {level}")
            print(get)

        # The canonical indices correspond to strategies at this level
        indices = list(get.canonical_free_positions())
        if indices:
            digit = nature[level]  # like random.choice(dice)

            # Go through every canonical placement, place the digit and recurse
            for index in indices:
                new_get = get.set_item(index, digit)
                yield from search_nature_fixed(new_get, nature, level=level + 1, verbose=verbose)

        # The GET instance has no free positions. We bottomed out, so yield.
        else:
            yield get

    random.seed(123)
    get = Get.get42()  # Get.get1000()
    nature = [random.choice(get.digits) for _ in range(get.shape[0] * get.shape[1])]

    ans = list(search_nature_fixed(get, nature, verbose=False))

    for get_i in ans:
        print(get_i)
        print(get_i.score())
        print()

    print(f"Nature: {nature}")
    print(f"Strategies: {len(ans)}")


if __name__ == "__main__":

    # =============================================================================
    # ============ THE STRATEGY IS FIXED, TRY EVERY POSSIBLE DICE ROLL ============
    # =============================================================================

    def search_strategy_fixed(get, nature: list, level=0, verbose=True):
        """Given a fixed strategy, look at all possible outcomes."""
        if verbose:
            print("  " * level, f"level = {level}")
            print(get)

        if list(get.canonical_free_positions()):

            # Every outcome from the dice roll
            for digit in get.digits:

                # This is the strategy: top left always!
                index = next(get.canonical_free_positions())

                new_get = get.set_item(index, digit)
                yield from search_strategy_fixed(new_get, nature, level=level + 1, verbose=verbose)

        # The GET instance has no free positions. We bottomed out, so yield.
        else:
            yield get

    get = Get.get42()  # Get.get1000()

    ans = list(search_strategy_fixed(get, nature, verbose=False))

    for get_i in ans:
        print(get_i)
        print(get_i.score())
        print()

    entries = get.shape[0] * get.shape[0]
    print(f"States of nature: {len(get.digits) ** entries}")
