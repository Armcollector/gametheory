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
import operator
import time
import statistics

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
    
    def expected_score(self):
        new_get = self.copy()
        expected_value = statistics.mean(new_get.digits)
        for index in new_get.free_positions():
            new_get = new_get.set_item(index, expected_value)
        return new_get.score()

    def set_item(self, index, value):
        """Set an item and return a copy."""
        new = self.copy()
        i, j = index
        assert min(self.digits) <= value <= max(self.digits)
        assert new.board[i][j] is None
        new.board[i][j] = value
        return new
    
    def set_item_at_column(self, column_j, value):
        """Set an item and return a copy."""
        new = self.copy()
        assert min(self.digits) <= value <= max(self.digits)
        
        for i, row in enumerate(self.board):
            for j, entry in enumerate(row):
                if j == column_j and entry is None:
                    return new.set_item((i, j), value)
                
        assert False, "No entry in column was free"

    def __hash__(self):
        return hash(tuple(tuple(r) for r in self.board))

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

    def __eq__(self, other):
        for attr in ["target", "shape", "board"]:
            if getattr(self, attr) != getattr(other, attr):
                return False

        return True
    
    def to_latex(self):
        out = r'\begin{bmatrix}'
        out += r'\\'.join(' & '.join(self.EMPTY_ENTRY_CHAR if e is None else str(e) for e in row) for row in self.board)
        out += r'\end{bmatrix}'
        return out

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
            score = 0
            for row in self.board:
                for j, entry in enumerate(reversed(row), 0):
                    score += entry * 10**j
            return score - self.target


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


if __name__ == "__main__":

    # =============================================================================
    # ============ THE STRATEGY IS FIXED, TRY EVERY POSSIBLE DICE ROLL ============
    # =============================================================================

    import functools

    @functools.lru_cache(maxsize=2 ** 26)
    def get_best_placement(get, digit, ell=1, level=0):
        """Returns (index, optimal_score).

        ell = expected l norm used in computation
                E[|x_i|**(ell)]**(1/ell)



        """
        if level > 3:
            return None, get.expected_score()
        
        

        free_positions = list(get.free_positions())

        # CASE: Bottom of recursion - there is only one placement available
        if len(free_positions) == 1:
            index = free_positions[0]
            new_get = get.set_item(index, digit)
            return index, new_get.score()

        # CASE: There are many placements available, we'll try each of them

        results = {}
        for index in get.canonical_free_positions():

            # Set this digit in the position
            new_get = get.set_item(index, digit)

            # How many digits should we try on recursion?
            # TODO: Better logic here, or try everything

            digits_try_to = random.choices(new_get.digits, k=max(5 - level, 1))
            digits_try_to = list(new_get.digits)

            prob = 1 / len(digits_try_to)

            indices_scores = list(
                get_best_placement(new_get, digit, ell=ell, level=level + 1) for digit in digits_try_to
            )
            scores = [score for (index, score) in indices_scores]

            # Here we take E[|x_i|**(ell)]**(1/ell) - an expected value l norm
            # The case ell = 1 corresponds to the expected value
            # As ell -> infty this corresponds to minimizing the maximum deviation from the target
            # Should be computed in log space for better numerical stability
            results[index] = (sum(abs(s) ** ell * prob for s in scores)) ** (1 / ell)

        return min(results.items(), key=operator.itemgetter(1))

    # Create some random games
    for seed in range(1):

        st = time.time()

        random.seed(seed)
        get = Get.get1000()  # Get.get1000()
        nature = [random.choice(get.digits) for _ in range(get.shape[0] * get.shape[1])]

        for digit in nature:
            opt_index, opt_score = get_best_placement(get, digit, ell=1)
            # print(f"Got digit {digit}. Placing it in {index} for expected score: {opt_score}")
            get = get.set_item(opt_index, digit)

            # print(get)

        print(f"Final score: {get.score()}")

        get = Get.get1000()  # Get.get1000()
        for digit in nature:

            #opt_index, opt_score = get_best_placement(get, digit, ell=2)
            # print(f"Got digit {digit}. Placing it in {index} for expected score: {opt_score}")
            #get = get.set_item(opt_index, digit)
            
            opt_index = random.choice(list(get.free_positions()))
            get = get.set_item(opt_index, digit)

            # print(get)

        print(f"Final score: {get.score()}")
        print(f"Time spent: {time.time() - st}")

        print()
        
        
        
if __name__ == "__main__":


    # =============================================================================
    # ============ THE STRATEGY IS FIXED, TRY EVERY POSSIBLE DICE ROLL ============
    # =============================================================================

    import functools

    def get_best_placement(get, digit:int, level=0):
        """Given a GET instance and a digit, returns (best_index, expected_score)."""
        assert digit in get.digits
    
        free_positions = list(get.free_positions())
    
        # CASE 1: Bottom of recursion - there is only one placement available.
        if len(free_positions) == 1:
            index = free_positions[0]
            return index, get.set_item(index, digit).score()
    
        # CASE: There are many placements available, try each of them.
        results = {}
        for index in get.canonical_free_positions():
    
            # Set this digit in the free position
            new_get = get.set_item(index, digit)
    
            # Given that the digit was placed in the position and nature yields
            # a new digit, what are the expected scores we can get?
            indices_and_scores = (
                get_best_placement(new_get, digit, level=level + 1) for digit in new_get.digits
            )
            scores = (score for (index, score) in indices_and_scores)
    
            # Here we take E[|x_i|**(ell)]**(1/ell) - an expected value l norm
            # The case ell = 1 corresponds to the expected value
            # The case ell -> infty corresponds to minimax
            ell = 1 # Set to ell >= 1
            prob = 1 / len(new_get.digits)
            results[index] = (sum(abs(s) ** ell * prob for s in scores)) ** (1 / ell)
    
        return min(results.items(), key=operator.itemgetter(1))
    
    algo_scores, random_scores = [], []
    import collections
    results = collections.defaultdict(int)

    # Create some random games
    import tqdm
    
    for seed in tqdm.tqdm(range(1000)):

        st = time.time()

        random.seed(seed)
        get = Get.get42()  # Get.get1000()
        nature = [random.choice(get.digits) for _ in range(get.shape[0] * get.shape[1])]

        for digit in nature:
            opt_index, opt_score = get_best_placement(get, digit)
            get = get.set_item(opt_index, digit)

        # print(f"Final score using algo: {get.score()}")
        algo_scores.append(get.score() + get.target)
        
        random.seed(seed)
        get2 = Get.get42()  # Get.get1000()
        nature = [random.choice(get.digits) for _ in range(get.shape[0] * get.shape[1])]

        for digit in nature:
            opt_index = random.choice(list(get2.free_positions()))
            get2 = get2.set_item(opt_index, digit)

        # print(f"Final score using random: {get.score()}")
        random_scores.append(get2.score() + get.target)
        
        if abs(get.score()) < abs(get2.score()):
            results["algo"] += 1
        elif abs(get.score()) > abs(get2.score()):
            results["random"] += 1
        else:
            results["draw"] += 1
        
        
    import matplotlib.pyplot as plt
    import collections
    import numpy as np
    
    data_to_plot = collections.Counter(algo_scores)
    x = np.arange(0, max(data_to_plot.keys()) + 1)
    y = np.zeros_like(x)
    for i, j in data_to_plot.items():
        y[i] = j
        
    plt.bar(x, y, label="Algorithm", alpha=0.5, zorder=15)
    
    
    data_to_plot = collections.Counter(random_scores)
    x = np.arange(0, max(data_to_plot.keys()) + 1)
    y = np.zeros_like(x)
    for i, j in data_to_plot.items():
        y[i] = j
        
    plt.bar(x, y, label="Random", alpha=0.5, zorder=15)
    
    plt.axvline(x=42, label="42", color="red", ls="--", zorder=25)

    plt.legend()
    plt.show()
    
    
    
if __name__ == "__main__":


    # =============================================================================
    # ============ THE STRATEGY IS FIXED, TRY EVERY POSSIBLE DICE ROLL ============
    # =============================================================================

    import functools

    def get_best_placement(get, digit:int, level=0, ell=1):
        """Given a GET instance and a digit, returns (best_index, expected_score)."""
        assert digit in get.digits
    
        free_positions = list(get.free_positions())
    
        # CASE 1: Bottom of recursion - there is only one placement available.
        if len(free_positions) == 1:
            index = free_positions[0]
            return index, get.set_item(index, digit).score()
    
        # CASE: There are many placements available, try each of them.
        results = {}
        for index in get.canonical_free_positions():
    
            # Set this digit in the free position
            new_get = get.set_item(index, digit)
    
            # Given that the digit was placed in the position and nature yields
            # a new digit, what are the expected scores we can get?
            indices_and_scores = (
                get_best_placement(new_get, digit, level=level + 1, ell=ell) for digit in new_get.digits
            )
            scores = (score for (index, score) in indices_and_scores)
    
            # Here we take E[|x_i|**(ell)]**(1/ell) - an expected value l norm
            # The case ell = 1 corresponds to the expected value
            # The case ell -> infty corresponds to minimax
            prob = 1 / len(new_get.digits)
            results[index] = (sum(abs(s) ** ell * prob for s in scores)) ** (1 / ell)
    
        return min(results.items(), key=operator.itemgetter(1))
    
    algo_scores, random_scores = [], []
    import collections
    results = collections.defaultdict(int)

    # Create some random games
    import tqdm
    
    for seed in tqdm.tqdm(range(10000)):

        st = time.time()

        random.seed(seed)
        get = Get.get42()  # Get.get1000()
        nature = [random.choice(get.digits) for _ in range(get.shape[0] * get.shape[1])]

        for digit in nature:
            opt_index, opt_score = get_best_placement(get, digit, ell=1)
            get = get.set_item(opt_index, digit)

        # print(f"Final score using algo: {get.score()}")
        algo_scores.append(get.score() + get.target)
        
        random.seed(seed)
        get2 = Get.get42()  # Get.get1000()
        nature = [random.choice(get.digits) for _ in range(get.shape[0] * get.shape[1])]

        for digit in nature:
            opt_index, opt_score = get_best_placement(get2, digit, ell=10)
            get2 = get2.set_item(opt_index, digit)

        # print(f"Final score using random: {get.score()}")
        random_scores.append(get2.score() + get.target)
        
        if abs(get.score()) < abs(get2.score()):
            results["algo"] += 1
        elif abs(get.score()) > abs(get2.score()):
            results["random"] += 1
        else:
            results["draw"] += 1
        
        
    import matplotlib.pyplot as plt
    import collections
    import numpy as np
    
    data_to_plot = collections.Counter(algo_scores)
    x = np.arange(0, max(data_to_plot.keys()) + 1)
    y = np.zeros_like(x)
    for i, j in data_to_plot.items():
        y[i] = j
        
    plt.bar(x, y, label="ell = 1", alpha=0.5, zorder=15)
    
    
    data_to_plot = collections.Counter(random_scores)
    x = np.arange(0, max(data_to_plot.keys()) + 1)
    y = np.zeros_like(x)
    for i, j in data_to_plot.items():
        y[i] = j
        
    plt.bar(x, y, label="high ELL", alpha=0.5, zorder=15)
    
    plt.axvline(x=42, label="42", color="red", ls="--", zorder=25)

    plt.legend()
    plt.show()

