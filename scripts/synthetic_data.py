"""Deterministic synthetic username and query generation."""

import random
from dataclasses import dataclass
from typing import Iterator, List, Sequence, Tuple


@dataclass
class QueryWorkload:
    """A deterministic ordered list of queries and their expected membership."""

    queries: List[Tuple[str, bool]]
    positive_count: int
    negative_count: int


class UsernameView(Sequence[str]):
    """Provide indexed generated usernames without storing the whole dataset."""

    def __init__(self, count: int, seed: int = 520) -> None:
        """Create a view of `count` unique usernames in the seed's namespace.

        Args:
            count: Number of usernames; may be zero.
            seed: Non-negative namespace identifier.
        """
        if count < 0 or seed < 0:
            raise ValueError("count and seed must be non-negative")
        self.count = count
        self.seed = seed

    def __len__(self) -> int:
        """Return the number of available usernames."""
        return self.count

    def __getitem__(self, index: int) -> str:
        """Return the username at an integer index, raising IndexError if invalid."""
        if not isinstance(index, int):
            raise TypeError("username index must be an integer")
        if index < 0:
            index += self.count
        if index < 0 or index >= self.count:
            raise IndexError("username index out of range")
        return "user_{:08d}_{:012d}".format(self.seed, index)


def generate_usernames(count: int, seed: int = 520) -> Iterator[str]:
    """Yield unique fixed-format usernames without retaining the full dataset.

    Args:
        count: Number of usernames to generate; may be zero.
        seed: Integer included in the generated namespace.

    Returns:
        An iterator yielding `count` usernames in stable order.
    """
    yield from UsernameView(count, seed)


def make_query_workload(
    items: Sequence[str],
    query_count: int,
    seed: int = 520,
    positive_fraction: float = 0.5,
) -> QueryWorkload:
    """Create a shuffled query mix with known positives and unique negatives.

    Args:
        items: Existing usernames used to choose positive queries.
        query_count: Total number of queries to generate.
        seed: Seed controlling positive selection, negative IDs, and query order.
        positive_fraction: Fraction of queries that should be successful.

    Returns:
        A workload containing `(username, expected_present)` pairs.
    """
    if query_count < 0:
        raise ValueError("query_count must be non-negative")
    if not 0 <= positive_fraction <= 1:
        raise ValueError("positive_fraction must be between 0 and 1")
    positive_count = round(query_count * positive_fraction)
    negative_count = query_count - positive_count
    if positive_count and not items:
        raise ValueError("positive queries require at least one existing item")

    rng = random.Random(seed)
    queries: List[Tuple[str, bool]] = []
    for _ in range(positive_count):
        queries.append((items[rng.randrange(len(items))], True))
    for index in rng.sample(range(10**12), negative_count):
        queries.append(("absent_{:08d}_{:012d}".format(seed, index), False))
    rng.shuffle(queries)
    return QueryWorkload(queries, positive_count, negative_count)
