"""A hash table using chaining."""

import math

from .hashing import stable_hash


class HashTable:
    """A resizable hash table using a list for each occupied bucket."""

    def __init__(self, initial_capacity: int = 8, max_load_factor: float = 0.75) -> None:
        """Create an empty table with a capacity and load threshold.

        Args:
            initial_capacity: Initial number of bucket slots.
            max_load_factor: Resizing load factor.
        """
        if initial_capacity < 1:
            raise ValueError("initial_capacity must be at least 1")
        if max_load_factor <= 0:
            raise ValueError("max_load_factor must be > 0")
        self._initial_capacity = initial_capacity
        self._max_load_factor = max_load_factor
        self._buckets = [None] * initial_capacity
        self.size = 0

    def build(self, items: list[str]) -> None:
        """Reset the table and insert each username from `items`.

        Args:
            items: Usernames to insert.
        """
        self._buckets = [None] * self._initial_capacity
        self.size = 0
        for item in items:
            self.insert(item)

    def insert(self, item: str) -> bool:
        """Insert `item` unless an equal username is already present.

        Args:
            item: Username to insert.

        Returns:
            True if a new entry was added, otherwise False.
        """
        bucket_index = self._bucket_index(item, len(self._buckets))
        bucket = self._buckets[bucket_index]
        if bucket is not None:
            for stored_item in bucket:
                if stored_item == item:
                    return False
        else:
            bucket = []
            self._buckets[bucket_index] = bucket

        bucket.append(item)
        self.size += 1
        if self.load_factor > self._max_load_factor:
            self._resize(max(len(self._buckets) * 2, math.ceil(self.size / self._max_load_factor)))
        return True

    def contains(self, item: str) -> bool:
        """Return whether `item` is present in its bucket's collision chain.

        Args:
            item: Username to find.

        Returns:
            True when an equal username is found, otherwise False.
        """
        bucket = self._buckets[self._bucket_index(item, len(self._buckets))]
        if bucket is None:
            return False
        for stored_item in bucket:
            if stored_item == item:
                return True
        return False

    def _bucket_index(self, item: str, capacity: int) -> int:
        """Return the bucket index for `item`."""
        return stable_hash(item) % capacity

    def _resize(self, new_capacity: int) -> None:
        """Allocate a larger bucket array and reinsert all existing entries."""
        buckets_old = self._buckets
        self._buckets = [None] * new_capacity
        for bucket in buckets_old:
            if bucket is None:
                continue
            for item in bucket:
                index = self._bucket_index(item, new_capacity)
                new_bucket = self._buckets[index]
                if new_bucket is None:
                    new_bucket = []
                    self._buckets[index] = new_bucket
                new_bucket.append(item)

    @property
    def capacity(self) -> int:
        """Return the number of bucket slots."""
        return len(self._buckets)

    @property
    def load_factor(self) -> float:
        """Return the number of stored usernames divided by bucket capacity."""
        return self.size / len(self._buckets)
