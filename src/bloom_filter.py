"""A compact Bloom filter with deterministic double hashing."""

from __future__ import annotations

import math

from .hashing import stable_hash


class BloomFilter:
    """Approximate membership structure backed by an explicitly managed bit array."""

    def __init__(
        self,
        expected_items: int,
        false_positive_rate: float = 0.01,
        *,
        bit_count: int | None = None,
        hash_count: int | None = None,
    ) -> None:
        """Create a filter, deriving missing bit and hash counts from `expected_items`.

        Args:
            expected_items: Positive expected number of inserted usernames.
            false_positive_rate: Target probability, strictly between zero and one.
            bit_count: Optional explicit number of bits in the array.
            hash_count: Optional explicit number of derived hash positions per query.
        """
        if expected_items <= 0:
            raise ValueError("expected_items must be at least 1")
        if not math.isfinite(false_positive_rate) or not 0 < false_positive_rate < 1:
            raise ValueError("false_positive_rate must be strictly between 0 and 1")
        if bit_count is None:
            bit_count = math.ceil(
                -expected_items * math.log(false_positive_rate) / (math.log(2) ** 2)
            )
        if bit_count < 1:
            raise ValueError("bit_count must be at least 1")
        if hash_count is None:
            hash_count = max(1, round((bit_count / expected_items) * math.log(2)))
        if hash_count < 1:
            raise ValueError("hash_count must be at least 1")

        self.expected_items = expected_items
        self.target_false_positive_rate = false_positive_rate
        self.bit_count = bit_count
        self.hash_count = hash_count
        self._bits = bytearray((bit_count + 7) // 8)
        self.size = 0

    def build(self, items: list[str]) -> None:
        """Clear the filter and add each username from `items`.

        Args:
            items: Usernames to insert.

        Returns:
            None.
        """
        self.clear()
        for item in items:
            self.add(item)

    def add(self, item: str) -> None:
        """Set the `hash_count` bits associated with `item`.

        Args:
            item: Username to insert.

        Returns:
            None.
        """
        first_hash = stable_hash(item)
        second_hash = stable_hash(item, 1) | 1
        for number in range(self.hash_count):
            position = (first_hash + number * second_hash) % self.bit_count
            byte_index, bit_offset = divmod(position, 8)
            self._bits[byte_index] |= 1 << bit_offset
        self.size += 1

    def contains(self, item: str) -> bool:
        """Return whether all Bloom-filter bits for `item` are set.

        Args:
            item: Username to query.

        Returns:
            False means definitely absent; True means possibly present.
        """
        first_hash = stable_hash(item)
        second_hash = stable_hash(item, 1) | 1
        for number in range(self.hash_count):
            position = (first_hash + number * second_hash) % self.bit_count
            byte_index, bit_offset = divmod(position, 8)
            if not self._bits[byte_index] & (1 << bit_offset):
                return False
        return True

    def clear(self) -> None:
        """Reset every bit and the insertion count."""
        self._bits = bytearray(len(self._bits))
        self.size = 0

    @property
    def bit_array_bytes(self) -> int:
        """Return allocated bytearray length, including its final partial byte."""
        return len(self._bits)

    @property
    def estimated_false_positive_rate(self) -> float:
        """Return the classic occupancy-based false-positive estimate."""
        if self.size == 0:
            return 0.0
        return (1.0 - math.exp(-self.hash_count * self.size / self.bit_count)) ** self.hash_count
