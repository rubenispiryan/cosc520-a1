"""A deterministic, non-deleting Cuckoo filter with partial-key relocation."""

from __future__ import annotations

import math
import random

from .hashing import stable_hash


def _next_power_of_two(value: int) -> int:
    """Return the smallest power of two at least `value`."""
    return 1 << (value - 1).bit_length()


class CuckooFilter:
    """Store short fingerprints in two candidate buckets with bounded kicking."""

    def __init__(
        self,
        bucket_count: int,
        bucket_capacity: int = 4,
        fingerprint_bits: int = 12,
        max_kicks: int = 500,
        seed: int = 0,
    ) -> None:
        """Create a filter with power-of-two bucket count and reproducible kicks.

        Args:
            bucket_count: Power-of-two number of buckets, at least two.
            bucket_capacity: Maximum fingerprints stored in each bucket.
            fingerprint_bits: Fingerprint width from 1 through 64 bits.
            max_kicks: Maximum relocations before a failed insert is rolled back.
            seed: Seed for deterministic relocation choices and hash separation.
        """
        if bucket_count < 2 or bucket_count & (bucket_count - 1):
            raise ValueError("bucket_count must be a power of two and at least 2")
        if bucket_capacity < 1:
            raise ValueError("bucket_capacity must be at least 1")
        if not 1 <= fingerprint_bits <= 64:
            raise ValueError("fingerprint_bits must be between 1 and 64")
        if max_kicks < 0:
            raise ValueError("max_kicks must be non-negative")
        self.bucket_count = bucket_count
        self.bucket_capacity = bucket_capacity
        self.fingerprint_bits = fingerprint_bits
        self.max_kicks = max_kicks
        self.seed = seed
        self._fingerprint_mask = (1 << fingerprint_bits) - 1
        self._fingerprint_bytes = (fingerprint_bits + 7) // 8
        self._rng = random.Random(seed)
        self._buckets: list[list[int]] = [[] for _ in range(bucket_count)]
        self.size = 0
        self.insertion_failures = 0
        self.relocation_count = 0

    @classmethod
    def for_expected_items(
        cls,
        expected_items: int,
        bucket_capacity: int = 4,
        fingerprint_bits: int = 12,
        target_load_factor: float = 0.90,
        max_kicks: int = 500,
        seed: int = 0,
    ) -> "CuckooFilter":
        """Create a power-of-two table sized for an expected item count.

        Args:
            expected_items: Positive number of usernames to plan for.
            bucket_capacity: Fingerprint slots per bucket.
            fingerprint_bits: Width of each stored fingerprint.
            target_load_factor: Requested maximum planned occupancy in (0, 1).
            max_kicks: Maximum relocations for one insertion.
            seed: Seed for deterministic relocation choices.

        Returns:
            A filter with enough bucket slots for the requested target load.
        """
        if expected_items <= 0:
            raise ValueError("expected_items must be at least 1")
        if not 0 < target_load_factor < 1:
            raise ValueError("target_load_factor must be strictly between 0 and 1")
        if bucket_capacity < 1:
            raise ValueError("bucket_capacity must be at least 1")
        needed = math.ceil(expected_items / (bucket_capacity * target_load_factor))
        bucket_count = max(2, _next_power_of_two(needed))
        return cls(bucket_count, bucket_capacity, fingerprint_bits, max_kicks, seed)

    def build(self, items: list[str]) -> int:
        """Reset the filter, insert `items`, and return failed insertion count.

        Args:
            items: Usernames to insert.

        Returns:
            Number of usernames whose insertion failed after bounded relocation.
        """
        self.clear()
        for item in items:
            self.insert(item)
        return self.insertion_failures

    def insert(self, item: str) -> bool:
        """Insert `item` or roll back and report failure when no slot is found.

        Args:
            item: Username to insert.

        Returns:
            True when the filter can represent the username, otherwise False.
        """
        fingerprint, first_index, second_index = self._item_location(item)
        candidate_indices = (first_index, second_index)
        for index in candidate_indices:
            if fingerprint in self._buckets[index]:
                # A non-deleting filter only needs one copy of this matching
                # fingerprint to answer membership queries for this item.
                return True
        for index in candidate_indices:
            bucket = self._buckets[index]
            if len(bucket) < self.bucket_capacity:
                bucket.append(fingerprint)
                self.size += 1
                return True

        current_index = self._rng.choice(candidate_indices)
        current_fingerprint = fingerprint
        changes: list[tuple[int, int, int]] = []
        for _ in range(self.max_kicks):
            bucket = self._buckets[current_index]
            slot = self._rng.randrange(len(bucket))
            evicted_fingerprint = bucket[slot]
            changes.append((current_index, slot, evicted_fingerprint))
            bucket[slot] = current_fingerprint
            self.relocation_count += 1
            current_fingerprint = evicted_fingerprint
            current_index = self._alternate_index(current_index, current_fingerprint)
            destination = self._buckets[current_index]
            if len(destination) < self.bucket_capacity:
                destination.append(current_fingerprint)
                self.size += 1
                return True

        # Keep failed insertion atomic: restore every kicked fingerprint so
        # previously inserted usernames retain their no-false-negative guarantee.
        for bucket_index, slot, previous_fingerprint in reversed(changes):
            self._buckets[bucket_index][slot] = previous_fingerprint
        self.insertion_failures += 1
        return False

    def contains(self, item: str) -> bool:
        """Return whether either candidate bucket contains `item`'s fingerprint.

        Args:
            item: Username to query.

        Returns:
            True when a matching fingerprint is present, otherwise False.
        """
        fingerprint, first_index, second_index = self._item_location(item)
        return (
            fingerprint in self._buckets[first_index]
            or fingerprint in self._buckets[second_index]
        )

    def _item_location(self, item: str) -> tuple[int, int, int]:
        """Return `(fingerprint, primary bucket, alternate bucket)` for an item."""
        salt = self.seed * 4
        fingerprint = stable_hash(item, salt + 2) & self._fingerprint_mask
        if fingerprint == 0:
            fingerprint = 1
        primary = stable_hash(item, salt + 3) & (self.bucket_count - 1)
        alternate = self._alternate_index(primary, fingerprint)
        return fingerprint, primary, alternate

    def _alternate_index(self, index: int, fingerprint: int) -> int:
        """Return a reversible alternate index based only on bucket and fingerprint."""
        fingerprint_bytes = fingerprint.to_bytes(self._fingerprint_bytes, "little")
        delta = stable_hash(fingerprint_bytes, self.seed * 4 + 4) & (self.bucket_count - 1)
        # A non-zero delta makes the two candidate buckets distinct. Because
        # delta depends only on the fingerprint, XOR remains its own inverse.
        if delta == 0:
            delta = 1
        return index ^ delta

    def clear(self) -> None:
        """Empty all buckets and reset deterministic relocation state."""
        self._buckets = [[] for _ in range(self.bucket_count)]
        self.size = 0
        self.insertion_failures = 0
        self.relocation_count = 0
        self._rng = random.Random(self.seed)

    @property
    def load_factor(self) -> float:
        """Return occupied fingerprint slots divided by configured slots."""
        return self.size / (self.bucket_count * self.bucket_capacity)

    @property
    def fingerprint_bytes_per_slot(self) -> int:
        """Return the whole-byte storage width for one configured fingerprint slot."""
        return self._fingerprint_bytes

    @property
    def configured_fingerprint_storage_bits(self) -> int:
        """Return theoretical fingerprint bits across all configured slots."""
        return self.bucket_count * self.bucket_capacity * self.fingerprint_bits
