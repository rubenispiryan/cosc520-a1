"""Correctness tests for all five membership implementations."""

import unittest

from src.binary_search import BinarySearch
from src.bloom_filter import BloomFilter
from src.cuckoo_filter import CuckooFilter
from src.hash_table import HashTable
from src.hashing import stable_hash
from src.linear_search import LinearSearch


SAMPLE_ITEMS = ["", "alice", "bob", "user_0007", "mahchakal", "punctuation!@#$", "with space"]


class ExactStructureTests(unittest.TestCase):
    """Check exact structures against a test-only Python set oracle."""

    def test_linear_search_empty_single_duplicate_and_unusual_strings(self):
        structure = LinearSearch()
        self.assertFalse(structure.contains("missing"))
        structure.build(["only"])
        self.assertTrue(structure.contains("only"))
        self.assertFalse(structure.contains("missing"))
        structure.build(["", "alice", "alice", "名字"])
        self.assertEqual(structure.size, 4)
        self.assertTrue(structure.contains(""))
        self.assertTrue(structure.contains("alice"))
        self.assertTrue(structure.contains("名字"))
        self.assertFalse(structure.contains("ALICE"))

    def test_binary_search_sorts_once_and_matches_oracle(self):
        empty = BinarySearch()
        self.assertFalse(empty.contains("missing"))
        single = BinarySearch(["only"])
        self.assertTrue(single.contains("only"))
        self.assertFalse(single.contains("missing"))

        structure = BinarySearch(SAMPLE_ITEMS + ["alice"])
        oracle = set(SAMPLE_ITEMS + ["alice"])
        for candidate in SAMPLE_ITEMS + ["absent", "ALICE"]:
            self.assertEqual(structure.contains(candidate), candidate in oracle)
        self.assertEqual(structure.size, len(SAMPLE_ITEMS) + 1)

    def test_hash_table_handles_collisions_duplicates_and_resize(self):
        collision_groups = [[] for _ in range(8)]
        candidate_number = 0
        while max(map(len, collision_groups)) < 3:
            value = "collision_{}".format(candidate_number)
            collision_groups[stable_hash(value) % 8].append(value)
            candidate_number += 1
        colliding_items = max(collision_groups, key=len)[:3]

        structure = HashTable(initial_capacity=8, max_load_factor=0.5)
        self.assertFalse(structure.contains("not-present"))
        self.assertTrue(structure.insert("only"))
        self.assertTrue(structure.contains("only"))
        structure.build([])
        self.assertFalse(structure.contains("only"))
        for item in colliding_items:
            self.assertTrue(structure.insert(item))
        self.assertFalse(structure.insert(colliding_items[0]))
        self.assertTrue(structure.contains(colliding_items[2]))
        self.assertFalse(structure.contains("not-present"))

        all_items = colliding_items + ["item_{}".format(index) for index in range(30)]
        structure.build(all_items)
        oracle = set(all_items)
        self.assertGreater(structure.capacity, 8)
        self.assertEqual(structure.size, len(oracle))
        for candidate in all_items + ["absent", "名字", ""]:
            self.assertEqual(structure.contains(candidate), candidate in oracle)

    def test_hash_table_rejects_invalid_configuration(self):
        with self.assertRaises(ValueError):
            HashTable(initial_capacity=0)
        with self.assertRaises(ValueError):
            HashTable(max_load_factor=0)


class BloomFilterTests(unittest.TestCase):
    """Check Bloom-filter invariants without assuming no false positives."""

    def test_inserted_items_never_become_false_negatives(self):
        structure = BloomFilter(expected_items=len(SAMPLE_ITEMS), false_positive_rate=0.01)
        self.assertFalse(structure.contains("absent-before-build"))
        single = BloomFilter(expected_items=1)
        single.add("")
        self.assertTrue(single.contains(""))
        structure.build(SAMPLE_ITEMS)
        for item in SAMPLE_ITEMS:
            self.assertTrue(structure.contains(item))
        self.assertEqual(structure.size, len(SAMPLE_ITEMS))
        self.assertGreaterEqual(structure.bit_count, 1)
        self.assertGreaterEqual(structure.bit_array_bytes, (structure.bit_count + 7) // 8)

    def test_absent_queries_have_reasonable_deterministic_rate(self):
        items = ["present_{}".format(index) for index in range(1000)]
        structure = BloomFilter(expected_items=1000, false_positive_rate=0.05)
        structure.build(items)
        absent = ["absent_{}".format(index) for index in range(2000)]
        false_positives = sum(structure.contains(item) for item in absent)
        self.assertLess(false_positives / len(absent), 0.15)
        self.assertGreaterEqual(structure.estimated_false_positive_rate, 0.0)
        self.assertLessEqual(structure.estimated_false_positive_rate, 1.0)

    def test_explicit_parameters_clear_and_validation(self):
        structure = BloomFilter(3, bit_count=97, hash_count=5)
        structure.add("one")
        self.assertTrue(structure.contains("one"))
        structure.clear()
        self.assertFalse(structure.contains("one"))
        with self.assertRaises(ValueError):
            BloomFilter(0)
        with self.assertRaises(ValueError):
            BloomFilter(5, false_positive_rate=1.0)


class CuckooFilterTests(unittest.TestCase):
    """Check fingerprint lookup, reversible relocation, and insertion failures."""

    def test_lookup_for_empty_single_multiple_and_unusual_strings(self):
        structure = CuckooFilter.for_expected_items(20, seed=4)
        self.assertFalse(structure.contains("absent"))
        self.assertTrue(structure.insert("single"))
        self.assertTrue(structure.contains("single"))
        size_after_first_insert = structure.size
        self.assertTrue(structure.insert("single"))
        self.assertEqual(structure.size, size_after_first_insert)
        for item in SAMPLE_ITEMS:
            self.assertTrue(structure.insert(item))
        for item in SAMPLE_ITEMS:
            self.assertTrue(structure.contains(item))
        self.assertEqual(structure.insertion_failures, 0)

    def test_alternate_index_is_reversible_for_all_tested_buckets(self):
        structure = CuckooFilter(bucket_count=64, fingerprint_bits=12, seed=19)
        for fingerprint in (1, 2, 17, 255, 2047, 4095):
            for index in range(structure.bucket_count):
                alternate = structure._alternate_index(index, fingerprint)
                self.assertNotEqual(alternate, index)
                self.assertEqual(structure._alternate_index(alternate, fingerprint), index)

    def test_deterministic_relocation_and_no_false_negatives_for_successes(self):
        items = ["relocate_{}".format(index) for index in range(45)]
        first = CuckooFilter(64, bucket_capacity=1, fingerprint_bits=8, max_kicks=100, seed=7)
        second = CuckooFilter(64, bucket_capacity=1, fingerprint_bits=8, max_kicks=100, seed=7)
        first_results = [first.insert(item) for item in items]
        second_results = [second.insert(item) for item in items]
        first_buckets = tuple(tuple(bucket) for bucket in first._buckets)
        second_buckets = tuple(tuple(bucket) for bucket in second._buckets)
        self.assertEqual(first_results, second_results)
        self.assertEqual(first_buckets, second_buckets)
        self.assertGreater(first.relocation_count, 0)
        self.assertGreater(len(first._buckets), 0)
        self.assertEqual(first.insertion_failures, first_results.count(False))
        for item, accepted in zip(items, first_results):
            if accepted:
                self.assertTrue(first.contains(item))

    def test_saturated_insert_fails_atomically(self):
        structure = CuckooFilter(2, bucket_capacity=1, fingerprint_bits=8, max_kicks=4, seed=2)
        accepted = []
        candidate = 0
        while structure.size < 2:
            item = "full_{}".format(candidate)
            candidate += 1
            if structure.insert(item):
                accepted.append(item)
        existing_fingerprints = set(
            fingerprint for bucket in structure._buckets for fingerprint in bucket
        )
        while True:
            rejected_candidate = "full_{}".format(candidate)
            candidate += 1
            fingerprint, first_index, second_index = structure._item_location(rejected_candidate)
            if (
                fingerprint not in existing_fingerprints
                and structure._buckets[first_index]
                and structure._buckets[second_index]
            ):
                break
        self.assertFalse(structure.insert(rejected_candidate))
        self.assertEqual(structure.insertion_failures, 1)
        self.assertEqual(structure.size, 2)
        for item in accepted:
            self.assertTrue(structure.contains(item))

    def test_rejects_non_power_of_two_bucket_count(self):
        with self.assertRaises(ValueError):
            CuckooFilter(3)


class HashingTests(unittest.TestCase):
    """Check a known digest and separate deterministic salts."""

    def test_hash_is_stable_and_salts_differ(self):
        self.assertEqual(stable_hash("hello"), 0x7D34E501A8EDB6A7)
        self.assertEqual(stable_hash("名字"), stable_hash("名字"))
        self.assertNotEqual(stable_hash("alice"), stable_hash("bob"))
        self.assertNotEqual(stable_hash("alice"), stable_hash("alice", 1))


if __name__ == "__main__":
    unittest.main()
