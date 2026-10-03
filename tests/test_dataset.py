"""Tests for reproducible synthetic dataset and query generation."""

import unittest

from scripts.synthetic_data import UsernameView, generate_usernames, make_query_workload


class DatasetTests(unittest.TestCase):
    """Check stable, unique usernames and known query labels."""

    def test_usernames_are_unique_and_reproducible(self):
        first = list(generate_usernames(30, seed=17))
        second = list(generate_usernames(30, seed=17))
        other_seed = list(generate_usernames(30, seed=18))
        self.assertEqual(first, second)
        self.assertNotEqual(first, other_seed)
        self.assertEqual(len(first), len(set(first)))
        self.assertEqual(list(generate_usernames(0)), [])
        view = UsernameView(30, seed=17)
        self.assertEqual(len(view), 30)
        self.assertEqual([view[index] for index in range(30)], first)
        self.assertEqual(view[-1], first[-1])
        with self.assertRaises(IndexError):
            view[30]

    def test_workload_is_mixed_and_reproducible(self):
        items = list(generate_usernames(20, seed=23))
        first = make_query_workload(items, 100, seed=9)
        second = make_query_workload(items, 100, seed=9)
        self.assertEqual(first.queries, second.queries)
        self.assertEqual(first.positive_count, 50)
        self.assertEqual(first.negative_count, 50)
        negative_items = [item for item, present in first.queries if not present]
        self.assertEqual(len(negative_items), len(set(negative_items)))
        for item, expected_present in first.queries:
            self.assertEqual(item in items, expected_present)

    def test_workload_validation(self):
        with self.assertRaises(ValueError):
            make_query_workload([], 1, positive_fraction=1.0)
        with self.assertRaises(ValueError):
            list(generate_usernames(-1))


if __name__ == "__main__":
    unittest.main()
