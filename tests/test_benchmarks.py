"""Safety checks for the benchmark command-line workflow."""

import csv

from scripts import benchmark


def test_large_run_is_skipped_before_generation(tmp_path, monkeypatch):
    """A size over the memory limit must not materialize any usernames."""
    original_generator = benchmark.generate_usernames

    def guarded_generator(count, seed):
        if count > 1:
            raise AssertionError("large dataset generation started")
        return original_generator(count, seed)

    monkeypatch.setattr(benchmark, "generate_usernames", guarded_generator)
    result = benchmark.main([
        "--sizes", "10000000", "--queries", "10", "--repetitions", "1",
        "--max-list-gib", "0.1",
        "--output-dir", str(tmp_path), "--prefix", "memory_guard",
    ])
    assert result == 0

    with (tmp_path / "memory_guard_raw.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(benchmark.ALGORITHMS)
    assert all("exceeds" in row["skip_reason"] for row in rows)
    assert all(row["trial"] == "0" for row in rows)

    with (tmp_path / "memory_guard_summary.csv").open(newline="") as stream:
        assert list(csv.DictReader(stream)) == []
