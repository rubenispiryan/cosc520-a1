"""Benchmark all five membership structures with repeatable query workloads."""

import argparse
import csv
import math
import platform
import random
import statistics
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from src.binary_search import BinarySearch
from src.bloom_filter import BloomFilter
from src.cuckoo_filter import CuckooFilter
from .synthetic_data import QueryWorkload, generate_usernames, make_query_workload
from src.hash_table import HashTable
from src.linear_search import LinearSearch


ALGORITHMS = (
    "linear_search",
    "binary_search",
    "hash_table",
    "bloom_filter",
    "cuckoo_filter",
)
PROFILES = {
    "quick": {
        "sizes": (1_000, 5_000, 10_000),
        "queries": 1_000,
        "repetitions": 2,
        "linear_max_n": 10_000,
    },
    "full": {
        "sizes": (
            1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000,
            500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000,
        ),
        "queries": 1_000,
        "repetitions": 3,
        "linear_max_n": 1_000_000,
    },
}
RAW_FIELDS = (
    "profile", "algorithm", "n", "trial", "query_count", "positive_query_count",
    "negative_query_count", "build_elapsed_ns", "elapsed_ns", "ns_per_query",
    "false_negatives", "false_positives", "false_positive_rate",
    "accuracy_negative_queries", "accuracy_false_positives", "accuracy_false_positive_rate",
    "all_input_false_negatives", "insertion_failures", "seed", "query_seed", "build_seed",
    "build_order", "max_list_gib", "dataset_list_min_gib", "target_false_positive_rate",
    "size", "hash_capacity", "hash_load_factor", "bloom_m_bits",
    "bloom_k", "bloom_bit_array_bytes", "bloom_estimated_false_positive_rate",
    "cuckoo_bucket_count", "cuckoo_bucket_capacity", "cuckoo_fingerprint_bits",
    "cuckoo_fingerprint_bytes_per_slot", "cuckoo_configured_fingerprint_bits",
    "cuckoo_load_factor", "cuckoo_max_kicks", "cuckoo_relocation_count",
    "skip_reason", "python_version", "platform",
)
SUMMARY_FIELDS = (
    "profile", "algorithm", "n", "repetitions", "query_count", "positive_query_count",
    "negative_query_count", "median_build_elapsed_ns", "mean_build_elapsed_ns",
    "median_elapsed_ns", "mean_elapsed_ns", "stdev_elapsed_ns", "median_ns_per_query",
    "mean_ns_per_query", "false_negatives", "false_positives", "false_positive_rate",
    "accuracy_negative_queries", "accuracy_false_positives", "accuracy_false_positive_rate",
    "all_input_false_negatives", "insertion_failures", "seed", "query_seed", "build_seed",
    "build_order", "max_list_gib", "dataset_list_min_gib", "target_false_positive_rate",
    "size", "hash_capacity", "hash_load_factor", "bloom_m_bits",
    "bloom_k", "bloom_bit_array_bytes", "bloom_estimated_false_positive_rate",
    "cuckoo_bucket_count", "cuckoo_bucket_capacity", "cuckoo_fingerprint_bits",
    "cuckoo_fingerprint_bytes_per_slot", "cuckoo_configured_fingerprint_bits",
    "cuckoo_load_factor", "cuckoo_max_kicks", "cuckoo_relocation_count", "python_version", "platform",
)


def _structure_factory(algorithm: str, n: int, seed: int, false_positive_rate: float):
    """Return a newly configured empty structure for `algorithm`."""
    if algorithm == "linear_search":
        return LinearSearch()
    if algorithm == "binary_search":
        return BinarySearch()
    if algorithm == "hash_table":
        return HashTable(initial_capacity=8, max_load_factor=0.75)
    if algorithm == "bloom_filter":
        return BloomFilter(n, false_positive_rate)
    if algorithm == "cuckoo_filter":
        return CuckooFilter.for_expected_items(
            n, bucket_capacity=4, fingerprint_bits=12,
            target_load_factor=0.90, max_kicks=500, seed=seed,
        )
    raise ValueError("unknown algorithm: " + algorithm)


def _structure_metrics(structure) -> Dict[str, object]:
    """Return explicit structural counts without estimating Python object overhead."""
    metrics: Dict[str, object] = {
        "size": structure.size,
        "hash_capacity": "",
        "hash_load_factor": "",
        "bloom_m_bits": "",
        "bloom_k": "",
        "bloom_bit_array_bytes": "",
        "bloom_estimated_false_positive_rate": "",
        "cuckoo_bucket_count": "",
        "cuckoo_bucket_capacity": "",
        "cuckoo_fingerprint_bits": "",
        "cuckoo_fingerprint_bytes_per_slot": "",
        "cuckoo_configured_fingerprint_bits": "",
        "cuckoo_load_factor": "",
        "cuckoo_max_kicks": "",
        "cuckoo_relocation_count": "",
        "insertion_failures": 0,
    }
    if isinstance(structure, HashTable):
        metrics["hash_capacity"] = structure.capacity
        metrics["hash_load_factor"] = structure.load_factor
    elif isinstance(structure, BloomFilter):
        metrics["bloom_m_bits"] = structure.bit_count
        metrics["bloom_k"] = structure.hash_count
        metrics["bloom_bit_array_bytes"] = structure.bit_array_bytes
        metrics["bloom_estimated_false_positive_rate"] = structure.estimated_false_positive_rate
    elif isinstance(structure, CuckooFilter):
        metrics["insertion_failures"] = structure.insertion_failures
        metrics["cuckoo_bucket_count"] = structure.bucket_count
        metrics["cuckoo_bucket_capacity"] = structure.bucket_capacity
        metrics["cuckoo_fingerprint_bits"] = structure.fingerprint_bits
        metrics["cuckoo_fingerprint_bytes_per_slot"] = structure.fingerprint_bytes_per_slot
        metrics["cuckoo_configured_fingerprint_bits"] = structure.configured_fingerprint_storage_bits
        metrics["cuckoo_load_factor"] = structure.load_factor
        metrics["cuckoo_max_kicks"] = structure.max_kicks
        metrics["cuckoo_relocation_count"] = structure.relocation_count
    return metrics


def _workload_outcomes(structure, workload: QueryWorkload) -> Tuple[int, int]:
    """Count false negatives and false positives over a known mixed workload."""
    false_negatives = 0
    false_positives = 0
    for item, expected_present in workload.queries:
        actual = structure.contains(item)
        if expected_present and not actual:
            false_negatives += 1
        elif not expected_present and actual:
            false_positives += 1
    return false_negatives, false_positives


def _accuracy_metrics(structure, items: List[str], seed: int, count: int) -> Dict[str, object]:
    """Check all inputs and a larger deterministic absent sample outside timing."""
    is_cuckoo = isinstance(structure, CuckooFilter)
    is_probabilistic = isinstance(structure, (BloomFilter, CuckooFilter))
    if not is_probabilistic:
        return {
            "accuracy_negative_queries": "",
            "accuracy_false_positives": "",
            "accuracy_false_positive_rate": "",
            "all_input_false_negatives": "",
            "insertion_failures": 0,
        }
    insertion_failures = structure.insertion_failures if is_cuckoo else 0
    all_input_false_negatives: object = ""
    if not is_cuckoo or insertion_failures == 0:
        all_input_false_negatives = sum(not structure.contains(item) for item in items)
        if all_input_false_negatives:
            raise RuntimeError("{} produced a false negative for an inserted item".format(type(structure).__name__))

    negative_ids = random.Random(seed).sample(range(10**12), count)
    negative_queries = [
        "accuracy_absent_{:08d}_{:012d}".format(seed, index)
        for index in negative_ids
    ]
    false_positives = sum(structure.contains(item) for item in negative_queries)
    return {
        "accuracy_negative_queries": count,
        "accuracy_false_positives": false_positives,
        "accuracy_false_positive_rate": false_positives / count if count else "",
        "all_input_false_negatives": all_input_false_negatives,
        "insertion_failures": insertion_failures,
    }


def _summarize(raw_rows: List[Dict[str, object]], seed: int) -> List[Dict[str, object]]:
    """Aggregate repeated raw rows by algorithm and dataset size."""
    groups: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for row in raw_rows:
        if row["skip_reason"]:
            continue
        groups.setdefault((str(row["algorithm"]), str(row["n"])), []).append(row)

    output = []
    constant_fields = (
        "profile", "algorithm", "n", "query_count", "positive_query_count", "negative_query_count",
        "false_negatives", "false_positives", "false_positive_rate", "accuracy_negative_queries",
        "accuracy_false_positives", "accuracy_false_positive_rate", "all_input_false_negatives",
        "insertion_failures", "query_seed", "build_seed", "build_order",
        "max_list_gib", "dataset_list_min_gib",
        "target_false_positive_rate", "size", "hash_capacity",
        "hash_load_factor", "bloom_m_bits", "bloom_k", "bloom_bit_array_bytes",
        "bloom_estimated_false_positive_rate", "cuckoo_bucket_count", "cuckoo_bucket_capacity",
        "cuckoo_fingerprint_bits", "cuckoo_fingerprint_bytes_per_slot",
        "cuckoo_configured_fingerprint_bits", "cuckoo_load_factor", "cuckoo_max_kicks",
        "cuckoo_relocation_count", "python_version", "platform",
    )
    for (_, _), rows in sorted(groups.items(), key=lambda entry: (entry[0][0], int(entry[0][1]))):
        build_times = [int(row["build_elapsed_ns"]) for row in rows]
        elapsed_times = [int(row["elapsed_ns"]) for row in rows]
        per_query = [float(row["ns_per_query"]) for row in rows]
        summary: Dict[str, object] = {field: rows[0][field] for field in constant_fields}
        summary["repetitions"] = len(rows)
        summary["seed"] = seed
        summary["median_build_elapsed_ns"] = statistics.median(build_times)
        summary["mean_build_elapsed_ns"] = statistics.mean(build_times)
        summary["median_elapsed_ns"] = statistics.median(elapsed_times)
        summary["mean_elapsed_ns"] = statistics.mean(elapsed_times)
        summary["stdev_elapsed_ns"] = statistics.stdev(elapsed_times) if len(elapsed_times) > 1 else 0.0
        summary["median_ns_per_query"] = statistics.median(per_query)
        summary["mean_ns_per_query"] = statistics.mean(per_query)
        output.append(summary)
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the selected experiment and write raw and summarized CSV files.

    Args:
        argv: Optional command-line arguments for testing or embedding.

    Returns:
        Process exit status, zero on success.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="quick")
    parser.add_argument("--sizes", type=int, nargs="+", help="override the profile's increasing n values")
    parser.add_argument("--queries", type=int, help="timed mixed-query count per size")
    parser.add_argument("--repetitions", type=int, help="timed repetitions per algorithm and size")
    parser.add_argument("--linear-max-n", type=int, help="largest n at which linear search is measured")
    parser.add_argument("--accuracy-negatives", type=int, default=100_000,
                        help="extra absent probes for filter accuracy, excluded from timing")
    parser.add_argument("--false-positive-rate", type=float, default=0.01)
    parser.add_argument("--max-list-gib", type=float, default=4.0,
                        help="skip n when the minimum username-list estimate exceeds this GiB limit")
    parser.add_argument("--seed", type=int, default=520)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--prefix", help="output filename prefix; defaults to a unique timestamp")
    args = parser.parse_args(argv)

    profile = PROFILES[args.profile]
    sizes = args.sizes or list(profile["sizes"])
    query_count = args.queries if args.queries is not None else profile["queries"]
    repetitions = args.repetitions if args.repetitions is not None else profile["repetitions"]
    linear_max_n = args.linear_max_n if args.linear_max_n is not None else profile["linear_max_n"]
    if any(size < 1 for size in sizes) or sizes != sorted(set(sizes)):
        parser.error("--sizes must be positive, strictly increasing integers")
    if query_count < 1 or repetitions < 1 or args.accuracy_negatives < 0:
        parser.error("query count and repetitions must be positive; accuracy probes cannot be negative")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if not math.isfinite(args.false_positive_rate) or not 0 < args.false_positive_rate < 1:
        parser.error("--false-positive-rate must be strictly between 0 and 1")
    if not math.isfinite(args.max_list_gib) or args.max_list_gib <= 0:
        parser.error("--max-list-gib must be a finite positive number")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or "benchmark_{}_{}".format(
        args.profile, datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    raw_path = args.output_dir / (prefix + "_raw.csv")
    summary_path = args.output_dir / (prefix + "_summary.csv")
    if raw_path.exists() or summary_path.exists():
        parser.error("benchmark output exists; choose a different --prefix")

    raw_rows: List[Dict[str, object]] = []
    raw_stream = raw_path.open("x", encoding="utf-8", newline="")
    raw_writer = csv.DictWriter(raw_stream, fieldnames=RAW_FIELDS)
    raw_writer.writeheader()

    def record_row(row: Dict[str, object]) -> None:
        """Keep a result for the summary and flush it to the raw CSV."""
        raw_rows.append(row)
        raw_writer.writerow(row)
        raw_stream.flush()

    def record_skip(algorithm: str, n: int, positives: int, list_gib: float, reason: str) -> None:
        """Record why an algorithm and size were not measured."""
        record_row({
            "profile": args.profile,
            "algorithm": algorithm,
            "n": n,
            "trial": 0,
            "query_count": query_count,
            "positive_query_count": positives,
            "negative_query_count": query_count - positives,
            "seed": args.seed,
            "query_seed": args.seed + n,
            "build_seed": args.seed + 2 * n,
            "build_order": "shuffled",
            "max_list_gib": args.max_list_gib,
            "dataset_list_min_gib": list_gib,
            "skip_reason": reason,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        })

    sample_username = next(generate_usernames(1, args.seed))
    minimum_bytes_per_item = sys.getsizeof(sample_username) + struct.calcsize("P")
    started = time.perf_counter()
    print("Profile {}, seed {}, sizes {}, {} queries, {} repetitions".format(
        args.profile, args.seed, ",".join(str(size) for size in sizes), query_count, repetitions
    ))
    for n in sizes:
        query_seed = args.seed + n
        build_seed = args.seed + 2 * n
        list_min_gib = n * minimum_bytes_per_item / (1024 ** 3)
        if list_min_gib > args.max_list_gib:
            reason = "minimum username list {:.1f} GiB exceeds {:.1f} GiB limit".format(
                list_min_gib, args.max_list_gib
            )
            for algorithm in ALGORITHMS:
                record_skip(algorithm, n, round(query_count * 0.5), list_min_gib, reason)
            print("n={} skipped ({})".format(n, reason))
            continue

        items = list(generate_usernames(n, args.seed))
        workload = make_query_workload(items, query_count, seed=query_seed, positive_fraction=0.5)
        random.Random(build_seed).shuffle(items)
        print("n={} workload ready ({} positive, {} negative)".format(
            n, workload.positive_count, workload.negative_count
        ))
        for algorithm in ALGORITHMS:
            if algorithm == "linear_search" and n > linear_max_n:
                record_skip(
                    algorithm, n, workload.positive_count, list_min_gib,
                    "linear_max_n={} exceeded".format(linear_max_n),
                )
                print("  linear_search skipped at n={} (cap {})".format(n, linear_max_n))
                continue

            accuracy: Optional[Dict[str, object]] = None
            for trial in range(1, repetitions + 1):
                build_start = time.perf_counter_ns()
                structure = _structure_factory(
                    algorithm, n, args.seed, args.false_positive_rate
                )
                structure.build(items)
                build_elapsed_ns = time.perf_counter_ns() - build_start
                metrics = _structure_metrics(structure)

                if accuracy is None:
                    accuracy = _accuracy_metrics(
                        structure, items, args.seed + n, args.accuracy_negatives
                    )
                false_negatives, false_positives = _workload_outcomes(structure, workload)
                if isinstance(structure, (LinearSearch, BinarySearch, HashTable)) and (
                    false_negatives or false_positives
                ):
                    raise RuntimeError("{} disagrees with the known workload".format(algorithm))
                if isinstance(structure, CuckooFilter) and structure.insertion_failures == 0 and false_negatives:
                    raise RuntimeError("Cuckoo filter returned a false negative without insertion failure")

                for item, _ in workload.queries[: min(100, query_count)]:
                    structure.contains(item)
                query_start = time.perf_counter_ns()
                for item, _ in workload.queries:
                    structure.contains(item)
                elapsed_ns = time.perf_counter_ns() - query_start

                row: Dict[str, object] = {field: "" for field in RAW_FIELDS}
                row.update({
                    "profile": args.profile,
                    "algorithm": algorithm,
                    "n": n,
                    "trial": trial,
                    "query_count": query_count,
                    "positive_query_count": workload.positive_count,
                    "negative_query_count": workload.negative_count,
                    "build_elapsed_ns": build_elapsed_ns,
                    "elapsed_ns": elapsed_ns,
                    "ns_per_query": elapsed_ns / query_count,
                    "false_negatives": false_negatives,
                    "false_positives": false_positives,
                    "false_positive_rate": false_positives / workload.negative_count if workload.negative_count else "",
                    "seed": args.seed,
                    "query_seed": query_seed,
                    "build_seed": build_seed,
                    "build_order": "shuffled",
                    "max_list_gib": args.max_list_gib,
                    "dataset_list_min_gib": list_min_gib,
                    "target_false_positive_rate": args.false_positive_rate if algorithm == "bloom_filter" else "",
                    "skip_reason": "",
                    "python_version": sys.version.split()[0],
                    "platform": platform.platform(),
                })
                row.update(metrics)
                row.update(accuracy)
                record_row(row)
                print("  {} trial {}/{}: {:.1f} ns/query, build {:.3f} ms".format(
                    algorithm, trial, repetitions, row["ns_per_query"], build_elapsed_ns / 1_000_000
                ))
        del structure, items, workload

    raw_stream.close()
    summary_rows = _summarize(raw_rows, args.seed)
    with summary_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("Raw CSV: {}".format(raw_path))
    print("Summary CSV: {}".format(summary_path))
    print("Total benchmark wall time: {:.1f} seconds".format(time.perf_counter() - started))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
