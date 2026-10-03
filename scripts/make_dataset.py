"""Write a reproducible username dataset and optional mixed query workload."""

import argparse
import csv
from pathlib import Path
from typing import Optional, Sequence

from .synthetic_data import UsernameView, generate_usernames, make_query_workload


def _default_dataset_path(count: int, seed: int) -> Path:
    """Build the conventional dataset filename for the requested settings."""
    return Path("data") / "usernames_n{}_seed{}.txt".format(count, seed)


def _default_query_path(count: int, query_seed: int, query_count: int) -> Path:
    """Build the conventional query filename for the requested settings."""
    return Path("data") / "queries_n{}_q{}_seed{}.csv".format(count, query_count, query_seed)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse options and write usernames plus an optional query CSV.

    Args:
        argv: Optional command-line arguments for testing or embedding.

    Returns:
        Process exit status, zero on success.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10_000, help="number of unique usernames")
    parser.add_argument("--seed", type=int, default=520, help="deterministic username namespace seed")
    parser.add_argument("--query-seed", type=int,
                        help="query sampling seed (defaults to --seed)")
    parser.add_argument("--output", type=Path, help="dataset output path")
    parser.add_argument("--query-count", type=int, default=0, help="optional mixed workload size")
    parser.add_argument("--queries-output", type=Path, help="query CSV path (requires --query-count)")
    parser.add_argument("--positive-fraction", type=float, default=0.5)
    parser.add_argument("--force", action="store_true", help="replace existing output files")
    args = parser.parse_args(argv)
    query_seed = args.seed if args.query_seed is None else args.query_seed

    if args.n < 0 or args.query_count < 0 or args.seed < 0 or query_seed < 0:
        parser.error("--n, --query-count, --seed, and --query-seed must be non-negative")
    if args.query_count and args.n == 0 and args.positive_fraction > 0:
        parser.error("positive query generation requires --n greater than zero")
    if not 0 <= args.positive_fraction <= 1:
        parser.error("--positive-fraction must be between 0 and 1")
    dataset_path = args.output or _default_dataset_path(args.n, args.seed)
    query_path = args.queries_output
    if args.query_count and query_path is None:
        query_path = _default_query_path(args.n, query_seed, args.query_count)
    if args.queries_output and not args.query_count:
        parser.error("--queries-output requires a positive --query-count")

    output_paths = [dataset_path] + ([query_path] if query_path is not None else [])
    existing = [str(path) for path in output_paths if path.exists()]
    if existing and not args.force:
        parser.error("output already exists (use --force to replace): " + ", ".join(existing))

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("w", encoding="utf-8", newline="\n") as stream:
        for username in generate_usernames(args.n, args.seed):
            stream.write(username + "\n")

    if query_path is not None:
        usernames = UsernameView(args.n, args.seed)
        workload = make_query_workload(
            usernames, args.query_count, seed=query_seed, positive_fraction=args.positive_fraction
        )
        query_path.parent.mkdir(parents=True, exist_ok=True)
        with query_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("username", "expected_present"))
            for username, expected_present in workload.queries:
                writer.writerow((username, int(expected_present)))

    print("Wrote {} unique usernames to {}".format(args.n, dataset_path))
    if query_path is not None:
        print(
            "Wrote {} queries ({} positive, {} negative) to {}".format(
                args.query_count, workload.positive_count, workload.negative_count, query_path
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
