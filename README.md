# COSC 520 Assignment 1: Login Checker

This project compares five implementations of username membership checks. Each data structure is implemented here; the only external packages are for tests and plots.

## Setup and tests

The measurements use Python 3.14.8 on macOS.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest
```

`src/` holds the algorithms and shared hashing. `scripts/` contains synthetic-data generation, benchmarking, and plotting. `tests/` contains correctness tests. `data/`, `results/`, and `plots/` hold the experiment artifacts.

## Implementations

- `LinearSearch` scans an unsorted list.
- `BinarySearch` sorts during construction, then searches the sorted list by halving the interval.
- `HashTable` uses separate chaining and grows when its load exceeds the configured limit.
- `BloomFilter` stores bits in a `bytearray`. It derives the bit count `m` and hash count `k` from expected items `n` and a target false-positive rate; both counts can also be set directly.
- `CuckooFilter` stores fingerprints in two candidate buckets, relocates them when necessary, and reports failed insertions. Failed relocations are rolled back so previous entries stay findable. Bucket capacity, fingerprint bits, and kick limit are configurable.

The structures use a small shared `hashlib.blake2b` helper rather than Python's randomized `hash()`. Separate salts give Bloom its two hashes and Cuckoo its fingerprint, primary bucket, and reversible XOR alternate bucket. The Cuckoo method follows [Fan et al. (2014)](https://www.istc-cc.cmu.edu/publications/papers/2014/cuckoo-conext2014.pdf). Bloom and Cuckoo positive answers mean *possibly present*; negative answers mean *absent* for successfully inserted usernames.

Cuckoo's `size` counts stored fingerprints. Two different usernames with the same fingerprint and candidate buckets can share one stored fingerprint and both remain findable.

## Dataset

Usernames have the form `user_<8-digit seed>_<12-digit index>`. They are unique and deterministic. Positive queries sample existing usernames; negative queries use a disjoint `absent_...` prefix. Seeds and workload settings are recorded in the CSVs. The saved files are ascending; the benchmark shuffles the build list with seed `520 + 2n`, then binary search sorts its own copy. Query order is shuffled with seed `520 + n`.

Generate a small dataset and mixed workload:

```sh
python -m scripts.make_dataset --n 10000 --seed 520 \
  --query-seed 10520 --query-count 1000
```

The largest retained artifact is `data/usernames_n10000000_seed520.txt.gz`, with query file `data/queries_n10000000_q1000_seed10000520.csv` and checksums/settings in `data/manifest_n10000000_seed520.json`. It was generated with:

```sh
python -m scripts.make_dataset --n 10000000 --seed 520 \
  --query-seed 10000520 --query-count 1000 \
  --output data/usernames_n10000000_seed520.txt \
  --queries-output data/queries_n10000000_q1000_seed10000520.csv
gzip -n -9 -f data/usernames_n10000000_seed520.txt
```

The generator streams usernames to disk; it does not create a billion-entry dataset by default. To change the size, change `--n` explicitly.

## Benchmarks and plots

```sh
python -m scripts.benchmark --profile quick
python -m scripts.benchmark --profile full --prefix my_full
python -m scripts.plot_benchmarks --input results/my_full_summary.csv --out-dir plots
```

The full profile uses `n = 1,000, 5,000, 10,000, 25,000, 50,000, 100,000, 250,000, 500,000, 1,000,000, 2,000,000, 5,000,000, 10,000,000`. All five methods run through one million; binary search, hash table, Bloom, and Cuckoo continue through ten million. At each size, every method gets the same shuffled usernames and the same 1,000 queries (500 present, 500 absent). Lookup time excludes dataset generation and structure construction. Three trials are timed with `perf_counter_ns`; the CSVs contain raw times and summary medians. Bloom and Cuckoo also receive 100,000 seeded absent probes for a more useful observed false-positive rate. The benchmark checks inserted items for false negatives outside lookup timing.

Results go to `results/*_raw.csv` and `results/*_summary.csv`; filenames are unique unless you supply an unused `--prefix`. CSVs include seeds, build and lookup times, accuracy counts, skipped sizes, and structural sizes. The plotting command saves PNG and PDF versions of a shared-range runtime plot, a log-scale runtime plot, and a filter false-positive plot in `plots/`. Use `--sizes`, `--queries`, or `--repetitions` for smaller experiments. The `--max-list-gib` guard rejects a size before generating usernames when its minimum list estimate exceeds the configured limit.

The final retained benchmark CSVs and run metadata are under `results/` with the prefix `final_full_20261002_seed520_blake2b`. The full run took about 1,005 seconds (17 minutes) on the measured Mac. The metadata preserves the original command used for that completed run; its former module name `scripts.run_benchmarks` is now `scripts.benchmark`. Replot the results with:

```sh
python -m scripts.plot_benchmarks \
  --input results/final_full_20261002_seed520_blake2b_summary.csv \
  --out-dir plots --title-suffix ' (seed 520, full profile)'
```
