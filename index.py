import argparse
import re
import subprocess
from bisect import bisect_right

import numpy as np
import pandas as pd

BASE_PATH='/mnt/lustre/users/inf/ms2420/non-temporal-comparison-datasets'

NUM_WALKS=10_000_000
MAX_WALK_LENGTH=80

TEMPORAL_DATASET_PATH=f"{BASE_PATH}/temporal"
THUNDERRW_DATASET_PATH=f"{BASE_PATH}/static_preprocessed_thunderrw"
FLOWWALKER_DATASET_PATH=f"{BASE_PATH}/static_preprocessed_flowwalker"

TEMPEST_OUTPUT_FILE=f"{BASE_PATH}/tempest_walks.txt"
THUNDERRW_OUTPUT_FILE=f"{BASE_PATH}/thunderrw_walks.txt"
FLOWWALKER_OUTPUT_FILE=f"{BASE_PATH}/flowwalker_walks.txt"


DATASET_FILE_NAMES = ['growth', 'delicious', 'ml_tgbl-coin', 'ml_tgbl-flight']


def build_temporal_index(csv_path, directed=False):
    print(f"Loading {csv_path}")

    df = pd.read_csv(csv_path)

    # Ensure correct types
    df["u"] = df["u"].astype(np.int64)
    df["i"] = df["i"].astype(np.int64)
    df["ts"] = df["ts"].astype(np.int64)

    if not directed:
        # For undirected graphs, normalize edge direction
        u = np.minimum(df["u"].values, df["i"].values)
        v = np.maximum(df["u"].values, df["i"].values)
    else:
        u = df["u"].values
        v = df["i"].values

    ts = df["ts"].values

    # Encode edge key as single uint64
    edge_keys = (u.astype(np.uint64) << 32) | v.astype(np.uint64)

    print("Sorting edges by (u,v,ts)...")

    order = np.lexsort((ts, edge_keys))
    edge_keys = edge_keys[order]
    ts = ts[order]

    print("Building index...")

    # Find unique edge keys
    unique_keys, start_indices = np.unique(edge_keys, return_index=True)

    # Build dictionary: key -> timestamps slice
    ts_map = {}

    for idx, key in enumerate(unique_keys):
        start = start_indices[idx]
        end = start_indices[idx + 1] if idx + 1 < len(start_indices) else len(ts)
        ts_map[key] = ts[start:end]  # already sorted

    print(f"Built index with {len(ts_map)} unique edges")

    return ts_map


def validate_walk(walk, ts_map):
    t_prev = -1
    for i in range(len(walk)-1):
        u, v = walk[i], walk[i+1]
        key = (min(u,v) << 32) | max(u,v)
        arr = ts_map.get(key)
        if arr is None:
            return False
        j = bisect_right(arr, t_prev)
        if j == len(arr):
            return False
        t_prev = arr[j]
    return True


def validate_walk_file(walk_file, ts_map, directed=False):
    correct_walks = 0

    with open(walk_file, "r") as f:
        for line in f:
            nodes = line.strip().split()
            if len(nodes) <= 1:
                continue

            t_prev = -1
            valid = True

            for i in range(len(nodes) - 1):
                u = int(nodes[i])
                v = int(nodes[i + 1])

                if not directed:
                    if u > v:
                        u, v = v, u

                key = (np.uint64(u) << 32) | np.uint64(v)

                arr = ts_map.get(key)
                if arr is None:
                    valid = False
                    break

                j = bisect_right(arr, t_prev)
                if j == len(arr):
                    valid = False
                    break

                t_prev = arr[j]

            if valid:
                correct_walks += 1

    return correct_walks


def get_flowwalker_metrics(dataset, is_directed, num_walks, n_runs):
    print(f"\nBuilding temporal index for {dataset}")
    ts_map = build_temporal_index(
        f"{TEMPORAL_DATASET_PATH}/{dataset}.csv",
        directed=is_directed
    )

    steps_per_sec_runs = []
    correct_walks_runs = []

    flowwalker_cmd = [
        "./../flowwalker-artifact/build/bin/flowwalker",
        "--input", f"{FLOWWALKER_DATASET_PATH}/{dataset}",
        "--deepwalk",
        f"--n={num_walks}",
        "--save_walks",
        "--walks_file", FLOWWALKER_OUTPUT_FILE
    ]

    for run in range(n_runs):
        print(f"\nRun {run+1}/{n_runs}")

        result = subprocess.run(
            flowwalker_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        output = result.stdout

        match = re.search(r"sampling time:([\d\.]+)\s*ms", output)
        if not match:
            raise RuntimeError("Could not find sampling time.")

        sampling_time_sec = float(match.group(1)) / 1000.0

        total_steps = 0
        with open(FLOWWALKER_OUTPUT_FILE, "r") as f:
            for line in f:
                nodes = line.strip().split()
                if len(nodes) > 1:
                    total_steps += len(nodes) - 1

        steps_per_sec = total_steps / sampling_time_sec
        steps_per_sec_runs.append(steps_per_sec)

        correct_walks = validate_walk_file(
            FLOWWALKER_OUTPUT_FILE,
            ts_map,
            directed=is_directed
        )

        correct_walks_runs.append(correct_walks)

        print(f"Steps/sec: {steps_per_sec:.4f}")
        print(f"Correct temporal walks: {correct_walks}")

    return steps_per_sec_runs, correct_walks_runs


def get_thunderrw_metrics(dataset, is_directed, num_walks, n_runs):
    print(f"\nBuilding temporal index for {dataset}")
    ts_map = build_temporal_index(
        f"{TEMPORAL_DATASET_PATH}/{dataset}.csv",
        directed=is_directed
    )

    steps_per_sec_runs = []
    correct_walks_runs = []

    thunderrw_cmd = [
        "./../ThunderRW/build/random_walk/genericwalk.out",
        "-f", f"{THUNDERRW_DATASET_PATH}/{dataset}",
        "-st", "2",
        "-wn", str(num_walks),
        "-wo", THUNDERRW_OUTPUT_FILE
    ]

    for run in range(n_runs):
        print(f"\nRun {run+1}/{n_runs}")

        result = subprocess.run(
            thunderrw_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        output = result.stdout

        matches = re.findall(
            r"Throughput \(steps per second\):\s*([\d\.]+)",
            output
        )

        if not matches:
            raise RuntimeError("Could not find throughput.")

        steps_per_sec = float(matches[-1])
        steps_per_sec_runs.append(steps_per_sec)

        correct_walks = validate_walk_file(
            THUNDERRW_OUTPUT_FILE,
            ts_map,
            directed=is_directed
        )

        correct_walks_runs.append(correct_walks)

        print(f"Steps/sec: {steps_per_sec:.4f}")
        print(f"Correct temporal walks: {correct_walks}")

    return steps_per_sec_runs, correct_walks_runs


def get_tempest_metrics(dataset, is_directed, num_walks, n_runs):
    print(f"\nBuilding temporal index for {dataset}")
    ts_map = build_temporal_index(
        f"{TEMPORAL_DATASET_PATH}/{dataset}.csv",
        directed=is_directed
    )

    steps_per_sec_runs = []
    correct_walks_runs = []

    tempest_cmd = [
        "./../temporal-random-walk/build/bin/walk_sampling_speed_test",
        f"{TEMPORAL_DATASET_PATH}/{dataset}.csv",
        TEMPEST_OUTPUT_FILE,
        "1" if is_directed else "0",
        str(num_walks),
        "-1",
        str(MAX_WALK_LENGTH),
        "ExponentialIndex",
        "Uniform"
    ]

    for run in range(n_runs):
        print(f"\nRun {run+1}/{n_runs}")

        result = subprocess.run(
            tempest_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        output = result.stdout

        match = re.search(r"Steps/sec:\s*([\d\.]+)", output)
        if not match:
            raise RuntimeError("Could not find Steps/sec in Tempest output.")

        steps_per_sec = float(match.group(1))
        steps_per_sec_runs.append(steps_per_sec)

        correct_walks = validate_walk_file(
            TEMPEST_OUTPUT_FILE,
            ts_map,
            directed=is_directed
        )

        correct_walks_runs.append(correct_walks)

        print(f"Steps/sec: {steps_per_sec:.4f}")
        print(f"Correct temporal walks: {correct_walks}")

    return steps_per_sec_runs, correct_walks_runs


def main(sampling_method, num_walks, is_directed, n_runs):
    for dataset in DATASET_FILE_NAMES:
        print(f"\n===== Dataset: {dataset} =====")

        if sampling_method == "tempest":
            steps, correct = get_tempest_metrics(
                dataset, is_directed, num_walks, n_runs
            )
        elif sampling_method == "thunderrw":
            steps, correct = get_thunderrw_metrics(
                dataset, is_directed, num_walks, n_runs
            )
        elif sampling_method == "flowwalker":
            steps, correct = get_flowwalker_metrics(
                dataset, is_directed, num_walks, n_runs
            )
        else:
            raise ValueError("Invalid sampling method")

        print("\nReturned:")
        print("Steps/sec runs:", steps)
        print("Correct temporal walks runs:", correct)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Non-Temporal Method Comparison")

    parser.add_argument('--sampling_method', type=str, required=True,
                        help='Random walk sampling method (tempest, thunderrw or flowwalker)')
    parser.add_argument('--num_walks', type=int, default=10_000_000,
                        help='Number of walks to sample')
    parser.add_argument('--directed', action='store_true',
                        help='Enable directed graphs')
    parser.add_argument('--n_runs', type=int, default=5,
                        help='Number of runs')

    args = parser.parse_args()

    main(args.sampling_method, args.num_walks, args.directed, args.n_runs)
