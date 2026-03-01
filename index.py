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


def _validate_one_direction(nodes, ts_map, directed=False):
    """
    Validate a single walk direction using greedy earliest-feasible timestamps.
    Returns: (hops, invalid_hops, is_walk_valid)
    Assumes ts_map[key] exists and is a sorted array of timestamps.
    """
    if len(nodes) <= 1:
        return 0, 0, True

    t_prev = -1
    invalid_hops = 0

    for j in range(len(nodes) - 1):
        u = int(nodes[j])
        v = int(nodes[j + 1])

        if not directed and u > v:
            u, v = v, u

        key = (np.uint64(u) << 32) | np.uint64(v)
        arr = ts_map[key]

        k = bisect_right(arr, t_prev)
        if k == len(arr):
            invalid_hops += 1
            # once broken, keep counting remaining hops as invalid
            t_prev = float("inf")
        else:
            t_prev = arr[k]

    hops = len(nodes) - 1
    return hops, invalid_hops, (invalid_hops == 0)


def validate_walk_file_bidir(walk_file, ts_map, directed=False):
    """
    Validates each walk both left->right and right->left.
    Returns aggregates for:
      - left_to_right
      - right_to_left
      - best (whichever has max correct_walks)
    """
    total_hops_lr = 0
    invalid_hops_lr = 0
    correct_walks_lr = 0

    total_hops_rl = 0
    invalid_hops_rl = 0
    correct_walks_rl = 0

    with open(walk_file, "r") as f:
        for line in f:
            nodes = line.strip().split()
            if len(nodes) <= 1:
                continue

            hops, inv, ok = _validate_one_direction(nodes, ts_map, directed=directed)
            total_hops_lr += hops
            invalid_hops_lr += inv
            if ok:
                correct_walks_lr += 1

            nodes_rev = nodes[::-1]
            hops, inv, ok = _validate_one_direction(nodes_rev, ts_map, directed=directed)
            total_hops_rl += hops
            invalid_hops_rl += inv
            if ok:
                correct_walks_rl += 1

    left_to_right = (total_hops_lr, invalid_hops_lr, correct_walks_lr)
    right_to_left = (total_hops_rl, invalid_hops_rl, correct_walks_rl)

    best = left_to_right if correct_walks_lr >= correct_walks_rl else right_to_left

    return {
        "left_to_right": left_to_right,
        "right_to_left": right_to_left,
        "best": best,
        "best_direction": "left_to_right" if best is left_to_right else "right_to_left",
    }


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

        res = validate_walk_file_bidir(
            FLOWWALKER_OUTPUT_FILE,
            ts_map,
            directed=is_directed
        )
        steps_lr, invalid_lr, correct_lr = res["left_to_right"]
        steps_rl, invalid_rl, correct_rl = res["right_to_left"]
        total_steps, total_invalid, total_correct = res["best"]
        best_dir = res["best_direction"]

        invalid_step_percent = (total_invalid / total_steps) * 100 if total_steps > 0 else 0
        invalid_walk_percent = ((num_walks - total_correct) / num_walks) * 100

        print(f"Steps/sec: {steps_per_sec:.4f}")
        print(f"Best direction: {best_dir}")
        print(f"L->R Correct walks: {correct_lr} | Invalid hops: {invalid_lr}")
        print(f"R->L Correct walks: {correct_rl} | Invalid hops: {invalid_rl}")
        print(f"Invalid Hop %: {invalid_step_percent:.2f}%")
        print(f"Invalid Walk %: {invalid_walk_percent:.2f}%")

        correct_walks_runs.append(total_correct)

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

        res = validate_walk_file_bidir(
            THUNDERRW_OUTPUT_FILE,
            ts_map,
            directed=is_directed
        )
        steps_lr, invalid_lr, correct_lr = res["left_to_right"]
        steps_rl, invalid_rl, correct_rl = res["right_to_left"]
        total_steps, total_invalid, total_correct = res["best"]
        best_dir = res["best_direction"]

        invalid_step_percent = (total_invalid / total_steps) * 100 if total_steps > 0 else 0
        invalid_walk_percent = ((num_walks - total_correct) / num_walks) * 100

        print(f"Steps/sec: {steps_per_sec:.4f}")
        print(f"Best direction: {best_dir}")
        print(f"L->R Correct walks: {correct_lr} | Invalid hops: {invalid_lr}")
        print(f"R->L Correct walks: {correct_rl} | Invalid hops: {invalid_rl}")
        print(f"Invalid Hop %: {invalid_step_percent:.2f}%")
        print(f"Invalid Walk %: {invalid_walk_percent:.2f}%")

        correct_walks_runs.append(total_correct)

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

        res = validate_walk_file_bidir(
            TEMPEST_OUTPUT_FILE,
            ts_map,
            directed=is_directed
        )
        steps_lr, invalid_lr, correct_lr = res["left_to_right"]
        steps_rl, invalid_rl, correct_rl = res["right_to_left"]
        total_steps, total_invalid, total_correct = res["best"]
        best_dir = res["best_direction"]

        invalid_step_percent = (total_invalid / total_steps) * 100 if total_steps > 0 else 0
        invalid_walk_percent = ((num_walks - total_correct) / num_walks) * 100

        print(f"Steps/sec: {steps_per_sec:.4f}")
        print(f"Best direction: {best_dir}")
        print(f"L->R Correct walks: {correct_lr} | Invalid hops: {invalid_lr}")
        print(f"R->L Correct walks: {correct_rl} | Invalid hops: {invalid_rl}")
        print(f"Invalid Hop %: {invalid_step_percent:.2f}%")
        print(f"Invalid Walk %: {invalid_walk_percent:.2f}%")

        correct_walks_runs.append(total_correct)

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
