import pandas as pd
import numpy as np
from bisect import bisect_right
import random

def compute_dataset_metrics(csv_path, directed=False, sample_size=1_000_000):
    print(f"\n===== Processing: {csv_path} =====")

    df = pd.read_csv(csv_path)

    df["u"] = df["u"].astype(np.int64)
    df["i"] = df["i"].astype(np.int64)
    df["ts"] = df["ts"].astype(np.int64)

    num_edges = len(df)
    num_nodes = pd.unique(df[["u", "i"]].values.ravel()).size

    # ============================
    # 1. Avg Static Degree
    # ============================

    if directed:
        avg_static_degree = num_edges / num_nodes
    else:
        avg_static_degree = (2 * num_edges) / num_nodes

    print(f"Total edges: {num_edges}")
    print(f"Total nodes: {num_nodes}")
    print(f"Avg Static Degree: {avg_static_degree:.4f}")

    # ============================
    # 2. Build Node → Sorted Timestamp Map
    # ============================

    print("Building node-level timestamp index...")

    df_sorted = df.sort_values(["u", "ts"])

    node_ts_map = {}
    for u, group in df_sorted.groupby("u"):
        node_ts_map[u] = group["ts"].values

    print("Index built.")

    # ============================
    # 3. Estimate Avg Γ_t(u) Under Sampling
    # ============================

    print(f"Sampling {sample_size} hop states...")

    total_gamma = 0
    counted = 0

    u_values = df["u"].values
    ts_values = df["ts"].values

    indices = np.random.choice(len(df), size=min(sample_size, len(df)), replace=False)

    for idx in indices:
        u = u_values[idx]
        t_prev = ts_values[idx]

        arr = node_ts_map.get(u)
        if arr is None:
            continue

        j = bisect_right(arr, t_prev)
        gamma_size = len(arr) - j

        total_gamma += gamma_size
        counted += 1

    avg_temporal_branching = total_gamma / counted if counted > 0 else 0

    print(f"Avg Temporal Γ_t(u) Size (Sampled): {avg_temporal_branching:.4f}")

    return {
        "Edges": num_edges,
        "Nodes": num_nodes,
        "Avg Static Degree": round(avg_static_degree, 4),
        "Avg Temporal Branching (Γ_t)": round(avg_temporal_branching, 4)
    }


if __name__ == "__main__":
    BASE_PATH = "/mnt/lustre/users/inf/ms2420/non-temporal-comparison-datasets/temporal"

    datasets = [
        "growth.csv",
        "delicious.csv",
        "ml_tgbl-coin.csv",
        "ml_tgbl-flight.csv"
    ]

    all_results = []

    for ds in datasets:
        path = f"{BASE_PATH}/{ds}"
        metrics = compute_dataset_metrics(path, directed=True)
        all_results.append({"Dataset": ds, **metrics})

    print("\n===== FINAL SUMMARY TABLE =====")
    print(pd.DataFrame(all_results).to_markdown(index=False))
