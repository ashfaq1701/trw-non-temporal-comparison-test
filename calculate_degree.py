import pandas as pd
import numpy as np

def compute_dataset_metrics(csv_path, directed=False):
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
    # 2. Avg Temporal-Valid Neighborhood Size
    # ============================

    # Sort by source and timestamp
    df_sorted = df.sort_values(["u", "ts"])

    # For each (u), count how many future edges remain
    df_sorted["future_count"] = (
        df_sorted.groupby("u").cumcount(ascending=False)
    )

    avg_temporal_valid = df_sorted["future_count"].mean()

    print(f"Avg Temporal-Valid Neighborhood Size: {avg_temporal_valid:.4f}")

    return {
        "Edges": num_edges,
        "Nodes": num_nodes,
        "Avg Static Degree": round(avg_static_degree, 4),
        "Avg Temporal Valid Neighborhood": round(avg_temporal_valid, 4)
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
