import pandas as pd
import numpy as np

def compute_dataset_metrics(csv_path, directed=False):
    print(f"\n===== Processing: {csv_path} =====")

    df = pd.read_csv(csv_path)

    df["u"] = df["u"].astype(np.int64)
    df["i"] = df["i"].astype(np.int64)
    df["ts"] = df["ts"].astype(np.int64)

    num_edges = len(df)

    # Unique node count
    num_nodes = pd.unique(df[["u", "i"]].values.ravel()).size

    print(f"Total edges: {num_edges}")
    print(f"Total nodes: {num_nodes}")

    # ============================
    # 1️⃣ Avg Static Degree
    # ============================

    if directed:
        # Average out-degree
        avg_static_degree = num_edges / num_nodes
    else:
        # Undirected degree
        avg_static_degree = (2 * num_edges) / num_nodes

    print(f"Avg Static Degree: {avg_static_degree:.4f}")

    # ============================
    # 2️⃣ Exact Avg Temporal Branching (Γ_t)
    # ============================

    # Count outgoing edges per node
    out_degree = df.groupby("u").size().values

    # Exact expectation:
    # sum_u k_u (k_u - 1) / 2 divided by total edges
    total_future_edges = np.sum(out_degree * (out_degree - 1) / 2)

    avg_temporal_branching = total_future_edges / num_edges

    print(f"Avg Temporal Branching (Exact Γ_t): {avg_temporal_branching:.4f}")

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
