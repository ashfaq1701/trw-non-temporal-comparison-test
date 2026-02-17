#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd


# -----------------------------
# Helpers
# -----------------------------
def gini_from_counts(x: np.ndarray) -> float:
    """
    Gini coefficient for non-negative values.
    """
    x = x.astype(np.float64)
    if x.size == 0:
        return 0.0
    if np.all(x == 0):
        return 0.0
    x_sorted = np.sort(x)
    n = x_sorted.size
    cumx = np.cumsum(x_sorted)
    # Gini = (n+1 - 2 * sum(cumx)/cumx[-1]) / n
    return float((n + 1 - 2.0 * np.sum(cumx) / cumx[-1]) / n)


def summarize_dist(x: np.ndarray, name: str) -> dict:
    """
    Robust distribution summary.
    """
    x = x.astype(np.float64)
    out = {
        f"{name}_mean": float(np.mean(x)) if x.size else 0.0,
        f"{name}_std": float(np.std(x)) if x.size else 0.0,
        f"{name}_cv": float(np.std(x) / np.mean(x)) if x.size and np.mean(x) > 0 else 0.0,
        f"{name}_median": float(np.median(x)) if x.size else 0.0,
        f"{name}_p90": float(np.quantile(x, 0.90)) if x.size else 0.0,
        f"{name}_p99": float(np.quantile(x, 0.99)) if x.size else 0.0,
        f"{name}_max": float(np.max(x)) if x.size else 0.0,
        f"{name}_gini": gini_from_counts(x) if x.size else 0.0,
    }
    return out


def top_mass_fraction(counts: pd.Series, frac: float) -> float:
    """
    Fraction of all mass contributed by top frac of keys in a count Series.
    """
    if counts.empty:
        return 0.0
    k = max(1, int(np.ceil(frac * counts.size)))
    top_sum = float(counts.nlargest(k).sum())
    total = float(counts.sum())
    return top_sum / total if total > 0 else 0.0


# -----------------------------
# Main profiling
# -----------------------------
def profile_temporal_csv(
    csv_path: str,
    directed: bool,
    compute_pair_metrics: bool,
    compute_temporal_node_metrics: bool,
) -> dict:
    print(f"\n===== Profiling: {csv_path} =====")
    df = pd.read_csv(csv_path, usecols=["u", "i", "ts"])

    # Types
    df["u"] = df["u"].astype(np.int64)
    df["i"] = df["i"].astype(np.int64)
    df["ts"] = df["ts"].astype(np.int64)

    # If undirected, canonicalize endpoints for pair-based metrics
    if not directed:
        uu = np.minimum(df["u"].values, df["i"].values)
        vv = np.maximum(df["u"].values, df["i"].values)
        df["u2"] = uu
        df["i2"] = vv
    else:
        df["u2"] = df["u"]
        df["i2"] = df["i"]

    E = int(len(df))
    nodes = pd.unique(df[["u2", "i2"]].to_numpy().ravel())
    V = int(nodes.size)

    # Timestamp stats
    ts = df["ts"].values
    ts_min = int(ts.min()) if E else 0
    ts_max = int(ts.max()) if E else 0
    ts_span = int(ts_max - ts_min) if E else 0
    n_ts = int(df["ts"].nunique()) if E else 0

    # Edges per timestamp
    edges_per_ts = df.groupby("ts", sort=False).size().to_numpy()
    edges_per_ts_stats = summarize_dist(edges_per_ts, "edges_per_ts")

    # Self-loops
    self_loops = int((df["u2"].values == df["i2"].values).sum())
    self_loop_frac = float(self_loops / E) if E else 0.0

    # Degree distributions
    out_counts = df.groupby("u2", sort=False).size()
    in_counts = df.groupby("i2", sort=False).size()

    out_stats = summarize_dist(out_counts.to_numpy(), "outdeg")
    in_stats = summarize_dist(in_counts.to_numpy(), "indeg")

    # Average (static) degree notions
    # directed: avg out-degree = E / V
    # undirected: avg degree = 2E / V
    avg_static_degree = float(E / V) if (directed and V > 0) else float((2 * E) / V) if V > 0 else 0.0

    # Concentration (skew) proxies that are often throughput-relevant
    top1_out_mass = top_mass_fraction(out_counts, 0.01)
    top01_out_mass = top_mass_fraction(out_counts, 0.001)

    # Reciprocity (directed only): fraction of unique pairs that have reverse
    reciprocity = None
    if directed and compute_pair_metrics:
        # Use a compact key to avoid tuple overhead
        u = df["u2"].astype(np.uint64).values
        v = df["i2"].astype(np.uint64).values
        key_uv = (u << 32) | v
        key_vu = (v << 32) | u
        s_uv = pd.Series(key_uv)
        s_vu = pd.Series(key_vu)
        # Use sets via unique arrays (memory heavy but manageable)
        uv_unique = np.unique(s_uv.values)
        vu_unique = np.unique(s_vu.values)
        # intersection size
        # use numpy intersect1d (sorted unique)
        inter = np.intersect1d(uv_unique, vu_unique, assume_unique=True)
        reciprocity = float(inter.size / uv_unique.size) if uv_unique.size else 0.0

    # Pair / multi-edge repetition
    pair_stats = {}
    if compute_pair_metrics:
        # count multiplicity of each (u,v) pair (after undirected canonicalization if needed)
        pair_counts = df.groupby(["u2", "i2"], sort=False).size()
        pair_mult = pair_counts.to_numpy()

        pair_stats.update({
            "unique_pairs": int(pair_counts.size),
            "multi_edge_pair_frac": float((pair_mult > 1).mean()) if pair_mult.size else 0.0,
            "avg_edges_per_pair": float(pair_mult.mean()) if pair_mult.size else 0.0,
        })
        pair_stats.update(summarize_dist(pair_mult, "pair_multiplicity"))

        # Also: timestamps-per-pair (how many distinct timestamps each pair appears on)
        pair_ts_counts = df.groupby(["u2", "i2"], sort=False)["ts"].nunique().to_numpy()
        pair_stats.update(summarize_dist(pair_ts_counts, "pair_unique_ts"))

    # Temporal-node activity metrics (can be expensive)
    temporal_node_stats = {}
    if compute_temporal_node_metrics:
        # unique timestamps per source node
        u_unique_ts = df.groupby("u2", sort=False)["ts"].nunique().to_numpy()
        temporal_node_stats.update(summarize_dist(u_unique_ts, "u_unique_ts"))

        # activity span per node (max ts - min ts per node)
        u_span = (df.groupby("u2", sort=False)["ts"].max() - df.groupby("u2", sort=False)["ts"].min()).to_numpy()
        temporal_node_stats.update(summarize_dist(u_span, "u_activity_span"))

    # Assemble
    result = {
        "dataset": csv_path.split("/")[-1],
        "directed": bool(directed),
        "E_edges": E,
        "V_nodes": V,
        "unique_timesteps": n_ts,
        "ts_min": ts_min,
        "ts_max": ts_max,
        "ts_span": ts_span,
        "avg_static_degree": round(avg_static_degree, 6),
        "self_loop_frac": round(self_loop_frac, 6),
        "top1pct_out_mass_frac": round(top1_out_mass, 6),
        "top0.1pct_out_mass_frac": round(top01_out_mass, 6),
    }

    if reciprocity is not None:
        result["reciprocity_frac"] = round(float(reciprocity), 6)

    # Add distribution summaries (lots of columns, but that’s the point)
    result.update({k: round(v, 6) for k, v in edges_per_ts_stats.items()})
    result.update({k: round(v, 6) for k, v in out_stats.items()})
    result.update({k: round(v, 6) for k, v in in_stats.items()})

    result.update({k: (round(v, 6) if isinstance(v, float) else v) for k, v in pair_stats.items()})
    result.update({k: round(v, 6) for k, v in temporal_node_stats.items()})

    # Print a compact log block (paste-friendly)
    print("---- Key indicators (paste-friendly) ----")
    key_fields = [
        "E_edges", "V_nodes", "avg_static_degree",
        "outdeg_p99", "outdeg_max", "outdeg_gini",
        "edges_per_ts_mean", "edges_per_ts_p99", "edges_per_ts_cv",
        "top1pct_out_mass_frac",
    ]
    if "unique_pairs" in result:
        key_fields += ["unique_pairs", "multi_edge_pair_frac", "avg_edges_per_pair"]
    if "u_unique_ts_mean" in result:
        key_fields += ["u_unique_ts_mean", "u_unique_ts_p99", "u_activity_span_mean"]
    if "reciprocity_frac" in result:
        key_fields += ["reciprocity_frac"]

    for k in key_fields:
        if k in result:
            print(f"{k}: {result[k]}")

    return result


def main():
    ap = argparse.ArgumentParser(description="Temporal dataset profiling (general metrics).")
    ap.add_argument("--base", type=str, required=True, help="Directory containing CSVs (u,i,ts)")
    ap.add_argument("--datasets", type=str, nargs="+", required=True, help="CSV filenames")
    ap.add_argument("--directed", action="store_true", help="Treat as directed (default: undirected canonicalization)")
    ap.add_argument("--pair_metrics", action="store_true", help="Compute pair/multi-edge metrics (can be heavy)")
    ap.add_argument("--temporal_node_metrics", action="store_true", help="Compute per-node temporal metrics (can be heavy)")
    ap.add_argument("--out_csv", type=str, default="", help="Optional: write results to CSV")
    args = ap.parse_args()

    results = []
    for ds in args.datasets:
        path = f"{args.base}/{ds}"
        results.append(profile_temporal_csv(
            path,
            directed=args.directed,
            compute_pair_metrics=args.pair_metrics,
            compute_temporal_node_metrics=args.temporal_node_metrics,
        ))

    df_out = pd.DataFrame(results)

    print("\n===== FINAL SUMMARY (markdown) =====")
    # Show a reasonable subset first
    subset_cols = [
        "dataset", "E_edges", "V_nodes", "unique_timesteps", "avg_static_degree",
        "outdeg_p99", "outdeg_max", "outdeg_gini",
        "edges_per_ts_mean", "edges_per_ts_p99", "edges_per_ts_cv",
        "top1pct_out_mass_frac",
    ]
    for c in ["unique_pairs", "multi_edge_pair_frac", "avg_edges_per_pair", "reciprocity_frac",
              "u_unique_ts_mean", "u_unique_ts_p99", "u_activity_span_mean"]:
        if c in df_out.columns:
            subset_cols.append(c)

    print(df_out[subset_cols].to_markdown(index=False))

    if args.out_csv:
        df_out.to_csv(args.out_csv, index=False)
        print(f"\nWrote: {args.out_csv}")


if __name__ == "__main__":
    main()
