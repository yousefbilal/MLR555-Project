"""Visualization utilities for MARFS experiment results.

Generates all plots needed for the paper:
- Convergence curves (reward + accuracy)
- Accuracy bar chart across configs
- Compression vs accuracy scatter
- Memory and speed comparison
- Scaling experiment (training time vs K)
- Attention weight visualization
- LaTeX table
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_results(results_dir: str) -> dict:
    path = os.path.join(results_dir, "ablation_results.json")
    if not os.path.exists(path):
        results = {}
        for f in sorted(os.listdir(results_dir)):
            if f.endswith(".json") and f != "ablation_results.json":
                with open(os.path.join(results_dir, f)) as fh:
                    results[f.replace(".json", "")] = json.load(fh)
        return results

    with open(path) as f:
        return json.load(f)


def plot_convergence_curves(results: dict, save_dir: str):
    """Plot reward and accuracy convergence curves."""
    n_datasets = len(results)
    fig, axes = plt.subplots(2, max(n_datasets, 1), figsize=(7 * max(n_datasets, 1), 10),
                             squeeze=False)

    for col, (dataset, configs) in enumerate(results.items()):
        for row, metric_key in enumerate(["reward_history", "accuracy_history"]):
            ax = axes[row, col]
            for config_name, data in configs.items():
                if "individual_runs" not in data:
                    continue
                histories = [r[metric_key] for r in data["individual_runs"]
                             if metric_key in r]
                if not histories:
                    continue

                min_len = min(len(h) for h in histories)
                trimmed = [h[:min_len] for h in histories]
                mean_vals = np.mean(trimmed, axis=0)
                std_vals = np.std(trimmed, axis=0)

                label = config_name.split("_")[0]
                ax.plot(np.arange(min_len), mean_vals, label=label, linewidth=1.5)
                ax.fill_between(np.arange(min_len),
                                mean_vals - std_vals, mean_vals + std_vals, alpha=0.12)

            metric_label = "Reward" if "reward" in metric_key else "Accuracy"
            ax.set_xlabel("Episode")
            ax.set_ylabel(metric_label)
            ax.set_title(f"{metric_label} — {dataset}")
            ax.legend(fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "convergence_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_accuracy_comparison(results: dict, save_dir: str):
    n_datasets = len(results)
    fig, axes = plt.subplots(1, max(n_datasets, 1),
                             figsize=(7 * max(n_datasets, 1), 5), squeeze=False)

    for col, (dataset, configs) in enumerate(results.items()):
        ax = axes[0, col]
        names, means_rf, stds_rf = [], [], []

        for config_name, data in configs.items():
            agg = data.get("aggregated", {})
            names.append(config_name.split("_")[0])
            if "downstream_rf_accuracy" in agg:
                means_rf.append(agg["downstream_rf_accuracy"]["mean"])
                stds_rf.append(agg["downstream_rf_accuracy"]["std"])
            else:
                means_rf.append(0)
                stds_rf.append(0)

        x = np.arange(len(names))
        ax.bar(x, means_rf, 0.6, yerr=stds_rf,
               label="RF", capsize=3, color="coral")

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Downstream RF Accuracy — {dataset}")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(save_dir, "accuracy_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_compression_vs_accuracy(results: dict, save_dir: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    markers = ["o", "s", "^", "D", "v", "P", "*", "X", "h", "p"]
    colors = sns.color_palette("husl", n_colors=12)
    seen_labels = set()

    for dataset, configs in results.items():
        for ci, (config_name, data) in enumerate(configs.items()):
            runs = data.get("individual_runs", [])
            label = config_name.split("_")[0]
            for run in runs:
                acc = run.get("downstream_rf_accuracy", 0)
                comp = run.get("compression", 0)
                lbl = label if label not in seen_labels else None
                ax.scatter(comp, acc,
                           marker=markers[ci % len(markers)],
                           color=colors[ci % len(colors)],
                           alpha=0.6, s=50, label=lbl)
                seen_labels.add(label)

    ax.legend(fontsize=8)
    ax.set_xlabel("Compression Ratio")
    ax.set_ylabel("Downstream Accuracy (RF)")
    ax.set_title("Compression vs Accuracy Trade-off")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "compression_vs_accuracy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_memory_and_speed(results: dict, save_dir: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for dataset, configs in results.items():
        names, memory, times = [], [], []
        for config_name, data in configs.items():
            agg = data.get("aggregated", {})
            names.append(config_name.split("_")[0])
            memory.append(agg.get("peak_memory_mb", {}).get("mean", 0))
            times.append(agg.get("total_time", {}).get("mean", 0))

        x = np.arange(len(names))
        ax1.bar(x, memory, color="steelblue")
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=45, ha="right")
        ax1.set_ylabel("Peak GPU Memory (MB)")
        ax1.set_title(f"Memory — {dataset}")
        ax1.grid(True, alpha=0.3, axis="y")

        ax2.bar(x, times, color="coral")
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=45, ha="right")
        ax2.set_ylabel("Training Time (s)")
        ax2.set_title(f"Time — {dataset}")
        ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(save_dir, "memory_and_speed.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_scaling(save_dir: str):
    """Plot training time vs K (from scaling experiment results)."""
    path = os.path.join(save_dir, "scaling_results.json")
    if not os.path.exists(path):
        print("No scaling_results.json found. Run ablation.py --scaling first.")
        return

    with open(path) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(7, 5))

    for method, label, color in [("ours", "Ours (AdaLN)", "steelblue"),
                                  ("eacfs", "EAC-FS (Collective)", "coral")]:
        ks = sorted([int(k) for k in data[method].keys() if data[method][k] is not None])
        times = [data[method][str(k)] for k in ks]
        ax.plot(ks, times, "o-", label=label, color=color, linewidth=2, markersize=8)

    ax.set_xlabel("Number of Agents (K)")
    ax.set_ylabel("Training Time (s)")
    ax.set_title("Scalability: Training Time vs Number of Agents")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "scaling_time_vs_k.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def generate_latex_table(results: dict, save_dir: str):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Ablation study results. Each row (A$\to$H) adds one component. "
        r"Config~H is the full system.}",
        r"\label{tab:ablation}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l l " + "c" * 5 + "}",
        r"\toprule",
        r"Config & Key Change & Acc (RF) & Compression & Conv.\ Step & Stability & Time (s) \\",
        r"\midrule",
    ]

    key_changes = {
        "A": "Per-feature binary",
        "B": "+ Local readout",
        "C": "+ Contrastive pretrain",
        "D": "+ K-Means assignment",
        "E": "+ GCN-spectral assign.",
        "F": "+ FiLM conditioning",
        "G": "+ AdaLN conditioning",
        "H": "+ Attention readout",
    }

    for dataset, configs in results.items():
        lines.append(r"\multicolumn{7}{l}{\textbf{" + dataset.replace("_", r"\_") + r"}} \\")
        lines.append(r"\midrule")

        for config_name, data in configs.items():
            agg = data.get("aggregated", {})

            def fmt(key):
                if key in agg:
                    return f"${agg[key]['mean']:.3f} \\pm {agg[key]['std']:.3f}$"
                return "---"

            short = config_name.split("_")[0]
            change = key_changes.get(short, "---")
            name_tex = config_name.replace("_", r"\_")
            row = (f"{short} & {change} & "
                   f"{fmt('downstream_rf_accuracy')} & "
                   f"{fmt('compression')} & "
                   f"{fmt('convergence_step')} & "
                   f"{fmt('stability')} & "
                   f"{fmt('total_time')} \\\\")
            lines.append(row)

        lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")

    latex = "\n".join(lines)
    path = os.path.join(save_dir, "ablation_table.tex")
    with open(path, "w") as f:
        f.write(latex)
    print(f"LaTeX table saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize MARFS results")
    parser.add_argument("--results-dir", type=str, default="results/ablation")
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()

    save_dir = args.save_dir or args.results_dir
    os.makedirs(save_dir, exist_ok=True)

    results = load_results(args.results_dir)
    if not results:
        print("No results found. Run the ablation study first.")
        return

    print("Generating visualizations...")
    plot_convergence_curves(results, save_dir)
    plot_accuracy_comparison(results, save_dir)
    plot_compression_vs_accuracy(results, save_dir)
    plot_memory_and_speed(results, save_dir)
    plot_scaling(save_dir)
    generate_latex_table(results, save_dir)
    print("Done!")


if __name__ == "__main__":
    main()
