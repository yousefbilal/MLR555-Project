"""Ablation study runner for MARFS.

Implements the full 10-config ablation table (A through J) from the paper.
"""

import argparse
import json
import os
import sys
import traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marfs.config import MARFSConfig
from marfs.trainer import MARFSTrainer
from marfs.metrics import compute_all_metrics


# -----------------------------------------------------------------------
# Full ablation table: A through J
# Each row isolates one variable from the previous row.
# -----------------------------------------------------------------------
ABLATION_CONFIGS = {
    # A: EAC-FS baseline reimplemented in our framework
    "A_eacfs_baseline": {
        "collective_action": True,
        "global_readout": "mean",
        "assignment": "random",
        "conditioning": "concat",
        "use_local_state": False,
        "contrastive_pretrain": False,
        "reward_type": "hierarchical",
        "description": "EAC-FS baseline: collective action, AE-style, random, independent",
    },
    # B: Per-feature binary control (our key departure from EAC-FS)
    "B_perfeat_binary": {
        "collective_action": False,
        "global_readout": "mean",
        "assignment": "random",
        "conditioning": "concat",
        "use_local_state": False,
        "contrastive_pretrain": False,
        "reward_type": "hierarchical",
        "description": "Per-feature binary action, concat, global-only, random",
    },
    # C: Add GCN global readout (vs AE-style)
    "C_gcn_global": {
        "collective_action": False,
        "global_readout": "mean",
        "assignment": "random",
        "conditioning": "concat",
        "use_local_state": False,
        "contrastive_pretrain": False,
        "reward_type": "hierarchical",
        "description": "GCN global readout only, concat, random",
    },
    # D: Add dual readout (global + local)
    "D_gcn_dual": {
        "collective_action": False,
        "global_readout": "mean",
        "assignment": "random",
        "conditioning": "concat",
        "use_local_state": True,
        "contrastive_pretrain": False,
        "reward_type": "hierarchical",
        "description": "GCN dual readout (global+local), concat, random",
    },
    # E: K-Means assignment (vs random)
    "E_kmeans": {
        "collective_action": False,
        "global_readout": "mean",
        "assignment": "kmeans",
        "conditioning": "concat",
        "use_local_state": True,
        "contrastive_pretrain": False,
        "reward_type": "hierarchical",
        "description": "GCN dual, K-Means assignment, concat",
    },
    # F: GCN-informed spectral clustering
    "F_gcn_spectral": {
        "collective_action": False,
        "global_readout": "mean",
        "assignment": "gcn_spectral",
        "conditioning": "concat",
        "use_local_state": True,
        "contrastive_pretrain": False,
        "reward_type": "hierarchical",
        "description": "GCN dual, GCN-informed spectral clustering, concat",
    },
    # G: FiLM conditioning (vs concat)
    "G_film": {
        "collective_action": False,
        "global_readout": "mean",
        "assignment": "gcn_spectral",
        "conditioning": "film",
        "use_local_state": True,
        "contrastive_pretrain": False,
        "reward_type": "hierarchical",
        "description": "GCN dual, GCN-spectral, FiLM conditioning",
    },
    # H: AdaLN conditioning (vs FiLM)
    "H_adaln": {
        "collective_action": False,
        "global_readout": "mean",
        "assignment": "gcn_spectral",
        "conditioning": "adaln",
        "use_local_state": True,
        "contrastive_pretrain": False,
        "reward_type": "hierarchical",
        "description": "GCN dual, GCN-spectral, AdaLN conditioning",
    },
    # I: Attention-based global readout (vs mean pooling)
    "I_attention": {
        "collective_action": False,
        "global_readout": "attention",
        "assignment": "gcn_spectral",
        "conditioning": "adaln",
        "use_local_state": True,
        "contrastive_pretrain": False,
        "reward_type": "hierarchical",
        "description": "GCN dual + attention readout, GCN-spectral, AdaLN",
    },
    # J: Full system (+ contrastive pre-training)
    "J_full": {
        "collective_action": False,
        "global_readout": "attention",
        "assignment": "gcn_spectral",
        "conditioning": "adaln",
        "use_local_state": True,
        "contrastive_pretrain": True,
        "reward_type": "hierarchical",
        "description": "FULL: attention + GCN-spectral + AdaLN + contrastive pre-training",
    },
}

# All 9 EAC-FS benchmark datasets
EACFS_DATASETS = [
    "wdbc", "control", "usps", "isolet", "orl",
    "yale", "coil20", "colon", "covertype",
]

DEFAULT_DATASETS = ["synthetic", "wdbc", "covertype"]
SEEDS = [42, 123, 456, 789, 1024]


def run_ablation(args):
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    configs_to_run = args.configs.split(",") if args.configs else list(ABLATION_CONFIGS.keys())
    datasets = args.datasets.split(",") if args.datasets else DEFAULT_DATASETS
    seeds = list(range(args.n_seeds))

    all_results = {}
    total_runs = len(configs_to_run) * len(datasets) * len(seeds)
    run_idx = 0

    for dataset in datasets:
        all_results[dataset] = {}
        for config_name in configs_to_run:
            if config_name not in ABLATION_CONFIGS:
                print(f"Warning: Unknown config '{config_name}', skipping.")
                continue

            ablation = ABLATION_CONFIGS[config_name]
            print(f"\n{'='*70}")
            print(f"Config: {config_name} | Dataset: {dataset}")
            print(f"  {ablation['description']}")
            print(f"{'='*70}")

            run_results = []
            for seed_idx, seed in enumerate(seeds):
                run_idx += 1
                print(f"\n--- Run {run_idx}/{total_runs}: seed={seed} ---")

                config = MARFSConfig(
                    dataset=dataset,
                    n_agents=args.n_agents,
                    assignment=ablation["assignment"],
                    conditioning=ablation["conditioning"],
                    global_readout=ablation["global_readout"],
                    collective_action=ablation["collective_action"],
                    use_local_state=ablation["use_local_state"],
                    use_global_state=True,
                    contrastive_pretrain=ablation["contrastive_pretrain"],
                    reward_type=ablation["reward_type"],
                    n_episodes=args.n_episodes,
                    n_steps_per_episode=args.n_steps,
                    seed=seed,
                    device=args.device,
                    use_wandb=args.use_wandb,
                    save_dir=save_dir,
                    run_name=f"{config_name}_{dataset}_seed{seed}",
                )

                try:
                    trainer = MARFSTrainer(config)
                    result = trainer.train()
                    run_results.append(result)
                except Exception as e:
                    print(f"  ERROR: {e}")
                    traceback.print_exc()
                    continue

            if run_results:
                agg = compute_all_metrics(run_results)
                all_results[dataset][config_name] = {
                    "aggregated": agg,
                    "individual_runs": run_results,
                    "description": ablation["description"],
                }

    # Save results
    results_path = os.path.join(save_dir, "ablation_results.json")

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)
    print(f"\nFull results saved to {results_path}")

    print_summary_table(all_results)


def print_summary_table(all_results: dict):
    print("\n" + "=" * 110)
    print("ABLATION STUDY RESULTS (A through J)")
    print("=" * 110)

    for dataset, configs in all_results.items():
        print(f"\nDataset: {dataset}")
        print("-" * 110)
        header = (f"{'Config':<22} {'Acc(Ridge)':<14} {'Acc(RF)':<14} "
                  f"{'Compress':<12} {'ConvStep':<12} {'Stability':<12} {'Time(s)':<12}")
        print(header)
        print("-" * 110)

        for config_name, data in configs.items():
            agg = data["aggregated"]

            def fmt(key):
                if key in agg:
                    return f"{agg[key]['mean']:.4f}±{agg[key]['std']:.4f}"
                return "N/A"

            short_name = config_name[:22]
            row = (f"{short_name:<22} "
                   f"{fmt('downstream_accuracy'):<14} "
                   f"{fmt('downstream_rf_accuracy'):<14} "
                   f"{fmt('compression'):<12} "
                   f"{fmt('convergence_step'):<12} "
                   f"{fmt('stability'):<12} "
                   f"{fmt('total_time'):<12}")
            print(row)

        print("-" * 110)


def run_scaling_experiment(args):
    """Experiment 1: Training time vs K (number of agents).

    Plots training time per episode for different K values,
    comparing our system vs EAC-FS collective action.
    """
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    k_values = [5, 10, 20, 30, 50]
    dataset = args.datasets.split(",")[0] if args.datasets else "synthetic"

    results = {"ours": {}, "eacfs": {}}

    for k in k_values:
        print(f"\n--- K={k} agents ---")

        # Our system (AdaLN, shared weights)
        config_ours = MARFSConfig(
            dataset=dataset, n_agents=k, assignment="random",
            conditioning="adaln", collective_action=False,
            n_episodes=10, n_steps_per_episode=20,
            seed=42, device=args.device, save_dir=save_dir,
        )
        try:
            trainer = MARFSTrainer(config_ours)
            r = trainer.train()
            results["ours"][k] = r["total_time"]
        except Exception as e:
            print(f"  Ours K={k} failed: {e}")
            results["ours"][k] = None

        # EAC-FS baseline (collective action)
        config_eacfs = MARFSConfig(
            dataset=dataset, n_agents=k, assignment="random",
            conditioning="concat", collective_action=True,
            n_episodes=10, n_steps_per_episode=20,
            seed=42, device=args.device, save_dir=save_dir,
        )
        try:
            trainer = MARFSTrainer(config_eacfs)
            r = trainer.train()
            results["eacfs"][k] = r["total_time"]
        except Exception as e:
            print(f"  EAC-FS K={k} failed: {e}")
            results["eacfs"][k] = None

    print("\n--- Scaling Results ---")
    print(f"{'K':<10} {'Ours (s)':<15} {'EAC-FS (s)':<15}")
    for k in k_values:
        ours_t = f"{results['ours'].get(k, 'N/A')}"
        eacfs_t = f"{results['eacfs'].get(k, 'N/A')}"
        print(f"{k:<10} {ours_t:<15} {eacfs_t:<15}")

    with open(os.path.join(save_dir, "scaling_results.json"), "w") as f:
        json.dump(results, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="MARFS Ablation Study")
    parser.add_argument("--configs", type=str, default=None,
                        help="Comma-separated config names (A_eacfs_baseline,...,J_full). Default: all.")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated dataset names. Default: synthetic,wdbc,covertype")
    parser.add_argument("--n-agents", type=int, default=10)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-episodes", type=int, default=200)
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--save-dir", type=str, default="results/ablation")
    parser.add_argument("--scaling", action="store_true",
                        help="Run scaling experiment (training time vs K)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.scaling:
        run_scaling_experiment(args)
    else:
        run_ablation(args)
