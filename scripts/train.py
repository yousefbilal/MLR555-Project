"""Main training script for MARFS."""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marfs.config import MARFSConfig
from marfs.trainer import MARFSTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="MARFS: Multi-Agent RL Feature Selection")

    # Dataset
    parser.add_argument("--dataset", type=str, default="synthetic",
                        choices=["wdbc", "usps", "isolet", "coil20", "colon", "covertype", "musk", "spambase", "mnist", "genomics", "synthetic"])
    parser.add_argument("--test-size", type=float, default=0.2)

    # Agent structure
    parser.add_argument("--n-agents", type=int, default=10)
    parser.add_argument("--assignment", type=str, default="random",
                        choices=["random", "kmeans", "spectral", "gcn_spectral"])

    # GCN
    parser.add_argument("--gcn-hidden", type=int, default=128)
    parser.add_argument("--gcn-out", type=int, default=64)
    parser.add_argument("--corr-threshold", type=float, default=0.3)
    parser.add_argument("--use-stat-descriptor", action="store_true")
    parser.add_argument("--global-readout", type=str, default="mean",
                        choices=["mean", "attention"])

    # Policy conditioning
    parser.add_argument("--conditioning", type=str, default="adaln",
                        choices=["concat", "film", "adaln"])
    parser.add_argument("--agent-emb-dim", type=int, default=16)
    parser.add_argument("--policy-hidden", type=int, default=128)

    # Action space
    parser.add_argument("--collective-action", action="store_true",
                        help="EAC-FS style collective select/deselect per group")

    # Initialization
    parser.add_argument("--init-strategy", type=str, default="all",
                        choices=["all", "none", "half", "random"])

    # Contrastive pre-training
    parser.add_argument("--contrastive-pretrain", action="store_true")
    parser.add_argument("--contrastive-epochs", type=int, default=100)

    # Exploration
    parser.add_argument("--exploration", type=str, default="entropy",
                        choices=["entropy", "eps_greedy"],
                        help="Exploration strategy: 'entropy' (Bernoulli + entropy bonus) or 'eps_greedy'")
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay-episodes", type=int, default=150)

    # Reward
    parser.add_argument("--reward-type", type=str, default="hierarchical",
                        choices=["simple", "hierarchical"])
    parser.add_argument("--reward-classifier", type=str, default="ridge",
                        choices=["ridge", "rf", "lightgbm", "xgboost"],
                        help="Classifier used to compute the reward signal")
    parser.add_argument("--w-acc", type=float, default=0.6)
    parser.add_argument("--w-size", type=float, default=0.1)
    parser.add_argument("--w-redundancy", type=float, default=0.1)

    # Training
    parser.add_argument("--lr-gcn", type=float, default=1e-4)
    parser.add_argument("--lr-policy", type=float, default=3e-4)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--mini-batch-size", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--n-episodes", type=int, default=200)
    parser.add_argument("--early-stop-patience", type=int, default=50)

    # Ablation flags
    parser.add_argument("--no-local-state", action="store_true")
    parser.add_argument("--no-global-state", action="store_true")

    # Experiment
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="marfs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="results")
    parser.add_argument("--log-freq", type=int, default=10)

    return parser.parse_args()


def main():
    args = parse_args()

    config = MARFSConfig(
        dataset=args.dataset,
        test_size=args.test_size,
        n_agents=args.n_agents,
        assignment=args.assignment,
        gcn_hidden=args.gcn_hidden,
        gcn_out=args.gcn_out,
        corr_threshold=args.corr_threshold,
        use_stat_descriptor=args.use_stat_descriptor,
        global_readout=args.global_readout,
        conditioning=args.conditioning,
        agent_emb_dim=args.agent_emb_dim,
        policy_hidden=args.policy_hidden,
        collective_action=args.collective_action,
        init_strategy=args.init_strategy,
        contrastive_pretrain=args.contrastive_pretrain,
        contrastive_epochs=args.contrastive_epochs,
        exploration=args.exploration,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_episodes=args.eps_decay_episodes,
        reward_type=args.reward_type,
        reward_classifier=args.reward_classifier,
        w_acc=args.w_acc,
        w_size=args.w_size,
        w_redundancy=args.w_redundancy,
        lr_gcn=args.lr_gcn,
        lr_policy=args.lr_policy,
        clip_ratio=args.clip_ratio,
        entropy_coeff=args.entropy_coeff,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        n_epochs=args.n_epochs,
        mini_batch_size=args.mini_batch_size,
        n_steps_per_episode=args.n_steps,
        n_episodes=args.n_episodes,
        early_stop_patience=args.early_stop_patience,
        use_local_state=not args.no_local_state,
        use_global_state=not args.no_global_state,
        seed=args.seed,
        device=args.device,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
        save_dir=args.save_dir,
        log_freq=args.log_freq,
    )

    trainer = MARFSTrainer(config)
    results = trainer.train()

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Downstream accuracy (Ridge):  {results['downstream_accuracy']:.4f}")
    print(f"  Downstream accuracy (RF):     {results['downstream_rf_accuracy']:.4f}")
    print(f"  Features selected:            {results['n_selected']}/{results['n_features']}")
    print(f"  Compression ratio:            {results['compression']:.1%}")
    print(f"  Convergence step:             {results['convergence_step']}")
    print(f"  Stability (Jaccard):          {results['stability']:.4f}")
    print(f"  Peak GPU memory:              {results['peak_memory_mb']:.1f} MB")
    print(f"  Total time:                   {results['total_time']:.1f}s")


if __name__ == "__main__":
    # import logging
    # logging.basicConfig(level=logging.INFO)
    main()
