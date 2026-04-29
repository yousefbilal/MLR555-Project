"""Configuration dataclasses for MARFS."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MARFSConfig:
    # --- Dataset ---
    dataset: str = "synthetic"  # wdbc, control, usps, isolet, orl, yale, coil20, colon, covertype, mnist, genomics, synthetic
    test_size: float = 0.2

    # --- Agent structure ---
    n_agents: int = 10  # M
    assignment: str = "random"  # random, kmeans, spectral, gcn_spectral

    # --- GCN ---
    gcn_input_dim: Optional[int] = None  # auto-set from data
    gcn_hidden: int = 128
    gcn_out: int = 64
    corr_threshold: float = 0.3
    use_stat_descriptor: bool = False
    stat_descriptor_dim: int = 16

    # --- Readout ---
    global_readout: str = "mean"  # "mean" or "attention"
    agent_emb_dim: int = 16

    # --- Policy conditioning ---
    conditioning: str = "adaln"  # "concat", "film", "adaln"
    policy_hidden: int = 128

    # --- Action space ---
    collective_action: bool = False  # True = EAC-FS style (select-all/deselect-all per group)

    # --- Initialization ---
    init_strategy: str = "all"  # "all", "none", "half", "random"

    # --- Contrastive pre-training ---
    contrastive_pretrain: bool = False
    contrastive_epochs: int = 100
    contrastive_lr: float = 1e-3
    contrastive_tau: float = 0.1

    # --- Reward ---
    reward_type: str = "hierarchical"  # "simple" or "hierarchical"
    reward_classifier: str = "ridge"  # "ridge", "rf", "lightgbm", "xgboost"
    w_acc: float = 0.6  # weight for accuracy in hierarchical reward
    w_size: float = 0.1  # weight for size penalty
    w_redundancy: float = 0.1  # weight for local redundancy penalty
    lambda_redundancy: float = 0.1  # for simple reward
    lambda_relevance: float = 0.1  # for simple reward

    # --- PPO ---
    lr_gcn: float = 1e-4
    lr_policy: float = 3e-4
    lr_embeds: float = 3e-4
    clip_ratio: float = 0.2
    entropy_coeff: float = 0.01
    value_loss_coeff: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    mini_batch_size: int = 64

    # --- Episode ---
    n_steps_per_episode: int = 500
    n_episodes: int = 200
    early_stop_patience: int = 50
    early_stop_epsilon: float = 1e-4

    # --- Ablation flags ---
    weight_sharing: bool = True
    use_local_state: bool = True
    use_global_state: bool = True

    # --- Experiment ---
    seed: int = 42
    device: str = "cuda"
    use_wandb: bool = False
    wandb_project: str = "marfs"
    run_name: Optional[str] = None
    save_dir: str = "results"
    n_eval_seeds: int = 5
    
    log_freq: int = 10

    def __post_init__(self):
        # Compute state_dim (GCN readout, without agent embedding)
        self.state_dim = 0
        if self.use_global_state:
            self.state_dim += self.gcn_out
        if self.use_local_state:
            self.state_dim += self.gcn_out
        if self.state_dim == 0:
            self.state_dim = self.gcn_out
            self.use_global_state = True

        # Policy input dim depends on conditioning method
        if self.conditioning == "concat":
            self.obs_dim = self.state_dim + self.agent_emb_dim
        else:
            # AdaLN/FiLM inject embedding via modulation, not concatenation
            self.obs_dim = self.state_dim
