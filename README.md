# MARFS — Multi-Agent RL for Feature Selection

A multi-agent PPO framework for unsupervised feature selection on tabular data. Features are partitioned across `M` agents; each agent learns per-feature on/off decisions over its assigned group. State representations come from a GCN over the feature-correlation graph, and agents are conditioned on a learned embedding via concat / FiLM / AdaLN.

## Pipeline

```
Dataset
  │
  ├─ (optional) Contrastive GCN pre-training
  │
  ├─ Feature → Agent assignment   (random | k-means | spectral | gcn-spectral)
  │
  └─ Multi-agent PPO loop
       ├─ GCN encodes the masked feature graph → per-agent state
       ├─ Policy (concat / FiLM / AdaLN-conditioned MLP) outputs per-feature Bernoulli
       ├─ Environment toggles features in the global mask
       ├─ Reward: classifier accuracy − size penalty − per-agent redundancy
       └─ PPO update on GCN + policy + agent embeddings
```

## Repository layout

```
marfs/
  config.py            MARFSConfig dataclass (all hyperparameters)
  trainer.py           MARFSTrainer: pipeline orchestration + PPO loop
  reward.py            Hierarchical / simple reward (Ridge / RF / LightGBM / XGBoost classifiers)
  metrics.py           Evaluation utilities and metric aggregation
  utils.py             Misc helpers (safe correlation, etc.)
  dataset/
    loader.py          Dataset loaders + balanced-ish undersampling
    assignment.py      Random / k-means / spectral / gcn-spectral feature assignment
    synthetic.py       Synthetic dataset generator
  environ/
    feature_env.py     PettingZoo-style multi-agent feature-selection env
  models/
    gcn.py             GCNStateEncoder, dynamic graph builder, contrastive pretrainer
    policy.py          PPOPolicy with concat / FiLM / AdaLN conditioning

scripts/
  train.py             Single-run training CLI
  ablation.py          A–H ablation table runner + scaling experiment
  visualize.py         Plots and LaTeX table generation
```

## Setup

```bash
pip install -r requirements.txt
```

PyTorch + PyG wheels are pinned to CUDA 12.8 in `requirements.txt`. Adjust the `--extra-index-url` / `--find-links` lines if your toolchain differs.

## Quick start

Train on the synthetic dataset:

```bash
python scripts/train.py --dataset synthetic --n-episodes 200
```

Train with the full system on WDBC:

```bash
python scripts/train.py \
  --dataset wdbc \
  --assignment gcn_spectral \
  --conditioning adaln \
  --global-readout attention \
  --contrastive-pretrain
```

Run the ablation table on a few datasets:

```bash
python scripts/ablation.py \
  --datasets synthetic,wdbc,covertype \
  --n-seeds 3 --n-episodes 200 \
  --save-dir results/ablation
```

Generate plots and a LaTeX table from the saved results:

```bash
python scripts/visualize.py --results-dir results/ablation
```

## Datasets

Built-in loaders (`scripts/train.py --dataset ...`):
`wdbc`, `usps`, `isolet`, `coil20`, `colon`, `covertype`, `musk`, `spambase`, `mnist`, `genomics`, `synthetic`.

Large datasets are undersampled to fit a row budget. Class balance is tunable via `max_imbalance_ratio` in `load_dataset`: minority classes are kept in full, majority classes are capped at `ratio × min_count` (default `3.0`, so up to 3:1 imbalance is preserved).

## Ablation table

`scripts/ablation.py` runs an A–H ladder. Each row flips one variable from the previous:

| Row | Change |
|---|---|
| A | Per-feature binary action, global-only readout, random assignment, concat |
| B | + dual readout (global + local) |
| C | + contrastive GCN pre-training (kept on for D–H) |
| D | + K-Means feature assignment |
| E | + GCN-spectral assignment (uses pretrained GCN embeddings) |
| F | + FiLM conditioning |
| G | + AdaLN conditioning |
| H | + attention-based global readout (full system) |

Final downstream accuracy is reported with a Random Forest (100 trees). The inner-loop reward classifier is selectable via `--reward-classifier {ridge, rf, lightgbm, xgboost}`.

## Configuration

All hyperparameters live in `marfs/config.MARFSConfig`. Key knobs:

- **Agent structure:** `n_agents`, `assignment`, `collective_action`
- **GCN:** `gcn_hidden`, `gcn_out`, `corr_threshold`, `global_readout` (`mean` / `attention`)
- **Policy:** `conditioning` (`concat` / `film` / `adaln`), `policy_hidden`, `agent_emb_dim`
- **Reward:** `reward_type`, `reward_classifier`, `w_acc`, `w_size`, `w_redundancy`
- **PPO:** `lr_gcn`, `lr_policy`, `clip_ratio`, `entropy_coeff`, `gamma`, `gae_lambda`, `n_epochs`
- **Pre-training:** `contrastive_pretrain`, `contrastive_epochs`, `contrastive_tau`

See `python scripts/train.py --help` for the CLI surface.

## Outputs

Each run writes to `--save-dir/<run_name>/`:
- `results.json` — final mask, downstream RF accuracy, compression, convergence step, stability, timings
- `episode_metrics.json` — per-episode reward / accuracy / loss / entropy
- (optional) wandb logs if `--use-wandb` is passed
