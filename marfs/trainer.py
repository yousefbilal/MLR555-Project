"""PPO trainer for multi-agent feature selection.

Handles the full pipeline:
1. Optional contrastive GCN pre-training
2. Optional GCN-informed feature clustering
3. Multi-agent PPO training with dual readout and AdaLN/FiLM/concat conditioning
"""

import time
import os
import json

import numpy as np
import torch
import torch.nn as nn

from .config import MARFSConfig
from .dataset import load_dataset, assign_features
from .models.gcn import GCNStateEncoder, DynamicGraphBuilder, ContrastivePretrainer
from .models.policy import PPOPolicy
from .environ.feature_env import GroupedFeatureEnv
from .metrics import EpisodeMetrics, compute_jaccard_similarity, evaluate_final_selection

try:
    import wandb
except ImportError:
    wandb = None


class RolloutBuffer:
    """Stores rollout data for PPO updates."""

    def __init__(self):
        self.states = []       # GCN state (without agent emb)
        self.embeddings = []   # agent embeddings
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.agent_ids = []
        self.group_sizes = []

    def add(self, state, emb, action, log_prob, reward, value, done, agent_id, group_size):
        self.states.append(state)
        self.embeddings.append(emb)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.agent_ids.append(agent_id)
        self.group_sizes.append(group_size)

    def compute_returns_and_advantages(self, gamma: float, gae_lambda: float):
        n = len(self.rewards)
        self.advantages = [0.0] * n
        self.returns = [0.0] * n

        last_gae = 0.0
        for t in reversed(range(n)):
            next_value = 0.0 if t == n - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]
            last_gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * last_gae
            self.advantages[t] = last_gae
            self.returns[t] = last_gae + self.values[t]

    def get_batches(self, batch_size: int, device: torch.device):
        n = len(self.states)
        indices = np.random.permutation(n)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]

            states = torch.stack([self.states[i] for i in batch_idx]).to(device)
            embs = torch.stack([self.embeddings[i] for i in batch_idx]).to(device)
            acts = torch.stack([self.actions[i] for i in batch_idx]).to(device)
            old_lp = torch.stack([self.log_probs[i] for i in batch_idx]).to(device)
            adv = torch.tensor([self.advantages[i] for i in batch_idx],
                               dtype=torch.float32, device=device)
            ret = torch.tensor([self.returns[i] for i in batch_idx],
                               dtype=torch.float32, device=device)
            gs = [self.group_sizes[i] for i in batch_idx]

            if len(adv) > 1:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            yield states, embs, acts, old_lp, adv, ret, gs

    def clear(self):
        self.__init__()


class MARFSTrainer:
    """Main trainer orchestrating multi-agent PPO for feature selection."""

    def __init__(self, config: MARFSConfig):
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() and config.device == "cuda"
            else "cpu"
        )
        print(f"Using device: {self.device}")

        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)

        # Load data
        print(f"Loading dataset: {config.dataset}")
        self.X_train, self.X_test, self.y_train, self.y_test, self.feature_names = \
            load_dataset(config.dataset, test_size=config.test_size, seed=config.seed)
        self.n_features = self.X_train.shape[1]
        print(f"  Features: {self.n_features}, Train: {len(self.X_train)}, Test: {len(self.X_test)}")

        # Determine GCN input dim
        if config.use_stat_descriptor:
            gcn_input_dim = config.stat_descriptor_dim
        else:
            gcn_input_dim = self.X_train.shape[0]
        config.gcn_input_dim = gcn_input_dim

        # Build GCN
        self.gcn = GCNStateEncoder(
            input_dim=gcn_input_dim,
            hidden_dim=config.gcn_hidden,
            output_dim=config.gcn_out,
            global_readout=config.global_readout,
        ).to(self.device)

        # Graph builder
        self.graph_builder = DynamicGraphBuilder(
            corr_threshold=config.corr_threshold,
            use_stat_descriptor=config.use_stat_descriptor,
        )

        # --- Phase 1: Contrastive pre-training (optional) ---
        if config.contrastive_pretrain:
            self._contrastive_pretrain()

        # --- Phase 2: Feature assignment ---
        if config.assignment == "gcn_spectral":
            gcn_embeddings = self._compute_gcn_embeddings()
            self.agent_groups = assign_features(
                self.X_train, config.n_agents, "gcn_spectral",
                config.seed, gcn_embeddings=gcn_embeddings
            )
        else:
            self.agent_groups = assign_features(
                self.X_train, config.n_agents, config.assignment, config.seed
            )

        self.max_k = max(len(g) for g in self.agent_groups)
        print(f"Assignment: {config.assignment} -> {config.n_agents} agents")
        print(f"  Group sizes: min={min(len(g) for g in self.agent_groups)}, "
              f"max={self.max_k}, mean={np.mean([len(g) for g in self.agent_groups]):.1f}")

        # --- Phase 3: Build remaining models ---
        action_size = 1 if config.collective_action else self.max_k
        self.policy = PPOPolicy(
            state_dim=config.state_dim,
            max_k=action_size,
            hidden_dim=config.policy_hidden,
            emb_dim=config.agent_emb_dim,
            conditioning=config.conditioning,
            collective_action=config.collective_action,
        ).to(self.device)

        self.agent_embeddings = nn.Embedding(
            config.n_agents, config.agent_emb_dim
        ).to(self.device)

        # Environment
        self.env = GroupedFeatureEnv(
            X_train=self.X_train, y_train=self.y_train,
            X_test=self.X_test, y_test=self.y_test,
            agent_groups=self.agent_groups, config=config,
        )

        # Separate optimizers
        self.optimizer_gcn = torch.optim.Adam(self.gcn.parameters(), lr=config.lr_gcn)
        self.optimizer_policy = torch.optim.Adam(self.policy.parameters(), lr=config.lr_policy)
        self.optimizer_embeds = torch.optim.Adam(
            self.agent_embeddings.parameters(), lr=config.lr_embeds
        )

        # Metrics tracking
        self.episode_metrics = []
        self.selected_masks_history = []
        self.attn_weights_history = []

        # Wandb
        self.wandb_run = None
        if config.use_wandb:
            if wandb is None:
                print("wandb not installed, skipping.")
            else:
                self.wandb_run = wandb.init(
                    project=config.wandb_project,
                    name=config.run_name,
                    config=vars(config),
                )

    def _contrastive_pretrain(self):
        """Run contrastive pre-training on the GCN."""
        print("Running contrastive GCN pre-training...")
        pretrainer = ContrastivePretrainer(
            gcn=self.gcn,
            graph_builder=self.graph_builder,
            tau=self.config.contrastive_tau,
            lr=self.config.contrastive_lr,
        )
        losses = pretrainer.train(
            self.X_train, self.y_train, self.device,
            n_epochs=self.config.contrastive_epochs,
        )
        if losses:
            print(f"  Pre-training done. Final loss: {losses[-1]:.4f}")

    def _compute_gcn_embeddings(self) -> np.ndarray:
        """Get GCN node embeddings for all features (for GCN-informed clustering)."""
        print("Computing GCN embeddings for clustering...")
        self.gcn.eval()
        with torch.no_grad():
            data = self.graph_builder.build_full(self.X_train, self.device)
            embeddings = self.gcn.get_embeddings(data)
        return embeddings.cpu().numpy()

    def _get_states_and_embs(self, mask: np.ndarray):
        """Compute GCN-based states and agent embeddings.

        Returns:
            states: dict mapping agent_id -> state tensor (state_dim,)
            embs: dict mapping agent_id -> embedding tensor (emb_dim,)
            attn_weights: attention weights or None
        """
        data = self.graph_builder.build(self.X_train, mask, self.device)
        global_state, node_embeddings, attn_weights = self.gcn(data)

        selected_indices = np.where(mask > 0)[0]
        states = self.gcn.compute_agent_states(
            global_state=global_state,
            node_embeddings=node_embeddings,
            selected_indices=selected_indices,
            agent_groups=self.agent_groups,
            use_local=self.config.use_local_state,
            use_global=self.config.use_global_state,
        )

        embs = {}
        for agent_id in range(self.config.n_agents):
            agent_key = f"agent_{agent_id}"
            aid = torch.tensor([agent_id], dtype=torch.long, device=self.device)
            embs[agent_key] = self.agent_embeddings(aid).squeeze(0)

        return states, embs, attn_weights

    def _collect_rollout(self) -> tuple[RolloutBuffer, dict]:
        """Collect one episode of experience."""
        buffer = RolloutBuffer()
        obs_dict, infos = self.env.reset(seed=self.config.seed)
        mask = self.env.get_selected_mask()

        episode_reward = 0.0
        episode_steps = 0

        while self.env.agents:
            states, embs, attn_weights = self._get_states_and_embs(mask)

            actions_dict = {}
            for agent_id_str in self.env.agents:
                agent_idx = int(agent_id_str.split("_")[1])
                k = len(self.agent_groups[agent_idx])
                state_tensor = states[agent_id_str]
                emb_tensor = embs[agent_id_str]

                with torch.no_grad():
                    action, log_prob, value = self.policy.get_action(
                        state_tensor, emb_tensor, k=k
                    )

                # Pad action for buffer storage
                if self.config.collective_action:
                    padded_action = action.detach()
                else:
                    padded_action = torch.zeros(self.max_k, device=self.device)
                    padded_action[:k] = action

                actions_dict[agent_id_str] = action.cpu().numpy().astype(np.int8)

                buffer.add(
                    state=state_tensor.detach(),
                    emb=emb_tensor.detach(),
                    action=padded_action,
                    log_prob=log_prob.detach(),
                    reward=0.0,
                    value=value.item(),
                    done=False,
                    agent_id=agent_idx,
                    group_size=k,
                )

            # Step environment
            _, rewards, terminations, truncations, infos = self.env.step(actions_dict)
            mask = self.env.get_selected_mask()

            buf_start = len(buffer.rewards) - len(actions_dict)
            for i, agent_id_str in enumerate(actions_dict.keys()):
                idx = buf_start + i
                buffer.rewards[idx] = rewards[agent_id_str]
                done = terminations[agent_id_str] or truncations[agent_id_str]
                buffer.dones[idx] = float(done)

            info_sample = next(iter(infos.values()))
            episode_reward += info_sample.get("global_reward", 0.0)
            episode_steps += 1

        buffer.compute_returns_and_advantages(self.config.gamma, self.config.gae_lambda)

        ep_info = {
            "episode_reward": episode_reward,
            "episode_steps": episode_steps,
            "n_selected": int(mask.sum()),
            "final_mask": mask.copy(),
            "accuracy": info_sample.get("accuracy", 0.0) if infos else 0.0,
        }

        if attn_weights is not None:
            self.attn_weights_history.append(attn_weights.detach().cpu().numpy())

        return buffer, ep_info

    def _ppo_update(self, buffer: RolloutBuffer) -> dict[str, float]:
        """Perform PPO update on collected rollout."""
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(self.config.n_epochs):
            for states, embs, acts, old_lp, adv, ret, group_sizes in \
                    buffer.get_batches(self.config.mini_batch_size, self.device):

                k = 1 if self.config.collective_action else self.max_k

                log_probs, values, entropy = self.policy.evaluate_actions(
                    states, embs, acts, k=k
                )

                ratio = torch.exp(log_probs - old_lp)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio,
                                    1 + self.config.clip_ratio) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(values, ret)
                entropy_loss = -entropy.mean()

                loss = (policy_loss
                        + self.config.value_loss_coeff * value_loss
                        + self.config.entropy_coeff * entropy_loss)

                self.optimizer_gcn.zero_grad()
                self.optimizer_policy.zero_grad()
                self.optimizer_embeds.zero_grad()

                loss.backward()

                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                nn.utils.clip_grad_norm_(self.gcn.parameters(), self.config.max_grad_norm)
                nn.utils.clip_grad_norm_(self.agent_embeddings.parameters(),
                                         self.config.max_grad_norm)

                self.optimizer_gcn.step()
                self.optimizer_policy.step()
                self.optimizer_embeds.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        n_updates = max(n_updates, 1)
        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }

    def train(self) -> dict:
        """Run the full training loop."""
        print(f"\nStarting training: {self.config.n_episodes} episodes")
        print(f"  Config: {self.config.n_agents} agents, "
              f"assignment={self.config.assignment}, "
              f"conditioning={self.config.conditioning}, "
              f"readout={self.config.global_readout}, "
              f"collective={self.config.collective_action}")
        print("-" * 70)

        best_reward = -np.inf
        best_mask = None
        start_time = time.time()

        for episode in range(self.config.n_episodes):
            ep_start = time.time()

            buffer, ep_info = self._collect_rollout()
            loss_info = self._ppo_update(buffer)
            buffer.clear()

            ep_time = time.time() - ep_start
            ep_reward = ep_info["episode_reward"]
            n_selected = ep_info["n_selected"]
            compression = (self.n_features - n_selected) / self.n_features

            if ep_reward > best_reward:
                best_reward = ep_reward
                best_mask = ep_info["final_mask"].copy()

            metrics = EpisodeMetrics(
                episode=episode,
                reward=ep_reward,
                accuracy=ep_info.get("accuracy", 0.0),
                n_selected=n_selected,
                compression=compression,
                policy_loss=loss_info["policy_loss"],
                value_loss=loss_info["value_loss"],
                entropy=loss_info["entropy"],
                steps=ep_info["episode_steps"],
                time=ep_time,
            )
            self.episode_metrics.append(metrics)
            self.selected_masks_history.append(ep_info["final_mask"])

            if episode % self.config.log_freq == 0 or episode == self.config.n_episodes - 1:
                fps = ep_info["episode_steps"] / max(ep_time, 1e-6)
                print(f"Ep {episode:4d} | R={ep_reward:+.4f} | "
                      f"Acc={ep_info.get('accuracy', 0):.3f} | "
                      f"Sel={n_selected:4d}/{self.n_features} ({compression:.1%}) | "
                      f"PL={loss_info['policy_loss']:.4f} | "
                      f"Ent={loss_info['entropy']:.3f} | "
                      f"FPS={fps:.0f} | {ep_time:.1f}s")

            if self.wandb_run:
                wandb.log({
                    "episode": episode,
                    "reward": ep_reward,
                    "accuracy": ep_info.get("accuracy", 0.0),
                    "n_selected": n_selected,
                    "compression": compression,
                    **loss_info,
                    "episode_time": ep_time,
                })

        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time:.1f}s")
        print(f"Best reward: {best_reward:.4f}")
        if best_mask is not None:
            print(f"Best selection: {int(best_mask.sum())}/{self.n_features} features")

        results = self._compile_results(best_mask, total_time)
        self._save_results(results)
        return results

    def _compile_results(self, best_mask: np.ndarray, total_time: float) -> dict:
        if best_mask is None:
            best_mask = np.ones(self.n_features, dtype=np.float32)

        eval_results = evaluate_final_selection(
            self.X_train, self.y_train,
            self.X_test, self.y_test,
            best_mask,
        )

        rewards = [m.reward for m in self.episode_metrics]
        accuracies = [m.accuracy for m in self.episode_metrics]
        final_reward = rewards[-1] if rewards else 0
        threshold = 0.95 * final_reward if final_reward > 0 else final_reward * 1.05
        convergence_step = len(rewards)
        for i, r in enumerate(rewards):
            if r >= threshold:
                convergence_step = i
                break

        if len(self.selected_masks_history) >= 2:
            recent_masks = self.selected_masks_history[-10:]
            stability = compute_jaccard_similarity(recent_masks)
        else:
            stability = 1.0

        results = {
            "config": {k: v for k, v in vars(self.config).items()
                       if isinstance(v, (int, float, str, bool))},
            "best_mask": best_mask.tolist(),
            "n_selected": int(best_mask.sum()),
            "n_features": self.n_features,
            "compression": (self.n_features - int(best_mask.sum())) / self.n_features,
            "downstream_rf_accuracy": eval_results["rf_accuracy"],
            "convergence_step": convergence_step,
            "stability": stability,
            "total_time": total_time,
            "reward_history": rewards,
            "accuracy_history": accuracies,
            "peak_memory_mb": self._get_peak_memory(),
        }
        return results

    def _get_peak_memory(self) -> float:
        if torch.cuda.is_available() and self.device.type == "cuda":
            return torch.cuda.max_memory_allocated(self.device) / 1e6
        return 0.0

    def _save_results(self, results: dict):
        save_dir = self.config.save_dir
        os.makedirs(save_dir, exist_ok=True)

        run_name = self.config.run_name or (
            f"{self.config.dataset}_{self.config.assignment}_"
            f"{self.config.conditioning}_{self.config.n_agents}ag_seed{self.config.seed}"
        )
        filepath = os.path.join(save_dir, f"{run_name}.json")

        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        serializable = json.loads(json.dumps(results, default=convert))
        with open(filepath, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"Results saved to {filepath}")
