"""PettingZoo ParallelEnv for multi-agent feature selection.

Supports per-feature binary actions (ours) and collective actions (EAC-FS baseline).
Supports multiple initialization strategies.
"""

from __future__ import annotations

import functools
import numpy as np
import gymnasium as gym
from pettingzoo import ParallelEnv

from ..reward import compute_reward


class GroupedFeatureEnv(ParallelEnv):
    """Multi-agent environment where M agents each manage K features."""

    metadata = {"name": "grouped_feature_selection_v0"}

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        agent_groups: list[list[int]],
        config,
    ):
        super().__init__()

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.agent_groups = agent_groups
        self.config = config

        self.n_features = X_train.shape[1]
        self.n_agents = len(agent_groups)
        self.collective_action = config.collective_action
        self.init_strategy = config.init_strategy

        self.possible_agents = [f"agent_{i}" for i in range(self.n_agents)]
        self.agents = self.possible_agents[:]

        self.group_sizes = {
            f"agent_{i}": len(group) for i, group in enumerate(agent_groups)
        }
        self.max_k = max(len(g) for g in agent_groups)

        # Episode state
        self.current_mask = None
        self.step_count = 0
        self.best_reward = -np.inf
        self.steps_without_improvement = 0
        self.prev_accuracy = None

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.config.obs_dim,),
            dtype=np.float32,
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        if self.collective_action:
            return gym.spaces.Discrete(2)  # 0=deselect-all, 1=select-all
        k = self.group_sizes[agent]
        return gym.spaces.MultiBinary(k)

    def reset(self, seed=None, options=None):
        """Reset environment with configurable initialization strategy."""
        self.agents = self.possible_agents[:]
        self.step_count = 0
        self.best_reward = -np.inf
        self.steps_without_improvement = 0
        self.prev_accuracy = None

        rng = np.random.RandomState(seed)

        if self.init_strategy == "all":
            self.current_mask = np.ones(self.n_features, dtype=np.float32)
        elif self.init_strategy == "none":
            # Start empty — force agents to build up
            self.current_mask = np.zeros(self.n_features, dtype=np.float32)
        elif self.init_strategy == "half":
            # Each agent selects ~half its features randomly
            self.current_mask = np.zeros(self.n_features, dtype=np.float32)
            for group in self.agent_groups:
                n_select = max(1, len(group) // 2)
                selected = rng.choice(group, n_select, replace=False)
                self.current_mask[selected] = 1.0
        elif self.init_strategy == "random":
            self.current_mask = rng.randint(0, 2, size=self.n_features).astype(np.float32)
        else:
            self.current_mask = np.ones(self.n_features, dtype=np.float32)

        observations = {
            agent: np.zeros(self.config.obs_dim, dtype=np.float32)
            for agent in self.agents
        }
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions: dict[str, np.ndarray]):
        """Execute one step."""
        # 1. Reconstruct full selection mask
        new_mask = np.zeros(self.n_features, dtype=np.float32)
        for agent_id, group in enumerate(self.agent_groups):
            agent_key = f"agent_{agent_id}"
            action = actions[agent_key]

            if self.collective_action:
                # EAC-FS style: select-all or deselect-all
                val = float(action) if np.isscalar(action) else float(action.item())
                for global_idx in group:
                    new_mask[global_idx] = val
            else:
                # Per-feature binary control
                for local_idx, global_idx in enumerate(group):
                    if local_idx < len(action):
                        new_mask[global_idx] = float(action[local_idx])

        self.current_mask = new_mask
        self.step_count += 1

        # 2. Compute reward
        n_selected = int(new_mask.sum())
        reward_info = compute_reward(
            self.X_train, self.y_train,
            self.X_test, self.y_test,
            new_mask,
            config=self.config,
            prev_accuracy=self.prev_accuracy,
            agent_groups=self.agent_groups,
        )

        self.prev_accuracy = reward_info["accuracy"]
        global_reward = reward_info["global"]

        # 3. Check termination
        if global_reward > self.best_reward + self.config.early_stop_epsilon:
            self.best_reward = global_reward
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1

        truncated = self.step_count >= self.config.n_steps_per_episode
        terminated = self.steps_without_improvement >= self.config.early_stop_patience

        # 4. Assign per-agent rewards (participation rule + local component)
        rewards = {}
        participating_agents = []
        for agent_id, group in enumerate(self.agent_groups):
            agent_key = f"agent_{agent_id}"
            action = actions[agent_key]
            if self.collective_action:
                has_selected = float(action) > 0 if np.isscalar(action) else action.item() > 0
            else:
                has_selected = np.any(action > 0)
            if has_selected:
                participating_agents.append(agent_id)

        n_participating = max(len(participating_agents), 1)
        for agent_id in range(self.n_agents):
            agent_key = f"agent_{agent_id}"
            if agent_id in participating_agents:
                local_r = reward_info["local"].get(agent_id, 0.0)
                rewards[agent_key] = global_reward / n_participating + local_r
            else:
                rewards[agent_key] = 0.0

        # 5. Dummy observations (filled by trainer via GCN)
        observations = {
            agent: np.zeros(self.config.obs_dim, dtype=np.float32)
            for agent in self.agents
        }

        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}
        infos = {
            agent: {
                "n_selected": n_selected,
                "global_reward": global_reward,
                "accuracy": reward_info["accuracy"],
                "step": self.step_count,
            }
            for agent in self.agents
        }

        if terminated or truncated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def get_selected_mask(self) -> np.ndarray:
        return self.current_mask.copy()

    def get_selected_indices(self) -> np.ndarray:
        return np.where(self.current_mask > 0)[0]
