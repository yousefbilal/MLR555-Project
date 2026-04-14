"""PPO policy network with AdaLN/FiLM/concat conditioning for MARFS."""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Bernoulli


class AdaLN(nn.Module):
    """Adaptive Layer Normalization: modulates hidden activations with agent embedding.

    h_out = (1 + gamma(emb)) * LayerNorm(h) + beta(emb)
    Initialized to identity (gamma=0, beta=0) so all agents start identical.
    """

    def __init__(self, hidden_dim: int, emb_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.gamma_proj = nn.Linear(emb_dim, hidden_dim)
        self.beta_proj = nn.Linear(emb_dim, hidden_dim)
        # Zero-initialize for identity modulation at start
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, h: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h_norm = self.norm(h)
        gamma = self.gamma_proj(emb)
        beta = self.beta_proj(emb)
        return (1 + gamma) * h_norm + beta


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: simpler than AdaLN (no LayerNorm)."""

    def __init__(self, hidden_dim: int, emb_dim: int):
        super().__init__()
        self.gamma_proj = nn.Linear(emb_dim, hidden_dim)
        self.beta_proj = nn.Linear(emb_dim, hidden_dim)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, h: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_proj(emb)
        beta = self.beta_proj(emb)
        return gamma * h + beta


class PPOPolicy(nn.Module):
    """PPO actor-critic with configurable agent conditioning.

    Conditioning modes:
    - "concat": concatenate agent embedding to state before first layer
    - "film": FiLM modulation at each hidden layer
    - "adaln": AdaLN modulation at each hidden layer (default)

    For collective_action=True (EAC-FS baseline), outputs a single
    select/deselect logit per agent group instead of K independent logits.
    """

    def __init__(self, state_dim: int, max_k: int, hidden_dim: int = 128,
                 emb_dim: int = 16, conditioning: str = "adaln",
                 collective_action: bool = False):
        super().__init__()
        self.conditioning = conditioning
        self.collective_action = collective_action
        self.max_k = max_k

        # Input dim depends on conditioning
        if conditioning == "concat":
            input_dim = state_dim + emb_dim
        else:
            input_dim = state_dim

        # Shared backbone layers
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)

        # Conditioning modules
        if conditioning == "adaln":
            self.cond1 = AdaLN(hidden_dim, emb_dim)
            self.cond2 = AdaLN(hidden_dim, emb_dim)
        elif conditioning == "film":
            self.cond1 = FiLM(hidden_dim, emb_dim)
            self.cond2 = FiLM(hidden_dim, emb_dim)
        else:
            self.cond1 = None
            self.cond2 = None

        # Actor head
        if collective_action:
            self.actor = nn.Linear(hidden_dim, 1)  # single logit for select/deselect
        else:
            self.actor = nn.Linear(hidden_dim, max_k)  # K independent logits

        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)

    def _backbone(self, state: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Shared backbone with conditioning."""
        if self.conditioning == "concat":
            h = torch.cat([state, emb], dim=-1)
            h = self.layer1(h)
            h = torch.relu(h)
            h = self.layer2(h)
            h = torch.relu(h)
        elif self.conditioning in ("adaln", "film"):
            h = self.layer1(state)
            h = self.cond1(h, emb)
            h = torch.relu(h)
            h = self.layer2(h)
            h = self.cond2(h, emb)
            h = torch.relu(h)
        else:
            raise ValueError(f"Unknown conditioning: {self.conditioning}")
        return h

    def forward(self, state: torch.Tensor, emb: torch.Tensor, k: int = None):
        """Forward pass.

        Args:
            state: (state_dim,) or (batch, state_dim) state from GCN readout.
            emb: (emb_dim,) or (batch, emb_dim) agent embedding.
            k: Number of features for this agent (used to trim logits).

        Returns:
            logits: action logits.
            value: value estimate.
        """
        h = self._backbone(state, emb)
        logits = self.actor(h)
        value = self.critic(h)

        if not self.collective_action and k is not None:
            if logits.dim() == 1:
                logits = logits[:k]
            else:
                logits = logits[:, :k]

        return logits, value

    def get_action(self, state: torch.Tensor, emb: torch.Tensor, k: int,
                   deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action and compute log probability.

        Returns:
            action: binary action vector.
            log_prob: scalar log probability.
            value: scalar value estimate.
        """
        logits, value = self.forward(state, emb, k=k)
        probs = torch.sigmoid(logits)

        if deterministic:
            action = (probs > 0.5).float()
        else:
            dist = Bernoulli(probs=probs)
            action = dist.sample()

        log_prob = self._log_prob(logits, action)
        return action, log_prob, value.squeeze(-1)

    def get_action_eps_greedy(self, state: torch.Tensor, emb: torch.Tensor, k: int,
                              epsilon: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select action using ε-greedy strategy.

        With probability ε: random binary action per feature.
        With probability 1-ε: greedy action (p > 0.5).
        Log probs are computed under the Bernoulli for valid PPO ratios.

        Returns:
            action, log_prob, value (same as get_action).
        """
        logits, value = self.forward(state, emb, k=k)
        probs = torch.sigmoid(logits)

        if np.random.random() < epsilon:
            # Random binary action
            action = torch.bernoulli(torch.full_like(probs, 0.5))
        else:
            # Greedy
            action = (probs > 0.5).float()

        log_prob = self._log_prob(logits, action)
        return action, log_prob, value.squeeze(-1)

    def evaluate_actions(self, state: torch.Tensor, emb: torch.Tensor,
                         actions: torch.Tensor, k: int
                         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update.

        Returns:
            log_probs: (batch,) log probabilities.
            values: (batch,) value estimates.
            entropy: (batch,) entropy of the policy.
        """
        logits, values = self.forward(state, emb, k=k)
        log_probs = self._log_prob(logits, actions)
        entropy = self._entropy(logits)
        return log_probs, values.squeeze(-1), entropy

    @staticmethod
    def _log_prob(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        dist = Bernoulli(logits=logits)
        lp = dist.log_prob(actions)
        if lp.dim() > 1:
            return lp.sum(dim=-1)
        return lp.sum()

    @staticmethod
    def _entropy(logits: torch.Tensor) -> torch.Tensor:
        dist = Bernoulli(logits=logits)
        ent = dist.entropy()
        if ent.dim() > 1:
            return ent.sum(dim=-1)
        return ent.sum()
