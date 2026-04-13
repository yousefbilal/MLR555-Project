"""GCN state encoder with dual readout and contrastive pre-training for MARFS."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from marfs.utils import safe_corrcoef


class StatDescriptor:
    """Compute a fixed-size statistical descriptor for each feature column."""

    def __init__(self, n_components: int = 4):
        self.n_components = n_components

    def transform(self, X: np.ndarray) -> np.ndarray:
        from scipy import stats as scipy_stats

        descs = []
        for j in range(X.shape[1]):
            col = X[:, j]
            desc = [
                np.mean(col), np.std(col), np.min(col),
                np.percentile(col, 25), np.median(col), np.percentile(col, 75),
                np.max(col),
                float(scipy_stats.skew(col)), float(scipy_stats.kurtosis(col)),
                np.mean(col == 0), len(np.unique(col)) / len(col),
                self._entropy(col),
            ]
            descs.append(desc)

        descs = np.array(descs, dtype=np.float32)

        from sklearn.decomposition import PCA
        if X.shape[0] >= self.n_components and X.shape[1] >= self.n_components:
            pca = PCA(n_components=self.n_components)
            pca.fit(X)
            pca_loadings = pca.components_.T
        else:
            pca_loadings = np.zeros((X.shape[1], self.n_components), dtype=np.float32)

        descs = np.hstack([descs, pca_loadings])
        return np.nan_to_num(descs, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _entropy(col: np.ndarray, n_bins: int = 50) -> float:
        counts, _ = np.histogram(col, bins=n_bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs + 1e-12))


class DynamicGraphBuilder:
    """Build a PyG graph from the currently selected feature columns."""

    def __init__(self, corr_threshold: float = 0.3, use_stat_descriptor: bool = False):
        self.corr_threshold = corr_threshold
        self.use_stat_descriptor = use_stat_descriptor
        self.stat_desc = StatDescriptor() if use_stat_descriptor else None

    def build(self, X: np.ndarray, selected_mask: np.ndarray,
              device: torch.device) -> Data:
        selected_indices = np.where(selected_mask > 0)[0]
        n_selected = len(selected_indices)

        # Determine node feature dim
        if self.use_stat_descriptor:
            feat_dim = 16
        else:
            feat_dim = X.shape[0]  # n_samples

        if n_selected == 0:
            x = torch.zeros(1, feat_dim, device=device)
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
            return Data(x=x, edge_index=edge_index,
                        batch=torch.zeros(1, dtype=torch.long, device=device))

        X_selected = X[:, selected_indices]

        if self.use_stat_descriptor:
            node_features = self.stat_desc.transform(X_selected)
        else:
            node_features = X_selected.T

        x = torch.tensor(node_features, dtype=torch.float32, device=device)

        if n_selected > 1:
            corr = safe_corrcoef(X_selected, axis=0)
            np.fill_diagonal(corr, 0.0)
            adj = np.abs(corr) > self.corr_threshold
            src, dst = np.where(adj)
            edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long, device=device)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)

        batch = torch.zeros(n_selected, dtype=torch.long, device=device)
        return Data(x=x, edge_index=edge_index, batch=batch)

    def build_full(self, X: np.ndarray, device: torch.device) -> Data:
        """Build graph from ALL features (for pre-training / GCN-informed clustering)."""
        mask = np.ones(X.shape[1], dtype=np.float32)
        return self.build(X, mask, device)


class AttentionReadout(nn.Module):
    """Learned attention-based global pooling over node embeddings."""

    def __init__(self, dim: int):
        super().__init__()
        self.W = nn.Linear(dim, dim)
        self.w = nn.Linear(dim, 1)

    def forward(self, Z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            Z: (n_nodes, dim) node embeddings.

        Returns:
            readout: (dim,) weighted sum of node embeddings.
            alpha: (n_nodes,) attention weights (for visualization).
        """
        scores = self.w(torch.tanh(self.W(Z)))  # (n_nodes, 1)
        alpha = F.softmax(scores, dim=0)  # (n_nodes, 1)
        readout = (alpha * Z).sum(dim=0)  # (dim,)
        return readout, alpha.squeeze(-1)


class GCNStateEncoder(nn.Module):
    """Two-layer GCN with configurable global readout (mean or attention)
    and dual readout (global + per-agent local pooling).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64,
                 global_readout: str = "mean"):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.output_dim = output_dim
        self.global_readout_type = global_readout

        if global_readout == "attention":
            self.attention = AttentionReadout(output_dim)
        else:
            self.attention = None

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Forward pass through GCN.

        Returns:
            global_state: (output_dim,) global pooled embedding.
            node_embeddings: (n_nodes, output_dim) per-node embeddings.
            attn_weights: (n_nodes,) attention weights if using attention readout, else None.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        node_embeddings = x

        if self.global_readout_type == "attention":
            global_state, attn_weights = self.attention(x)
        else:
            global_state = global_mean_pool(x, batch).squeeze(0)
            attn_weights = None

        return global_state, node_embeddings, attn_weights

    def compute_agent_states(
        self,
        global_state: torch.Tensor,
        node_embeddings: torch.Tensor,
        selected_indices: np.ndarray,
        agent_groups: list[list[int]],
        use_local: bool = True,
        use_global: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Compute per-agent state vectors (without agent embedding — that goes to policy).

        Returns:
            Dict mapping agent_id string to state tensor (state_dim,).
        """
        device = global_state.device
        idx_to_pos = {int(idx): pos for pos, idx in enumerate(selected_indices)}

        states = {}
        for agent_id, group in enumerate(agent_groups):
            parts = []

            if use_global:
                parts.append(global_state)

            if use_local:
                local_positions = [idx_to_pos[f] for f in group if f in idx_to_pos]
                if local_positions:
                    local_state = node_embeddings[local_positions].mean(dim=0)
                else:
                    local_state = torch.zeros(self.output_dim, device=device)
                parts.append(local_state)

            if not parts:
                parts.append(global_state)

            states[f"agent_{agent_id}"] = torch.cat(parts, dim=-1)

        return states

    def get_embeddings(self, data: Data) -> torch.Tensor:
        """Get node embeddings without readout (for clustering / pre-training)."""
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        return x


class ContrastivePretrainer:
    """Pre-train the GCN with supervised contrastive loss on the feature graph.

    Positive pairs: features with high relevance to target + low mutual redundancy.
    Negative pairs: features with high mutual redundancy.
    """

    def __init__(self, gcn: GCNStateEncoder, graph_builder: DynamicGraphBuilder,
                 tau: float = 0.1, lr: float = 1e-3):
        self.gcn = gcn
        self.graph_builder = graph_builder
        self.tau = tau
        self.optimizer = torch.optim.Adam(gcn.parameters(), lr=lr)

    def compute_pair_labels(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Determine positive and negative feature pairs.

        Returns:
            pos_pairs: (n_pos, 2) array of positive pair indices.
            neg_pairs: (n_neg, 2) array of negative pair indices.
        """
        n_features = X.shape[1]

        # Feature-target relevance: abs correlation with label
        y_float = y.astype(np.float64)
        relevance = np.zeros(n_features)
        for j in range(n_features):
            col = X[:, j].astype(np.float64)
            if np.std(col) > 1e-10 and np.std(y_float) > 1e-10:
                r = np.corrcoef(col, y_float)[0, 1]
                relevance[j] = abs(r) if not np.isnan(r) else 0.0

        # Feature-feature redundancy: abs pairwise correlation
        corr = safe_corrcoef(X, axis=0)
        np.fill_diagonal(corr, 0.0)
        redundancy = np.abs(corr)

        # Positive pairs: both features relevant (top 30%) and low mutual redundancy
        rel_threshold = np.percentile(relevance, 70)
        relevant_features = np.where(relevance >= rel_threshold)[0]

        pos_pairs = []
        for i in range(len(relevant_features)):
            for j in range(i + 1, len(relevant_features)):
                fi, fj = relevant_features[i], relevant_features[j]
                if redundancy[fi, fj] < 0.3:  # low redundancy
                    pos_pairs.append([fi, fj])

        # Negative pairs: high mutual redundancy
        neg_pairs = []
        high_red = np.argwhere(redundancy > 0.7)
        for idx in range(len(high_red)):
            i, j = high_red[idx]
            if i < j:
                neg_pairs.append([i, j])

        # Limit size for efficiency
        max_pairs = 500
        if len(pos_pairs) > max_pairs:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(pos_pairs), max_pairs, replace=False)
            pos_pairs = [pos_pairs[i] for i in idx]
        if len(neg_pairs) > max_pairs:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(neg_pairs), max_pairs, replace=False)
            neg_pairs = [neg_pairs[i] for i in idx]

        return np.array(pos_pairs) if pos_pairs else np.zeros((0, 2), dtype=int), \
               np.array(neg_pairs) if neg_pairs else np.zeros((0, 2), dtype=int)

    def train(self, X: np.ndarray, y: np.ndarray, device: torch.device,
              n_epochs: int = 100) -> list[float]:
        """Run contrastive pre-training.

        Returns:
            List of per-epoch losses.
        """
        pos_pairs, neg_pairs = self.compute_pair_labels(X, y)

        if len(pos_pairs) == 0 and len(neg_pairs) == 0:
            print("  No contrastive pairs found, skipping pre-training.")
            return []

        data = self.graph_builder.build_full(X, device)
        losses = []

        for epoch in range(n_epochs):
            self.gcn.train()
            self.optimizer.zero_grad()

            embeddings = self.gcn.get_embeddings(data)  # (n_features, dim)
            embeddings = F.normalize(embeddings, dim=-1)

            loss = torch.tensor(0.0, device=device, requires_grad=True)

            # InfoNCE-style loss over positive pairs
            if len(pos_pairs) > 0:
                for pi, pj in pos_pairs:
                    zi = embeddings[pi]
                    zj = embeddings[pj]
                    pos_sim = torch.dot(zi, zj) / self.tau

                    # Negatives: all other features
                    all_sims = (embeddings @ zi) / self.tau
                    # Mask out self
                    mask = torch.ones(len(embeddings), device=device, dtype=torch.bool)
                    mask[pi] = False
                    neg_sims = all_sims[mask]
                    logits = torch.cat([pos_sim.unsqueeze(0), neg_sims])
                    labels = torch.zeros(1, dtype=torch.long, device=device)
                    loss = loss + F.cross_entropy(logits.unsqueeze(0), labels)

                loss = loss / max(len(pos_pairs), 1)

            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

            if epoch % 20 == 0:
                print(f"  Contrastive epoch {epoch}: loss={loss.item():.4f}")

        return losses
