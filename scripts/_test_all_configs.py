"""Test all assignment strategies and ablation configs work."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marfs.config import MARFSConfig
from marfs.trainer import MARFSTrainer

configs = [
    ("random, global only", {"assignment": "random", "use_local_state": False, "use_agent_emb": False}),
    ("random, dual readout", {"assignment": "random", "use_local_state": True, "use_agent_emb": True}),
    ("kmeans, dual readout", {"assignment": "kmeans", "use_local_state": True, "use_agent_emb": True}),
    ("spectral, dual readout", {"assignment": "spectral", "use_local_state": True, "use_agent_emb": True}),
]

for name, overrides in configs:
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    config = MARFSConfig(
        dataset="synthetic",
        n_agents=5,
        n_episodes=5,
        n_steps_per_episode=10,
        early_stop_patience=100,
        seed=42,
        device="cuda",
        save_dir="results/integration_test",
        **overrides,
    )
    trainer = MARFSTrainer(config)
    results = trainer.train()
    print(f"  -> Acc={results['downstream_rf_accuracy']:.4f}, Selected={results['n_selected']}/{results['n_features']}")

print("\n\nAll configs PASSED!")
