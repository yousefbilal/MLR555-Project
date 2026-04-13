"""Quick smoke test — runs 3 episodes on synthetic data to verify end-to-end."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marfs.config import MARFSConfig
from marfs.trainer import MARFSTrainer

config = MARFSConfig(
    dataset="synthetic",
    n_agents=5,
    assignment="random",
    n_episodes=3,
    n_steps_per_episode=10,
    early_stop_patience=100,
    seed=42,
    device="cuda",
    use_wandb=False,
    save_dir="results/smoke_test",
)

print("Running smoke test...")
trainer = MARFSTrainer(config)
results = trainer.train()

print(f"\nSmoke test PASSED!")
print(f"  Accuracy: {results['downstream_accuracy']:.4f}")
print(f"  RF Accuracy: {results['downstream_rf_accuracy']:.4f}")
print(f"  Selected: {results['n_selected']}/{results['n_features']}")
