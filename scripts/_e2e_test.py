"""End-to-end test of all key configurations after the big update."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marfs.config import MARFSConfig
from marfs.trainer import MARFSTrainer

BASE = dict(
    dataset="synthetic", n_agents=5, n_episodes=3, n_steps_per_episode=10,
    early_stop_patience=100, seed=42, device="cuda",
    save_dir="results/_e2e_test", use_wandb=False,
)

configs = [
    ("A: EAC-FS collective", dict(
        collective_action=True, assignment="random", conditioning="concat",
        use_local_state=False, global_readout="mean", contrastive_pretrain=False,
        reward_type="hierarchical")),
    ("D: dual readout + concat", dict(
        collective_action=False, assignment="random", conditioning="concat",
        use_local_state=True, global_readout="mean", contrastive_pretrain=False,
        reward_type="hierarchical")),
    ("H: AdaLN + gcn_spectral", dict(
        collective_action=False, assignment="gcn_spectral", conditioning="adaln",
        use_local_state=True, global_readout="mean", contrastive_pretrain=False,
        reward_type="hierarchical")),
    ("I: attention readout", dict(
        collective_action=False, assignment="random", conditioning="adaln",
        use_local_state=True, global_readout="attention", contrastive_pretrain=False,
        reward_type="hierarchical")),
    ("J: full system + contrastive", dict(
        collective_action=False, assignment="gcn_spectral", conditioning="adaln",
        use_local_state=True, global_readout="attention", contrastive_pretrain=True,
        contrastive_epochs=10, reward_type="hierarchical")),
    ("Simple reward", dict(
        collective_action=False, assignment="random", conditioning="adaln",
        use_local_state=True, global_readout="mean", contrastive_pretrain=False,
        reward_type="simple")),
    ("FiLM conditioning", dict(
        collective_action=False, assignment="random", conditioning="film",
        use_local_state=True, global_readout="mean", contrastive_pretrain=False,
        reward_type="hierarchical")),
]

passed = 0
failed = 0

for name, overrides in configs:
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    try:
        cfg = MARFSConfig(**{**BASE, **overrides})
        trainer = MARFSTrainer(cfg)
        results = trainer.train()
        print(f"  PASSED | Acc={results['downstream_accuracy']:.4f}, Sel={results['n_selected']}/{results['n_features']}")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
        failed += 1

print(f"\n{'='*60}")
print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
print(f"{'='*60}")
