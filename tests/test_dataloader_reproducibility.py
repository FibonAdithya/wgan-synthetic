"""The training loader's batch order must be a function of the seed alone.

The ladder discipline reads a difference between two overlaid runs as caused by
the single config change between them. That inference only holds if everything
*else* is pinned -- including the order the loader hands out data, which would
otherwise drift with `num_workers` and turn a machine difference into an
apparent rung difference.
"""

import numpy as np
import torch

from src.data.dataset import NumpyTensorDataset
from src.train.train_wgan_gp import build_dataloader, seed_dataloader_worker, train


def make_config(out_dir, num_workers: int) -> dict:
    """The tiny synthetic-data config from the training smoke test, kept local
    so a parallel edit to that module cannot silently change what this file
    holds constant."""
    return {
        "seed": 0,
        "device": "cpu",
        "output_dir": str(out_dir),
        "data": {
            "real_path": None,
            "format": "npy",
            "descriptor_dim": 16,
            "holdout_fraction": 0.2,
            "synthetic_if_missing": True,
            "synthetic_num_vectors": 256,
            "preprocess": {"center": False, "whiten": False, "l2_normalize": True},
        },
        "model": {
            "latent_dim": 8,
            "generator_hidden_dims": [16, 16],
            "critic_hidden_dims": [16, 16],
            "negative_slope": 0.2,
        },
        "training": {
            "batch_size": 32,
            "num_gen_steps": 4,
            "n_critic": 2,
            "lr_g": 1.0e-4,
            "lr_d": 1.0e-4,
            "betas": [0.0, 0.9],
            "lambda_gp": 5.0,
            "ema_decay": 0.9,
            "num_workers": num_workers,
            "distance_reg_alpha": 0.1,
            "distance_reg_max_points": 16,
            "amp": False,
            "log_every": 1,
            "eval_every": 2,
            "save_every": 4,
        },
    }


def make_dataset(num_vectors: int = 64, descriptor_dim: int = 8) -> NumpyTensorDataset:
    rng = np.random.default_rng(7)
    # Distinct rows so a batch is identifiable by its contents: the loader hands
    # back tensors, not indices, so identity of order is checked through values.
    return NumpyTensorDataset(
        rng.standard_normal((num_vectors, descriptor_dim), dtype=np.float32)
    )


def collect_batches(
    dataset: NumpyTensorDataset, num_workers: int, seed: int, epochs: int = 1
):
    loader = build_dataloader(dataset, batch_size=8, num_workers=num_workers, seed=seed)
    return [batch.clone() for _ in range(epochs) for batch in loader]


def assert_same_order(left, right, message: str) -> None:
    assert len(left) == len(right), message
    for i, (a, b) in enumerate(zip(left, right)):
        assert torch.equal(a, b), f"{message} (first differing batch: index {i})"


def test_shuffle_order_is_identical_across_different_worker_counts():
    """Same seed, `num_workers=0` vs `num_workers=2`: the batches must match.

    This is the claim the explicit `generator=` exists to make true -- the
    permutation is drawn in the parent process, so worker count cannot move it.
    """
    dataset = make_dataset()
    single = collect_batches(dataset, num_workers=0, seed=1234)
    multi = collect_batches(dataset, num_workers=2, seed=1234)
    assert_same_order(
        single, multi, "shuffle order changed between num_workers=0 and num_workers=2"
    )


def test_shuffle_order_stays_identical_past_the_first_epoch_boundary():
    """Reproducibility must survive the epoch boundary, not just the first pass.

    The training loop re-creates the iterator on StopIteration and the seeded
    generator is consumed as it goes, so later epochs only line up if the
    generator advances the same way under both worker counts -- and with
    `persistent_workers` on, only the multi-worker run keeps a live pool across
    that boundary.
    """
    dataset = make_dataset()
    single = collect_batches(dataset, num_workers=0, seed=99, epochs=3)
    multi = collect_batches(dataset, num_workers=2, seed=99, epochs=3)
    assert_same_order(single, multi, "shuffle order diverged after epoch 1")


def test_shuffle_order_ignores_the_global_torch_rng_state():
    """Whatever the process did to the global RNG before the loader was built
    must not reach the batch order -- otherwise unrelated model-init changes
    would silently reshuffle the data."""
    dataset = make_dataset()

    torch.manual_seed(0)
    first = collect_batches(dataset, num_workers=0, seed=555)
    torch.manual_seed(12345)
    torch.randn(64)
    second = collect_batches(dataset, num_workers=0, seed=555)

    assert_same_order(
        first, second, "batch order moved with the global torch RNG state"
    )


def test_different_seeds_produce_different_shuffle_orders():
    """Guards the tests above against being vacuously true: if the loader were
    not shuffling at all, every order would match every other one."""
    dataset = make_dataset()
    a = collect_batches(dataset, num_workers=0, seed=1)
    b = collect_batches(dataset, num_workers=0, seed=2)

    assert any(not torch.equal(x, y) for x, y in zip(a, b)), (
        "two different seeds produced the same batch order -- shuffling is inert"
    )


def test_worker_init_seeds_each_worker_from_the_base_seed_and_worker_id():
    """Workers inherit the parent's RNG state on fork, so without this hook two
    workers would draw identical 'random' numbers."""
    drawn = []
    for worker_id in (0, 1):
        seed_dataloader_worker(worker_id, base_seed=4242)
        drawn.append(np.random.rand())

    assert drawn[0] != drawn[1], (
        "workers 0 and 1 were seeded identically -- worker_id is not being used"
    )

    seed_dataloader_worker(0, base_seed=4242)
    assert np.random.rand() == drawn[0], (
        "re-seeding worker 0 from the same base seed did not reproduce its draw"
    )


def test_training_run_is_bit_identical_across_worker_counts(tmp_path):
    """End to end: the same config trained with 0 and with 2 loader workers must
    land on the same generator weights."""
    ckpt_single, _ = train(make_config(tmp_path / "single", num_workers=0))
    ckpt_multi, _ = train(make_config(tmp_path / "multi", num_workers=2))

    single = torch.load(ckpt_single, weights_only=False)["generator_state_dict"]
    multi = torch.load(ckpt_multi, weights_only=False)["generator_state_dict"]
    for key in single:
        assert torch.equal(single[key], multi[key]), (
            f"generator weight {key!r} differs between num_workers=0 and 2"
        )
