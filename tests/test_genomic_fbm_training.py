from argparse import Namespace

import jax.numpy as jnp
import numpy as np

import genomic_fbm_training as genomic_fbm


def test_train_records_piquasso_coverage_at_checkpoints(monkeypatch):
    def fake_estimator(**_):
        return lambda weights: (jnp.asarray(0.25), jnp.zeros_like(weights))

    sample_calls = []

    def fake_sample_keys(weights, args, seed):
        sample_calls.append((np.asarray(weights), seed))
        return np.asarray([3, 7])

    monkeypatch.setattr(genomic_fbm, "get_loss_and_grad_estimator", fake_estimator)
    monkeypatch.setattr(genomic_fbm, "sample_keys", fake_sample_keys)

    args = Namespace(
        N=1,
        discarded_qubits=(),
        lr=0.001,
        n_z_samples=4,
        length_cutoff=1,
        steps=5,
        checkpoint_every=2,
        seed=9,
    )
    task = {
        "train": np.asarray([[0, 0, 0, 0]], dtype=np.int8),
        "unseen_keys": np.asarray([3, 5]),
    }

    _, losses, checkpoints = genomic_fbm.train(
        task, jnp.zeros(3), args, sigma=1.0
    )

    np.testing.assert_allclose(losses, 0.25)
    assert checkpoints == [(2, 0.5), (4, 0.5)]
    assert [seed for _, seed in sample_calls] == [9, 9]
