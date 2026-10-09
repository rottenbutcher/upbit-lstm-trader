"""Small diagonal Gaussian HMM with causal (forward-only) forecasts."""

from __future__ import annotations

import logging

import numpy as np

LOG = logging.getLogger(__name__)
HMM_FEATURES = ["ret1", "ret12", "vol12", "volume_z"]
DEFAULTS = dict(
    states=3,
    iterations=40,
    restarts=2,
    tolerance=0.01,
    max_fit_rows=60000,
    block_size=2048,
    shrinkage=100.0,
    threads=4,
)


def settings(cfg):
    result = DEFAULTS | cfg.get("hmm", {})
    for key in ("states", "iterations", "restarts", "block_size", "threads"):
        if not isinstance(result[key], int) or result[key] < 1:
            raise ValueError(f"hmm.{key} must be a positive integer")
    if result["states"] < 2:
        raise ValueError("HMM needs at least two states")
    if not isinstance(result["max_fit_rows"], int) or result["max_fit_rows"] < 0:
        raise ValueError("hmm.max_fit_rows must be zero (all) or a positive integer")
    for key in ("tolerance", "shrinkage"):
        if not np.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f"hmm.{key} must be finite and positive")
    return result


def training_blocks(x, mask, cap, block_size):
    """Sample contiguous blocks across history; never invent transitions over gaps."""
    indexes = np.flatnonzero(mask & np.isfinite(x).all(axis=1))
    runs = np.split(indexes, np.flatnonzero(np.diff(indexes) != 1) + 1)
    blocks = [r[i : i + block_size] for r in runs for i in range(0, len(r), block_size)]
    if not blocks:
        raise ValueError("No finite HMM training observations")
    if cap and sum(map(len, blocks)) > cap:
        count = min(len(blocks), max(1, int(np.ceil(cap / block_size))))
        blocks = [blocks[i] for i in np.linspace(0, len(blocks) - 1, count, dtype=int)]
        remaining, chosen = cap, []
        for block in blocks:
            chosen.append(block[:remaining])
            remaining -= len(chosen[-1])
            if remaining <= 0:
                break
        blocks = chosen
    return np.concatenate([x[b] for b in blocks]), [len(b) for b in blocks]


def forward_filter(x, parameters):
    """P(state[t] | x[:t+1]); future observations cannot change earlier results.

    Nonfinite rows reset the filter instead of joining disconnected sequences.
    Log emissions and per-row normalization avoid underflow in extreme candles.
    """
    start = np.asarray(parameters["startprob"], dtype=float)
    transition = np.asarray(parameters["transmat"], dtype=float)
    means = np.asarray(parameters["means"], dtype=float)
    variance = np.asarray(parameters["variances"], dtype=float)
    result = np.full((len(x), len(start)), np.nan)
    alpha = None
    for offset in range(0, len(x), 65536):
        values = x[offset : offset + 65536]
        log_emission = -0.5 * (
            np.log(2 * np.pi * variance).sum(axis=1)
            + (((values[:, None, :] - means) ** 2) / variance).sum(axis=2)
        )
        for j, likelihood in enumerate(log_emission):
            if not np.isfinite(likelihood).all():
                alpha = None
                continue
            prior = start if alpha is None else alpha @ transition
            joint = np.log(np.maximum(prior, np.finfo(float).tiny)) + likelihood
            alpha = np.exp(joint - joint.max())
            alpha /= alpha.sum()
            result[offset + j] = alpha
    return result


def fit_forecaster(values, train_mask, label_indexes, labels, options, seed):
    from hmmlearn.hmm import GaussianHMM
    from threadpoolctl import threadpool_limits

    training = values[train_mask & np.isfinite(values).all(axis=1)]
    if len(training) < max(50, options["states"] * 10):
        raise ValueError("Insufficient HMM training history")
    mean, std = training.mean(axis=0), np.maximum(training.std(axis=0), 1e-6)
    x = np.clip((values - mean) / std, -10, 10)
    fit_x, lengths = training_blocks(x, train_mask, options["max_fit_rows"], options["block_size"])
    if len(fit_x) < max(50, options["states"] * 10):
        raise ValueError("hmm.max_fit_rows too small for the selected number of states")
    best, best_score, attempts = None, -np.inf, []
    with threadpool_limits(limits=options["threads"]):
        for restart in range(options["restarts"]):
            LOG.info(
                "HMM initialization %d/%d, %d observations", restart + 1, options["restarts"], len(fit_x)
            )
            model = GaussianHMM(
                n_components=options["states"],
                covariance_type="diag",
                n_iter=options["iterations"],
                tol=options["tolerance"],
                random_state=seed + restart,
                implementation="log",
                startprob_prior=1.01,
                transmat_prior=1.01,
                min_covar=1e-3,
            )
            model.fit(fit_x, lengths)
            score = float(model.score(fit_x, lengths))
            if not np.isfinite(score):
                raise ValueError("HMM fitting produced a nonfinite likelihood")
            history = list(model.monitor_.history)
            attempts.append(
                {
                    "seed": seed + restart,
                    "log_likelihood": score,
                    "iterations": model.monitor_.iter,
                    "tolerance_reached": bool(
                        len(history) > 1 and 0 <= history[-1] - history[-2] < options["tolerance"]
                    ),
                }
            )
            if score > best_score:
                best_score = score
                best = {
                    "startprob": model.startprob_.tolist(),
                    "transmat": model.transmat_.tolist(),
                    "means": model.means_.tolist(),
                    "variances": np.diagonal(model.covars_, axis1=1, axis2=2).tolist(),
                }
    # State-to-return mapping uses only purged training labels and forward probabilities.
    probabilities = forward_filter(x, best)
    weights = probabilities[label_indexes]
    y = labels[label_indexes].astype(float)
    if not np.isfinite(weights).all() or not np.isfinite(y).all():
        raise ValueError("Invalid HMM calibration labels/probabilities")
    prior = float(np.mean(y))
    shrink = options["shrinkage"]
    returns = (weights.T @ y + shrink * prior) / (weights.sum(axis=0) + shrink)
    best.update(
        scaler={"mean": mean.tolist(), "std": std.tolist()},
        features=HMM_FEATURES,
        state_returns_bps=returns.tolist(),
        training_mean_bps=prior,
        fit_rows=len(fit_x),
        fit_sequences=len(lengths),
        attempts=attempts,
        selected_log_likelihood=best_score,
        calibration_rows=len(y),
    )
    return best, probabilities @ returns
