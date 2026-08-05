from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_fvecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size == 0:
        raise ValueError(f"Empty fvecs file: {path}")

    dim = np.frombuffer(np.array([raw[0]], dtype=np.float32).tobytes(), dtype=np.int32)[0]
    if dim <= 0:
        raise ValueError(f"Invalid dimension in fvecs header: {dim}")

    row_width = dim + 1
    if raw.size % row_width != 0:
        raise ValueError(
            f"Corrupt fvecs layout for {path}: size={raw.size}, row_width={row_width}"
        )

    matrix = raw.reshape(-1, row_width)
    dims = matrix[:, 0].view(np.int32)
    if not np.all(dims == dim):
        raise ValueError(f"Inconsistent dimensions in fvecs file {path}")

    return matrix[:, 1:].astype(np.float32, copy=False)


def _load_npy(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array in npy file {path}, got shape {arr.shape}")
    return arr.astype(np.float32, copy=False)


METRICS = ("l2", "angular")


@dataclass
class PreprocessConfig:
    center: bool = False
    whiten: bool = False
    l2_normalize: bool = True
    eps: float = 1.0e-8
    metric: str = "l2"

    def __post_init__(self) -> None:
        if self.metric not in METRICS:
            raise ValueError(
                f"Unknown metric {self.metric!r}; expected one of {METRICS}. "
                "This is the distance the real corpus is searched under, not a "
                "preprocessing step."
            )


@dataclass
class PreprocessState:
    descriptor_dim: int
    config: PreprocessConfig
    mean: Optional[np.ndarray] = None
    whitening_matrix: Optional[np.ndarray] = None

    def to_serializable(self) -> Dict:
        payload = asdict(self)
        payload["config"] = asdict(self.config)
        payload["mean"] = None if self.mean is None else self.mean.tolist()
        payload["whitening_matrix"] = (
            None if self.whitening_matrix is None else self.whitening_matrix.tolist()
        )
        return payload

    @classmethod
    def from_serializable(cls, payload: Dict) -> "PreprocessState":
        cfg = PreprocessConfig(**payload["config"])
        mean = payload.get("mean")
        whitening_matrix = payload.get("whitening_matrix")
        return cls(
            descriptor_dim=payload["descriptor_dim"],
            config=cfg,
            mean=None if mean is None else np.asarray(mean, dtype=np.float32),
            whitening_matrix=(
                None
                if whitening_matrix is None
                else np.asarray(whitening_matrix, dtype=np.float32)
            ),
        )


def _fit_preprocess_state(
    x_train: np.ndarray,
    descriptor_dim: int,
    cfg: PreprocessConfig,
) -> PreprocessState:
    state = PreprocessState(descriptor_dim=descriptor_dim, config=cfg)
    transformed = x_train

    if cfg.center:
        state.mean = transformed.mean(axis=0).astype(np.float32)
        transformed = transformed - state.mean

    if cfg.whiten:
        cov = np.cov(transformed, rowvar=False)
        u, s, _ = np.linalg.svd(cov + cfg.eps * np.eye(cov.shape[0]), hermitian=True)
        inv_sqrt = np.diag(1.0 / np.sqrt(s + cfg.eps))
        state.whitening_matrix = (u @ inv_sqrt @ u.T).astype(np.float32)

    return state


def apply_preprocess(x: np.ndarray, state: PreprocessState) -> np.ndarray:
    out = x.astype(np.float32, copy=False)
    cfg = state.config

    if state.mean is not None:
        out = out - state.mean
    if state.whitening_matrix is not None:
        out = out @ state.whitening_matrix
    if cfg.l2_normalize:
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        out = out / np.clip(norm, cfg.eps, None)
    return out.astype(np.float32, copy=False)


def invert_preprocess(x: np.ndarray, state: PreprocessState) -> np.ndarray:
    """Undo centering and whitening, in reverse order of `apply_preprocess`.

    Needed because a config with `whiten: true` trains in a transformed space:
    the generator's output lives there too, and has to be mapped back before
    anything compares it against the real corpus.

    L2 normalization is deliberately NOT inverted: it discards each vector's
    norm, so the information needed to undo it is gone. That is not a
    limitation for the families this matters to -- DEEP vectors are unit-norm
    to begin with and are compared angularly.

    The whitening matrix is symmetric (`u @ diag(1/sqrt(s)) @ u.T` over a
    symmetric covariance), so its inverse is likewise symmetric. pinv rather
    than inv, because the eps-regularized covariance can still be
    near-singular on the tail dimensions of a PCA-derived set like DEEP.

    Exactness, and the one case where it fails: `sample_generator`
    L2-normalizes its raw output, so what callers invert is `normalize(x @ W)`.
    With `state.mean is None` this reduces to
    `normalize(x @ W) @ W^-1 = x / ||x @ W||` -- the original direction `x`
    times a positive per-vector scalar, which any angular comparison is
    invariant to. Directions come back exactly.

    That breaks the moment `state.mean is not None`: the inverse then computes
    `normalize(x @ W) @ W^-1 + mean`, and the mean's relative contribution to
    that sum differs for every row (it does not scale with `1 / ||x @ W||` the
    way the direction term does), so re-normalizing downstream no longer
    recovers a uniformly scaled direction. This function raises nothing for
    that case -- it cannot know whether its input was L2-normalized upstream.
    Callers needing the exactness guarantee must reject
    `state.mean is not None and state.config.l2_normalize` first; see
    `src/eval/compare_variants.py::invert_samples`.
    """
    out = np.asarray(x, dtype=np.float32)
    if state.whitening_matrix is not None:
        out = out @ np.linalg.pinv(state.whitening_matrix).astype(np.float32)
    if state.mean is not None:
        out = out + state.mean
    return np.ascontiguousarray(out, dtype=np.float32)


def load_descriptors(path: Path, file_format: str = "auto") -> np.ndarray:
    if file_format == "auto":
        suffix = path.suffix.lower()
        if suffix == ".npy":
            file_format = "npy"
        elif suffix == ".fvecs":
            file_format = "fvecs"
        else:
            raise ValueError(f"Unknown extension for auto format detection: {suffix}")

    if file_format == "npy":
        return _load_npy(path)
    if file_format == "fvecs":
        return _load_fvecs(path)
    raise ValueError(f"Unsupported file format: {file_format}")


def train_holdout_split(
    x: np.ndarray, holdout_fraction: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.0 < holdout_fraction < 1.0):
        raise ValueError(f"holdout_fraction must be in (0,1), got {holdout_fraction}")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(x.shape[0])
    n_holdout = max(1, int(round(x.shape[0] * holdout_fraction)))
    holdout_idx = idx[:n_holdout]
    train_idx = idx[n_holdout:]
    if train_idx.size == 0:
        raise ValueError("Holdout split leaves no training data")
    return x[train_idx], x[holdout_idx]


class NumpyTensorDataset(Dataset):
    def __init__(self, x: np.ndarray):
        self.x = torch.from_numpy(x.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.x[idx]


def build_training_data(
    descriptor_path: Optional[str],
    file_format: str,
    descriptor_dim: int,
    holdout_fraction: float,
    preprocess_cfg: PreprocessConfig,
    seed: int,
    synthetic_if_missing: bool = False,
    synthetic_num_vectors: int = 100000,
) -> Tuple[np.ndarray, np.ndarray, PreprocessState]:
    if descriptor_path is None:
        if not synthetic_if_missing:
            raise ValueError("descriptor_path is null and synthetic_if_missing is false")
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(synthetic_num_vectors, descriptor_dim)).astype(np.float32)
    else:
        x = load_descriptors(Path(descriptor_path), file_format=file_format)

    if x.shape[1] != descriptor_dim:
        raise ValueError(f"Expected descriptor dim {descriptor_dim}, got {x.shape[1]}")

    x_train_raw, x_holdout_raw = train_holdout_split(
        x, holdout_fraction=holdout_fraction, seed=seed
    )
    state = _fit_preprocess_state(
        x_train=x_train_raw, descriptor_dim=descriptor_dim, cfg=preprocess_cfg
    )
    x_train = apply_preprocess(x_train_raw, state)
    x_holdout = apply_preprocess(x_holdout_raw, state)
    return x_train, x_holdout, state
