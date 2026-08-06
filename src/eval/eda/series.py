"""The datasets a report compares, loaded and preprocessed.

A `Series` is one already-subsampled, already-normalized set with the colour
it is drawn in. Everything downstream works in terms of these, so no figure
or panel touches a file path or a preprocessing mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data.dataset import load_descriptors
from src.eval.eda.config import EdaConfig

REAL_NAME = "real"
REAL_COLOR = "#2b6cb0"
# Colors for synthetic sets, in order. Deliberately distinct from REAL_COLOR so
# the reference curve stays identifiable when several overlays are present.
SYNTH_PALETTE = [
    "#dd6b20",
    "#38a169",
    "#805ad5",
    "#d53f8c",
    "#00897b",
    "#a0522d",
]


@dataclass
class Series:
    """One dataset to plot, already subsampled and preprocessed."""

    name: str
    x: np.ndarray
    color: str

    @property
    def is_real(self) -> bool:
        return self.name == REAL_NAME


def parse_synthetic_spec(spec: str) -> tuple[str, Path]:
    """Split a '[LABEL=]PATH' argument into its label and path.

    Only the first '=' separates, so paths containing '=' still work when a
    label is supplied. A bare path falls back to the file stem as its label.
    """
    if "=" in spec:
        label, _, raw = spec.partition("=")
        label = label.strip()
        if label:
            return label, Path(raw)
    path = Path(spec)
    return path.stem, path


def subsample(x: np.ndarray, max_vectors: int, seed: int) -> np.ndarray:
    if max_vectors <= 0 or x.shape[0] <= max_vectors:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=max_vectors, replace=False)
    return x[np.sort(idx)]


def maybe_l2_normalize(x: np.ndarray, mode: str, eps: float = 1.0e-8) -> np.ndarray:
    if mode == "none":
        return x
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.clip(norm, eps, None)).astype(np.float32, copy=False)


def load_series(cfg: EdaConfig) -> list[Series]:
    real_x = load_descriptors(Path(cfg.real_path), file_format=cfg.real_format)
    real_x = maybe_l2_normalize(
        subsample(real_x, cfg.max_vectors, cfg.seed), cfg.preprocess
    )
    series = [Series(REAL_NAME, real_x, REAL_COLOR)]

    seen = {REAL_NAME}
    for i, spec in enumerate(cfg.synthetic_path or []):
        label, path = parse_synthetic_spec(spec)
        if label in seen:
            raise ValueError(
                f"Duplicate series label {label!r}; use LABEL=PATH to rename"
            )
        seen.add(label)
        x = load_descriptors(path, file_format=cfg.synthetic_format)
        if x.shape[1] != real_x.shape[1]:
            raise ValueError(
                f"Dimension mismatch for {label!r}: real has {real_x.shape[1]}, "
                f"got {x.shape[1]}"
            )
        x = maybe_l2_normalize(subsample(x, cfg.max_vectors, cfg.seed), cfg.preprocess)
        series.append(Series(label, x, SYNTH_PALETTE[i % len(SYNTH_PALETTE)]))
    return series
