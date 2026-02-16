#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_PATH="${1:-$SCRIPT_DIR/sift_base.fvecs}"
OUTPUT_PATH="${2:-$SCRIPT_DIR/sift_base_sample_100k.npy}"
NUM_SAMPLES="${3:-100000}"
SEED="${SEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" - "$INPUT_PATH" "$OUTPUT_PATH" "$NUM_SAMPLES" "$SEED" <<'PY'
import sys
from pathlib import Path

import numpy as np


def load_fvecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size == 0:
        raise ValueError(f"Empty fvecs file: {path}")

    dim = np.frombuffer(np.array([raw[0]], dtype=np.float32).tobytes(), dtype=np.int32)[0]
    if dim <= 0:
        raise ValueError(f"Invalid fvecs dimension header: {dim}")

    row_width = dim + 1
    if raw.size % row_width != 0:
        raise ValueError(
            f"Corrupt fvecs layout for {path}: size={raw.size}, row_width={row_width}"
        )

    matrix = raw.reshape(-1, row_width)
    dims = matrix[:, 0].view(np.int32)
    if not np.all(dims == dim):
        raise ValueError(f"Inconsistent dimensions in fvecs file: {path}")

    return matrix[:, 1:].astype(np.float32, copy=False)


def main() -> None:
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    num_samples = int(sys.argv[3])
    seed = int(sys.argv[4])

    if num_samples <= 0:
        raise ValueError(f"num_samples must be > 0, got {num_samples}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    vectors = load_fvecs(input_path)
    total = vectors.shape[0]
    if num_samples > total:
        raise ValueError(
            f"Requested {num_samples} samples but dataset has only {total} vectors."
        )

    rng = np.random.default_rng(seed)
    indices = rng.choice(total, size=num_samples, replace=False)
    sampled = vectors[indices].astype(np.float32, copy=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, sampled)

    print(
        f"Saved {sampled.shape[0]} vectors (dim={sampled.shape[1]}) "
        f"to {output_path} from {input_path}"
    )


if __name__ == "__main__":
    main()
PY
