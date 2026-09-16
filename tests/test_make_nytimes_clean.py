"""`clean_nytimes_rows` rebuilds the cleaned NYTimes corpus the
`nytimes_v*_seed42_job.sh` scripts read as `$CLEAN`. The GPU box's copy was
wiped in the rebuild, so this is exercised against a small fixture built in
the test rather than the real 250k-row file, which this suite cannot see.
"""

import numpy as np
from scripts.make_nytimes_clean import clean_nytimes_rows

# Row 2 is exact zero (must be dropped). Row 3 is an exact duplicate of row 1
# (must be dropped, keeping row 1). Row 4 differs from row 1 by 1e-3 in one
# coordinate -- a near-duplicate, not an exact one, so it must survive. Row 5
# is unrelated. Rows 0, 1, 4, 5 are the expected survivors, in that order.
_ROW0 = [1.0, 0.0, 0.0, 0.0]
_ROW1 = [0.0, 1.0, 0.0, 0.0]
_ZERO = [0.0, 0.0, 0.0, 0.0]
_DUPLICATE_OF_ROW1 = [0.0, 1.0, 0.0, 0.0]
_NEAR_DUPLICATE_OF_ROW1 = [0.0, 1.0, 0.0, 1.0e-3]
_ROW5 = [2.0, 2.0, 0.0, 0.0]


def _fixture() -> np.ndarray:
    return np.array(
        [_ROW0, _ROW1, _ZERO, _DUPLICATE_OF_ROW1, _NEAR_DUPLICATE_OF_ROW1, _ROW5],
        dtype=np.float32,
    )


def _unit(row: list[float]) -> np.ndarray:
    v = np.array(row, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_drops_zero_dedupes_and_normalises():
    cleaned, stats = clean_nytimes_rows(_fixture())

    assert stats.rows_in == 6
    assert stats.zero_dropped == 1
    assert stats.duplicates_dropped == 1
    assert stats.rows_out == 4
    assert cleaned.shape == (4, 4)
    assert cleaned.dtype == np.float32

    # Every surviving row is unit-norm.
    norms = np.linalg.norm(cleaned, axis=1)
    np.testing.assert_allclose(norms, np.ones(4), atol=1e-6)

    # First occurrences, in input order: row0, row1 (not its later exact
    # duplicate), the near-duplicate, then row5.
    np.testing.assert_allclose(cleaned[0], _unit(_ROW0), atol=1e-6)
    np.testing.assert_allclose(cleaned[1], _unit(_ROW1), atol=1e-6)
    np.testing.assert_allclose(cleaned[2], _unit(_NEAR_DUPLICATE_OF_ROW1), atol=1e-6)
    np.testing.assert_allclose(cleaned[3], _unit(_ROW5), atol=1e-6)

    # The zero row is gone: nothing in the output is all-zero, and no row
    # sits at the origin's normalised (undefined) direction.
    assert (norms > 0).all()
