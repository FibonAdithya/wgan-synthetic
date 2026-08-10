"""Settings for one EDA report run, as a typed value object.

`compare_variants` builds the argparse Namespace that `pipeline.run` consumes
by hand, so the Namespace stays the public contract. Everything downstream of
`run` takes an `EdaConfig` instead: a panel that needs `bins` should not be
able to reach `sys.argv`-shaped state.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

# Single source of truth for the ANN-difficulty flag defaults, shared with
# compare_variants.py so its hand-built Namespace cannot silently drift from
# what the CLI's own --ann-* / --ivf-nlist flags default to.
ANN_K_DEFAULT = 100
ANN_HUB_K_DEFAULT = 10
ANN_MAX_ROWS_DEFAULT = 20000
IVF_NLIST_DEFAULT = 256
# The within-set k-NN distance panel is not an ANN-difficulty panel, so it
# gets its own knob rather than riding on --ann-max-rows: tuning the cost of
# the difficulty metrics should not silently move a pre-existing panel. The
# default is the same number, so nothing changes unless a flag is passed.
KNN_MAX_ROWS_DEFAULT = ANN_MAX_ROWS_DEFAULT

GLYPH_SAMPLES_DEFAULT = 8

# Above this width the per-dimension marginals and correlation panels are
# dropped. Both scale with the square of the dimension -- the marginals
# dropdown carries one visibility flag per (button, trace) pair, and the
# correlation heatmap is dim x dim -- so at 1536 they cost tens of megabytes
# of report each. They also say least at that width: a 1536-entry dropdown is
# not something a reader pages through, and per-dimension marginals of a text
# embedding carry almost no signal.
#
# 256 keeps both panels for sift (128), deep (96), glove (100) and nytimes
# (256), and drops them for gist (960) and openai (1536).
MAX_PANEL_DIM_DEFAULT = 256


@dataclass(frozen=True)
class EdaConfig:
    """One run's settings. Field names match the CLI flags exactly."""

    real_path: str
    real_format: str
    synthetic_path: list[str]
    synthetic_format: str
    output_dir: str
    preprocess: str
    max_vectors: int
    num_pairs: int
    knn: int
    ann_k: int
    ann_hub_k: int
    ann_max_rows: int
    knn_max_rows: int
    ivf_nlist: int
    bins: int
    top_divergent: int
    seed: int
    no_png: bool
    glyph_samples: int
    plotlyjs: str
    max_panel_dim: int

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> EdaConfig:
        """Build from the Namespace `pipeline.run` was handed.

        `glyph_samples` and `max_panel_dim` are read defensively because
        `compare_variants` has built Namespaces without them; every other
        field is required, so a Namespace missing one fails here rather than
        deep inside a panel.
        """
        return cls(
            real_path=args.real_path,
            real_format=args.real_format,
            synthetic_path=list(args.synthetic_path or []),
            synthetic_format=args.synthetic_format,
            output_dir=args.output_dir,
            preprocess=args.preprocess,
            max_vectors=args.max_vectors,
            num_pairs=args.num_pairs,
            knn=args.knn,
            ann_k=args.ann_k,
            ann_hub_k=args.ann_hub_k,
            ann_max_rows=args.ann_max_rows,
            knn_max_rows=args.knn_max_rows,
            ivf_nlist=args.ivf_nlist,
            bins=args.bins,
            top_divergent=args.top_divergent,
            seed=args.seed,
            no_png=args.no_png,
            glyph_samples=getattr(args, "glyph_samples", GLYPH_SAMPLES_DEFAULT),
            plotlyjs=args.plotlyjs,
            max_panel_dim=getattr(args, "max_panel_dim", MAX_PANEL_DIM_DEFAULT),
        )
