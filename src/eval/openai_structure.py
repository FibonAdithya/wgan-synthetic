"""Structural EDA for the openai family, as design input for its WGAN-GP.

Deliberately outside src/eval/eda/: these questions are about one family's
geometry, not about comparing a generator against a corpus, and the report
package should not grow to answer them until they prove general.

Answers four questions the v0 config currently guesses at:

1. Is `latent_dim: 512` sized to the data? -- PCA spectrum and intrinsic
   dimension.
2. Is `generator_type: mlp` with post-hoc L2 normalization the right family,
   or does the corpus want a native spherical parameterisation? -- norms and
   anisotropy.
3. Should `preprocess.center` be turned on for a v1 rung? -- the spectrum
   before and after centering.
4. Which of the four gate statistics can carry a band at all? -- the noise
   floor across disjoint draws of the same real data.

Plus one question that decides whether any of the above survives: this
family is searched under `angular` but ann_difficulty.py measures under L2
(issue #16). Section 5 measures how much that actually changes.

    python -m src.eval.openai_structure \
        --real-path data/openai_250k.npy \
        --output-dir runs/openai/structure
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors

from src.eval import ann_difficulty
from src.eval.eda import html
from src.eval.eda.series import REAL_COLOR, SYNTH_PALETTE

# The gate's locked conditions, from gates/openai.yaml. Hardcoded rather than
# read from the gate file because this script reports alongside the gate, not
# against it -- if the two ever disagree the gate wins, and the mismatch
# should be visible in a diff rather than silently inherited.
CANONICAL_N = 20_000
CANONICAL_K = 100
CANONICAL_K_HUB = 10
CANONICAL_NLIST = 256

GATE_STATS = (
    "lid_median",
    "relative_contrast_median",
    "hubness_skew",
    "ivf_gini",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--draws",
        type=int,
        default=10,
        help=(
            "Disjoint draws for the noise floor. Each is CANONICAL_N rows, so "
            "the corpus must hold draws * 20000 rows to supply them."
        ),
    )
    parser.add_argument(
        "--pca-rows",
        type=int,
        default=50_000,
        help=(
            "Rows for the eigenvalue spectrum. Needs comfortably more than "
            "1536 for the covariance to be full rank and stable; the default "
            "is ~32 rows per dimension."
        ),
    )
    parser.add_argument("--num-pairs", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument(
        "--plotlyjs", type=str, default="inline", choices=["inline", "cdn", "directory"]
    )
    return parser.parse_args()


# --------------------------------------------------------------------------
# measurements
# --------------------------------------------------------------------------


def sample_rows(x: np.ndarray, count: int, seed: int) -> np.ndarray:
    """A random `count`-row subsample. Never a prefix.

    openai_250k.npy is written by subset_parquet, which sorts its random
    draw back into corpus order before saving. Its rows are therefore in
    ascending original-corpus index order, so `x[:50000]` is the *first*
    fifth of the corpus as DBpedia orders it, not a sample of it. DBpedia
    entities are not shuffled, so that prefix can be topically skewed --
    the same failure the fetcher avoids by sampling across all 26 shards
    instead of reading the first few.
    """
    if count <= 0 or x.shape[0] <= count:
        return x
    rng = np.random.default_rng(seed)
    return x[np.sort(rng.choice(x.shape[0], size=count, replace=False))]


def norm_facts(x: np.ndarray) -> dict[str, float]:
    """Whether the corpus really is unit-norm, rather than said to be.

    docs/datasets/openai.md asserts "already unit-norm". If that holds, the
    v0 config's l2_normalize is a no-op on the real side and the only thing
    it constrains is the generator's output.
    """
    norms = np.linalg.norm(x, axis=1)
    return {
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_max_abs_deviation_from_1": float(np.abs(norms - 1.0).max()),
    }


def anisotropy_facts(x: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    """How far the corpus is from being centred on the origin.

    Text embeddings are famously anisotropic: they occupy a narrow cone
    rather than the whole sphere, because a large shared component survives
    in every vector. The mean vector's norm measures that directly -- on an
    isotropic unit sphere it would be ~0, and at 1.0 every vector points the
    same way.

    This is the central mlp-versus-spherical evidence. A narrow cone is a
    harder target for an mlp that reaches the sphere only by dividing at the
    end, because almost all of its output space maps outside the cone.
    """
    mean_vector = x.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean_vector))
    direction = mean_vector / max(mean_norm, 1e-12)
    cos_to_mean = x @ direction
    return (
        {
            "mean_vector_norm": mean_norm,
            "cos_to_mean_median": float(np.median(cos_to_mean)),
            "cos_to_mean_min": float(cos_to_mean.min()),
        },
        cos_to_mean,
    )


def pairwise_cosine(x: np.ndarray, num_pairs: int, seed: int) -> np.ndarray:
    """Cosine similarity between random pairs -- the angular geometry itself.

    `data.metric: angular` names this distribution. On an isotropic sphere at
    this width it would concentrate hard on 0; anything well above 0 is the
    cone showing up again.
    """
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    i = rng.integers(0, n, size=num_pairs)
    j = rng.integers(0, n, size=num_pairs)
    keep = i != j
    a = x[i[keep]]
    b = x[j[keep]]
    return np.einsum("ij,ij->i", a, b) / (
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    )


def spectrum_facts(
    x: np.ndarray, label: str, *, center: bool
) -> tuple[dict[str, float], np.ndarray]:
    """Eigenvalue spectrum, and the component counts that bracket `latent_dim`.

    `center` is the whole point of having this twice, and it is why this does
    not use sklearn's PCA: PCA *always* subtracts the mean, so asking it for
    a centered and an uncentered spectrum returns the same numbers twice.
    The two matrices are genuinely different questions:

    - center=False: eigenvalues of the second-moment matrix X'X/n, about the
      origin. This is the geometry a generator faces under v0's
      `preprocess.center: false` -- the mean direction is part of what it has
      to reproduce.
    - center=True: eigenvalues of the covariance, about the mean. The
      geometry it would face under `center: true`.

    The gap between them is how much of this corpus's apparent spread is just
    the shared mean direction, which is the evidence for or against a
    centering rung.

    Going through the d x d Gram matrix rather than an SVD of the n x d data
    also makes this seconds rather than minutes at 1536 dimensions, so the
    full corpus can be used instead of a subsample.

    The participation ratio is a spectrum-shape summary that needs no
    variance cutoff: (sum l)^2 / sum(l^2). For a flat spectrum over d
    directions it is d; for one dominant direction it is 1. It reads as "how
    many directions this corpus effectively uses".
    """
    x = np.asarray(x, dtype=np.float64)
    if center:
        x = x - x.mean(axis=0)
    gram = (x.T @ x) / x.shape[0]
    eigenvalues = np.linalg.eigvalsh(gram)[::-1]
    eigenvalues = np.clip(eigenvalues, 0.0, None)

    ratio = eigenvalues / eigenvalues.sum()
    cumulative = np.cumsum(ratio)
    participation = float(eigenvalues.sum() ** 2 / np.sum(eigenvalues**2))

    facts = {
        f"{label}_components_90pct": int(np.searchsorted(cumulative, 0.90) + 1),
        f"{label}_components_95pct": int(np.searchsorted(cumulative, 0.95) + 1),
        f"{label}_components_99pct": int(np.searchsorted(cumulative, 0.99) + 1),
        f"{label}_participation_ratio": participation,
        f"{label}_top_component_share": float(ratio[0]),
    }
    return facts, ratio


def two_nn_dimension(x: np.ndarray, seed: int, max_rows: int = 20_000) -> float:
    """Facco two-NN global intrinsic dimension.

    Uses only each point's two nearest neighbours, so it is far less
    sensitive to the curse of dimensionality than a full-neighbourhood
    estimator. Complements LID, which is the same quantity measured locally
    and averaged.

        mu_i = r2/r1,  ID = mean(log mu)^-1 over the surviving points
    """
    rng = np.random.default_rng(seed)
    if x.shape[0] > max_rows:
        x = x[np.sort(rng.choice(x.shape[0], size=max_rows, replace=False))]
    nn = NearestNeighbors(n_neighbors=3, algorithm="brute").fit(x)
    dist, _ = nn.kneighbors(x)
    r1, r2 = dist[:, 1], dist[:, 2]
    keep = r1 > 0
    mu = r2[keep] / r1[keep]
    return float(1.0 / np.mean(np.log(mu[mu > 1.0])))


def disjoint_draws(x: np.ndarray, count: int, rows: int, seed: int) -> list[np.ndarray]:
    """`count` non-overlapping row blocks of `rows` each, in random order.

    Disjoint is the point: two draws that share rows share their hubs and
    their duplicate structure, which would understate the spread this is
    trying to measure.
    """
    rng = np.random.default_rng(seed)
    needed = count * rows
    if x.shape[0] < needed:
        raise ValueError(
            f"noise floor needs {needed} rows for {count} disjoint draws of "
            f"{rows}, corpus has {x.shape[0]}"
        )
    order = rng.permutation(x.shape[0])[:needed]
    return [x[np.sort(order[i * rows : (i + 1) * rows])] for i in range(count)]


def gate_statistics(x: np.ndarray, seed: int) -> dict[str, float]:
    """The four gate statistics for one draw, at the locked conditions."""
    metrics = ann_difficulty.compute(
        x,
        k=CANONICAL_K,
        k_hub=CANONICAL_K_HUB,
        nlist=CANONICAL_NLIST,
        max_rows=CANONICAL_N,
        seed=seed,
    )
    full = ann_difficulty.summary(metrics)
    stats = {k: full[k] for k in GATE_STATS}
    # summary() returns None for the two median statistics when every query
    # was discarded. Numpy would turn that into a silent nan two calls later
    # and the noise floor would report a nan spread as though it had measured
    # something.
    degenerate = [k for k, v in stats.items() if v is None]
    if degenerate:
        raise ValueError(
            f"{', '.join(degenerate)} came back None: every query in this draw "
            "was discarded, which means the draw is degenerate (all duplicates "
            "or all equidistant) rather than merely hard."
        )
    return stats


def noise_floor(draws: list[np.ndarray], seed: int) -> dict[str, dict[str, float]]:
    """Spread of each gate statistic across disjoint draws of the same corpus.

    This is the measurement that decides whether a band is meaningful. A
    statistic whose spread across redraws of *real* data is as wide as the
    distance a generator would have to close cannot separate a good
    generator from a bad one, however carefully the band is chosen. SIFT's
    relative contrast and IVF Gini failed this test, and GloVe's hubness
    skew swung 3.46-8.33 (issue #29).
    """
    per_draw = [gate_statistics(d, seed) for d in draws]
    out: dict[str, dict[str, float]] = {}
    for stat in GATE_STATS:
        values = np.array([p[stat] for p in per_draw], dtype=np.float64)
        median = float(np.median(values))
        out[stat] = {
            "min": float(values.min()),
            "median": median,
            "max": float(values.max()),
            "spread": float(values.max() - values.min()),
            # The spread as a fraction of the level. A band has to be wider
            # than this to admit real data, so it is the floor on how tight
            # any band for this statistic can be.
            "spread_pct_of_median": float(
                100.0 * (values.max() - values.min()) / abs(median)
            )
            if median
            else float("nan"),
        }
    return out


def _cosine_knn(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Cosine k-NN with self-exclusion by index, not by dropping column 0.

    Same care ann_difficulty.knn takes, for the same reason: exact duplicate
    rows tie with the query at distance 0 and sklearn does not promise the
    query sorts first. Dropping the first column would then leave a point in
    its own neighbour list, inflating its k-occurrence and deflating its LID
    -- and DBpedia does contain near-identical entities, so this is not a
    hypothetical.
    """
    n = x.shape[0]
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
    dist, idx = nn.fit(x).kneighbors(x)

    rows = np.arange(n)[:, None]
    keep = idx != rows
    surplus = keep.sum(axis=1) > k
    if np.any(surplus):
        last_true = (keep.shape[1] - 1) - np.argmax(keep[:, ::-1], axis=1)
        keep[surplus, last_true[surplus]] = False

    selected = np.where(keep)
    return dist[selected].reshape(n, k), idx[selected].reshape(n, k)


def angular_vs_l2(
    x: np.ndarray, seed: int, k: int = CANONICAL_K, k_hub: int = CANONICAL_K_HUB
) -> dict[str, float]:
    """How much the L2-versus-angular choice changes what gets measured.

    For unit-norm vectors ||a-b||^2 = 2 - 2cos(a,b), a strictly monotone map,
    so the two metrics should induce identical neighbour *sets* -- which
    would make hubness exactly invariant and leave only the distance-ratio
    statistics to rescale. Measured rather than asserted, because if it does
    not hold, every number this family records under L2 has to be remeasured
    when phase (c) lands rather than merely reinterpreted.

    Each statistic is measured at the k the gate measures it at -- LID at
    CANONICAL_K, hubness at CANONICAL_K_HUB -- so the numbers here are
    comparable with the ones in the profile table. Measuring both at one
    convenient k would produce an internally consistent comparison whose
    LID was not the LID anything else reports.
    """
    x = sample_rows(x, CANONICAL_N, seed)

    l2_dist, l2_idx, _ = ann_difficulty.knn(x, k)
    cos_dist, cos_idx = _cosine_knn(x, k)

    agreement = float(
        np.mean([len(set(a) & set(b)) / k for a, b in zip(l2_idx, cos_idx)])
    )

    l2_survivors = ann_difficulty.survivor_mask(l2_dist)
    cos_survivors = ann_difficulty.survivor_mask(cos_dist)
    return {
        "angular_k": k,
        "angular_k_hub": k_hub,
        "neighbour_set_agreement": agreement,
        # Sliced to k_hub from the same cache, exactly as ann_difficulty.compute
        # does, so these are the gate's hubness numbers under each metric.
        "hubness_skew_l2": ann_difficulty.hubness_skew(
            ann_difficulty.k_occurrence(l2_idx, x.shape[0], k_hub)
        ),
        "hubness_skew_cosine": ann_difficulty.hubness_skew(
            np.bincount(cos_idx[:, :k_hub].ravel(), minlength=x.shape[0])
        ),
        "lid_median_l2": float(
            np.median(ann_difficulty.lid_mle(l2_dist[l2_survivors]))
        ),
        "lid_median_cosine": float(
            np.median(ann_difficulty.lid_mle(cos_dist[cos_survivors]))
        ),
    }


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------


def _layout(fig: go.Figure, title: str, x_title: str, y_title: str) -> go.Figure:
    """Recessive grid and axes, per the house style in eda/figures.py."""
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        template="plotly_white",
        height=420,
        margin=dict(l=60, r=30, t=60, b=50),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def fig_histogram(
    values: np.ndarray, bins: int, title: str, x_title: str, color: str
) -> go.Figure:
    """Overlaid density histogram, tolerant of a constant series.

    The norms panel is the reason for that tolerance: this corpus is
    supposed to be exactly unit-norm, and if it is, every value is 1.0 and
    numpy raises "Too many bins for data range" rather than drawing a spike.
    A constant series is a *result* here -- it is the family page's claim
    coming out true -- so it gets a one-bin figure that says so, not a
    crash.
    """
    low, high = float(np.min(values)), float(np.max(values))
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError(f"{title}: values contain nan or inf")

    if high - low <= 0.0:
        fig = go.Figure()
        fig.add_bar(x=[low], y=[1.0], marker_color=color, name=x_title)
        fig = _layout(fig, f"{title} (constant at {low:.6g})", x_title, "density")
        fig.update_xaxes(range=[low - 1.0, high + 1.0])
        return fig

    counts, edges = np.histogram(values, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig = go.Figure()
    fig.add_bar(x=centers, y=counts, marker_color=color, name=x_title)
    return _layout(fig, title, x_title, "density")


def fig_spectrum(spectra: list[tuple[str, np.ndarray, str]]) -> go.Figure:
    """Cumulative explained variance. One axis, one question: how many
    directions does this corpus actually use?"""
    fig = go.Figure()
    for name, ratio, color in spectra:
        cumulative = np.cumsum(ratio)
        fig.add_scatter(
            x=np.arange(1, ratio.size + 1),
            y=cumulative,
            name=name,
            line=dict(color=color, width=2),
        )
    fig.add_hline(y=0.95, line=dict(color="#718096", width=1, dash="dot"))
    _layout(fig, "Cumulative explained variance", "components", "fraction of variance")
    fig.update_xaxes(type="log")
    return fig


def fig_noise_floor(floor: dict[str, dict[str, float]]) -> go.Figure:
    """Each statistic's spread across disjoint real draws, as a fraction of
    its own level -- the only form in which the four are comparable."""
    names = list(floor)
    values = [floor[n]["spread_pct_of_median"] for n in names]
    fig = go.Figure()
    fig.add_bar(
        x=values,
        y=names,
        orientation="h",
        marker_color=REAL_COLOR,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
    )
    _layout(fig, "Noise floor across disjoint real draws", "spread (% of median)", "")
    fig.update_layout(height=320)
    return fig


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def facts_table(facts: dict[str, float]) -> str:
    rows = "".join(
        f"<tr><th>{k}</th><td>{html.format_stat(v)}</td></tr>" for k, v in facts.items()
    )
    return f"<table><tbody>{rows}</tbody></table>"


def noise_floor_table(floor: dict[str, dict[str, float]]) -> str:
    head = "<tr><th>statistic</th><th>min</th><th>median</th><th>max</th><th>spread</th><th>% of median</th></tr>"
    rows = []
    for stat, f in floor.items():
        rows.append(
            f"<tr><th>{stat}</th>"
            + "".join(
                f"<td>{html.format_stat(f[key])}</td>"
                for key in ("min", "median", "max", "spread", "spread_pct_of_median")
            )
            + "</tr>"
        )
    return f"<table><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x = np.load(args.real_path, mmap_mode="r")
    x = np.ascontiguousarray(x, dtype=np.float32)
    print(f"loaded {x.shape} from {args.real_path}")

    facts: dict[str, float] = {"rows": x.shape[0], "dim": x.shape[1]}
    facts.update(norm_facts(x))
    aniso, cos_to_mean = anisotropy_facts(x)
    facts.update(aniso)

    pairs = pairwise_cosine(x, args.num_pairs, args.seed)
    facts["pairwise_cos_median"] = float(np.median(pairs))
    facts["pairwise_cos_p99"] = float(np.percentile(pairs, 99))

    pca_rows = sample_rows(x, args.pca_rows, args.seed)
    raw_facts, raw_ratio = spectrum_facts(pca_rows, "raw", center=False)
    centered_facts, centered_ratio = spectrum_facts(pca_rows, "centered", center=True)
    facts.update(raw_facts)
    facts.update(centered_facts)

    facts["two_nn_intrinsic_dim"] = two_nn_dimension(x, args.seed)
    print("spectrum and intrinsic dimension done")

    angular = angular_vs_l2(x, args.seed)
    facts.update(angular)
    print("angular-vs-L2 done")

    floor = noise_floor(
        disjoint_draws(x, args.draws, CANONICAL_N, args.seed), args.seed
    )
    print("noise floor done")

    sections = [
        (
            "Vector norms",
            "The family page calls this corpus already unit-norm. If that "
            "holds, v0's <code>l2_normalize</code> constrains only the "
            "generator's output, not the real side.",
            fig_histogram(
                np.linalg.norm(pca_rows, axis=1),
                args.bins,
                "L2 norm",
                "norm",
                REAL_COLOR,
            ),
        ),
        (
            "Anisotropy: cosine to the mean direction",
            "How tightly the corpus clusters around a single shared "
            "direction. Mass concentrated well above zero means a narrow "
            "cone rather than a sphere -- the case where an mlp that reaches "
            "the sphere only by dividing at the end is aiming most of its "
            "output space outside the data.",
            fig_histogram(
                sample_rows(cos_to_mean, args.pca_rows, args.seed),
                args.bins,
                "cosine to mean direction",
                "cos",
                SYNTH_PALETTE[0],
            ),
        ),
        (
            "Pairwise cosine similarity",
            "The geometry <code>data.metric: angular</code> actually names. "
            "On an isotropic sphere at 1536 dimensions this would concentrate "
            "hard on zero.",
            fig_histogram(
                pairs, args.bins, "cosine similarity", "cos", SYNTH_PALETTE[1]
            ),
        ),
        (
            "PCA spectrum",
            "Where <code>latent_dim: 512</code> gets tested against the data. "
            "Centering is shown alongside because it is the cheapest v1 rung "
            "available, and its effect on the spectrum is the evidence for "
            "or against it.",
            fig_spectrum(
                [
                    ("raw", raw_ratio, REAL_COLOR),
                    ("centered", centered_ratio, SYNTH_PALETTE[0]),
                ]
            ),
        ),
        (
            "Noise floor",
            "Spread of each gate statistic across "
            f"{args.draws} disjoint {CANONICAL_N}-row draws of the same real "
            "corpus. A band narrower than this bar would reject real data, so "
            "this is the floor on how tight any band for that statistic can "
            "be. No band is set here: gates/openai.yaml stays unset.",
            fig_noise_floor(floor),
        ),
    ]

    meta_html = (
        f'<div class="meta">real: <code>{args.real_path}</code>'
        f" &middot; {x.shape[0]} rows &times; {x.shape[1]} dims"
        f" &middot; measured at N={CANONICAL_N}, k={CANONICAL_K},"
        f" k_hub={CANONICAL_K_HUB}, nlist={CANONICAL_NLIST}</div>"
        + facts_table(facts)
        + "<h2>Noise floor detail</h2>"
        + noise_floor_table(floor)
    )

    report = html.build_report(
        sections,
        meta_html,
        html.plotlyjs_head(args.plotlyjs, out_dir),
        heading=f"openai structure: {Path(args.real_path).stem}",
    )
    report_path = out_dir / "openai_structure.html"
    report_path.write_text(report, encoding="utf-8")

    (out_dir / "structure.json").write_text(
        json.dumps({"facts": facts, "noise_floor": floor}, indent=2), encoding="utf-8"
    )
    print(f"Wrote {report_path}")
    print(f"Wrote {out_dir / 'structure.json'}")


if __name__ == "__main__":
    main()
