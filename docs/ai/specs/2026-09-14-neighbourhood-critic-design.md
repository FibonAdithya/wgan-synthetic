# Neighbourhood-aware critic

Design, 2026-09-14. Not authoritative (see `AGENTS.md`); the code and
`PROJECT_DOCUMENTATION.md` win on any conflict once built.

Three approaches are specified. Approach 1 is the rung to build first
(`v2`). Approaches 2 and 3 (`v2b`, `v2c`) are written to the same level of
detail so a second agent can build and run them in parallel, or so they can
be taken up if `v2` misses the bar. All three share the corpus decision, the
distance floor, the gradient-penalty formulation, the success bar and most of
the tests, in the "Shared" section. Each approach section then says only
what differs.

## Problem

`v1` (linear-skip generator, `docs/ai/specs/2026-09-11-linear-skip-generator-design.md`)
kept the output full rank but not the trunk-to-skip balance: the trunk's
output energy grew 38x over 30k steps while the skip's held still, the
residual's share of each output fell from 91% to 20%, and LID drifted
monotonically through the real value. Continued to 100k steps, `W` itself
lost rank (254 to 50 singular values above 0.1) and the sample ended at LID
5.2, effective rank 6.4. Throughout, the Wasserstein estimate stayed under
0.06 and the gradient penalty under 0.01 (`docs/datasets/nytimes.md`,
"Continued to 100,000 steps"). The critic was satisfied while the geometry
collapsed.

The critic scores one row at a time. Local intrinsic dimension is a property
of a row's neighbours, not of the row, so a six-dimensional sheet with the
right mean and covariance envelope is, to a per-vector critic, the corpus.
The fixed-statistic version of the fix already exists: `lid_reg` (SIFT `v4`)
matches the within-batch log-ratio profile as a generator regulariser. It
moved contrast and hubness, but it fits one statistic we chose. A
neighbourhood-aware critic gets neighbourhood information as input and
learns whatever separates real from fake, including the balance drift.

### Measured before this design

`batch_profile_probe.py` and `duplicate_probe.py` (session scratchpad,
numpy on CPU, throwaway; results below are the record). Cleaned corpus
`data/nytimes_250k_l2_clean.npy`, 60k-row subsample, 40 batches of 512,
populations: real; real projected onto its top-6 principal directions and
renormalised (the `v1` step-100k geometry); the same at rank 30; a Gaussian
with the real covariance, renormalised.

Per-row logistic regression on the within-batch log-ratio profile alone,
held-out accuracy (0.5 is no separation):

| real vs | k=20 | k=100 |
|---|---|---|
| rank-6 sheet | 0.998 | 1.000 |
| rank-30 sheet | 0.976 | 1.000 |
| Gaussian, real covariance | 0.619 | 0.553 |
| real (control) | 0.505 | 0.491 |

Within-batch Hill LID at batch 512, k=20: real 64.7, rank-6 sheet 5.7,
Gaussian 73.6, batch-to-batch std about 1. So the per-row within-batch
profile carries the collapse signal almost perfectly, and the
real-versus-Gaussian ordering weakly per row and clearly per batch. k=20
separates the Gaussian end better than k=100 and matches the `lid_reg_k`
default, so it is the default here.

Duplicates and zero rows, shipped normalised 250k file, 200 batches of 512:
0.44 exact-zero rows per batch (max 3); 0.75 rows per batch whose batch-mate
is an exact copy (0.15% of rows); among rows that are not copies, 16 of
102k have a batch-mate closer than 0.1, and the nearest-neighbour distance
has median 1.22 and first percentile 0.90. Exact 1-NN over a 20k subsample
finds a copy for 3.4% of rows, so the copy rate seen by a critic scales with
how many rows it can see: 250k rows show 14%, a 512 batch shows 0.15%.
The expanded-square distance form rounds an exact copy to about 1e-8, not
0, which is why the floor below is a clamp on distance and not a test for
zero.

## Shared

### Corpus: duplicates kept, exact-zero rows dropped

Exact duplicate rows are part of the search target. The corpus is built from
documents, 14% of its rows are exact copies, the gate keeps them as
legitimate queries (`gate_statistics` drops only zero rows), and a
generator that never produces near-copies has the wrong density. They stay.

Exact-zero rows are an empty document after preprocessing. The gate drops
them, the trainer's `normalize_l2` clamp means the generator cannot emit
one, and to a neighbourhood critic a zero row is a row at L2 exactly 1.0
from everything: a profile no fake can have, i.e. a shortcut with nothing to
emulate. They are dropped at load.

New key `data.preprocess.drop_zero_rows`, default `false` (every existing
config unchanged). When true, `build_training_data` drops rows with exact
L2 norm zero from the loaded array before the train/holdout split and
records the count as `dropped_zero_rows` in `run_metadata.json`. Before
the split, so the holdout is also clean and the gate reference measured on
it matches what the critic saw. `PreprocessConfig` gains the field;
`PreprocessState.to_serializable` carries it so `run_config.yaml` and the
checkpoint say what happened.

### The distance floor

Every within-batch or bank distance the critic reads is clamped from below
at `critic_distance_floor`, default `0.01`. This sits inside the measured
gap between exact copies (about 1e-8 after rounding) and the nearest
genuine pair (first percentile 0.90, 16 in 102k below 0.1), so it changes
the feature of a copy and of nothing else. A copy then reads as a bounded
"tight pair" feature, log(0.01 / r_k) about -4.9 against a typical
log(r_1 / r_k) of -0.08 at k=20, which a continuous generator can match by
producing near-copies within 0.01. That is the duplication behaviour the
generator should learn, expressed as something reachable by gradient
rather than an infinity it can only chase.

The floor is applied to squared distances as `clamp(min=floor**2)` after
the diagonal is masked, in float32 (see "Numerics").

Two module-level functions in `src/models/critic.py` carry this so the
tests can target them directly and so approaches 1 to 3 share one
implementation: `neighbourhood_distances(x, k, floor) -> (n, k)`, the
sorted, floored within-batch distances with self excluded (approach 2
passes a bank as a second argument); and `profile_features(r) -> (n, k)`,
the log-ratio profile plus log scale defined in approach 1.

### Critic interface and factory

`src/models/critic.py` keeps `Critic` unchanged: `forward(x) -> (n,)`, rows
scored independently, and `tests/test_critic.py` keeps asserting that. New
classes in the same module share the calling convention
`forward(x: (n, D)) -> (n,)` but are batch-dependent by design.

`build_critic(model_cfg, input_dim) -> nn.Module` mirrors
`build_generator`: `critic_type` in `model`, default `per_vector` (today's
`Critic`), or `neighbourhood` (approach 1), `neighbourhood_bank`
(approach 2), `neighbourhood_set` (approach 3); anything else raises
`ValueError` naming the offending value. The trainer's one construction
site calls the factory. `critic_hidden_dims` and `negative_slope` mean the
same for every type. Common keys:

| key | default | meaning |
|---|---|---|
| `critic_type` | `per_vector` | which class |
| `critic_k` | `20` | neighbour depth; the profile has `critic_k` entries (see each approach) |
| `critic_distance_floor` | `0.01` | lower clamp on every neighbour distance the critic reads |

### Gradient penalty

`gradient_penalty` in `src/train/train_wgan_gp.py` does not change. It
interpolates real and fake row-wise, runs the critic on the interpolated
batch, and takes the gradient of the summed scores with respect to each
row. For a per-vector critic that is each row's own score gradient. For a
batch-dependent critic it is the gradient of the batch's summed score with
respect to row i, which includes how row i moves every other row's
neighbourhood features. The penalty therefore constrains the Lipschitz
constant of the summed batch score, one row at a time. This is the standard
formulation for minibatch-discrimination critics and is what the spec
intends; the docstring says so, so nobody "fixes" it back to per-row later.

The interpolated batch mixes real and fake rows, so its neighbourhoods are
mixed. That is fine: the penalty is about smoothness of the critic, not
about the population the batch came from. The only approach where this
needs a decision is approach 2 (which bank the interpolates query), settled
in its section.

### Numerics

The neighbour computation runs in float32 with autocast disabled, for the
reasons the `lid_reg` block in the trainer records: the expanded-square form
cancels catastrophically in fp16, and the floor is far below fp16's useful
range. Distances use the expanded-square form with `clamp(min=0)` rather
than `torch.cdist`, because cdist's gradient is undefined at zero distance
and a single copy would poison the backward pass. Self is excluded by
index, by writing `inf` on the diagonal, not by dropping the nearest column,
so an exact copy is still a neighbour. `torch.topk(..., largest=False,
sorted=True)` on squared distances gives sorted `r_1 .. r_k` with gradients
through the selected entries. The MLP itself runs under whatever autocast
the trainer has.

`v1_seed42` runs with `amp: false`, so on the rung itself this is
belt-and-braces; the tests cover it on CPU.

### Duplication diagnostic

`AnnMetrics` in `src/eval/ann_difficulty.py` gains `nearest_distance`
(`dist[:, 0]`, one entry per row, before the survivor mask). `gate_statistics`
in `src/train/selection.py` logs `near_duplicate_fraction`, the fraction of
measured rows with `nearest_distance <= critic_distance_floor` (the same
constant, imported from one place), on the real holdout and on the fake
sample, as `eval.gate_real_near_duplicate_fraction` and
`eval.gate_fake_near_duplicate_fraction`. Logged, not scored: it says
whether the run learned to make near-copies; it does not decide selection.
The real holdout figure will be well under 14% because copies are counted
against the holdout only, and the page must say so where it quotes it.

### Success bar

Same as the `v1` spec, unchanged: all four gate statistics of the selected
checkpoint, measured against the cleaned real corpus at the canonical
conditions, inside the real ten-draw range or, for LID and contrast, within
3% of the real median (`docs/datasets/nytimes_noise_floor.json`): LID
55.0 to 56.9, contrast 1.265 to 1.276, hubness 2.32 to 2.78, Gini 0.78 to
0.82. In addition, the mechanism `v1` failed on must be visibly gone: the
trunk and skip output energies logged per evaluation (the `v1` page shows
how they were measured on checkpoints) must not drift monotonically across
the run, and the gate-selected checkpoint must not be a transient, meaning
the selection score at the steps either side of it is within a factor of
two of it rather than 40x worse as in `v1`.

Any approach that meets the bar is a rung; the ladder table on
`docs/datasets/nytimes.md` names it. If none does, the page records which
statistics each missed and the loss traces, and the "next rung" decision
is a human one.

### Tests shared by all three approaches

Each test names the mutation it catches. A test that catches nothing is
not written.

| test | catches |
|---|---|
| `forward` on `(n, D)` returns shape `(n,)` | wrong squeeze / reduction axis |
| permuting the rows of a batch permutes the scores identically (`atol` 1e-5) | topk / gather index bugs that pair a row with the wrong neighbours |
| replacing one row changes at least one *other* row's score by more than 1e-6 | features computed but not concatenated, or the class silently degenerating to per-vector |
| a batch of two exact copies plus filler yields finite scores, and `neighbourhood_distances` reads the copies' `r_1` as exactly the floor | lost clamp (would be `-inf` / `nan`), floor applied before the diagonal mask |
| self is excluded: on a batch where every other row is farther than 0.5, the smallest value `neighbourhood_distances` returns is above 0.5 | missing diagonal mask (would read 0, then floor) |
| `gradient_penalty(critic, real, fake)` returns a finite scalar and `d loss / d critic params` is finite (double backward) | a non-differentiable op in the profile path |
| `build_critic` defaults to `Critic`, dispatches each type, raises on an unknown string | factory not wired; typo in a config silently training the old critic |
| `test_critic.py` independence test still passes on `Critic`; the mirror test on the new class asserts dependence | the change leaking into the per-vector class |
| one training step end-to-end with `critic_type: neighbourhood` at `test_train_smoke.py` sizes writes a checkpoint whose `critic_state_dict` loads into the factory-built class | trainer construction site not using the factory; state-dict key mismatch on resume |
| `drop_zero_rows: true` on an array with two zero rows returns two fewer rows and `dropped_zero_rows == 2`; `false` keeps them | flag ignored; count off by one |
| `gate_statistics` on a sample with 3 of 100 rows within the floor of another logs `near_duplicate_fraction == 0.03` | diagnostic reading the wrong column or counting the survivor-masked set |

Mutation checks after implementing, recorded in the PR: remove the
diagonal mask and confirm the self-exclusion test fails; remove the clamp
and confirm the copies test fails; drop the concatenation and confirm the
dependence test fails.

## Approach 1: within-batch profile features (`v2`)

### `NeighbourhoodCritic`

For each row i of the batch, the profile is computed among the other rows
of the same batch:

    r_1 <= ... <= r_k        k nearest within-batch distances, floored
    phi_i = [ log(r_1 / r_k), ..., log(r_{k-1} / r_k), log r_k ]

`k - 1` shape entries (the log-ratio profile the Hill estimator reduces to
LID, and that `lid_reg` already matches) plus one scale entry, `critic_k`
features in all. The critic is the existing MLP on the concatenation
`[x_i, phi_i]`, input width `D + critic_k`. Real rows are profiled among
real batch-mates, fake among fake, so the within-batch bias (distances far
larger than true k-NN distances in 250k rows) is the same on both sides and
cancels, the argument `batch_log_ratio_profile` records.

Cost: one `n x n` distance matrix per critic call, 512 x 512 x 256, and
`topk` at k=20. Negligible against the MLP.

No batch-level features (mean profile, batch LID). The probe shows the
per-row profile separates the sheet nearly perfectly on its own, and the
MLP is nonlinear where the probe's regression was not; batch pooling is
approach 3's territory.

`critic_k` must be at most `batch_size - 1`; the constructor cannot know
the batch size, so `forward` raises if `n <= critic_k` rather than
silently truncating. The training smoke sizes satisfy this.

### The rung

`configs/nytimes/v2.yaml` and the box instrument `v2_seed42.yaml`: copies
of `v1.yaml` / `v1_seed42.yaml` plus exactly two stated changes, listed in
the config comment the way the `v1_seed42` comment lists its own:

1. `model.critic_type: neighbourhood` (with `critic_k: 20`,
   `critic_distance_floor: 0.01`, both written out so the file says what
   ran).
2. `data.preprocess.drop_zero_rows: true`.

Everything else identical: linear-skip generator, 30k steps, `n_critic 3`,
`lambda_gp 5`, `select_on: gate`, `amp: false`, output dirs
`runs/nytimes/v2` / `runs/nytimes/v2_seed42`. Job script
`scripts/nytimes_v2_seed42_job.sh` follows the `v1` one. Reported the way
`v1` is: selected and final checkpoints sampled and measured against the
cleaned corpus at canonical conditions, trunk/skip energies per checkpoint,
the loss traces, and `near_duplicate_fraction` real and fake.

Budget: the profile adds under 5% to a `v1` step, so about 40 minutes on
the RTX 3060.

### What approach 1 does not do

- No change to the generator, the selector, the gradient penalty code, or
  any other family's configs.
- No batch-level pooling and no learned neighbour features.
- No attempt to reproduce the 14% copy rate: at batch 512 the critic sees a
  copy for 0.15% of real rows, so `near_duplicate_fraction` is expected to
  stay near zero on the fake side; that is a measurement, not a target.

## Approach 2: bank neighbourhoods (`v2b`)

Same features as approach 1, but the neighbours of a row come from a large
detached bank instead of its batch-mates, so the profile is measured at a
scale closer to the gate's (100-NN in 250k) and copies become visible at a
rate a critic can learn from.

### `BankNeighbourhoodCritic`

Two banks on the device, both detached, both of size `critic_bank_size`
(default 16,384, i.e. 16k x 256 floats, 16 MB each):

- **Real bank**: a fixed random subset of the training split (not the
  holdout), drawn once with the run seed and stored in the checkpoint by
  index so a resume rebuilds the same bank.
- **Fake bank**: a ring buffer of the most recent generator outputs, written
  (detached, after `normalize_l2`) at every generator step from the fake
  batch the generator step already produces. At 512 per step it holds the
  last 32 generator steps. Until it has been filled once, fake rows are
  profiled within-batch (approach 1's rule), and `run_metadata.json`
  records the step at which the bank first filled.

For a real row, the k nearest rows of the real bank, excluding itself if it
is in the bank (by index, not by distance, so a copy is kept). For a fake
row, the k nearest rows of the fake bank. Gradients flow through the query
row only; the bank is data. `phi_i` and the MLP are exactly approach 1's.

Gradient penalty: interpolated rows query the concatenation of both banks.
An interpolate is neither population and the penalty only needs the critic
to be smooth in the row, so the union is the neutral choice; the docstring
records it so the parallel agent does not need to re-decide.

Cost: 512 x 16,384 x 256 per critic call, about 4 GFLOP, plus topk over
16k columns. Small next to the MLP at batch 512; measure it in the smoke
run and record it.

### What it measures that approach 1 cannot

At bank size 16k the real bank shows a copy for roughly 3% of real rows
(measured 3.4% at 20k), so `near_duplicate_fraction` on the fake side is a
live target here rather than a diagnostic, and the run report must quote
both sides. Staleness: the fake bank lags the generator by up to 32 steps,
which is well inside the timescale of the balance drift (thousands of
steps), so it is not expected to matter; the report says whether the
Wasserstein estimate shows a sawtooth at the ring period.

### Extra tests

| test | catches |
|---|---|
| a real row that is in the bank does not see itself: with the bank equal to the batch, the smallest distance read is above the floor | self-match by index not excluded |
| before the fake bank has filled, fake scores equal approach 1's scores on the same batch (same weights) | fallback path wired to the wrong feature |
| after `critic_bank_size / batch_size` writes, the oldest entry is overwritten (write a marker row, confirm it is gone) | ring pointer off by one |
| checkpoint round-trip restores the real-bank indices | bank redrawn on resume |

### The rung

`configs/nytimes/v2b.yaml` / `v2b_seed42.yaml`: `v2` plus
`critic_type: neighbourhood_bank`, `critic_bank_size: 16384`, output dirs
`runs/nytimes/v2b*`. Everything else as `v2`, so `v2b` against `v2` is one
change, the neighbour source.

## Approach 3: learned set critic (`v2c`)

Instead of a hand-chosen profile, the critic sees the raw within-batch
neighbour geometry and learns its own features. One EdgeConv layer
(Wang et al., DGCNN) over each row's k within-batch neighbours.

### `SetNeighbourhoodCritic`

For row i with within-batch neighbours j_1 .. j_k (topk as in approach 1,
self excluded, floor applied to the distances used for selection only):

    e_ij = MLP_edge([ x_i, x_j - x_i ])         (2D -> H, one hidden layer, LeakyReLU)
    a_i  = max_j e_ij                            (H)
    s_i  = MLP_out([ x_i, a_i ])                 (D + H -> 1, today's critic_hidden_dims)

`x_j - x_i` is the local difference; its spread over j is the local tangent
structure, and its singular spectrum is exactly what a sheet loses. Max
aggregation is permutation-invariant over neighbours and is what DGCNN
uses; mean is the fallback if max trains badly (config key
`critic_edge_pool: max | mean`). `H` is `critic_edge_dim`, default 128.

The gradient penalty flows through both the query row and the neighbour
differences, so the summed-score formulation in "Shared" is doing real work
here: it is the only thing bounding how fast the critic's score can change
when a neighbour moves.

Cost: `n x k x 2D x H` for the edge MLP, 512 x 20 x 512 x 128, about
0.7 GFLOP per call. Fine.

### Extra tests

| test | catches |
|---|---|
| permuting the *neighbour order* (shuffle columns of the topk result) leaves scores unchanged | pooling not permutation-invariant (e.g. a flatten crept in) |
| a translated batch (`x + c` for all rows) changes only the `x_i` path: with `MLP_out`'s `x_i` weights zeroed, scores are invariant to `c` | edge features using absolute `x_j` instead of the difference |
| `critic_edge_pool: mean` and `max` both build and differ on a random batch | pool key ignored |

### The rung

`configs/nytimes/v2c.yaml` / `v2c_seed42.yaml`: `v2` plus
`critic_type: neighbourhood_set`, `critic_edge_dim: 128`,
`critic_edge_pool: max`, output dirs `runs/nytimes/v2c*`. One change
against `v2`, the feature extractor.

## Running the three in parallel

Each rung has its own config, output dir, job script and result page
under `docs/results/nytimes-v2{,b,c}-seed42/`. They share code, so the
order of merging matters: the "Shared" section (factory, floor,
`drop_zero_rows`, the diagnostic, the shared tests) is one change that
lands first on `nytimes-eda`; each approach is then a branch off that with
its own critic class, tests and config. An agent taking `v2b` or `v2c`
starts after the shared change is merged, reads this spec and its own
section, and does not edit `NeighbourhoodCritic`.

The box has one GPU and the queue serialises jobs; three 40-minute runs
are two hours of GPU. Submit through `gpuq` as `v1` was.

## What needs a human

- Whether the zero-row drop becomes part of the family's locked measurement
  conditions in `gates/nytimes.yaml` is still the open decision on the
  family page. This spec drops zero rows from *training* only; the gate
  bands stay unset.
- If two approaches meet the bar, which one is the rung the ladder
  continues from.
