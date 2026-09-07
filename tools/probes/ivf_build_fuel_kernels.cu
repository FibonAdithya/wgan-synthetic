// IVF-Flat index BUILD for c004 vector_search, written to be fuel-metered.
//
// This is the build half of an IVF-Flat index the way cuVS constructs one:
// k-means (Lloyd) over a training subset of the database, then one assignment
// pass over the whole database, CSR inverted lists, and a copy of the vectors
// into list order. It exists so that TIG's fuel instrumentation (build_ptx's
// static per-PTX-instruction cost table, injected per basic block) can be run
// over a real index build; cuVS ships SASS and cannot be metered.
//
// Determinism, since the verifier regenerates and compares: no float atomics.
// Initial centroids are a strided sample; the Lloyd update sums each list in
// a fixed order with one thread per dimension; the CSR scatter uses an integer
// atomic cursor so ids WITHIN a list are unordered, but the update sums over a
// list in cursor order -- so the sum's order is NOT reproducible across runs.
// That matters for a shipped algorithm and does not matter here: fuel is a
// count of executed instructions and does not depend on the summation order.
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

#define A_DB_TILE   8
#define A_CENT_TILE 32
// 129, not 128: an odd stride rotates the 32 lanes across shared-memory banks.
#define A_PAD       129
#define UPD_THREADS 128

// ---------------------------------------------------------------------------
// Straight-line calibration kernel: no branches, so it is exactly one basic
// block and every thread executes the whole of it. The counter should read
// threads * (static block cost) * (however many times the injector adds it).
// ---------------------------------------------------------------------------
extern "C" __global__ void fuel_calibrate(const float* __restrict__ in, float* __restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float a = in[i];
    float b = a - 1.5f;
    float c = b - 0.25f;
    out[i] = c - a;
}

// ---------------------------------------------------------------------------
// Initial centroids: training vector c * (n_train / n_cent). Deterministic.
// ---------------------------------------------------------------------------
extern "C" __global__ void ivf_pick_centroids(
    const float* __restrict__ db, int n_db, int dims, int n_cent,
    float* __restrict__ cent)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cent * dims) return;
    int c = i / dims, k = i % dims;
    long long stride = (long long)n_db / (long long)n_cent;
    cent[i] = db[(long long)c * stride * dims + k];
}

// ---------------------------------------------------------------------------
// Assign every vector to its nearest centroid. 256 threads = A_DB_TILE(8)
// vectors x A_CENT_TILE(32) centroids, so the 32 threads sharing a vector are
// one warp and the reduction is a shuffle. Work: n_db * n_cent * dims.
// ---------------------------------------------------------------------------
extern "C" __global__ void ivf_assign(
    const float* __restrict__ db, int n_db,
    const float* __restrict__ cent, int n_cent, int dims,
    unsigned int* __restrict__ assign)
{
    __shared__ float s_db[A_DB_TILE][A_PAD];
    __shared__ float s_cent[A_CENT_TILE][A_PAD];
    const int tid = threadIdx.x;
    const int dbi = tid / A_CENT_TILE;
    const int ci  = tid % A_CENT_TILE;
    const int db_base = blockIdx.x * A_DB_TILE;
    for (int i = tid; i < A_DB_TILE * dims; i += blockDim.x) {
        int r = i / dims, k = i % dims;
        int g = db_base + r;
        s_db[r][k] = (g < n_db) ? db[(long long)g * dims + k] : 0.0f;
    }
    __syncthreads();
    float best = FLT_MAX;
    int   best_c = 0;
    for (int c0 = 0; c0 < n_cent; c0 += A_CENT_TILE) {
        for (int i = tid; i < A_CENT_TILE * dims; i += blockDim.x) {
            int r = i / dims, k = i % dims;
            int g = c0 + r;
            s_cent[r][k] = (g < n_cent) ? cent[(long long)g * dims + k] : 0.0f;
        }
        __syncthreads();
        int cg = c0 + ci;
        if (cg < n_cent) {
            float acc = 0.0f;
            for (int k = 0; k < dims; ++k) {
                float d = s_db[dbi][k] - s_cent[ci][k];
                acc = fmaf(d, d, acc);
            }
            if (acc < best || (acc == best && cg < best_c)) { best = acc; best_c = cg; }
        }
        __syncthreads();
    }
    for (int off = 16; off > 0; off >>= 1) {
        float od = __shfl_down_sync(0xffffffffu, best, off);
        int   oc = __shfl_down_sync(0xffffffffu, best_c, off);
        if (od < best || (od == best && oc < best_c)) { best = od; best_c = oc; }
    }
    if (ci == 0) {
        int g = db_base + dbi;
        if (g < n_db) assign[g] = (unsigned int)best_c;
    }
}

// ---------------------------------------------------------------------------
// CSR inverted lists: count -> exclusive scan -> scatter. Integer atomics only.
// ---------------------------------------------------------------------------
extern "C" __global__ void ivf_count(
    const unsigned int* __restrict__ assign, int n_db,
    unsigned int* __restrict__ counts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_db) atomicAdd(&counts[assign[i]], 1u);
}

extern "C" __global__ void ivf_scan(
    const unsigned int* __restrict__ counts, int n_cent,
    unsigned int* __restrict__ offsets)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    unsigned int run = 0;
    for (int i = 0; i < n_cent; ++i) { offsets[i] = run; run += counts[i]; }
    offsets[n_cent] = run;
}

extern "C" __global__ void ivf_scatter(
    const unsigned int* __restrict__ assign, int n_db,
    const unsigned int* __restrict__ offsets,
    unsigned int* __restrict__ cursor,
    unsigned int* __restrict__ ids)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_db) return;
    unsigned int l = assign[i];
    unsigned int p = atomicAdd(&cursor[l], 1u);
    ids[offsets[l] + p] = (unsigned int)i;
}

// ---------------------------------------------------------------------------
// Lloyd update: one block per centroid, thread k sums dimension k over the
// list's members in list order. No float atomics. An empty list keeps its old
// centroid. Work: n_train * dims.
// ---------------------------------------------------------------------------
extern "C" __global__ void kmeans_update(
    const float* __restrict__ db, int dims,
    const unsigned int* __restrict__ offsets,
    const unsigned int* __restrict__ ids,
    int n_cent, float* __restrict__ cent)
{
    int c = blockIdx.x;
    if (c >= n_cent) return;
    unsigned int s = offsets[c], e = offsets[c + 1];
    if (e == s) return;
    float inv = 1.0f / (float)(e - s);
    for (int k = threadIdx.x; k < dims; k += blockDim.x) {
        float acc = 0.0f;
        for (unsigned int t = s; t < e; ++t)
            acc += db[(long long)ids[t] * dims + k];
        cent[(long long)c * dims + k] = acc * inv;
    }
}

// ---------------------------------------------------------------------------
// Copy the vectors into list order, which is what an IVF-Flat index stores.
// One block per row (grid-strided), threads over dimensions. Work: n_db * dims.
// ---------------------------------------------------------------------------
extern "C" __global__ void ivf_gather(
    const float* __restrict__ db, int dims,
    const unsigned int* __restrict__ ids, int n_db,
    float* __restrict__ out)
{
    for (int row = blockIdx.x; row < n_db; row += gridDim.x) {
        const float* src = db + (long long)ids[row] * dims;
        float* dst = out + (long long)row * dims;
        for (int k = threadIdx.x; k < dims; k += blockDim.x) dst[k] = src[k];
    }
}
