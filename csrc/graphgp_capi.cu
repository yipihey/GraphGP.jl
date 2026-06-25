// graphgp_capi.cu — a thin C ABI around the vendored graphgp_cuda per-point device kernels, so the
// hand-written CUDA path is callable from GraphGP.jl (via ccall on CuArray device pointers) as an
// optional accelerator alongside the portable KernelAbstractions kernels.
//
// Only the per-point, ORDER-INDEPENDENT operations are exposed here (refine_logdet, refine_inv):
// the Vecchia logdet is a sum over refined points and the inverse recovers each xi independently,
// so we launch ONE kernel over all M = N - n0 refined points (start_idx = n0) — no depth-batch
// loop / device offsets needed. Single covariance batch (n_batches = 1).
//
// Layout (matches GraphGP.jl): `points` is point-major, D contiguous per point — i.e. GraphGP.jl's
// (D,N) column-major coords cast to Float32 and multiplied by `scale` (physical positions), so the
// Euclidean distance + cov_lookup here reproduce the integer-lattice lookup to f32. `neighbors` is
// point-major (a point's k neighbors contiguous) = GraphGP.jl's (K,M) column-major, 0-based.

#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>

#include "vendor/common.h"
#include "vendor/linalg.h"
#include "vendor/covariance.h"
#include "vendor/refine_logdet.h"
#include "vendor/refine_inv.h"
// Graph-build kernels (the hand-written shallow on-GPU build used by gp.build_graph(cuda=True)).
#include "vendor/sort.h"
#include "vendor/tree.h"
#include "vendor/query.h"
#include "vendor/depth.h"

using i_t = int64_t;   // GraphGP.jl neighbors are Int64 (0-based here)
using f_t = float;

template <size_t MAX_K, size_t N_DIM>
static inline void launch_logdet(cudaStream_t s, const f_t* pts, const i_t* nbr,
        const f_t* bins, const f_t* vals, f_t* out, size_t n0, size_t k, size_t N, size_t ncov) {
    cudaMemsetAsync(out, 0, sizeof(f_t), s);
    size_t M = N - n0;
    if (M == 0) return;
    refine_logdet_kernel<MAX_K, N_DIM, i_t, f_t><<<cld(M, 256), 256, 0, s>>>(
        pts, nbr, bins, vals, out, n0, k, N, ncov, /*n_batches=*/1, /*start_idx=*/n0, /*n_threads=*/M);
}

template <size_t MAX_K, size_t N_DIM>
static inline void launch_inv(cudaStream_t s, const f_t* pts, const i_t* nbr, const f_t* bins,
        const f_t* vals, const f_t* values, f_t* xi, size_t n0, size_t k, size_t N, size_t ncov) {
    size_t M = N - n0;
    if (M == 0) return;
    refine_inv_kernel<MAX_K, N_DIM, i_t, f_t><<<cld(M, 256), 256, 0, s>>>(
        pts, nbr, bins, vals, values, xi, n0, k, N, ncov, /*n_batches=*/1, /*start_idx=*/n0, /*n_threads=*/M);
}

// Pick the smallest compiled MAX_K >= k (matches the original DISPATCH_K_DIM), for N_DIM in {2,3,4}.
#define DISPATCH(CALL)                                                            \
    do {                                                                          \
        if (ndim == 2) {                                                          \
            if      (k <= 4)  CALL(4, 2);  else if (k <= 8)  CALL(8, 2);          \
            else if (k <= 16) CALL(16, 2); else if (k <= 32) CALL(32, 2);         \
            else if (k <= 64) CALL(64, 2); else return 2;                         \
        } else if (ndim == 3) {                                                   \
            if      (k <= 4)  CALL(4, 3);  else if (k <= 8)  CALL(8, 3);          \
            else if (k <= 16) CALL(16, 3); else if (k <= 32) CALL(32, 3);         \
            else if (k <= 64) CALL(64, 3); else return 2;                         \
        } else if (ndim == 4) {                                                   \
            if      (k <= 4)  CALL(4, 4);  else if (k <= 8)  CALL(8, 4);          \
            else if (k <= 16) CALL(16, 4); else if (k <= 32) CALL(32, 4);         \
            else if (k <= 64) CALL(64, 4); else return 2;                         \
        } else { return 3; }                                                      \
    } while (0)

extern "C" {

// refine_logdet: out[0] = sum over refined points of log(std). Returns 0 on success.
int gpcuda_refine_logdet_f32(const f_t* points, const i_t* neighbors, const f_t* cov_bins,
        const f_t* cov_vals, f_t* out, int64_t n0, int64_t k, int64_t n_points, int64_t n_cov,
        int64_t ndim, void* stream) {
    cudaStream_t s = (cudaStream_t)stream;
    size_t N = (size_t)n_points, K = (size_t)k, n0u = (size_t)n0, ncov = (size_t)n_cov;
#define CALL_LOGDET(MK, ND) launch_logdet<MK, ND>(s, points, neighbors, cov_bins, cov_vals, out, n0u, K, N, ncov)
    DISPATCH(CALL_LOGDET);
#undef CALL_LOGDET
    return (int)cudaPeekAtLastError();
}

// refine_inv: xi[0..M-1] = recovered unit-normal parameters from `values` (length N). Returns 0 ok.
int gpcuda_refine_inv_f32(const f_t* points, const i_t* neighbors, const f_t* cov_bins,
        const f_t* cov_vals, const f_t* values, f_t* xi, int64_t n0, int64_t k, int64_t n_points,
        int64_t n_cov, int64_t ndim, void* stream) {
    cudaStream_t s = (cudaStream_t)stream;
    size_t N = (size_t)n_points, K = (size_t)k, n0u = (size_t)n0, ncov = (size_t)n_cov;
#define CALL_INV(MK, ND) launch_inv<MK, ND>(s, points, neighbors, cov_bins, cov_vals, values, xi, n0u, K, N, ncov)
    DISPATCH(CALL_INV);
#undef CALL_INV
    return (int)cudaPeekAtLastError();
}

// Fast SHALLOW on-GPU graph build — the same pipeline as gp.build_graph(cuda=True):
// build_tree (special heap order) → query preceding k-NN → compute depths → order by depth.
// `points_in`/`points` are point-major (D contiguous), `neighbors` is point-major (k contiguous,
// 0-based), `depths` doubles as the split-dims scratch during the tree build. `temp` is 2N int32.
int gpcuda_build_graph_f32(const f_t* points_in, f_t* points, int32_t* indices, int32_t* neighbors,
        int32_t* depths, int32_t* temp, int64_t n0, int64_t k, int64_t n_points, int64_t ndim,
        void* stream) {
    cudaStream_t s = (cudaStream_t)stream;
    size_t N = (size_t)n_points, K = (size_t)k, n0u = (size_t)n0, D = (size_t)ndim;
    int32_t* temp_int = temp;
    f_t* temp_float = reinterpret_cast<f_t*>(temp);
    cudaMemcpyAsync(points, points_in, N * D * sizeof(f_t), cudaMemcpyDeviceToDevice, s);
    build_tree(s, points_in, points, depths, indices, temp_int, temp_float + N, D, N);
    size_t Q = N - n0u;
#define CALL_Q(MK, ND) query_preceding_neighbors_kernel<MK, ND, int32_t, f_t><<<cld(Q, 256), 256, 0, s>>>(points, depths, neighbors, n0u, K, Q)
    DISPATCH(CALL_Q);
#undef CALL_Q
    compute_depths_parallel(s, neighbors, depths, temp_int, n0u, K, N);
    order_by_depth(s, points, indices, neighbors, depths, temp_int, temp_int + N, temp_float + N,
        n0u, K, N, D);
    return (int)cudaPeekAtLastError();
}

} // extern "C"
