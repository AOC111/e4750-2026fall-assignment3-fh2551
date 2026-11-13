"""
EECS E4750 Fall 2026 - Assignment 3 - PyCUDA template

##############################################################################
#                        READ THIS BEFORE YOU START                          #
##############################################################################
# 1. ONLY write code inside the blocks marked:                               #
#        # ============ STUDENT CODE STARTS HERE ============                #
#        # ============= STUDENT CODE ENDS HERE =============                #
#    and inside the kernel bodies in CUDA_CODE.                              #
#    DO NOT touch anything else: the kernel signatures, the function         #
#    signatures/return values, __init__(), check_inputs(), cpu(), and the    #
#    entire TEST HARNESS section are used for grading and must remain        #
#    exactly as provided.                                                    #
# 2. Make all times returned in SECONDS and NOT milliseconds.                #
# 3. Run modes:                                                             #
#        python3 pycuda_assignment3.py -m test                               #
#        python3 pycuda_assignment3.py -m bench                              #
#        python3 pycuda_assignment3.py -m shapes                             #
#        python3 pycuda_assignment3.py -m tile                               #
#        python3 pycuda_assignment3.py -m profile-ncu                        #
#    All five must run error-free, and every correctness test must print     #
#    PASS. The bench, shapes and tile modes save the plots for your report.  #
##############################################################################

You must complete:
    - bmm_naive     body in CUDA_CODE
    - bmm_optimized body in CUDA_CODE
    - cudaModule.naive()
    - cudaModule.optimized()

------------------------------------------------------------------------------
THE PROBLEM
------------------------------------------------------------------------------
Matrix multiplication is the basic operator of a neural network: a fully
connected layer is a matmul, a convolution is usually lowered to one, and an
attention layer is two of them. Training and inference rarely run one matmul
at a time. They run many independent matmuls of the same shape together (one
per sample, one per attention head, one per micro-batch) as a single BATCHED
matmul, because one small matmul does not give the GPU enough independent work
to fill all of its SMs.

    Input   A : shape (batch, M, K), float32
    Input   B : shape (batch, K, N), float32
    Output  C : shape (batch, M, N), float32

    C[b] = A[b] @ B[b]                         for every b in [0, batch)
    C[b, m, n] = sum over k of A[b, m, k] * B[b, k, n]

    FLOPs = 2 * batch * M * N * K

Worked example (batch = 2, M = 2, K = 3, N = 2):

    A[0] = [[1, 2, 3],      B[0] = [[1, 0],      C[0] = [[ 4,  5],
            [4, 5, 6]]              [0, 1],              [10, 11]]
                                    [1, 1]]

    A[1] = [[1, 0, 0],      B[1] = [[1, 2],      C[1] = [[1, 2],
            [0, 1, 0]]              [3, 4],              [3, 4]]
                                    [5, 6]]

Each batch entry is an independent matmul. Nothing is shared across b.

You implement two kernels and profile them with NSight Compute:

    1. bmm_naive      - global memory only, no shared memory.
    2. bmm_optimized  - coalesced global memory accesses and shared-memory
                        tiling with tile size TILE.

------------------------------------------------------------------------------
LAUNCH CONFIGURATION - THIS IS FIXED, AND YOU MUST FOLLOW IT
------------------------------------------------------------------------------
Every array is row-major (C order, C-contiguous), as it is in C and in a
default numpy array. The kernels receive flat pointers to the whole batched
tensors.

    bmm_naive       block = (16, 16, 1)
        grid x spans the M rows of C
        grid y spans the N columns of C
        grid z spans the batch

    bmm_optimized   block = (TILE, TILE, 1)
        grid x spans the N columns of C
        grid y spans the M rows of C
        grid z spans the batch

The two kernels deliberately use x differently. Theory question A2 asks you
to explain what that does to the global memory accesses.

batch, M, K and N may be ANY positive integers within the limits below,
including 1 and including values that are not multiples of 16 or of TILE.

------------------------------------------------------------------------------
TILE IS A COMPILE-TIME CONSTANT
------------------------------------------------------------------------------
TILE is not a kernel argument. CUDA_CODE is compiled once per supported tile
size with -DTILE=<value> (see __init__). bmm_optimized must take its tile size
from TILE. optimized() must launch self.tiled_kernels[tile], the bmm_optimized
compiled for the requested tile.

------------------------------------------------------------------------------
SIZE LIMITS
------------------------------------------------------------------------------
check_inputs() enforces batch <= 512, M, N, K <= 1024, and at most 2^27
elements (512 MiB) in each of A, B and C. Theory question C3 asks why limits
of this kind exist.

------------------------------------------------------------------------------
A NOTE ON THE CORRECTNESS TOLERANCE
------------------------------------------------------------------------------
The tests compare against a float64 reference with
np.allclose(C, ref, rtol=1e-5, atol=1e-6 * K). Your kernel accumulates K
products in float32, and each addition rounds; the accumulated rounding error
grows with K, so the absolute tolerance scales with K. Inputs are drawn from
[0, 1), so there is no cancellation near zero. A kernel that drops, repeats or
misplaces a single product fails at this tolerance.
"""
"""
Date <11/11/2026>
Written by <Fanghao Hu> <fh2551@columbia.edu>
The code has not benn generated by AI tools, or copied from an external resources.
"""
import time
import argparse
import subprocess
import numpy as np

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt


# Tile sizes the optimized kernel is compiled for, and the default one.
TILES = (8, 16, 32)
DEFAULT_TILE = 16

# Block edge of the naive kernel. Fixed; do not change.
NAIVE_BLOCK = 16

# Size limits enforced by check_inputs(): at most 512 matrices, no matrix
# dimension above 1024, and no tensor above 2^27 elements (512 MiB).
MAX_BATCH = 512
MAX_DIM = 1024
MAX_ELEMS = 2 ** 27


# ===================== CUDA KERNELS ====================
# Do NOT change the kernel names or their parameters - only fill in the bodies.
CUDA_CODE = """
// TILE is supplied at compile time with -DTILE=<value>. This default only
// applies if the source is compiled without that option.
#ifndef TILE
#define TILE 16
#endif

// ===================== bmm_naive =====================
// C[b, m, n] = sum_k A[b, m, k] * B[b, k, n]
//
// Requirements:
//   - global memory only: no shared memory;
//   - launch configuration as in the header: block (16, 16, 1), grid x
//     spans the rows of C, y the columns, z the batch;
//   - correct for every shape within the size limits;
//   - no out-of-bounds memory accesses.
__global__ void bmm_naive(
    float* C, const float* A, const float* B,
    const unsigned int batch, const unsigned int M,
    const unsigned int K, const unsigned int N
) {
    int x = blockIdx.x * 16 + threadIdx.x;  // row
    int y = blockIdx.y * 16 + threadIdx.y;  // col
    int z = blockIdx.z + threadIdx.z;   // batch
    float Cvalue = 0;
    
    if (x<M && y<N && z<batch){
        for(int d=0; d<K; d++){
            Cvalue += A[z*M*K + x*K + d] * B[z*K*N + d*N + y];
        }
        C[z*M*N + x*N + y] = Cvalue;
    }
}


// ===================== bmm_optimized =====================
// C[b, m, n] = sum_k A[b, m, k] * B[b, k, n]
//
// Requirements:
//   - shared-memory tiling with tile size TILE;
//   - coalesced global memory accesses;
//   - launch configuration as in the header: block (TILE, TILE, 1), grid x
//     spans the columns of C, y the rows, z the batch;
//   - correct for every shape within the size limits, including dimensions
//     that are not multiples of TILE;
//   - no out-of-bounds memory accesses.
__global__ void bmm_optimized(
    float* C, const float* A, const float* B,
    const unsigned int batch, const unsigned int M,
    const unsigned int K, const unsigned int N
) {
    __shared__ float ds_A[TILE][TILE];
    __shared__ float ds_B[TILE][TILE];
    int bx = blockIdx.x;  int by = blockIdx.y; int bz = blockIdx.z;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int Row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;
    float Cvalue = 0;
    
    for (int t=0; t<(K+TILE-1) / TILE; t++){
    // loop over tiles
        if (Row<M && t*TILE + tx < K)
            ds_A[ty][tx] = A[bz*M*K + Row*K + t*TILE + tx];
        else
            ds_A[ty][tx] = 0.0f;
        if (Col<N && t*TILE + ty < K)
            ds_B[ty][tx] = B[bz*K*N + (t*TILE+ty)*N + Col];
        else
            ds_B[ty][tx] = 0.0f;
        __syncthreads();
        for (int i=0; i<TILE; i++){
        // loop over TILE
            Cvalue += ds_A[ty][i] * ds_B[i][tx];
        }
        __syncthreads();
    } // End for (int t = 0...
    if (Row<M && Col<N && bz<batch)
        C[bz*M*N + Row*N + Col] = Cvalue;
}
"""


class cudaModule:
    def __init__(self):
        """
        **Do not modify this code**
        CUDA_CODE is compiled once for the naive kernel, and once per tile size
        with -DTILE=<value>, so that TILE is a compile-time constant inside
        bmm_optimized.

        The kernel handles are retrieved once here. self.tiled_kernels[tile]
        is bmm_optimized from the module compiled for that tile.
        """
        self.naive_mod = SourceModule(CUDA_CODE)
        self.tiled_mods = {
            t: SourceModule(CUDA_CODE, options=["-DTILE=%d" % t])
            for t in TILES
        }
        self.naive_kernel = self.naive_mod.get_function("bmm_naive")
        self.tiled_kernels = {
            t: mod.get_function("bmm_optimized")
            for t, mod in self.tiled_mods.items()
        }
        self.naive_block = (NAIVE_BLOCK, NAIVE_BLOCK, 1)

    def check_inputs(self, A, B):
        """
        **Do not modify this code**
        Validates the inputs and returns (batch, M, K, N).
        """
        assert A.dtype == np.float32 and B.dtype == np.float32, \
            "A and B must be float32"
        assert A.ndim == 3 and B.ndim == 3, "A and B must be 3D"
        assert A.flags['C_CONTIGUOUS'] and B.flags['C_CONTIGUOUS'], \
            "A and B must be C-contiguous"
        batch, M, K = A.shape
        assert B.shape[0] == batch and B.shape[1] == K, \
            "shape mismatch: A is (batch, M, K), B must be (batch, K, N)"
        N = B.shape[2]
        assert 1 <= batch <= MAX_BATCH, "batch must be in [1, %d]" % MAX_BATCH
        assert all(1 <= d <= MAX_DIM for d in (M, K, N)), \
            "M, K and N must be in [1, %d]" % MAX_DIM
        assert max(batch * M * K, batch * K * N, batch * M * N) <= MAX_ELEMS, \
            "each of A, B and C must have at most %d elements" % MAX_ELEMS
        return batch, M, K, N

    # ------------------------------------------------------------------
    # Naive kernel
    # ------------------------------------------------------------------
    def naive(self, A, B):
        """
        Compute C[b] = A[b] @ B[b] for every b with bmm_naive.

            alloc + H2D:  A, B
            alloc:        C
            run:          bmm_naive  ->  C on device
            D2H:          C  ->  host

        Requirements (these fix the choices; follow them exactly):
          - Device memory: cuda.mem_alloc() only, no gpuarray. Transfers:
            cuda.memcpy_htod() / cuda.memcpy_dtoh() only.
          - Launch self.naive_kernel exactly once, with block
            self.naive_block and the launch configuration in the file
            header, and no other kernel.
          - No GPU library calls (no cuBLAS, skcuda, CuPy, torch) and no
            host-side matmul: C must be computed by bmm_naive.

        Time the operation twice, both with CUDA events:
          - kernel_time: the kernel duration ONLY.
          - total_time : the whole sequence above, end to end, including
            every allocation and both transfers.

        Arguments:
            A           :   3D numpy array of np.float32, shape (batch, M, K),
                            C-contiguous
            B           :   3D numpy array of np.float32, shape (batch, K, N),
                            C-contiguous
        Returns:
            C           :   3D numpy array of np.float32, shape (batch, M, N)
            kernel_time :   kernel-only execution time, in seconds
            total_time  :   total end-to-end execution time, in seconds
        """
        batch, M, K, N = self.check_inputs(A, B)
        batch_u, M_u, K_u, N_u = (np.uint32(batch), np.uint32(M),
                                  np.uint32(K), np.uint32(N))

        # ============ STUDENT CODE STARTS HERE ============
        [total_start, total_end, kernel_start, kernel_end] = [cuda.Event() for _ in range(4)]

        grid_size = ((M-1)//self.naive_block[0] + 1, (N-1)//self.naive_block[1] + 1, batch)

        total_start.record()

        C = np.empty((batch, M, N), dtype=np.float32)
        C_gpu = cuda.mem_alloc(C.nbytes)
        A_gpu = cuda.mem_alloc(A.nbytes)
        B_gpu = cuda.mem_alloc(B.nbytes)
        cuda.memcpy_htod(A_gpu, A)
        cuda.memcpy_htod(B_gpu, B)

        kernel_start.record()
        self.naive_kernel(C_gpu, A_gpu, B_gpu, batch_u, M_u, K_u, N_u, block=self.naive_block, grid=grid_size)
        kernel_end.record()

        cuda.memcpy_dtoh(C, C_gpu)

        total_end.record()
        total_end.synchronize()

        total_time = total_end.time_since(total_start) * 1e-3
        kernel_time = kernel_end.time_since(kernel_start) * 1e-3
        # ============= STUDENT CODE ENDS HERE =============

        return C, kernel_time, total_time

    # ------------------------------------------------------------------
    # Optimized kernel: coalesced + shared-memory tiled
    # ------------------------------------------------------------------
    def optimized(self, A, B, tile=DEFAULT_TILE):
        """
        Compute C[b] = A[b] @ B[b] for every b with bmm_optimized, compiled
        for the given tile size.

            alloc + H2D:  A, B
            alloc:        C
            run:          bmm_optimized  ->  C on device
            D2H:          C  ->  host

        Requirements (these fix the choices; follow them exactly):
          - Device memory: cuda.mem_alloc() only, no gpuarray. Transfers:
            cuda.memcpy_htod() / cuda.memcpy_dtoh() only.
          - Launch self.tiled_kernels[tile] exactly once, with block
            (tile, tile, 1) and the launch configuration in the file
            header, and no other kernel.
          - No GPU library calls (no cuBLAS, skcuda, CuPy, torch) and no
            host-side matmul: C must be computed by bmm_optimized.

        Time the operation twice, both with CUDA events:
          - kernel_time: the kernel duration ONLY.
          - total_time : the whole sequence above, end to end, including
            every allocation and both transfers.

        Arguments:
            A           :   3D numpy array of np.float32, shape (batch, M, K),
                            C-contiguous
            B           :   3D numpy array of np.float32, shape (batch, K, N),
                            C-contiguous
            tile        :   one of TILES (8, 16, 32)
        Returns:
            C           :   3D numpy array of np.float32, shape (batch, M, N)
            kernel_time :   kernel-only execution time, in seconds
            total_time  :   total end-to-end execution time, in seconds
        """
        batch, M, K, N = self.check_inputs(A, B)
        if tile not in self.tiled_kernels:
            raise ValueError("tile must be one of %s" % (TILES,))
        batch_u, M_u, K_u, N_u = (np.uint32(batch), np.uint32(M),
                                  np.uint32(K), np.uint32(N))

        # ============ STUDENT CODE STARTS HERE ============
        [total_start, total_end, kernel_start, kernel_end] = [cuda.Event() for _ in range(4)]

        block_size = (tile, tile, 1)
        grid_size = ((N - 1) // block_size[0] + 1, (M - 1) // block_size[1] + 1, batch)

        total_start.record()

        C = np.empty((batch, M, N), dtype=np.float32)
        C_gpu = cuda.mem_alloc(C.nbytes)
        A_gpu = cuda.mem_alloc(A.nbytes)
        B_gpu = cuda.mem_alloc(B.nbytes)
        cuda.memcpy_htod(A_gpu, A)
        cuda.memcpy_htod(B_gpu, B)

        kernel_start.record()
        self.tiled_kernels[tile](C_gpu, A_gpu, B_gpu, batch_u, M_u, K_u, N_u, block=block_size, grid=grid_size)
        kernel_end.record()

        cuda.memcpy_dtoh(C, C_gpu)

        total_end.record()
        total_end.synchronize()

        total_time = total_end.time_since(total_start) * 1e-3
        kernel_time = kernel_end.time_since(kernel_start) * 1e-3
        # ============= STUDENT CODE ENDS HERE =============

        return C, kernel_time, total_time

    # ------------------------------------------------------------------
    # CPU baseline
    # ------------------------------------------------------------------
    def cpu(self, A, B):
        """
        **Do not modify this code**
        The numpy baseline used in the timing plots. np.matmul on 3D arrays
        is a batched matmul in float32, backed by a multithreaded BLAS.

        Arguments:
            A       :   3D numpy array of np.float32, shape (batch, M, K)
            B       :   3D numpy array of np.float32, shape (batch, K, N)
        Returns:
            C       :   3D numpy array of np.float32, shape (batch, M, N)
            time    :   execution time in seconds
        """
        start = time.time()
        C = np.matmul(A, B)
        end = time.time()
        return C, end - start


###############################################################################
#                       TEST HARNESS - DO NOT MODIFY                          #
###############################################################################

# Correctness tolerance against the float64 reference. The absolute part
# scales with K because float32 accumulation error grows with the number of
# products summed. See the note in the module docstring.
RTOL = 1e-5
ATOL_PER_K = 1e-6

# Every timed point is the MEDIAN of REPS runs, after one discarded warm-up.
REPS = 5

# The fixed shape used by the NSight Compute mode: batch = 32, M = N = K = 256.
PROFILE_BATCH = 32
PROFILE_N = 256

# -m bench: square matrices at a fixed batch.
BENCH_BATCH = 16
BENCH_SIZES = [32, 64, 128, 256, 512, 1024]

# -m shapes, left panel: four configurations that all perform exactly 2^31
# FLOPs (2 * batch * n^3 = 2^31).
EQUAL_FLOPS_SHAPES = [(512, 128), (64, 256), (8, 512), (1, 1024)]

# -m shapes, right panel: the batch sweep at a fixed matrix size.
BATCH_SWEEP_N = 256
BATCH_SWEEP = [1, 2, 4, 8, 16, 32, 64]

# -m tile: the tile-size sweep, at the profiling batch.
TILE_SWEEP_BATCH = PROFILE_BATCH
TILE_SWEEP_SIZES = [128, 256, 512, 1024]


# Plot colours: one per implementation; dashed lines and hatched bars are
# end-to-end totals, solid ones are kernel-only.
NAIVE_C, OPT_C, CPU_C = 'tab:blue', 'tab:orange', 'tab:green'


def set_log_ticks(ax, values):
    """Label a log x axis at exactly the swept values."""
    ax.set_xticks(values)
    ax.set_xticklabels([str(v) for v in values])
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())


def gflops(batch, M, K, N, seconds):
    """Achieved GFLOP/s for one batched matmul that took `seconds`."""
    return 2.0 * batch * M * N * K / seconds / 1e9


def make_inputs(rng, batch, M, K, N):
    """Random inputs in [0, 1)."""
    A = rng.random((batch, M, K), dtype=np.float32)
    B = rng.random((batch, K, N), dtype=np.float32)
    return A, B


def reference(A, B):
    """float64 reference result, used only for the correctness check."""
    return np.matmul(A.astype(np.float64), B.astype(np.float64))


def is_correct(C, ref, K):
    return (C.shape == ref.shape and C.dtype == np.float32
            and np.allclose(C, ref, rtol=RTOL, atol=ATOL_PER_K * K))


def time_median(fn):
    """
    Call fn() once as a warm-up and discard it, then REPS more times.
    fn returns (result, kernel_time, total_time). Returns the medians.
    """
    fn()
    ks, ts = [], []
    for _ in range(REPS):
        _, k, t = fn()
        ks.append(k)
        ts.append(t)
    return float(np.median(ks)), float(np.median(ts))


def time_cpu_median(module, A, B):
    module.cpu(A, B)
    return float(np.median([module.cpu(A, B)[1] for _ in range(REPS)]))


def report_gpu_clocks(tag):
    """
    Print the GPU name, current SM / memory clocks, temperature and power.
    GPU clocks drop under sustained load as the card reaches its power and
    thermal limits, so record these lines next to your timings.
    """
    query = ("name,clocks.sm,clocks.max.sm,clocks.mem,"
             "temperature.gpu,power.draw")
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + query, "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        print("[nvidia-smi %s] %s" % (tag, out.stdout.strip()))
    except Exception as exc:
        print("[nvidia-smi %s] unavailable (%s)" % (tag, exc))


def make_test_cases():
    """
    Build the correctness test suite: (name, A, B) tuples.

    The shapes cover the degenerate case, dimensions that are not multiples
    of 8, 16 or 32, M != K != N (which catches swapped dimensions and wrong
    batch offsets), K = 1, a single row, the largest batch, and the largest
    matrix.
    """
    rng = np.random.default_rng(4750)
    tests = []

    # The worked example straight out of the assignment text.
    A = np.array([[[1, 2, 3], [4, 5, 6]],
                  [[1, 0, 0], [0, 1, 0]]], dtype=np.float32)
    B = np.array([[[1, 0], [0, 1], [1, 1]],
                  [[1, 2], [3, 4], [5, 6]]], dtype=np.float32)
    tests.append(("worked_example", A, B))

    for batch, M, K, N in [
        (1,    1,    1,    1),
        (1,   17,   33,    9),
        (3,   31,   64,   47),
        (4,  100,  129,   77),
        (2,  257,  255,  256),
        (128,  16,   16,   16),
        (5,    1,  300,    7),
        (2,   64,    1,   64),
        (1, 1024, 1024, 1024),
    ]:
        name = "b%d_m%d_k%d_n%d" % (batch, M, K, N)
        tests.append((name, *make_inputs(rng, batch, M, K, N)))

    return tests


def run_tests(module):
    """Run the correctness suite against the naive kernel and every tile."""
    print("=" * 100)
    print("Part 1: Correctness tests (PyCUDA)")
    print("        tolerance: np.allclose(rtol=%g, atol=%g * K) against a "
          "float64 reference" % (RTOL, ATOL_PER_K))
    print("=" * 100)

    all_passed = True
    for name, A, B in make_test_cases():
        batch, M, K = A.shape
        N = B.shape[2]
        ref = reference(A, B)

        results = [("naive", module.naive(A, B)[0])]
        for t in TILES:
            results.append(("tile%d" % t, module.optimized(A, B, tile=t)[0]))

        oks = [(label, is_correct(C, ref, K)) for label, C in results]
        ok = all(o for _, o in oks)
        all_passed = all_passed and ok

        detail = "   ".join("%s: %s" % (label, "PASS" if o else "FAIL")
                            for label, o in oks)
        print("[%s] %-22s %s" % ("PASS" if ok else "FAIL", name, detail))

    print("-" * 100)
    if all_passed:
        print("All correctness tests PASSED.")
    else:
        print("Some correctness tests FAILED - fix your implementation before "
              "looking at any timing or profiler number.")
    print()
    return all_passed


def run_bench(module):
    """
    Square matrices n x n x n at a fixed batch, n swept from 32 to 1024.
    Saves a3_bench_cuda.png: runtime on the left, achieved GFLOP/s on the
    right.
    """
    print("=" * 100)
    print("Part 2: Size sweep (PyCUDA), batch = %d, M = N = K = n, "
          "tile = %d" % (BENCH_BATCH, DEFAULT_TILE))
    print("=" * 100)
    report_gpu_clocks("before")

    rng = np.random.default_rng(2026)
    nv_k, nv_t, op_k, op_t, cpu_t = [], [], [], [], []
    for n in BENCH_SIZES:
        A, B = make_inputs(rng, BENCH_BATCH, n, n, n)
        k, t = time_median(lambda: module.naive(A, B))
        nv_k.append(k); nv_t.append(t)
        k, t = time_median(lambda: module.optimized(A, B, DEFAULT_TILE))
        op_k.append(k); op_t.append(t)
        cpu_t.append(time_cpu_median(module, A, B))

        print("n=%5d | naive %.3e/%.3es (%7.1f GFLOP/s) | optimized "
              "%.3e/%.3es (%7.1f GFLOP/s) | cpu %.3e s   (kernel/total)"
              % (n, nv_k[-1], nv_t[-1],
                 gflops(BENCH_BATCH, n, n, n, nv_k[-1]),
                 op_k[-1], op_t[-1],
                 gflops(BENCH_BATCH, n, n, n, op_k[-1]), cpu_t[-1]))

    report_gpu_clocks("after")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.set_title('Batched matmul runtime, batch=%d (PyCUDA)' % BENCH_BATCH)
    ax1.loglog(BENCH_SIZES, nv_k, 'o-', color=NAIVE_C, label='naive (kernel only)')
    ax1.loglog(BENCH_SIZES, nv_t, 'o--', color=NAIVE_C, label='naive (total)')
    ax1.loglog(BENCH_SIZES, op_k, 's-', color=OPT_C,
               label='optimized, tile %d (kernel only)' % DEFAULT_TILE)
    ax1.loglog(BENCH_SIZES, op_t, 's--', color=OPT_C,
               label='optimized, tile %d (total)' % DEFAULT_TILE)
    ax1.loglog(BENCH_SIZES, cpu_t, '^-', color=CPU_C, label='CPU (numpy)')
    ax1.set_xlabel('n  (M = N = K = n)')
    ax1.set_ylabel('Median runtime over %d runs (seconds)' % REPS)
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, which='both', alpha=0.3)

    ax2.set_title('Achieved throughput, batch=%d (PyCUDA)' % BENCH_BATCH)
    ax2.semilogx(BENCH_SIZES,
                 [gflops(BENCH_BATCH, n, n, n, s)
                  for n, s in zip(BENCH_SIZES, nv_k)],
                 'o-', color=NAIVE_C, label='naive (kernel only)')
    ax2.semilogx(BENCH_SIZES,
                 [gflops(BENCH_BATCH, n, n, n, s)
                  for n, s in zip(BENCH_SIZES, op_k)],
                 's-', color=OPT_C, label='optimized, tile %d (kernel only)' % DEFAULT_TILE)
    ax2.semilogx(BENCH_SIZES,
                 [gflops(BENCH_BATCH, n, n, n, s)
                  for n, s in zip(BENCH_SIZES, op_t)],
                 's--', color=OPT_C, label='optimized, tile %d (total)' % DEFAULT_TILE)
    ax2.semilogx(BENCH_SIZES,
                 [gflops(BENCH_BATCH, n, n, n, s)
                  for n, s in zip(BENCH_SIZES, cpu_t)],
                 '^-', color=CPU_C, label='CPU (numpy)')
    ax2.set_xlabel('n  (M = N = K = n)')
    set_log_ticks(ax1, BENCH_SIZES)
    set_log_ticks(ax2, BENCH_SIZES)
    ax2.set_ylabel('GFLOP/s')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig('a3_bench_cuda.png', dpi=300)
    print("\nPlot saved as 'a3_bench_cuda.png' - include it in your report.")
    print()


def run_shapes(module):
    """
    Left panel: four shapes with exactly the same FLOP count (2^31), from
    many small matrices to one large one. Right panel: a batch sweep at a
    fixed matrix size, from a single matmul up to batch = 64. Theory
    question B1 is about both. Saves a3_shapes_cuda.png.
    """
    print("=" * 100)
    print("Part 3: Shape sweeps (PyCUDA), tile = %d" % DEFAULT_TILE)
    print("=" * 100)
    report_gpu_clocks("before")

    rng = np.random.default_rng(2026)

    print("Equal-FLOPs configurations (2 * batch * n^3 = 2^31 FLOPs):")
    eq = []
    for batch, n in EQUAL_FLOPS_SHAPES:
        A, B = make_inputs(rng, batch, n, n, n)
        nk, nt = time_median(lambda: module.naive(A, B))
        ok, ot = time_median(lambda: module.optimized(A, B, DEFAULT_TILE))
        eq.append((nk, nt, ok, ot))
        print("batch=%4d n=%5d | naive %.3e/%.3es (%7.1f GFLOP/s) | "
              "optimized %.3e/%.3es (%7.1f GFLOP/s)   (kernel/total)"
              % (batch, n, nk, nt, gflops(batch, n, n, n, nk),
                 ok, ot, gflops(batch, n, n, n, ok)))

    print("\nBatch sweep at n = %d:" % BATCH_SWEEP_N)
    n = BATCH_SWEEP_N
    sw_nk, sw_ok = [], []
    for batch in BATCH_SWEEP:
        A, B = make_inputs(rng, batch, n, n, n)
        nk, _ = time_median(lambda: module.naive(A, B))
        ok, _ = time_median(lambda: module.optimized(A, B, DEFAULT_TILE))
        sw_nk.append(nk); sw_ok.append(ok)
        print("batch=%4d n=%5d | naive kernel %.3e s (%7.1f GFLOP/s) | "
              "optimized kernel %.3e s (%7.1f GFLOP/s)"
              % (batch, n, nk, gflops(batch, n, n, n, nk),
                 ok, gflops(batch, n, n, n, ok)))

    report_gpu_clocks("after")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    labels = ['batch=%d\nn=%d' % s for s in EQUAL_FLOPS_SHAPES]
    x = np.arange(len(EQUAL_FLOPS_SHAPES))
    w = 0.2
    ax1.bar(x - 1.5 * w, [e[0] for e in eq], w, color=NAIVE_C, label='naive (kernel only)')
    ax1.bar(x - 0.5 * w, [e[1] for e in eq], w, color=NAIVE_C,
            alpha=0.45, hatch='//', label='naive (total)')
    ax1.bar(x + 0.5 * w, [e[2] for e in eq], w, color=OPT_C,
            label='optimized, tile %d (kernel only)' % DEFAULT_TILE)
    ax1.bar(x + 1.5 * w, [e[3] for e in eq], w, color=OPT_C,
            alpha=0.45, hatch='//',
            label='optimized, tile %d (total)' % DEFAULT_TILE)
    ax1.set_yscale('log')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title('Equal FLOPs (2^31), different shapes (PyCUDA)')
    ax1.set_ylabel('Median runtime over %d runs (seconds)' % REPS)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, which='both', axis='y', alpha=0.3)

    ax2.semilogx(BATCH_SWEEP,
                 [gflops(b, n, n, n, s) for b, s in zip(BATCH_SWEEP, sw_nk)],
                 'o-', color=NAIVE_C, label='naive (kernel only)')
    ax2.semilogx(BATCH_SWEEP,
                 [gflops(b, n, n, n, s) for b, s in zip(BATCH_SWEEP, sw_ok)],
                 's-', color=OPT_C, label='optimized, tile %d (kernel only)' % DEFAULT_TILE)
    set_log_ticks(ax2, BATCH_SWEEP)
    ax2.set_title('Batch sweep at n=%d (PyCUDA)' % n)
    ax2.set_xlabel('batch')
    ax2.set_ylabel('GFLOP/s')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig('a3_shapes_cuda.png', dpi=300)
    print("\nPlot saved as 'a3_shapes_cuda.png' - include it in your report.")
    print()


def run_tile(module):
    """
    Tile-size sweep: the optimized kernel at every tile size, plus the naive
    kernel, at batch = 32 and square n from 128 to 1024. The n = 256 row is
    the NSight Compute profiling shape. Theory question A3 is about this
    sweep. Saves a3_tile_cuda.png.
    """
    print("=" * 100)
    print("Part 4: Tile-size sweep (PyCUDA), batch = %d" % TILE_SWEEP_BATCH)
    print("=" * 100)
    report_gpu_clocks("before")

    rng = np.random.default_rng(2026)
    batch = TILE_SWEEP_BATCH
    naive_k = []
    tile_k = {t: [] for t in TILES}
    for n in TILE_SWEEP_SIZES:
        A, B = make_inputs(rng, batch, n, n, n)
        k, _ = time_median(lambda: module.naive(A, B))
        naive_k.append(k)
        line = "n=%5d | naive %.3e s (%7.1f GFLOP/s)" % (
            n, k, gflops(batch, n, n, n, k))
        for t in TILES:
            k, _ = time_median(lambda: module.optimized(A, B, t))
            tile_k[t].append(k)
            line += " | tile %2d %.3e s (%7.1f GFLOP/s)" % (
                t, k, gflops(batch, n, n, n, k))
        mark = "   <-- profiling shape" if n == PROFILE_N else ""
        print(line + "   (kernel only)" + mark)

    report_gpu_clocks("after")

    plt.figure(figsize=(8, 6))
    plt.title('Tile-size sweep, batch=%d (PyCUDA)' % batch)
    plt.semilogx(TILE_SWEEP_SIZES,
                 [gflops(batch, n, n, n, s)
                  for n, s in zip(TILE_SWEEP_SIZES, naive_k)],
                 'o-', color=NAIVE_C, label='naive')
    tile_colors = {8: 'tab:red', 16: OPT_C, 32: 'tab:purple'}
    for t, marker in zip(TILES, ['s-', 'd-', '^-']):
        plt.semilogx(TILE_SWEEP_SIZES,
                     [gflops(batch, n, n, n, s)
                      for n, s in zip(TILE_SWEEP_SIZES, tile_k[t])],
                     marker, color=tile_colors[t],
                     label='optimized, tile %d' % t)
    set_log_ticks(plt.gca(), TILE_SWEEP_SIZES)
    plt.xlabel('n  (M = N = K = n)')
    plt.ylabel('GFLOP/s (kernel only, median over %d runs)' % REPS)
    plt.legend(loc='upper left', fontsize=8)
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('a3_tile_cuda.png', dpi=300)
    print("\nPlot saved as 'a3_tile_cuda.png' - include it in your report.")
    print()


def run_profile_ncu(module):
    """
    NSight Compute mode. Runs each kernel configuration EXACTLY ONCE at the
    fixed profiling shape, in this order, with no warm-up:

        ID 0  bmm_naive                  block (16, 16, 1)
        ID 1  bmm_optimized, tile  8     block ( 8,  8, 1)
        ID 2  bmm_optimized, tile 16     block (16, 16, 1)
        ID 3  bmm_optimized, tile 32     block (32, 32, 1)

    The three bmm_optimized entries share a kernel name; tell them apart by
    their ID or by the Block Size column in the NSight Compute GUI.

    Profile this with:

        ncu --set full -o a3_ncu python3 pycuda_assignment3.py -m profile-ncu

    then open a3_ncu.ncu-rep in the NSight Compute GUI. See the README for the
    exact metrics you must pull and report.
    """
    batch, n = PROFILE_BATCH, PROFILE_N
    print("=" * 100)
    print("NSight Compute profiling run: batch=%d, M=N=K=%d" % (batch, n))
    print("        FLOPs = %s   bytes per tensor = %s"
          % (format(2 * batch * n ** 3, ','), format(batch * n * n * 4, ',')))
    print("=" * 100)

    rng = np.random.default_rng(2026)
    A, B = make_inputs(rng, batch, n, n, n)

    _, k, t = module.naive(A, B)
    print("ID 0  bmm_naive               : kernel %.6e s   total %.6e s"
          % (k, t))
    for i, tile in enumerate(TILES, start=1):
        _, k, t = module.optimized(A, B, tile)
        print("ID %d  bmm_optimized, tile %2d : kernel %.6e s   total %.6e s"
              % (i, tile, k, t))
    print()
    print("These timings are taken while NSight Compute is attached. Use the")
    print("n = %d row of -m tile, run WITHOUT ncu, as your event timings." % n)
    print()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description="EECS E4750 Fall 2026 Assignment 3 - PyCUDA: batched "
                    "matrix multiplication with shared-memory tiling")
    arg_parser.add_argument(
        '-m', '--mode', required=True,
        choices=['test', 'bench', 'shapes', 'tile', 'profile-ncu'],
        help="test: correctness suite. bench: size sweep at batch=16. "
             "shapes: equal-FLOPs shapes and batch sweep. "
             "tile: tile-size sweep. "
             "profile-ncu: fixed shape for NSight Compute.")
    args = arg_parser.parse_args()

    module = cudaModule()

    if args.mode == 'test':
        run_tests(module)
    elif args.mode == 'bench':
        run_bench(module)
    elif args.mode == 'shapes':
        run_shapes(module)
    elif args.mode == 'tile':
        run_tile(module)
    elif args.mode == 'profile-ncu':
        run_profile_ncu(module)
