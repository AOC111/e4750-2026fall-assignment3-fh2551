# EECS E4750: Heterogeneous Computing for Signal and Data Processing (Fall 2026)

## Assignment-3: Shared-Memory Tiling in PyCUDA & PyOpenCL — Batched Matrix Multiplication

Due date: See in the courseworks.

Total points: 100

### <span style="color:red"><strong>TODO:</strong></span> (Re)naming of the student repository (for submitting the assignments)

***INSTRUCTIONS***
* You have to write the Code with thorough and clear comments, and write a report in PDF format which includes your plots, your profiler screenshots, and your answers to the theory questions.
* Students need to change and use the following name for the repository with their solutions: `e4750-2026fall-assignment3-UNI`
  * Good Example: `e4750-2026fall-assignment3-zz9999`
  * Bad example: `e4750-2026fall-assignments-e4750-2026fall-assign3-zz9999`
* This change can be done from the "Settings" tab which is located on the repository page.


### Primer

The goal of this assignment is to implement shared-memory tiling, and to use NVIDIA NSight Compute to explain how much it helps and why.

Matrix multiplication is the basic operator of a neural network. A fully connected layer is a matmul, a convolution is usually lowered to one, and an attention layer is two of them. Training and inference rarely run one matmul at a time. They run many independent matmuls of the same shape together — one per sample, one per attention head, one per micro-batch — as a single **batched** matmul. A single small matmul does not give the GPU enough independent work to fill all of its SMs; a batch of them does.

You will implement batched matrix multiplication twice in each API: once naively, reading every operand straight from global memory, and once with coalesced accesses and shared-memory tiling. You will benchmark both across matrix sizes, batch sizes and tile sizes, and profile the PyCUDA versions with NSight Compute.

The assignment should be completed in Google Cloud Hosted VMs only.

Please share any questions you come across on Ed Discussion for everyone to see.

### Relevant Documentation

1. Course slides on tiled matrix multiplication: [2.6.HC.TiledMatrixMultiplicationKernel](https://drive.google.com/drive/folders/1i9Ri_3Ow1ZVeNDeJG9QNF2YkhSIr7AeL)
2. [CUDA C++ Programming Guide — Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
3. [CUDA C++ Programming Guide — Compute Capabilities](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities) — per-SM limits (resident threads, blocks, shared memory, registers) for your GPU's compute capability.
4. [NSight Compute User Guide](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html) and [NSight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
5. [TechPowerUp GPU database](https://www.techpowerup.com/gpu-specs/) — look up the GPU you ran on; you need its SM count, FP32 cores, clocks, memory bandwidth and L2 size for the theory questions.

NSight Compute recitation session from the previous offering: [Slides](https://docs.google.com/presentation/d/1IIfkEgLrWg2hHBw2EWk1lFV9xAZiTqeyL4AcixYQMHM/edit?slide=id.g38407867325_0_11#slide=id.g38407867325_0_11)

Install the NSight Compute GUI on your own computer.

### **Submission Instructions on deliverables**

1. Complete the code templates provided in this folder for both PyOpenCL and PyCUDA. Keep the file names `pyopencl_assignment3.py` and `pycuda_assignment3.py` respectively.
    - All nine commands listed under **Running the code** must run without errors.
    - Every correctness test must print `PASS` for the naive kernel and for all three tile sizes, in both files.
2. Create a report in PDF format which includes:
    - A screenshot of the correctness-test output (all `PASS`) for both files.
    - All six plots: `a3_bench_cuda.png`, `a3_shapes_cuda.png`, `a3_tile_cuda.png`, `a3_bench_opencl.png`, `a3_shapes_opencl.png`, `a3_tile_opencl.png`.
    - The `[nvidia-smi ...]` clock lines printed by each sweep, next to the corresponding plot.
    - NSight Compute screenshots covering the metrics listed in the profiling section, including the roofline chart for `bmm_naive` and for `bmm_optimized` at tile 16.
    - A thorough analysis/comparison of the results shown in the plots.
    - Your answers to all eight theory questions (A1–A3, B1–B2, C1–C3).
    - Report File Name: `Assignment3_Report_{First Name}_{Last Name}_{uni}.pdf`
3. Commit the six `.png` plots and your `.ncu-rep` profiler report to your repository.


## Programming Problem (80 points)

### Problem set up

| Symbol | Shape | dtype | Meaning |
|---|---|---|---|
| `A` | `(batch, M, K)` | `float32` | `batch` left-hand matrices, each `M × K` |
| `B` | `(batch, K, N)` | `float32` | `batch` right-hand matrices, each `K × N` |
| `C` | `(batch, M, N)` | `float32` | the output, `C[b] = A[b] @ B[b]` |

$$C[b, m, n] = \sum_{k=0}^{K-1} A[b, m, k] \cdot B[b, k, n] \qquad \text{FLOPs} = 2 \cdot \text{batch} \cdot M \cdot N \cdot K$$

Worked example (`batch = 2`, `M = 2`, `K = 3`, `N = 2`):

```
A[0] = [[1, 2, 3],      B[0] = [[1, 0],      C[0] = [[ 4,  5],
        [4, 5, 6]]              [0, 1],              [10, 11]]
                                [1, 1]]

A[1] = [[1, 0, 0],      B[1] = [[1, 2],      C[1] = [[1, 2],
        [0, 1, 0]]              [3, 4],              [3, 4]]
                                [5, 6]]
```

Each batch entry is an independent matmul; nothing is shared across `b`.

Every array is row-major (C order, C-contiguous), as in C and in a default numpy array. The kernels receive flat pointers to the whole batched tensors.

The `cpu()` function in the templates is the numpy baseline used in the timing plots (`np.matmul`, float32, multithreaded BLAS). The correctness tests compare against a separate float64 reference.

### The two kernels (required in both PyCUDA and PyOpenCL)

1. **`bmm_naive`** — global memory only, no shared memory.
2. **`bmm_optimized`** — coalesced global memory accesses and shared-memory (local-memory) tiling with tile size `TILE`. It must be correct for every shape within the size limits, including dimensions that are not multiples of `TILE`.

No other optimizations are permitted in `bmm_optimized`: no register tiling, no vector types (`float4`), no `#pragma unroll`, no double buffering, no shared-memory padding, no streams. Theory question C2 asks you to reason about these.

### Launch configuration

| Kernel | Block / local size | grid x / NDRange dim 0 spans | grid y / NDRange dim 1 spans | grid z / NDRange dim 2 spans |
|---|---|---|---|---|
| `bmm_naive` | `16 × 16 × 1` | the `M` rows of `C` | the `N` columns of `C` | the batch |
| `bmm_optimized` | `TILE × TILE × 1` | the `N` columns of `C` | the `M` rows of `C` | the batch |

The two kernels deliberately use dimension 0 differently. Theory question A2 asks what that does to the global memory accesses.

`batch`, `M`, `K` and `N` may be any positive integers within the size limits below, including `1` and including values that are not multiples of 16 or of `TILE`.

### `TILE` is a compile-time constant

`TILE` is not a kernel argument. The provided `__init__()` compiles the kernel source once per tile size, `TILE ∈ {8, 16, 32}`, with `-DTILE=<value>`. `bmm_optimized` must take its tile size from `TILE`. The kernel handles for every tile size are retrieved once in `__init__()`: `self.tiled_kernels[tile]` is `bmm_optimized` compiled for that tile.

### Size limits

`check_inputs()` (provided) enforces `batch ≤ 512`, `M, N, K ≤ 1024`, and at most 2²⁷ elements (512 MiB) in each of `A`, `B` and `C`. Theory question C3 asks why limits of this kind exist.

### The two host functions (required in both PyCUDA and PyOpenCL)

**`naive(A, B)`** and **`optimized(A, B, tile=16)`** both implement the same sequence and return `(C, kernel_time, total_time)`:

```
alloc + H2D:  A, B
alloc:        C
run:          bmm_naive / bmm_optimized  ->  C on device
D2H:          C  ->  host
```

### Implementation requirements

These fix the choices that would otherwise differ between submissions. Follow them exactly.

* **Device memory and transfers.**
  * PyCUDA: allocate every device array with `cuda.mem_alloc()` and move data with `cuda.memcpy_htod()` / `cuda.memcpy_dtoh()`. Do not use `gpuarray`.
  * PyOpenCL: allocate every device array as a `cl.Buffer`, and copy the result back with `cl.enqueue_copy()`. Host-to-device data may be supplied either with the `COPY_HOST_PTR` flag at buffer creation or with `cl.enqueue_copy()`. Do not use `pyopencl.array`.
* **No libraries on the GPU path.** No cuBLAS, skcuda, CuPy, torch or CLBlast, and no host-side matmul inside `naive()` / `optimized()`. `C` must be computed by your kernel.
* **Precision.** FP32 inputs, FP32 accumulation, FP32 output. No `double`, no `half`.
* **Launch configuration.** Use exactly the block / local sizes and grid layout in the table above. `naive()` launches `self.naive_kernel`; `optimized()` launches `self.tiled_kernels[tile]`.
* **Kernel launches.** Each host function launches exactly one kernel, once. `-m profile-ncu` relies on there being exactly four kernel launches in total.
* **Memory safety.** No out-of-bounds memory accesses, for any shape within the size limits.

### Timing requirements

* **`kernel_time`** — the kernel duration only. No copies, no allocations.
* **`total_time`** — the entire sequence end to end: every allocation, both transfers and the kernel.

All times must be returned in **seconds**, NOT milliseconds. In PyCUDA use CUDA events for both measurements. In PyOpenCL use OpenCL event profiling for `kernel_time` and `time.time()` for `total_time`.

The harness times every point as the **median** of 5 runs after one discarded warm-up run.

### Correctness tolerance

The test harness compares against a float64 reference with `np.allclose(C, ref, rtol=1e-5, atol=1e-6 * K)`.

Your kernel adds `K` float32 products one at a time, and every addition rounds. The accumulated rounding error grows with `K`, so the absolute tolerance scales with `K`. Inputs are drawn from `[0, 1)`, so there is no cancellation near zero. A kernel that drops, repeats or misplaces even a single product fails at this tolerance. Do not disable FMA contraction.

### Running the code

PyCUDA, five modes:

```bash
python3 pycuda_assignment3.py -m test          # correctness suite (naive + tiles 8/16/32)
python3 pycuda_assignment3.py -m bench         # n sweep at batch=16            -> a3_bench_cuda.png
python3 pycuda_assignment3.py -m shapes        # equal-FLOPs shapes + batch sweep -> a3_shapes_cuda.png
python3 pycuda_assignment3.py -m tile          # tile-size sweep at batch=32    -> a3_tile_cuda.png
python3 pycuda_assignment3.py -m profile-ncu   # batch=32, n=256: four launches for NSight Compute
```

PyOpenCL, four modes:

```bash
python3 pyopencl_assignment3.py -m test        # correctness suite (naive + tiles 8/16/32)
python3 pyopencl_assignment3.py -m bench       # n sweep at batch=16            -> a3_bench_opencl.png
python3 pyopencl_assignment3.py -m shapes      # equal-FLOPs shapes + batch sweep -> a3_shapes_opencl.png
python3 pyopencl_assignment3.py -m tile        # tile-size sweep at batch=32    -> a3_tile_opencl.png
```

All nine commands must run error-free. Profiling is required for the PyCUDA half only, matching the convention from previous assignments in this course.

The three sweeps (all square, `M = N = K = n`):

* **`-m bench`** — `batch = 16`, `n` from 32 to 1024, tile 16. Runtime (left panel) and achieved GFLOP/s (right panel) for both kernels in kernel-only and total form, plus the CPU baseline.
* **`-m shapes`** — left panel: four configurations that all perform exactly 2³¹ FLOPs, `(batch, n) = (512, 128), (64, 256), (8, 512), (1, 1024)`. Right panel: a batch sweep at `n = 256`, `batch` from 1 to 64. Theory question B1 is about both.
* **`-m tile`** — `batch = 32`, `n` from 128 to 1024, the naive kernel and the optimized kernel at all three tile sizes. The `n = 256` row is the NSight Compute profiling shape. Theory question A3 is about this sweep.

Each sweep prints `[nvidia-smi before]` / `[nvidia-smi after]` lines with the SM clock, temperature and power. GPU clocks drop under sustained load as the card reaches its power and thermal limits; include these lines in your report next to the plot they belong to.

### Task-1: PyCUDA (40 points)

| Item | Points |
|---|---|
| `bmm_naive`: correct, global memory only, required launch configuration | 8 |
| `bmm_optimized`: correct for every shape, coalesced, shared-memory tiled with `TILE`, required launch configuration | 14 |
| `naive()` with the required allocations, transfers, grid and dual timing | 6 |
| `optimized()` with the kernel for the requested tile, grid derived from the tile, and dual timing | 6 |
| All correctness tests PASS, for the naive kernel and all three tile sizes | 6 |

### Task-2: PyOpenCL (40 points)

Identical split to Task 1, for `pyopencl_assignment3.py`.


## Profiling (PyCUDA only)

```bash
ncu --set full -o a3_ncu python3 pycuda_assignment3.py -m profile-ncu
```

Open `a3_ncu.ncu-rep` in the NSight Compute GUI.

`-m profile-ncu` runs every kernel configuration exactly once at `batch = 32`, `M = N = K = 256`, with no warm-up, in this order:

| ID | Kernel | Block size |
|---|---|---|
| 0 | `bmm_naive` | `(16, 16, 1)` |
| 1 | `bmm_optimized`, tile 8 | `(8, 8, 1)` |
| 2 | `bmm_optimized`, tile 16 | `(16, 16, 1)` |
| 3 | `bmm_optimized`, tile 32 | `(32, 32, 1)` |

The three `bmm_optimized` entries share a kernel name; tell them apart by ID or by the *Block Size* column. Use the GUI's baseline feature (right-click a result → *Set Baseline*) to compare `bmm_naive` against tile 16 side by side.

Collect the following:

| Metric | Collect for | Location in the NSight Compute GUI | CLI metric name |
|---|---|---|---|
| Kernel duration | all four | Speed Of Light → *Duration* | `gpu__time_duration.sum` |
| SM frequency | all four | Speed Of Light → *SM Frequency* | `gpc__cycles_elapsed.avg.per_second` |
| Global load instructions | all four | Memory Workload Analysis → Memory Tables → L1/TEX Cache, *Global Load*, Instructions | `smsp__inst_executed_op_global_ld.sum` |
| Global load sectors per request | naive, tile 16 | Memory Workload Analysis → L1/TEX Cache, *Global Load*, Sectors/Req | `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` ÷ `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` |
| L1/TEX hit rate | naive, tile 16 | Memory Workload Analysis → *L1/TEX Hit Rate* | `l1tex__t_sector_hit_rate.pct` |
| L2 hit rate | naive, tile 16 | Memory Workload Analysis → *L2 Hit Rate* | `lts__t_sector_hit_rate.pct` |
| DRAM bytes read / written | naive, tile 16 | Memory Workload Analysis → *Device Memory* row | `dram__bytes_read.sum`, `dram__bytes_write.sum` |
| Shared load instructions and bank conflicts | tiles 8, 16, 32 | Memory Workload Analysis → Shared Memory table | `smsp__inst_executed_op_shared_ld.sum`, `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` |
| Registers per thread, theoretical and achieved occupancy | all four | Launch Statistics; Occupancy | `launch__registers_per_thread`, `sm__maximum_warps_per_active_cycle_pct`, `sm__warps_active.avg.pct_of_peak_sustained_active` |
| Top three warp stall reasons | all four | Warp State Statistics | `smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio` (e.g. `long_scoreboard`, `barrier`, `mio_throttle`, `short_scoreboard`) |
| Roofline chart | naive, tile 16 | Speed Of Light → *Roofline* | screenshot |

CLI metric names change between NSight Compute versions. If a metric name is rejected on your VM, use `ncu --query-metrics | grep <keyword>` to find the current name, and fall back on the GUI section named in the table above, since the section names are stable. State which version of `ncu` you used (`ncu --version`) in your report.

`--set full` replays each kernel many times to collect all sections. With exactly four launches this takes a minute or two. Do not run `ncu --set full` on `-m bench`, `-m shapes` or `-m tile`: they launch hundreds of kernels, and each one would be replayed.


## Theory Problems (20 points)

Answer these using your own profiled numbers, plots and printed timings. Answers that do not reference your own measurements where a question asks for them will not receive credit.

The profiling shape throughout is `batch = 32`, `M = N = K = 256`. Wherever a question needs hardware figures, use those of the GPU you ran on, from the documentation linked above: SM count, FP32 cores, base and boost clock, memory bandwidth, L2 size, and per SM the maximum resident threads, resident blocks and shared memory.

### Part A — NSight Compute (9 points)

**A1. Global loads: predicted versus measured, and why tiling pays off less than it saves. (4 pts)**

* Before opening the profiler, derive the number of **warp-level global load instructions** issued by `bmm_naive` and by `bmm_optimized` in terms of `batch`, `M`, `N`, `K` and `TILE`, assuming every dimension is a multiple of 16 and of `TILE` (they are at the profiling shape). Evaluate for the naive kernel and for tiles 8, 16 and 32 at the profiling shape.
* Compare against `smsp__inst_executed_op_global_ld.sum` for all four launches. State predicted, measured and the naive : tiled ratio for each tile.
* Report the measured kernel-duration speedup of tile 16 over naive, compare it with the ratio of their global load counts, and explain the difference. Support your explanation with at least three of: L1/TEX hit rate, L2 hit rate, DRAM bytes, shared load instructions, and warp stall reasons, for both kernels.

**A2. Coalescing at warp granularity. (2 pts)**

* For `bmm_naive` under its required launch configuration, state which `(m, n)` elements of `C` the 32 threads of one warp work on. For each of its global loads and its store, give the byte stride between consecutive threads' addresses and the number of 32-byte sectors the warp-level request touches at the profiling shape. Do the same for every global load and store of your `bmm_optimized` at tile 16.
* Report the global load sectors per request for `bmm_naive` and for tile 16, and reconcile them with your derivation.

**A3. Tile-size sweep. (3 pts)**

* Before measuring, compute for tiles 8, 16 and 32: threads per block, shared memory per block, the maximum number of resident blocks per SM on your GPU and which limit sets it, the resulting theoretical occupancy (check registers per thread from Launch Statistics), the number of blocks and waves at the profiling shape, and global loads per FMA. Predict which tile is fastest.
* From the `n = 256` row of `-m tile` and from NSight Compute, report kernel time, achieved occupancy and the top warp stall reasons for each tile. State whether your prediction held, and explain the result. In particular, explain why the largest tile, which issues the fewest global loads, is not automatically the fastest.

### Part B — Measured performance (6 points)

**B1. Shapes and batching. (3 pts)**

* For each of the four equal-FLOPs configurations in `-m shapes`, compute the compulsory bytes (each of `A`, `B` read once and `C` written once), the arithmetic intensity relative to those bytes, and the number of `bmm_optimized` blocks and waves at tile 16. Report the kernel-only and total times from your plot. All four perform the same arithmetic: explain why their total times differ, and whether their kernel times differ, and why.
* From the batch sweep at `n = 256`: for `batch = 1`, compute the number of blocks at tile 16, the number of waves, and how full the last wave is. Using your measured GFLOP/s, explain how throughput changes with `batch` and where it saturates. Explain why one launch of a batched kernel is preferable to a host-side loop of `batch` launches of a single-matrix kernel.

**B2. Roofline and clocks. (3 pts)**

* Compute your GPU's FP32 peak and its ridge point (FLOP per DRAM byte) at the boost clock. Using the FLOP count of the profiling shape and the measured DRAM bytes, compute the arithmetic intensity of `bmm_naive` and of tile 16, place both on the roofline (include the NSight Compute roofline screenshots), and report each kernel's achieved FP32 throughput as a percentage of peak. State which roof bounds each kernel according to the DRAM roofline, and explain why both nonetheless sit far below it: what on-chip resource is actually limiting each one?
* Compare NSight Compute's *Duration* for each of the four launches with the event-timed kernel time for the same configuration from the `n = 256` row of `-m tile` (run without `ncu`). Report the ratio, report the *SM Frequency* NSight Compute shows, and explain the discrepancy. State which measurement you should use for "percent of peak" and at which clock.

### Part C — General (5 points)

**C1. Barriers. (1 pt)**

For every `__syncthreads()` / `barrier()` in your `bmm_optimized`, state precisely which hazard it prevents. Then explain what goes wrong if threads whose output element lies outside `C` return at the top of the kernel, before any shared-memory work.

**C2. Beyond the required optimizations. (2 pts)**

Choose **two** of the following and, for each, explain the mechanism, the bottleneck it targets (refer to your A1–A3 measurements), its expected effect on your GPU, and its cost or constraint. You do not implement them.

* Register tiling (each thread computes a `TM × TN` block of `C`)
* `float4` vectorized global loads
* Loop unrolling of the inner product loop
* Double buffering of the shared tiles
* Shared-memory padding (`[TILE][TILE + 1]`)
* CUDA streams with pinned host memory, chunking the batch

**C3. Scaling limits: 32-bit indexing. (2 pts)**

The harness caps the problem size. Suppose instead that the kernels computed their flat indices in 32-bit `int`.

* At how many elements does a flat index overflow? Give one `(batch, M, K, N)` for which `A` alone overflows, and show that this problem still fits in your GPU's memory, so the overflow happens before the GPU runs out of memory.
* Which index expression in your kernels overflows first, and what change to the index arithmetic prevents it?
* Name one other limit in the launch configuration that would stop this kernel from scaling to a very large `batch`, and how you would work around it.


## Code Templates

You **must** adhere to the templates given in the starter code — this is essential for all assignments to be graded fairly and equally. You are free to add helper functions and to modify comments, but the class structure, `__init__()`, `check_inputs()`, the kernel signatures, the function signatures/return values, `cpu()`, and the entire `TEST HARNESS` section must remain as provided.

Write your code **only** inside the blocks marked `STUDENT CODE STARTS HERE` / `STUDENT CODE ENDS HERE`, and inside the kernel bodies in `CUDA_CODE` / `KERNEL_CODE`. The comment above each kernel and the docstring of each function state its requirements; the sections above state the required sequence, launch configuration and implementation requirements.

#### PyCuda Starter Code
[pycuda_assignment3.py](./pycuda_assignment3.py)

#### PyOpenCL Starter Code
[pyopencl_assignment3.py](./pyopencl_assignment3.py)
