# Top-K Per Row on ROCm (MI300A)

This repo contains a ROCm/HIP implementation of a **top-k-per-row** kernel for large logits matrices, with support for prefix tokens (prefill). It is designed for MI300A (gfx942), but can be adapted to other CDNA GPUs.

## Build

Requirements:

- ROCm with `hipcc`
- A ROCm-enabled PyTorch / ATen installation (if you use the ATen version)
- GPU architecture set via `GPU_ARCH` in the Makefile (default: `gfx942` for MI300A)

Compile both baseline and optimized versions:

```bash
make
````

This produces:

* `test_topk_rocm_base`   – baseline implementation
* `test_topk_rocm_air`    – optimized implementation (AIR top-k)

## Run

Run the baseline:

```bash
make run_base
# or
./test_topk_rocm_base
```

Run the optimized version:

```bash
make run_air
# or
./test_topk_rocm_air
```

Profiling with rocprof (example):

```bash
make profile_base
make profile_air
```

## Example Results

All tests use:

* `num_rows = 8192`
* `top_k    = 2048`
* `stride1  = 1`
* `stride0  = context_len`
* Time is the average kernel time over multiple iterations (in microseconds).

| # | num_prefix | context_len | top_k | Time (us) | CPU check                 |
| - | ---------- | ----------: | ----: | --------: | ------------------------- |
| 1 | 8,000      |      16,192 |  2048 |    626.77 | All checked results match |
| 2 | 24,000     |      32,192 |  2048 |  1,457.96 | All checked results match |
| 3 | 8,000      |      16,192 |  2048 |    636.02 | Skipping CPU check        |
| 4 | 16,000     |      24,192 |  2048 |  1,039.99 | Skipping CPU check        |
| 5 | 24,000     |      32,192 |  2048 |  1,462.47 | Skipping CPU check        |
| 6 | 32,000     |      40,192 |  2048 |  1,947.59 | Skipping CPU check        |
| 7 | 40,000     |      48,192 |  2048 |  2,494.06 | Skipping CPU check        |
| 8 | 48,000     |      56,192 |  2048 |  2,963.71 | Skipping CPU check        |
| 9 | 56,000     |      64,192 |  2048 |  3,435.84 | Skipping CPU check        |

Example index output (first row, first 10 indices) confirms correctness for the checked cases, e.g.:

```text
indices[0, 0:10]: 68 69 71 74 75 82 84 89 92 103
indices[0, 0:10]: 196 208 211 216 249 4 5 27 39 40
```

