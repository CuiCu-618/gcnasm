**HIP GEMM microbench (ROCm).**
Benchmarks three FP32 GEMM kernels on AMD GPUs: baseline tiled, tiled+XCD swizzle, and a register-blocked tiled kernel (also swizzled).

**Build & Run**

```bash
# Build (example hipcc flags)
make

# Run: ./gemm M N K reps check
./matrix_multiplication 2048 2048 4096 100 1
```

**Result (example)**
The program prints an aligned table per variant:

```
./matrix_multiplication
> Device: AMD Instinct MI300A | Arch: gfx942:sramecc+:xnack- | CUs: 228
GEMM: [2048x4096] * [4096x2048], tiles: [16x16], block: 128x128, reps: 100, check: on

Variant              |     Avg ms |    TFLOP/s |  GB/s(min) |   Reps |    Blk | Check
---------------------+------------+------------+------------+--------+--------+------
tiled                |    11.0538 |      3.108 |      7.068 |    100 |  32x32 |   YES
+ swizzle            |     4.3594 |      7.882 |     17.921 |    100 |  32x32 |   YES
+ swzl + regs (nt)   |     1.9055 |     18.031 |     40.999 |    100 |  32x32 |   YES

Validation passed.
```

