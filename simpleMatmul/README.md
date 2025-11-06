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
GEMM: [2048x4096] * [4096x2048], tiles: [64x64], block: 32x32, reps: 100, check: on

Variant          |     Avg ms |    TFLOP/s |  GB/s(min) |   Reps |    Blk | Check
-----------------+------------+------------+------------+--------+--------+------
tiled            |    10.4814 |      3.278 |      7.454 |    100 |  32x32 |   YES
+ swizzle        |     9.3270 |      3.684 |      8.376 |    100 |  32x32 |   YES
+ swzl + regs    |     1.9475 |     17.643 |     40.116 |    100 |  32x32 |   YES

Validation passed.
```

