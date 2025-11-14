// MIT License
// (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
#include <algorithm>
#include <cassert>
#include <ck_tile/core/numeric/vector_type.hpp>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <iostream>
#include <string>
#include <vector>

#if defined(__gfx942__)
#define N_XCD 6
#elif defined(__gfx950__)
#define N_XCD 8
#else
#define N_XCD 1
#endif

using namespace ck_tile;

constexpr int kErrorExitCode = -1;

// HIP error guard.
#define HIP_CHECK(x)                                                           \
  do {                                                                         \
    hipError_t _e = (x);                                                       \
    if (_e != hipSuccess) {                                                    \
      std::cerr << "HIP error: \"" << hipGetErrorString(_e) << "\" at "        \
                << __FILE__ << ':' << __LINE__ << '\n';                        \
      std::exit(kErrorExitCode);                                               \
    }                                                                          \
  } while (0)

/* Map a linear block id onto XCD buckets for more uniform CU spread. */
__device__ __forceinline__ int remap_xcd(int linear_bid, int grid_sz) {
  const int bucket = linear_bid % N_XCD;
  const int max_buckets = (grid_sz + N_XCD - 1) / N_XCD;
  const int remainder = grid_sz % N_XCD;
  const int offset = linear_bid / N_XCD;

  const int subtract =
      (remainder == 0 || bucket < remainder) ? 0 : (bucket - remainder);
  return bucket * max_buckets + offset - subtract;
}

template <unsigned int BlockSize>
__global__ void matrix_multiplication_base(const float *A, const float *B,
                                           float *C,
                                           const unsigned int a_cols) {
  const unsigned int tx = threadIdx.x;
  const unsigned int ty = threadIdx.y;
  const unsigned int bx = blockIdx.x;
  const unsigned int by = blockIdx.y;

  // b_cols must match the number of output matrix columns.
  const unsigned int b_cols = blockDim.x * gridDim.x;

  // The number of tiles is determined by A's columns (which is equal to B's
  // rows).
  const unsigned int steps = a_cols / BlockSize;

  // thread_result is the accumulation variable.
  float thread_result = 0.0F;
  for (unsigned int step = 0; step < steps; step++) {
    // Shared memory is used to cache the tile from both input matrices.
    // The tile is a square of BlockSize*BlockSize.
    __shared__ float a_values[BlockSize][BlockSize];
    __shared__ float b_values[BlockSize][BlockSize];

    // Index of the top-left element of the tile in A.
    // "BlockSize * a_cols * by" is the number of elements to move "down".
    // "BlockSize * step" is the number of elements to move "right".
    const unsigned int a_idx = BlockSize * (a_cols * by + step);

    // Index of the top-left element of the tile in B.
    // "BlockSize * b_cols * step" is the number of elements to move "down".
    // "BlockSize * bx" is the number of elements to move "right".
    const unsigned int b_idx = BlockSize * (b_cols * step + bx);

    // Load each element in the tile to shared memory.
    a_values[ty][tx] = A[a_idx + a_cols * ty + tx];
    b_values[ty][tx] = B[b_idx + b_cols * ty + tx];

    // Synchronization is needed to make sure that all elements are loaded
    // before starting the calculation.
    __syncthreads();

    // Each thread calculates the scalar product of the tile and increments the
    // thread-individual thread_result.
    for (unsigned int i = 0; i < BlockSize; i++) {
      thread_result += a_values[ty][i] * b_values[i][tx];
    }

    // Synchronize to ensure that the calculation is finished before the next
    // tile's elements start to load.
    __syncthreads();
  }

  // Calculate the index of the top-left element of the output block.
  const unsigned block_offset = b_cols * BlockSize * by + BlockSize * bx;

  // Every thread stores the final result to global memory.
  C[block_offset + b_cols * ty + tx] = thread_result;
}

// Tiled GEMM: C[MxN] = A[MxK] * B[KxN]; row-major.
template <unsigned BlockSize>
__global__ void gemm_tiled(const float *__restrict__ A,
                           const float *__restrict__ B, float *__restrict__ C,
                           unsigned a_cols) {
  const int grid_sz = gridDim.x * gridDim.y;
  const int linear_bid = blockIdx.x + blockIdx.y * gridDim.x;
  const int remapped_bid = remap_xcd(linear_bid, grid_sz);
  const int bx = remapped_bid % gridDim.x;
  const int by = remapped_bid / gridDim.x;

  //     // filing curve remap
  //   const int block_dim = 4;
  //   const int block_sz = block_dim * block_dim;
  //   const int block_id = remapped_bid / block_sz;
  //   const int global_row = block_id / (gridDim.x / block_dim);
  //   const int global_col = block_id % (gridDim.x / block_dim);
  //   const int bx = global_col * block_dim + (remapped_bid % block_sz) %
  //   block_dim; const int by = global_row * block_dim + (remapped_bid %
  //   block_sz) / block_dim;

  __shared__ float As[BlockSize][BlockSize];
  __shared__ float Bs[BlockSize][BlockSize];

  const unsigned tx = threadIdx.x;
  const unsigned ty = threadIdx.y;

  const unsigned b_cols = blockDim.x * gridDim.x; // N
  const unsigned steps = a_cols / BlockSize;      // K tiles

  float acc = 0.0f;

  for (unsigned s = 0; s < steps; ++s) {
    const unsigned a_idx = BlockSize * (a_cols * by + s);
    const unsigned b_idx = BlockSize * (b_cols * s + bx);

    As[ty][tx] = A[a_idx + a_cols * ty + tx];
    Bs[ty][tx] = B[b_idx + b_cols * ty + tx];
    __syncthreads();

#pragma unroll
    for (unsigned k = 0; k < BlockSize; ++k)
      acc += As[ty][k] * Bs[k][tx];

    __syncthreads();
  }

  const unsigned c_cols = b_cols; // N
  const unsigned block_off = c_cols * BlockSize * by + BlockSize * bx;
  C[block_off + c_cols * ty + tx] = acc;
}

/**
 * M x N x K matrix multiplication kernel
 * C = A * B
 * M = 128, N = 128, K = 8
 */
#define TILE_M 128
#define TILE_N 128
#define TILE_K 8
#define SubM 16
__global__ void gemm_tiled_regs(const float *__restrict__ A,
                                const float *__restrict__ B, float *C,
                                unsigned wA, unsigned wB) {
  // Block index
  const int grid_size    = gridDim.x * gridDim.y;
  const int linear_bid   = blockIdx.x + blockIdx.y * gridDim.x;
  const int remapped_bid = remap_xcd(linear_bid, grid_size);
  const int bx           = remapped_bid % gridDim.x;
  const int by           = remapped_bid / gridDim.x;

  // Thread index
  int th = threadIdx.x;
  int tid = threadIdx.x % 32;
  int warpId = threadIdx.x / 32;

  // Thread index for local computation
  int tx = threadIdx.x % SubM;
  int ty = threadIdx.x / SubM;

  // Index of the first sub-matrix of A processed by the block
  int aBegin = wA * TILE_M * by;

  // Index of the last sub-matrix of A processed by the block
  int aEnd = aBegin + wA - 1;

  // Step size used to iterate through the sub-matrices of A
  int aStep = TILE_K;

  // Index of the first sub-matrix of B processed by the block
  int bBegin = TILE_N * bx;

  // Step size used to iterate through the sub-matrices of B
  int bStep = TILE_K * wB;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float regC[TILE_K][TILE_K] = {{0}};

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ float As[TILE_M][TILE_K];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ float Bs[TILE_K][TILE_N];

    // Load the matrices from device memory
    // to shared memory; each thread
    // loads one element of each matrix
    *((fp32x4_t *)&As[th / 2][(th % 2) * 4]) =
        *((fp32x4_t *)&A[a + wA * (th / 2) + (th % 2) * 4]);
    *((fp32x4_t *)&Bs[warpId][tid * 4]) =
        *((fp32x4_t *)&B[b + wB * warpId + tid * 4]);

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    float regAs[TILE_M / SubM];
    float regBs[TILE_N / SubM];

    // Load the shared memory sub-matrices into registers
    for (int i = 0; i < TILE_K; i++) {

      for (int ii = 0; ii < TILE_M / SubM; ++ii) {
        regAs[ii] = As[ty + ii * SubM][i];
        regBs[ii] = Bs[i][tx + ii * SubM];
      }

      // Multiply the two matrices together;
      // each thread computes one element
      for (int ii = 0; ii < TILE_K; ++ii) {
        for (int jj = 0; jj < TILE_K; jj++) {
          regC[ii][jj] += regAs[ii] * regBs[jj];
        }
      }
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the computed values to the output matrix C
  int c = wB * TILE_N * by + TILE_M * bx;
  for (int i = 0; i < TILE_K; i++) {
    for (int j = 0; j < TILE_K; j++) {
      C[c + wB * (ty + i * SubM) + (tx + j * SubM)] = regC[i][j];
    }
  }
}
#undef TILE_K


#define TILE_K 32
#define WARP_M 16
#define WARP_N 16
#define WARP_K 16

#define WARP_GROUP 256

__global__ void gemm_tiled_regs_nt(const float *__restrict__ A,
                                const float *__restrict__ B, float *C,
                                unsigned wAB, unsigned wC) {
  // Block index
  const int grid_size    = gridDim.x * gridDim.y;
  const int linear_bid   = blockIdx.x + blockIdx.y * gridDim.x;
  const int remapped_bid = remap_xcd(linear_bid, grid_size);
  const int bx           = remapped_bid % gridDim.x;
  const int by           = remapped_bid / gridDim.x;

  constexpr int elem_per_thread = 16 / sizeof(float);
  constexpr int threads_per_row = TILE_K / elem_per_thread;
  constexpr int threads_per_col = WARP_GROUP / threads_per_row;
  constexpr int n_reps_A = TILE_M / threads_per_col;
  constexpr int n_reps_B = TILE_N / threads_per_col;

  constexpr int swizzle_base = TILE_K / 4;

  int a_begin = wAB * TILE_M * by;
  int a_end = a_begin + wAB - 1;
  int a_step = TILE_K;

  int b_begin = wAB * TILE_N * bx;
  int b_step = TILE_K;

  // Thread index
  int tidx = threadIdx.x % threads_per_row;
  int tidy = threadIdx.x / threads_per_row;

  // Thread index for local computation
  int tx = threadIdx.x % WARP_M;
  int ty = threadIdx.x / WARP_N;

  float regC[TILE_M / WARP_M][TILE_N / WARP_N] = {{0}};

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int a = a_begin, b = b_begin; a <= a_end; a += a_step, b += b_step) {

    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_N][TILE_K];

    #pragma unroll
    for (int i = 0; i < n_reps_A; ++i)
      *((fp32x4_t *)&As[tidy + i * threads_per_col][tidx * elem_per_thread]) = 
            *((fp32x4_t *)&A[a + wAB * (tidy + i * threads_per_col) + tidx * elem_per_thread]);

    #pragma unroll
    for (int i = 0; i < n_reps_B; ++i)
      *((fp32x4_t *)&Bs[tidy + i * threads_per_col][tidx * elem_per_thread]) = 
            *((fp32x4_t *)&B[a + wAB * (tidy + i * threads_per_col) + tidx * elem_per_thread]);
    __syncthreads();

    float regAs[TILE_M / WARP_M];
    float regBs[TILE_N / WARP_N];

    // Load the shared memory sub-matrices into registers
    for (int i = 0; i < TILE_K; i++) {

    #pragma unroll
      for (int ii = 0; ii < TILE_M / WARP_M; ++ii) 
        regAs[ii] = As[ty + ii * WARP_M][i];

    #pragma unroll
      for (int ii = 0; ii < TILE_N / WARP_N; ++ii) 
        regBs[ii] = Bs[tx + ii * WARP_N][i];

      // Multiply the two matrices together;
      // each thread computes one element
      for (int ii = 0; ii < TILE_M / WARP_M; ++ii) {
        for (int jj = 0; jj < TILE_N / WARP_N; jj++) {
          regC[ii][jj] += regAs[ii] * regBs[jj];
        }
      }
    }

    __syncthreads();
  }

  // Write the computed values to the output matrix C
  int c = wC * TILE_N * by + TILE_M * bx;
  for (int i = 0; i < TILE_M / WARP_M; i++) {
    for (int j = 0; j < TILE_N / WARP_N; j++) {
      C[c + wC * (ty + i * WARP_M) + (tx + j * WARP_N)] = regC[i][j];
    }
  }
}


// ----- Metrics helpers -----
static inline double tflops(double m, double n, double k, double ms) {
  const double flops = 2.0 * m * n * k;
  return (flops / (ms * 1e-3)) / 1e12;
}
static inline double min_gbytes(double m, double n, double k) {
  const double bytes = (m * k + k * n + m * n) * sizeof(float);
  return bytes / (1024.0 * 1024.0 * 1024.0);
}

constexpr unsigned BS = 32;
// ----- Launch wrappers -----
using KernelLaunch = void (*)(unsigned, unsigned, const float *, const float *,
                              float *, unsigned);

template <unsigned BS>
void launch_tiled(unsigned C_rows, unsigned C_cols, const float *A,
                  const float *B, float *C, unsigned K) {
  const dim3 block(BS, BS);
  const dim3 grid(C_cols / BS, C_rows / BS);
  matrix_multiplication_base<BS><<<grid, block>>>(A, B, C, K);
}

template <unsigned BS>
void launch_tiled_swizzle(unsigned C_rows, unsigned C_cols, const float *A,
                         const float *B, float *C, unsigned K) {
  const dim3 block(BS, BS);
  const dim3 grid(C_cols / BS, C_rows / BS);
  gemm_tiled<BS><<<grid, block>>>(A, B, C, K);
}

void launch_tiled_regs(unsigned C_rows, unsigned C_cols, const float *A,
                      const float *B, float *C, unsigned K) {
  const dim3 block(256);
  const dim3 grid(C_cols / TILE_N, C_rows / TILE_M);
  gemm_tiled_regs<<<grid, block>>>(A, B, C, K, C_cols);
}

void launch_tiled_regs_nt(unsigned C_rows, unsigned C_cols, const float *A,
                      const float *B, float *C, unsigned K) {
  const dim3 block(WARP_GROUP);
  const dim3 grid(C_cols / TILE_N, C_rows / TILE_M);
  gemm_tiled_regs<<<grid, block>>>(A, B, C, K, C_cols);
}

int main(int argc, const char *argv[]) {

  // Args: M N K reps check
  unsigned M = 2048, N = 2048, K = 4096;
  int reps = 100;
  int check_flag = 1; // 1 = do CPU check, 0 = skip
  if (argc >= 2)
    M = std::max(1, std::atoi(argv[1]));
  if (argc >= 3)
    N = std::max(1, std::atoi(argv[2]));
  if (argc >= 4)
    K = std::max(1, std::atoi(argv[3]));
  if (argc >= 5)
    reps = std::max(1, std::atoi(argv[4]));
  if (argc >= 6)
    check_flag = std::atoi(argv[5]) ? 1 : 0;
  const bool do_check = (check_flag != 0);

  if ((M % BS) || (N % BS) || (K % BS)) {
    std::cerr << "Matrix sizes must be multiples of block_size (" << BS
              << ")\n";
    return kErrorExitCode;
  }

  const unsigned A_rows = M, A_cols = K; // A: MxK
  const unsigned B_rows = K, B_cols = N; // B: KxN
  const unsigned C_rows = M, C_cols = N; // C: MxN

  // Device info
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::cout << "> Device: " << prop.name << " | Arch: " << prop.gcnArchName
            << " | CUs: " << prop.multiProcessorCount << "\n";

  std::cout << "GEMM: [" << M << 'x' << K << "] * [" << K << 'x' << N
            << "], tiles: [" << C_rows / TILE_N << "x" << C_cols / TILE_M
            << "], block: " << TILE_N << "x" << TILE_M << ", reps: " << reps
            << ", check: " << (do_check ? "on" : "off") << "\n\n";

  // Host data (A=1, B=const => C=K*const)
  std::vector<float> A(A_rows * A_cols, 1.0f);
  constexpr float b_val = 0.02f;
  std::vector<float> B(B_rows * B_cols, b_val);
  std::vector<float> C; // 仅在需要校验时分配
  if (do_check)
    C.resize(C_rows * C_cols, 0.0f);

  // Device buffers
  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  const size_t a_bytes = A.size() * sizeof(float);
  const size_t b_bytes = B.size() * sizeof(float);
  const size_t c_bytes = static_cast<size_t>(C_rows) * C_cols * sizeof(float);
  HIP_CHECK(hipMalloc(&dA, a_bytes));
  HIP_CHECK(hipMalloc(&dB, b_bytes));
  HIP_CHECK(hipMalloc(&dC, c_bytes));
  HIP_CHECK(hipMemcpy(dA, A.data(), a_bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dB, B.data(), b_bytes, hipMemcpyHostToDevice));

  // Variants
  struct Variant {
    const char *name;
    KernelLaunch launch;
  };
  Variant variants[] = {
      {"tiled", &launch_tiled<BS>},
      {"+ swizzle", &launch_tiled_swizzle<BS>},
      // {"+ swzl + regs", &launch_tiled_regs},
      {"+ swzl + regs (nt)", &launch_tiled_regs_nt},
  };

  // Timing infra
  hipEvent_t ev_start{}, ev_stop{};
  HIP_CHECK(hipEventCreate(&ev_start));
  HIP_CHECK(hipEventCreate(&ev_stop));

  // Header
  std::printf("%-20s | %10s | %10s | %10s | %6s | %6s | %5s\n", "Variant",
              "Avg ms", "TFLOP/s", "GB/s(min)", "Reps", "Blk", "Check");
  std::printf("%-20s-+-%10s-+-%10s-+-%10s-+-%6s-+-%6s-+-%5s\n",
              std::string(20, '-').c_str(), std::string(10, '-').c_str(),
              std::string(10, '-').c_str(), std::string(10, '-').c_str(),
              std::string(6, '-').c_str(), std::string(6, '-').c_str(),
              std::string(5, '-').c_str());

  bool all_ok = true;

  for (const auto &v : variants) {
    // Warmup
    v.launch(C_rows, C_cols, dA, dB, dC, K);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Time
    HIP_CHECK(hipEventRecord(ev_start));
    for (int i = 0; i < reps; ++i) {
      v.launch(C_rows, C_cols, dA, dB, dC, K);
      HIP_CHECK(hipGetLastError());
    }
    HIP_CHECK(hipEventRecord(ev_stop));
    HIP_CHECK(hipEventSynchronize(ev_stop));

    float ms_total = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms_total, ev_start, ev_stop));
    const double ms_avg = ms_total / reps;

    // Metrics
    const double m = static_cast<double>(M);
    const double n = static_cast<double>(N);
    const double k = static_cast<double>(K);
    const double gmin = min_gbytes(m, n, k);
    const double gbps = gmin * (1000.0 / ms_avg);
    const double tfs = tflops(m, n, k, ms_avg);

    // Optional validation (host copy only when needed)
    bool ok = true;
    if (do_check) {
      HIP_CHECK(hipMemcpy(C.data(), dC, c_bytes, hipMemcpyDeviceToHost));
      const float expect = static_cast<float>(K) * b_val;
      const float tol = 1e-3f;
      ok = std::all_of(C.begin(), C.end(),
                       [&](float x) { return std::fabs(x - expect) / expect < tol; });
      all_ok &= ok;
    }

    std::printf("%-20s | %10.4f | %10.3f | %10.3f | %6d |  %2ux%2u | %5s%s\n",
                v.name, ms_avg, tfs, gbps, reps, BS, BS,
                do_check ? "YES" : "NO", (do_check && !ok) ? " *FAIL*" : "");
  }

  if (do_check)
    std::cout << (all_ok ? "\nValidation passed.\n" : "\nValidation failed.\n");
  else
    std::cout << "\nValidation skipped (user disabled).\n";

  // Cleanup
  HIP_CHECK(hipEventDestroy(ev_start));
  HIP_CHECK(hipEventDestroy(ev_stop));
  HIP_CHECK(hipFree(dA));
  HIP_CHECK(hipFree(dB));
  HIP_CHECK(hipFree(dC));
  return (!do_check || all_ok) ? EXIT_SUCCESS : kErrorExitCode;
}
