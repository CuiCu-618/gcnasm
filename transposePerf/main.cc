#include <hip/hip_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <iostream>

#if defined(__gfx942__)
  #define N_XCD 6
  #define N_GROUPS 4
#elif defined(__gfx950__)
  #define N_XCD 8
  #define N_GROUPS 4
#else
  #define N_XCD 1
  #define N_GROUPS 1
#endif

#define TILE_DIM   64
#define BLOCK_ROWS 16
#define NUM_REPS   100

#define HIP_CALL(cmd)                                                                                                                            \
  do {                                                                                                                                           \
    hipError_t err_ = (cmd);                                                                                                                     \
    if (err_ != hipSuccess) {                                                                                                                    \
      std::cout << "'" << hipGetErrorString(err_) << "'(" << err_ << ") at " << __FILE__ << ":" << __LINE__ << std::endl;                        \
      std::exit(EXIT_FAILURE);                                                                                                                   \
    }                                                                                                                                            \
  } while (0)

/* Map a linear block id onto XCD buckets for more uniform CU spread. */
__device__ __forceinline__ int remap_xcd(int linear_bid, int grid_sz = 1) {
  const int bucket      = linear_bid % N_XCD;
  const int max_buckets = (grid_sz + N_XCD - 1) / N_XCD;
  const int remainder   = grid_sz % N_XCD;
  const int offset      = linear_bid / N_XCD;

  const int subtract = (remainder == 0 || bucket < remainder) ? 0 : (bucket - remainder);
  return bucket * max_buckets + offset - subtract;
}

/* Like remap_xcd but inside groups of size group_sz. */
__device__ __forceinline__ int remap_xcd_group(int linear_bid, int grid_sz, int group_sz = 1) {
  const int group_id    = linear_bid / (N_XCD * group_sz);
  const int is_last_group = group_id == grid_sz / (N_XCD * group_sz);
  const int local_grid_sz = is_last_group ? grid_sz - group_id * (N_XCD * group_sz) : (N_XCD * group_sz);
  const int local_bid   = linear_bid % (N_XCD * group_sz);
  const int bucket      = local_bid % N_XCD;
  const int max_buckets = (local_grid_sz + N_XCD - 1) / N_XCD;
  const int remainder   = local_grid_sz % N_XCD;
  const int offset      = local_bid / N_XCD;

  const int subtract = (remainder == 0 || bucket < remainder) ? 0 : (bucket - remainder);
  return group_id * (N_XCD * group_sz) + bucket * max_buckets + offset - subtract;
}

/* Plain copy tile (baseline). */
__global__ void kernel_copy(float* __restrict__ out, const float* __restrict__ in,
                            int width, int height, int /*group_sz*/) {
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  const int x  = bx * TILE_DIM + threadIdx.x;
  const int y0 = by * TILE_DIM + threadIdx.y;
  int index    = x + width * y0;

  #pragma unroll
  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    out[index + i * width] = in[index + i * width];
  }
}

/* Copy with XCD remap (better CU spread). */
__global__ void kernel_copy_swizzled(float* __restrict__ out, const float* __restrict__ in,
                                     int width, int height, int /*group_sz*/) {
  const int grid_sz      = gridDim.x * gridDim.y;
  const int linear_bid   = blockIdx.x + blockIdx.y * gridDim.x;
  const int remapped_bid = remap_xcd(linear_bid, grid_sz);
  const int bx           = remapped_bid % gridDim.x;
  const int by           = remapped_bid / gridDim.y;

  const int x  = bx * TILE_DIM + threadIdx.x;
  const int y0 = by * TILE_DIM + threadIdx.y;
  int index    = x + width * y0;

  #pragma unroll
  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    out[index + i * width] = in[index + i * width];
  }
}

/* Copy with grouped XCD remap. */
__global__ void kernel_copy_swizzled_group(float* __restrict__ out, const float* __restrict__ in,
                                           int width, int height, int group_sz) {
  const int grid_sz      = gridDim.x * gridDim.y;
  const int linear_bid   = blockIdx.x + blockIdx.y * gridDim.x;
  const int remapped_bid = remap_xcd_group(linear_bid, grid_sz, group_sz);
  const int bx           = remapped_bid % gridDim.x;
  const int by           = remapped_bid / gridDim.x;

  const int x  = bx * TILE_DIM + threadIdx.x;
  const int y0 = by * TILE_DIM + threadIdx.y;
  int index    = x + width * y0;

  #pragma unroll
  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    out[index + i * width] = in[index + i * width];
  }
}

/* Naive transpose (uncoalesced writes). */
__global__ void kernel_transpose_naive(float* __restrict__ out, const float* __restrict__ in,
                                       int width, int height, int /*group_sz*/) {
  const int x0 = blockIdx.x * TILE_DIM + threadIdx.x;
  const int y0 = blockIdx.y * TILE_DIM + threadIdx.y;

  const int in_base  = x0 + width  * y0;      // row-major
  const int out_base = y0 + height * x0;      // transposed index base

  #pragma unroll
  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    out[out_base + i] = in[in_base + i * width];
  }
}

/* Naive transpose with XCD remap. */
__global__ void kernel_transpose_naive_swizzled(float* __restrict__ out, const float* __restrict__ in,
                                                int width, int height, int /*group_sz*/) {
  const int grid_size    = gridDim.x * gridDim.y;
  const int linear_bid   = blockIdx.x + blockIdx.y * gridDim.x;
  const int remapped_bid = remap_xcd(linear_bid, grid_size);
  const int bx           = remapped_bid % gridDim.x;
  const int by           = remapped_bid / gridDim.x;

  const int x0 = bx * TILE_DIM + threadIdx.x;
  const int y0 = by * TILE_DIM + threadIdx.y;

  const int in_base  = x0 + width  * y0;
  const int out_base = y0 + height * x0;

  #pragma unroll
  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    out[out_base + i] = in[in_base + i * width];
  }
}

/* Transpose via shared tile (coalesced). */
__global__ void kernel_transpose_coalesced(float* __restrict__ out, const float* __restrict__ in,
                                           int width, int height, int /*group_sz*/) {
  __shared__ float tile[TILE_DIM][TILE_DIM];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;

  const int in_base = x + y * width;

  #pragma unroll
  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    tile[threadIdx.y + i][threadIdx.x] = in[in_base + i * width];
  }

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;
  const int out_base = x + y * height;

  #pragma unroll
  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    out[out_base + i * height] = tile[threadIdx.x][threadIdx.y + i];
  }
}

/* Transpose via shared tile + XCD remap. */
__global__ void kernel_transpose_coalesced_swizzled(float* __restrict__ out, const float* __restrict__ in,
                                                    int width, int height, int /*group_sz*/) {
  const int grid_size    = gridDim.x * gridDim.y;
  const int linear_bid   = blockIdx.x + blockIdx.y * gridDim.x;
  const int remapped_bid = remap_xcd(linear_bid, grid_size);
  const int bx           = remapped_bid % gridDim.x;
  const int by           = remapped_bid / gridDim.x;

  __shared__ float tile[TILE_DIM][TILE_DIM];

  int x = bx * TILE_DIM + threadIdx.x;
  int y = by * TILE_DIM + threadIdx.y;

  const int in_base = x + y * width;

  #pragma unroll
  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    tile[threadIdx.y + i][threadIdx.x] = in[in_base + i * width];
  }

  __syncthreads();

  x = by * TILE_DIM + threadIdx.x;
  y = bx * TILE_DIM + threadIdx.y;
  const int out_base = x + y * height;

  #pragma unroll
  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    out[out_base + i * height] = tile[threadIdx.x][threadIdx.y + i];
  }
}

/* Transpose via padded shared tile (avoid bank conflicts). */
__global__ void kernel_transpose_nobank(float* __restrict__ out, const float* __restrict__ in,
                                        int width, int height, int /*group_sz*/) {
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;

  const int in_base = x + y * width;

  #pragma unroll
  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    tile[threadIdx.y + i][threadIdx.x] = in[in_base + i * width];
  }

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;
  const int out_base = x + y * height;

  #pragma unroll
  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    out[out_base + i * height] = tile[threadIdx.x][threadIdx.y + i];
  }
}

/* Transpose via padded shared tile + XCD remap. */
__global__ void kernel_transpose_nobank_swizzled(float* __restrict__ out, const float* __restrict__ in,
                                                 int width, int height, int /*group_sz*/) {
  const int grid_size    = gridDim.x * gridDim.y;
  const int linear_bid   = blockIdx.x + blockIdx.y * gridDim.x;
  const int remapped_bid = remap_xcd(linear_bid, grid_size);
  const int bx           = remapped_bid % gridDim.x;
  const int by           = remapped_bid / gridDim.x;

  __shared__ float tile[TILE_DIM][TILE_DIM + 1];

  int x = bx * TILE_DIM + threadIdx.x;
  int y = by * TILE_DIM + threadIdx.y;

  const int in_base = x + y * width;

  #pragma unroll
  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    tile[threadIdx.y + i][threadIdx.x] = in[in_base + i * width];
  }

  __syncthreads();

  x = by * TILE_DIM + threadIdx.x;
  y = bx * TILE_DIM + threadIdx.y;
  const int out_base = x + y * height;

  #pragma unroll
  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    out[out_base + i * height] = tile[threadIdx.x][threadIdx.y + i];
  }
}

__global__ void kernel_transpose_diagnoal(float* __restrict__ out, const float* __restrict__ in,
                                        int width, int height, int /*group_sz*/) 
{
    // Handle to thread block group
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int blockIdx_x, blockIdx_y;

    // do diagonal reordering
    if (width == height) {
        blockIdx_y = blockIdx.x;
        blockIdx_x = (blockIdx.x + blockIdx.y) % gridDim.x;
    }
    else {
        int bid    = blockIdx.x + gridDim.x * blockIdx.y;
        blockIdx_y = bid % gridDim.y;
        blockIdx_x = ((bid / gridDim.y) + blockIdx_y) % gridDim.x;
    }

    // from here on the code is same as previous kernel except blockIdx_x replaces
    // blockIdx.x and similarly for y

    int xIndex   = blockIdx_x * TILE_DIM + threadIdx.x;
    int yIndex   = blockIdx_y * TILE_DIM + threadIdx.y;
    int index_in = xIndex + (yIndex)*width;

    xIndex        = blockIdx_y * TILE_DIM + threadIdx.x;
    yIndex        = blockIdx_x * TILE_DIM + threadIdx.y;
    int index_out = xIndex + (yIndex)*height;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        tile[threadIdx.y + i][threadIdx.x] = in[index_in + i * width];
    }

    __syncthreads();

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        out[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
    }
}

__global__ void kernel_transpose_diagnoal_swizzled(float* __restrict__ out, const float* __restrict__ in,
                                        int width, int height, int /*group_sz*/) 
{
  const int grid_size    = gridDim.x * gridDim.y;
  const int linear_bid   = blockIdx.x + blockIdx.y * gridDim.x;
  const int remapped_bid = remap_xcd(linear_bid, grid_size);
  const int bx           = remapped_bid % gridDim.x;
  const int by           = remapped_bid / gridDim.x;

    // Handle to thread block group
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int blockIdx_x, blockIdx_y;

    // do diagonal reordering
    if (width == height) {
        blockIdx_y = bx;
        blockIdx_x = (bx + by) % gridDim.x;
    }
    else {
        int bid    = bx + gridDim.x * bx;
        blockIdx_y = bid % gridDim.y;
        blockIdx_x = ((bid / gridDim.y) + blockIdx_y) % gridDim.x;
    }

    // from here on the code is same as previous kernel except blockIdx_x replaces
    // blockIdx.x and similarly for y

    int xIndex   = blockIdx_x * TILE_DIM + threadIdx.x;
    int yIndex   = blockIdx_y * TILE_DIM + threadIdx.y;
    int index_in = xIndex + (yIndex)*width;

    xIndex        = blockIdx_y * TILE_DIM + threadIdx.x;
    yIndex        = blockIdx_x * TILE_DIM + threadIdx.y;
    int index_out = xIndex + (yIndex)*height;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        tile[threadIdx.y + i][threadIdx.x] = in[index_in + i * width];
    }

    __syncthreads();

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        out[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
    }
}

/* Transpose via shared tile + XCD remap. */
__global__ void kernel_transpose_swizzle_swizzled(float* __restrict__ out, const float* __restrict__ in,
                                                    int width, int height, int /*group_sz*/) {
  const int grid_size    = gridDim.x * gridDim.y;
  const int linear_bid   = blockIdx.x + blockIdx.y * gridDim.x;
  const int remapped_bid = remap_xcd(linear_bid, grid_size);

  // filing curve remap
  const int block_dim = 16;
  const int block_sz = block_dim * block_dim;
  const int block_id = remapped_bid / block_sz;
  const int global_row = block_id / (gridDim.x / block_dim);
  const int global_col = block_id % (gridDim.x / block_dim);
  const int bx = global_col * block_dim + (remapped_bid % block_sz) % block_dim;
  const int by = global_row * block_dim + (remapped_bid % block_sz) / block_dim;

  __shared__ float tile[TILE_DIM][TILE_DIM];

  int x = bx * TILE_DIM + threadIdx.x;
  int y = by * TILE_DIM + threadIdx.y;

  const int in_base = x + y * width;

  #pragma unroll
  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    tile[threadIdx.y + i][threadIdx.x] = in[in_base + i * width];
  }

  __syncthreads();

  x = by * TILE_DIM + threadIdx.x;
  y = bx * TILE_DIM + threadIdx.y;
  const int out_base = x + y * height;

  #pragma unroll
  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    out[out_base + i * height] = tile[threadIdx.x][threadIdx.y + i];
  }
}

template <class T, class S>
inline bool arrays_equal_eps(const T* ref, const T* got,
                             unsigned len, S eps, float threshold) {
  assert(eps >= 0);
  bool ok = true;
  unsigned err_cnt = 0;

  for (unsigned i = 0; i < len; ++i) {
    float diff = static_cast<float>(ref[i]) - static_cast<float>(got[i]);
    bool pass  = (diff <= eps) && (diff >= -eps);
    ok &= pass;
    err_cnt += !pass;
  }

  if (threshold == 0.0f) return ok;

  if (err_cnt) {
    std::printf("%4.2f(%%) of elements mismatched (count=%u)\n",
                static_cast<float>(err_cnt) * 100.0f / static_cast<float>(len),
                err_cnt);
  }
  return (len * threshold > err_cnt);
}

void transpose_host(float* dst, const float* src, int w, int h) {
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      dst[x * h + y] = src[y * w + x];
}

int main(int argc, char** argv) {
  int width     = 2048;
  int height    = 2048;
  int group_sz  = 228;

  if (argc >= 2) width    = std::atoi(argv[1]);
  if (argc >= 3) height   = std::atoi(argv[2]);
  if (argc >= 4) group_sz = std::atoi(argv[3]);

  hipDeviceProp_t prop{};
  HIP_CALL(hipGetDeviceProperties(&prop, /*device*/ 0));
  std::printf("> Device:        %s\n", prop.name);
  std::printf("> gcnArchName:   %s\n", prop.gcnArchName);
  std::printf("> Number of CUs: %d\n", prop.multiProcessorCount);

  using KernelFn = void(*)(float*, const float*, int, int, int);

  struct Variant {
    KernelFn fn;
    const char* name;
  } variants[] = {
    {kernel_copy,                         "copy (baseline)            "},
    {kernel_copy_swizzled,                "copy + xcd swizzle         "},
    {kernel_copy_swizzled_group,          "copy + xcd swizzle (group) "},
    {kernel_transpose_naive,              "transpose naive            "},
    {kernel_transpose_naive_swizzled,     "transpose naive + swizzle  "},
    {kernel_transpose_coalesced,          "transpose coalesced        "},
    {kernel_transpose_coalesced_swizzled, "transpose coalesced + swzl "},
    {kernel_transpose_nobank,             "transpose no-bank          "},
    {kernel_transpose_nobank_swizzled,    "transpose no-bank + swzl   "},
    {kernel_transpose_diagnoal,           "transpose diagnoal         "},
    {kernel_transpose_diagnoal_swizzled,  "transpose diagnoal + swzl  "},
    {kernel_transpose_swizzle_swizzled,   "transpose swizzle + swzl  "},
  };

  const dim3 grid(width / TILE_DIM, height / TILE_DIM);
  const dim3 block(TILE_DIM, BLOCK_ROWS);

  const size_t elem_cnt = static_cast<size_t>(width) * height;
  const size_t bytes    = elem_cnt * sizeof(float);

  float *h_in = (float*) std::malloc(bytes);
  float *h_out = (float*) std::malloc(bytes);
  float *h_ref = (float*) std::malloc(bytes);

  float *d_in = nullptr, *d_out = nullptr;
  HIP_CALL(hipMalloc(&d_in,  bytes));
  HIP_CALL(hipMalloc(&d_out, bytes));

  for (size_t i = 0; i < elem_cnt; ++i) h_in[i] = static_cast<float>(i);

  HIP_CALL(hipMemcpy(d_in, h_in, bytes, hipMemcpyHostToDevice));
  transpose_host(h_ref, h_in, width, height);

  std::printf("\nMatrix %dx%d, tiles %dx%d, tile %dx%d, block %dx%d\n\n",
              width, height,
              width / TILE_DIM, height / TILE_DIM,
              TILE_DIM, TILE_DIM,
              TILE_DIM, BLOCK_ROWS);

  hipEvent_t ev_start, ev_stop;
  HIP_CALL(hipEventCreate(&ev_start));
  HIP_CALL(hipEventCreate(&ev_stop));

  bool all_ok = true;

  std::printf("%-30s | %12s | %10s | %12s | %6s\n",
            "Variant", "BW (GB/s)", "Time (ms)", "Elements", "WG");
  std::printf("%-30s-+-%12s-+-%10s-+-%12s-+-%6s\n",
            std::string(30,'-').c_str(),
            std::string(12,'-').c_str(),
            std::string(10,'-').c_str(),
            std::string(12,'-').c_str(),
            std::string(6,'-').c_str());

  for (const auto& v : variants) {
    HIP_CALL(hipMemset(d_out, 0.0f, bytes));
    // Warm-up
    v.fn<<<grid, block>>>(d_out, d_in, width, height, group_sz);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());

    // Time loop
    HIP_CALL(hipEventRecord(ev_start));
    for (int i = 0; i < NUM_REPS; ++i) {
      v.fn<<<grid, block>>>(d_out, d_in, width, height, group_sz);
      HIP_CALL(hipGetLastError());
    }
    HIP_CALL(hipEventRecord(ev_stop));
    HIP_CALL(hipEventSynchronize(ev_stop));

    float ms = 0.0f;
    HIP_CALL(hipEventElapsedTime(&ms, ev_start, ev_stop));
    const float avg_ms = ms / NUM_REPS;

    HIP_CALL(hipMemcpy(h_out, d_out, bytes, hipMemcpyDeviceToHost));

    const bool is_copy = (std::string(v.name).find("copy") != std::string::npos);
    const float* ref   = is_copy ? h_in : h_ref;

    const bool ok = arrays_equal_eps(ref, h_out, static_cast<unsigned>(elem_cnt), 0.01f, 0.0f);
    if (!ok) {
      std::printf("*** %s FAILED ***\n", v.name);
      all_ok = false;
    }

    const double gb = 2.0 * static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
    const double gbps = gb * 1000.0 / avg_ms;
    std::printf("%-30s | %12.3f | %10.4f | %12zu | %6u\n",
            v.name, gbps, avg_ms, elem_cnt, TILE_DIM * BLOCK_ROWS);
  }

  std::free(h_in);
  std::free(h_out);
  std::free(h_ref);
  HIP_CALL(hipFree(d_in));
  HIP_CALL(hipFree(d_out));
  HIP_CALL(hipEventDestroy(ev_start));
  HIP_CALL(hipEventDestroy(ev_stop));

  std::printf("%-30s-+-%12s-+-%10s-+-%12s-+-%6s\n",
            std::string(30,'-').c_str(),
            std::string(12,'-').c_str(),
            std::string(10,'-').c_str(),
            std::string(12,'-').c_str(),
            std::string(6,'-').c_str());

  if (!all_ok) {
    std::puts("Test failed!");
    return EXIT_FAILURE;
  }
  std::puts("Test passed");
  return EXIT_SUCCESS;
}
