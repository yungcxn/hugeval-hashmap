#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "warped-hashset.cu"

#define NVALS 16
#define ELEMLEN 128

__global__ void kern_murmur(
  warped_hashset_t map,
  const uint32_t* vals,
  uint32_t n,
  int32_t* out
) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) return;
  const uint32_t* val = &vals[tid * ELEMLEN];
  out[tid] = dev_warped_hashset_insert_nonduped<murmurhash3_32>(&map,val);
}

__global__ void kern_xxhash(
  warped_hashset_t map,
  const uint32_t* vals,
  uint32_t n,
  int32_t* out
) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) return;
  const uint32_t* val = &vals[tid * ELEMLEN];
  out[tid] = dev_warped_hashset_insert_nonduped<xxhash32>(&map,val);
}

__global__ void kern_murmur_warped(
  warped_hashset_t map,
  const uint32_t* vals,
  uint32_t n,
  int32_t* out
) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) return;
  const uint32_t* val = &vals[tid * ELEMLEN];
  out[tid] = dev_warped_hashset_insert_nonduped_warped<murmurhash3_32x32>(&map,val);
}

__global__ void kern_xxhash_warped(
  warped_hashset_t map,
  const uint32_t* vals,
  uint32_t n,
  int32_t* out
) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) return;
  const uint32_t* val = &vals[tid * ELEMLEN];
  out[tid] = dev_warped_hashset_insert_nonduped_warped<xxhash32x32>(&map,val);
}

static void run_test(
  const char* name,
  void (*kernel)(warped_hashset_t,const uint32_t*,uint32_t,int32_t*),
  warped_hashset_t map,
  uint32_t* d_vals,
  int32_t* d_out,
  int32_t* h_out,
  int n
) {
  kernel<<<(n+31)/32,32>>>(map, d_vals, n, d_out);
  cudaMemcpy(h_out, d_out, sizeof(int32_t)*n, cudaMemcpyDeviceToHost);
  printf("%s:\n", name);
  for (int i = 0; i < n; i++) {
    printf(" insert[%d] -> %d\n", i, h_out[i]);
  }
}

int main() {
  srand(time(0));
  uint32_t h_vals[NVALS][ELEMLEN];
  for (int i = 0; i < NVALS; i++) {
    for (int j = 0; j < ELEMLEN; j++) {
      if (i == 0) h_vals[i][j] = j;
      else if (i == 1) h_vals[i][j] = j;
      else if (i == 2) h_vals[i][j] = 0;
      else h_vals[i][j] = rand();
    }
  }

  uint32_t* d_vals;
  int32_t* d_out;
  int32_t h_out[NVALS];
  cudaMalloc(&d_vals, sizeof(h_vals));
  cudaMalloc(&d_out, sizeof(h_out));
  cudaMemcpy(d_vals, h_vals, sizeof(h_vals), cudaMemcpyHostToDevice);

  warped_hashset_t map1 = warped_hashset_create<>(ELEMLEN, NVALS*2);
  run_test("murmur", kern_murmur, map1, d_vals, d_out, h_out, NVALS);
  warped_hashset_destroy(&map1);

  warped_hashset_t map2 = warped_hashset_create<>(ELEMLEN, NVALS*2);
  run_test("xxhash", kern_xxhash, map2, d_vals, d_out, h_out, NVALS);
  warped_hashset_destroy(&map2);

  warped_hashset_t map3 = warped_hashset_create<true>(ELEMLEN, NVALS*2);
  run_test("murmur warped", kern_murmur_warped, map3, d_vals, d_out, h_out, NVALS);
  warped_hashset_destroy(&map3);

  warped_hashset_t map4 = warped_hashset_create<true>(ELEMLEN, NVALS*2);
  run_test("xxhash warped", kern_xxhash_warped, map4, d_vals, d_out, h_out, NVALS);
  warped_hashset_destroy(&map4);

  cudaFree(d_vals);
  cudaFree(d_out);
  return 0;
}
