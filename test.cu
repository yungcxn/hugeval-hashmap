#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "warped-hashset.cu"

#define NVALS 32768
#define ELEMLEN 128
#define BLOCK 256

__global__ void kern_murmur(warped_hashset_t map, const uint32_t* vals, uint32_t n, int32_t* out) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) return;
  const uint32_t* val = &vals[tid * ELEMLEN];
  out[tid] = dev_warped_hashset_insert_nonduped<murmurhash3_32>(&map,val);
}

__global__ void kern_xxhash(warped_hashset_t map, const uint32_t* vals, uint32_t n, int32_t* out) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) return;
  const uint32_t* val = &vals[tid * ELEMLEN];
  out[tid] = dev_warped_hashset_insert_nonduped<xxhash32>(&map,val);
}

__global__ void kern_murmur_warped(warped_hashset_t map, const uint32_t* vals, uint32_t n, int32_t* out) {
  uint32_t warp_id = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
  if (warp_id >= n/32) return;
  const uint32_t* val = &vals[warp_id * 32 * ELEMLEN];
  uint32_t result = dev_warped_hashset_insert_nonduped_warped<murmurhash3_32x32>(&map,val);
  if ((threadIdx.x % 32) == 0) out[warp_id] = result;
}

__global__ void kern_xxhash_warped(warped_hashset_t map, const uint32_t* vals, uint32_t n, int32_t* out) {
  uint32_t warp_id = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
  if (warp_id >= n/32) return;
  const uint32_t* val = &vals[warp_id * 32 * ELEMLEN];
  uint32_t result = dev_warped_hashset_insert_nonduped_warped<xxhash32x32>(&map,val);
  if ((threadIdx.x % 32) == 0) out[warp_id] = result;
}

template <bool per_insert_print=false>
static void benchmark(const char* name, void (*kernel)(warped_hashset_t,const uint32_t*,uint32_t,int32_t*), warped_hashset_t map, uint32_t* d_vals, int32_t* d_out, int32_t* h_out, int n) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  kernel<<<(n+BLOCK-1)/BLOCK,BLOCK>>>(map,d_vals,n,d_out);
  cudaDeviceSynchronize();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0;
  cudaEventElapsedTime(&ms,start,stop);
  cudaMemcpy(h_out,d_out,sizeof(int32_t)*n,cudaMemcpyDeviceToHost);
  int success=0, duplicate=0, error=0;
  printf("%s Results:\n",name);
  for (int i=0;i<n;i++) {
    int32_t r = h_out[i];
    if (r==-1) { success++; if constexpr (per_insert_print) printf(" insert[%d] -> success\n", i); }
    else if (r==-2) { error++; if constexpr (per_insert_print) printf(" insert[%d] -> error\n", i); }
    else { duplicate++; if constexpr (per_insert_print) printf(" insert[%d] -> duplicate at %d\n", i, r); }
  }
  printf("Summary for %s:\n", name);
  printf(" total inserts attempted: %d\n", n);
  printf(" successful inserts: %d\n", success);
  printf(" duplicates: %d\n", duplicate);
  printf(" errors: %d\n", error);
  printf(" total time: %.3f ms\n", ms);
  printf(" throughput: %.3f Mops/s\n", n/(ms*1e-3)/1e6);
  printf("---------------------------------------------------\n");
}

int main() {
  srand(time(0));
  uint32_t* h_vals = (uint32_t*)malloc(sizeof(uint32_t)*NVALS*ELEMLEN);
  for (int i=0;i<NVALS;i++) for (int j=0;j<ELEMLEN;j++) {
    if (i==0) h_vals[i*ELEMLEN+j]=j;
    else if (i==1) h_vals[i*ELEMLEN+j]=j;
    else if (i==2) h_vals[i*ELEMLEN+j]=0;
    else h_vals[i*ELEMLEN+j]=rand();
  }

  uint32_t* d_vals;
  int32_t* d_out;
  int32_t* h_out = (int32_t*)malloc(sizeof(int32_t)*NVALS);
  cudaMalloc(&d_vals,sizeof(uint32_t)*NVALS*ELEMLEN);
  cudaMalloc(&d_out,sizeof(int32_t)*NVALS);
  cudaMemcpy(d_vals,h_vals,sizeof(uint32_t)*NVALS*ELEMLEN,cudaMemcpyHostToDevice);

  warped_hashset_t map1 = warped_hashset_create<>(ELEMLEN,NVALS*2);
  benchmark("murmur", kern_murmur, map1, d_vals, d_out, h_out, NVALS);
  warped_hashset_destroy(&map1);

  warped_hashset_t map2 = warped_hashset_create<>(ELEMLEN,NVALS*2);
  benchmark("xxhash", kern_xxhash, map2, d_vals, d_out, h_out, NVALS);
  warped_hashset_destroy(&map2);

  uint32_t warped_size = ((NVALS*2+31)/32)*32;
  warped_hashset_t map3 = warped_hashset_create<true>(ELEMLEN, warped_size);
  benchmark("murmur warped", kern_murmur_warped, map3, d_vals, d_out, h_out, NVALS);
  warped_hashset_destroy(&map3);

  warped_hashset_t map4 = warped_hashset_create<true>(ELEMLEN, warped_size);
  benchmark("xxhash warped", kern_xxhash_warped, map4, d_vals, d_out, h_out, NVALS);
  warped_hashset_destroy(&map4);

  cudaFree(d_vals);
  cudaFree(d_out);
  free(h_vals);
  free(h_out);
  return 0;
}
