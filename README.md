# warped-hashset

This hashset is built upon the following assumptions:
1. All values (elements) are huge and of the same size => No pointers to elements needed, consecutive array of elements
2. Elements do not differ much => Full hashing of elements
4. All CUDA-Threads may insert, so one warp inserts nothing or 1-32 elements together
5. element is multiple of 32 uint32s (or other equally sized data types)

I utilize this for high performance state space exploration of petri nets for my master thesis.

## Underlying work

I looked into `murmurhash3` and implementations of `xxhash*` both being a high-performer in mass-hashing long arrays.
I implemented custom version of `murmurhash_32 -> murmurhash_32x32` and `xxhash32 -> xxhash32x32` for a warped-size parallel hashing function.

The warped versions of the insertion ops are done under the assumption that the whole warp is active. 
So make sure that you enter with an active mask of `0xFFFFFFFF`.

## Usage

copy `warped-hashset.cu` into your project.
Linkage through headers is highly discouraged due to impractible use of the provided templating.
If you nevertheless try to link against some custom header, use `nvcc -rdc=true ...` since `__device__` functions get relocatable by linkage through that.

In Code, you would do something like:

```c++
#include "warped-hashset.cu"

#define NVALS 32768      /* number of elements */
#define ELEMLEN 128      /* element size in uint32s */
#define BLOCK 256

__global__ void insert_warped(warped_hashset_t map, const uint32_t* data, 
                               uint32_t n, uint32_t* results) {
  uint32_t warp_id = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
  uint32_t lane_id = threadIdx.x % 32;
  if (warp_id >= n/32) return;
  
  uint32_t element_id = warp_id * 32 + lane_id;
  const uint32_t* element = (element_id < n) ? &data[element_id * ELEMLEN] : nullptr;
  
  uint32_t result = dev_warped_hashset_insert_nonduped_warped<murmurhash3_32x32>(&map, element);
  
  if (element_id < n) results[element_id] = result;
}

int main() {
  uint32_t* d_data;
  uint32_t* d_results;
  
  /* alloc mem on gpu */
  cudaMalloc(&d_data, sizeof(uint32_t) * NVALS * ELEMLEN);
  cudaMalloc(&d_results, sizeof(uint32_t) * NVALS);
  
  /* fill arrays ..... */
  
  /* create 32-element (warp-aligned) data array */
  uint32_t capacity = ((NVALS * 2 + 31) / 32) * 32;
  warped_hashset_t map = warped_hashset_create<true>(ELEMLEN, capacity);
  
  /* insertion */
  insert_warped<<<(NVALS+BLOCK-1)/BLOCK, BLOCK>>>(map, d_data, NVALS, d_results);
  cudaDeviceSynchronize();
  
  /* cleanup */
  warped_hashset_destroy(&map);
  
  cudaFree(d_data);
  cudaFree(d_results);
  return 0;
}
```

## Requirements

CUDA 13.0 installation which should include `nvcc`.
