#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define DEFAULT_C1 0xcc9e2d51
#define DEFAULT_C2 0x1b873593
#define DEFAULT_C3 0xe6546b64

#define XXH32_PRIME1 0x9E3779B1u
#define XXH32_PRIME2 0x85EBCA77u
#define XXH32_PRIME3 0xC2B2AE3Du
#define XXH32_PRIME4 0x27D4EB2Fu
#define XXH32_PRIME5 0x165667B1u

#define LOAD_MULTIPLIER 4
#define EMPTYVAL 0xFF
#define UNUSED 0xFFFFFFFF
#define USED   0xFFFFFFFE

#define NONDUPE_INSERT_ERR 0xFFFFFFFF
#define NONDUPE_UNINSERTED 0xFFFFFFFE
#define NONDUPE_INSERT_DUPE 0xFFFFFFFD

typedef struct {
  uint32_t linelength;
  uint32_t elements;
  uint32_t* data;
} whashset_t;

static __forceinline__ __device__
void _memcpy(uint32_t* dst, const uint32_t* src, uint32_t len) {
  for (uint32_t i = 0; i < len; i++) dst[i] = src[i];
}

static __forceinline__ __device__
void _warp32_memcpy(uint32_t* dst, const uint32_t* src, uint32_t len, uint32_t lane_id) {
  for (int i = lane_id; i < len; i+=32) dst[i] = src[i];
}

static __forceinline__ __device__
bool _memcmp(const uint32_t* a, const uint32_t* b, uint32_t len) {
  for (uint32_t i = 0; i < len; i++) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

static __forceinline__ __device__
bool _warp32_memcmp(const uint32_t* a, const uint32_t* b, uint32_t len,
                    uint32_t lane_id) {
  bool eq = true;
  for (int i = lane_id; i < len; i+=32) if (a[i] != b[i]) eq = false;
  eq = __all_sync(0xFFFFFFFF, eq);
  return eq;
}

static __forceinline__ __device__
uint32_t _fmix32(uint32_t h)
{
  h ^= h >> 16;
  h *= 0x85ebca6b;
  h ^= h >> 13;
  h *= 0xc2b2ae35;
  h ^= h >> 16;
  return h;
}

template <uint32_t c1 = DEFAULT_C1, uint32_t c2 = DEFAULT_C2, uint32_t c3 = DEFAULT_C3>
static __device__ __forceinline__
void _murmur_manip_32(uint32_t* k, uint32_t* h) {
  *k *= c1;
  *k = __funnelshift_lc(*k, *k, 15);
  *k *= c2;
  *h ^= *k;
  *h = __funnelshift_lc(*h, *h, 13);
  *h = *h * 5 + c3;
}

static __device__ __forceinline__
uint32_t murmurhash3_32(const uint32_t* val, uint32_t len)
{
  uint32_t h1 = 0;

  for (int i = 0; i < len; i++) {
    uint32_t k = val[i];
    _murmur_manip_32(&k, &h1);
  }

  h1 ^= len;
  h1 = _fmix32(h1);

  return h1;
}

static __device__ __forceinline__
void murmurhash3_32x32(const uint32_t* val, uint32_t len, uint32_t* out, const uint8_t lane_id)
{
  uint32_t h1 = 0;

  for (uint32_t base = 0; base < len; base += 32) {
    uint32_t idx = base + lane_id;
    uint32_t k = (idx < len) ? val[idx] : 0;

    if (idx < len) {
      _murmur_manip_32(&k, &h1);
    }

    h1 = __shfl_sync(0xFFFFFFFF, h1, 31);
  }

  if (lane_id == 0) {
    h1 ^= len;
    h1 = _fmix32(h1);
  }
  
  *out = __shfl_sync(0xFFFFFFFF, h1, 0);
}


static __device__ __forceinline__
uint32_t _xxh32_round(uint32_t acc, uint32_t input)
{
  acc += input * XXH32_PRIME2;
  acc = __funnelshift_lc(acc, acc, 13);
  acc *= XXH32_PRIME1;
  return acc;
}

static __device__ __forceinline__
uint32_t _xxh32_avalanche(uint32_t h32)
{
  h32 ^= h32 >> 15;
  h32 *= XXH32_PRIME2;
  h32 ^= h32 >> 13;
  h32 *= XXH32_PRIME3;
  h32 ^= h32 >> 16;
  return h32;
}

static __device__ __forceinline__
uint32_t xxhash32(const uint32_t* val, uint32_t len)
{
  uint32_t h32 = 0;
  uint32_t i = 0;

  if (len >= 4) {
    uint32_t v1 = XXH32_PRIME1 + XXH32_PRIME2;
    uint32_t v2 = XXH32_PRIME2;
    uint32_t v3 = 0;
    uint32_t v4 = -XXH32_PRIME1;

    for (; i + 4 <= len; i += 4) {
      v1 = _xxh32_round(v1, val[i+0]);
      v2 = _xxh32_round(v2, val[i+1]);
      v3 = _xxh32_round(v3, val[i+2]);
      v4 = _xxh32_round(v4, val[i+3]);
    }

    h32 = __funnelshift_lc(v1, v1, 1)
        + __funnelshift_lc(v2, v2, 7)
        + __funnelshift_lc(v3, v3, 12)
        + __funnelshift_lc(v4, v4, 18);

    h32 = _xxh32_round(h32, v1);
    h32 = _xxh32_round(h32, v2);
    h32 = _xxh32_round(h32, v3);
    h32 = _xxh32_round(h32, v4);
  } else {
    h32 = XXH32_PRIME5;
  }

  for (; i < len; i++) {
    h32 += val[i] * XXH32_PRIME3;
    h32 = __funnelshift_lc(h32, h32, 17) * XXH32_PRIME4;
  }

  h32 += len;
  return _xxh32_avalanche(h32);
}

static __device__ __forceinline__
void xxhash32x32(const uint32_t* val, uint32_t len, uint32_t* out, const uint8_t lane_id)
{
  uint32_t h_i = 0;
  uint32_t h_i_next = 0;

  for (int i = 0; i < len; i += 32) {
    uint32_t k = val[i + lane_id];
    h_i = _xxh32_round(h_i, k);
    h_i += h_i_next;
    h_i_next = __shfl_sync(0xFFFFFFFF, h_i, (lane_id + 1) & 31);
  }

  h_i += len;
  h_i = _xxh32_avalanche(h_i);
  uint32_t h_xor = __reduce_xor_sync(0xFFFFFFFF, h_i);

  *out = _xxh32_avalanche(h_xor ^ len);
}

template <bool warped_mode_guide = false>
whashset_t whashset_create(
  const uint32_t elementlen,
  const uint32_t elements
) {
  if constexpr (warped_mode_guide) {
    assert(elements % 32 == 0);
    assert(elementlen > 0 && elementlen % 32 == 0); 
  }

  whashset_t map = {};
  map.linelength = elementlen;
  map.elements = elements * LOAD_MULTIPLIER;
  cudaMalloc(&map.data, map.linelength * elements 
                        * LOAD_MULTIPLIER * sizeof(uint32_t));
  if (map.data != 0) {
    cudaMemset(map.data, EMPTYVAL, map.linelength * elements 
                                   * LOAD_MULTIPLIER * sizeof(uint32_t));
  }
  return map;
}

/* return: 1=failure */
uint8_t whashset_destroy(whashset_t* map) {
  if (map->data != 0) {
    cudaFree(map->data);
    map->data = 0;
    map->elements = 0;
    map->linelength = 0;
    return 0;
  }
  return 1;
}

/* return: ptr => 0=failure */
template <bool guide = true>
__device__ __forceinline__
uint32_t* dev_whashset_get(
  whashset_t* map,
  const uint32_t hashkey
) {
  if constexpr (guide) {
    if (hashkey >= map->elements) return 0;
  }
  return &map->data[hashkey * map->linelength];
}

__device__ __forceinline__
bool dev_whashset_remove(
  whashset_t* map,
  const uint32_t* hashkey
) {
  uint32_t* elementstart = dev_whashset_get(map, *hashkey);
  if (elementstart == 0) return 1;
  elementstart[0] = UNUSED; /* rest is garbage */
  return 0;
}

/* return: 1=failure */
/*                           value ptr        len                             */
template <uint32_t (*HASH32)(const uint32_t*, uint32_t)>
__device__ __forceinline__
uint32_t dev_whashset_insert(
  whashset_t* map,
  const uint32_t* value
) {
  uint32_t idx = HASH32(value, map->linelength) % map->elements;
  uint32_t* elementstart = dev_whashset_get(map, idx);
  if (elementstart == 0) return 1;
  while (atomicCAS(elementstart, UNUSED, USED) <= USED) {
    idx = (idx + 1) % map->elements;
    elementstart = dev_whashset_get(map, idx);
  }
  if (elementstart != 0) {
    _memcpy(elementstart, value, (map->linelength));
    return 0;
  }
  return 1;
}

/* return: 1=failure */
/*                          value ptr        len       out ptr    lane id     */
template <void (*HASH32x32)(const uint32_t*, uint32_t, uint32_t*, const uint8_t)>
__device__ __forceinline__
bool dev_whashset_insert_warped(
  whashset_t* map,
  const uint32_t* value
) {
  const uint8_t lane_id = threadIdx.x % 32;
  uint32_t hash; /* literally idx, but register reuse so no 2 vars */
  for (uint32_t i = 0; i < 32; i++) {
    uint32_t* val_i = (uint32_t*) __shfl_sync(0xFFFFFFFF, (uintptr_t) value, i);
    if (val_i == 0) continue;
    HASH32x32(val_i, map->linelength, &hash, lane_id);
    hash = hash % map->elements;
    uint32_t* elementstart;
    if (lane_id == 0) {
      elementstart =  dev_whashset_get(map, hash);
      while (atomicCAS(elementstart, UNUSED, USED) <= USED) {
        hash = (hash + 1) % map->elements;
        elementstart = dev_whashset_get(map, hash);
        if (elementstart == 0) return 1;
      }
    }
    __syncwarp(0xFFFFFFFF);
    elementstart = (uint32_t*) __shfl_sync(0xFFFFFFFF, (uintptr_t) elementstart, 0);
    if (elementstart == 0) return 1;
    _warp32_memcpy(elementstart, val_i, (map->linelength), lane_id);
  }
  
  return 0;
}


/* return: hash or amount of collisions and NONDUPE_* codes above */
/*                           value ptr        len                             */
template <uint32_t (*HASH32)(const uint32_t*, uint32_t), bool hashret = false>
__device__ __forceinline__
uint32_t dev_whashset_insert_nonduped_nonwarped(
  whashset_t* map,
  const uint32_t* value
) {
  uint32_t idx = HASH32(value, map->linelength) % map->elements;
  uint32_t* elementstart = dev_whashset_get(map, idx);
  if (elementstart == 0) return NONDUPE_INSERT_ERR;
  
  for (uint32_t probe = 0; probe < map->elements; probe++) {
    uint32_t old = atomicCAS(elementstart, UNUSED, USED);
    
    if (old == UNUSED) {
      _memcpy(elementstart, value, map->linelength);
      if constexpr (hashret) return idx;
      else return probe;
    }
    
    if (old == USED) while (atomicAdd(elementstart, 0) == USED);
                     /* spin-wait (forever) */

    if (_memcmp(elementstart, value, map->linelength)) 
      return NONDUPE_INSERT_DUPE;
    
    idx = (idx + 1) % map->elements;
    elementstart = dev_whashset_get(map, idx);
    if (elementstart == 0) return NONDUPE_INSERT_ERR;
  }
  
  return NONDUPE_UNINSERTED;
}

/* return: hash or amount of collisions and NONDUPE_* codes above */
/*                          value ptr        len       out ptr    lane id     */
template <void (*HASH32x32)(const uint32_t*, uint32_t, uint32_t*, const uint8_t), bool hashret = false>
__device__ __forceinline__
uint32_t dev_whashset_insert_nonduped(
  whashset_t* map,
  const uint32_t* value
) {
  const uint8_t lane_id = threadIdx.x % 32;
  uint32_t retcode = NONDUPE_UNINSERTED; 
  /* it could be that some do not have a probe */
  
  for (int i = 0; i < 32; i++) {
    const uint32_t* val_i = (const uint32_t*) __shfl_sync(0xFFFFFFFF,
                                                          (uintptr_t) value,
                                                          i);
    if (val_i == 0) continue;
    
    uint32_t hash;
    HASH32x32(val_i, map->linelength, &hash, lane_id);
    hash = hash % map->elements;
    
    for (uint32_t probe = 0; probe < map->elements; 
         probe++, hash = (hash + 1) % map->elements) {
      uint32_t* elementstart = dev_whashset_get(map, hash);
      if (elementstart == 0) return NONDUPE_INSERT_ERR; /* alltogether */
      
      uint32_t claim;
      if (lane_id == 0) claim = atomicCAS(elementstart, UNUSED, USED);
      __syncwarp(0xFFFFFFFF);
      claim = __shfl_sync(0xFFFFFFFF, claim, 0);

      if (claim == UNUSED) { /* alltogether */
        _warp32_memcpy(elementstart, val_i, map->linelength, lane_id);
        if (lane_id == i) {
          if constexpr (hashret) retcode = hash;
          else retcode = probe;
        }
        break; /* process next */
      } else if (claim == USED) { 
        if (lane_id == 0) while (atomicAdd(elementstart, 0) == USED);
        __syncwarp(0xFFFFFFFF);
      } else if (_warp32_memcmp(elementstart, val_i, map->linelength, lane_id)) {
        /* was inserted before */
        if (lane_id == i) retcode = NONDUPE_INSERT_DUPE;
        break; /* process next */
      } /* else continue probing */
    }
  } /* left alltogether so just return the saved retcode */
  return retcode;
}

/* we abstract away the hashes, the user should not be in trouble with them   */
/* therefore, no direct find or search ops for values to get hashes are needed*/