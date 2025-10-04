#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define LOAD_MULTIPLIER 4
#define EMPTYVAL 0xFF
#define UNUSED 0xFFFFFFFF
#define USED   0xFFFFFFFE

#define DEFAULT_C1 0xcc9e2d51
#define DEFAULT_C2 0x1b873593
#define DEFAULT_C3 0xe6546b64

#define XXH32_PRIME1 0x9E3779B1u  // 2654435761
#define XXH32_PRIME2 0x85EBCA77u  // 2246822519
#define XXH32_PRIME3 0xC2B2AE3Du  // 3266489917
#define XXH32_PRIME4 0x27D4EB2Fu  //  668265263
#define XXH32_PRIME5 0x165667B1u  //  374761393

typedef struct {
  uint32_t linelength; /* used/unused element + uint32s in one line */
  uint32_t elements;
  uint32_t* data;
} warped_hashset_t;

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
  eq = __all_sync(0xFFFFFFFF, eq); /* outside loop */
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
  uint32_t h_i = 0;
  uint32_t h_i_next = 0;

  for(int i = 0; i < len; i+=32)
  {
    uint32_t k = val[i + lane_id];
    _murmur_manip_32(&k, &h_i);
    h_i += h_i_next;
    h_i_next = __shfl_sync(0xFFFFFFFF, h_i, (lane_id + 1) & 31);
  }

  h_i ^= len;
  h_i = _fmix32(h_i);
  uint32_t h_xor = __reduce_xor_sync(0xFFFFFFFF, h_i);
  *out = _fmix32(h_xor ^ len); /* all do it */
}


static __device__ __forceinline__
uint32_t _xxh32_round(uint32_t acc, uint32_t input)
{
  acc += input * XXH32_PRIME2;
  acc = __funnelshift_lc(acc, acc, 13);
  acc *= XXH32_PRIME1;
  return acc;
}


/* finalization function */
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

  if (len >= 4) {
    uint32_t v1 = XXH32_PRIME1 + XXH32_PRIME2;
    uint32_t v2 = XXH32_PRIME2;
    uint32_t v3 = 0;
    uint32_t v4 = -XXH32_PRIME1;

    for (int i = 0; i + 4 <= len; i += 4) {
      v1 = _xxh32_round(v1, val[i+0]);
      v2 = _xxh32_round(v2, val[i+1]);
      v3 = _xxh32_round(v3, val[i+2]);
      v4 = _xxh32_round(v4, val[i+3]);
    }

    h32 = (__funnelshift_lc(v1, v1, 1) +
           __funnelshift_lc(v2, v2, 7) +
           __funnelshift_lc(v3, v3, 12) +
           __funnelshift_lc(v4, v4, 18));
  } else {
    h32 = XXH32_PRIME5;
  }

  h32 += len * 4;

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

  h_i += len * 4;
  h_i = _xxh32_avalanche(h_i);
  uint32_t h_xor = __reduce_xor_sync(0xFFFFFFFF, h_i);

  *out = _xxh32_avalanche(h_xor ^ (len * 4)); /* all do it */
}

/******************************************************************************/
/*** MAIN PART ****************************************************************/
/******************************************************************************/

template <bool warped_mode_guide = false>
warped_hashset_t warped_hashset_create(
  const uint32_t elementlen,
  const uint32_t elements
) {
  if constexpr (warped_mode_guide) {
    /* 32 warps have their own elements*/
    assert(elements % 32 == 0);
    /* one warp traverses one element*/
    assert(elementlen > 0 && elementlen % 32 == 0); 
  }

  warped_hashset_t map = {};
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


/* 0 on success */
uint8_t warped_hashset_destroy(warped_hashset_t* map) {
  if (map->data != 0) {
    cudaFree(map->data);
    map->data = 0;
    map->elements = 0;
    map->linelength = 0;
    return 0;
  }
  return 1;
}

/* 0 if out of bounds */
__device__ __forceinline__
uint32_t* dev_warped_hashset_get(
  warped_hashset_t map,
  const uint32_t hashkey /* moduled */
) {
  if (hashkey >= map.elements) return 0;
  return &map.data[hashkey * map.linelength];
}


/* 0 on success */
/*                           data in          len                             */
template <uint32_t (*HASH32)(const uint32_t*, uint32_t)>
__device__ __forceinline__
uint32_t dev_warped_hashset_insert_nonduped(
  warped_hashset_t* map,
  const uint32_t* value
) {
  uint32_t hash = HASH32(value, map->linelength);
  uint32_t idx = hash % map->elements;

  for (uint32_t probe = 0; probe < map->elements; probe++, idx = (idx + 1) % map->elements) {
    uint32_t* elementstart = dev_warped_hashset_get(*map, idx);
    if (elementstart == 0) return NONDUPE_INSERT_ERROR;

    /* try to claim slot: UNUSED -> USED */
    uint32_t old = atomicCAS(&elementstart[0], UNUSED, USED);

    if (old == UNUSED) {
      /* We claimed the slot. Write payload except word 0 first. */
      uint32_t len = map->linelength;
      for (uint32_t i = 1; i < len; i++) elementstart[i] = value[i];

      /* ensure stores to payload are visible before publishing first word */
      __threadfence_block();

      /* publish first word (this also ends the 'USED' marker) */
      elementstart[0] = value[0];

      return NONDUPE_INSERT_SUCCESS;
    }

    if (old == USED) {
      /* Another writer is publishing right now; wait until it's done */
      while (atomicAdd(&elementstart[0], 0) == USED) { }
      /* now elementstart should contain a finished value; compare below */
    }

    /* If old is some occupied value (or we just waited), compare for duplicate */
    if (_memcmp(elementstart, value, map->linelength)) {
      return idx; /* duplicate found */
    }

    /* otherwise continue probing */
  }

  /* table full / couldn't insert */
  return NONDUPE_INSERT_ERROR;
}


#define NONDUPE_INSERT_SUCCESS 0xFFFFFFFF
#define NONDUPE_INSERT_ERROR   0xFFFFFFFE


template <void (*HASH32x32)(const uint32_t*, uint32_t, uint32_t*, const uint8_t)>
__device__ __forceinline__
uint32_t dev_warped_hashset_insert_nonduped_warped(
  warped_hashset_t* map,
  const uint32_t* value
) {
  const uint8_t lane_id = threadIdx.x % 32;

  for (int i = 0; i < 32; i++) {
    const uint32_t* val_i = (const uint32_t*)__shfl_sync(0xFFFFFFFF, (uintptr_t)value, i);
    if (val_i == 0) continue;

    uint32_t hash;
    HASH32x32(val_i, map->linelength, &hash, lane_id);

    uint32_t idx = hash % map->elements;

    for (uint32_t probe = 0; probe < map->elements; probe++, idx = (idx + 1) % map->elements) {
      uint32_t* elementstart = dev_warped_hashset_get(*map, idx);
      if (elementstart == 0) return NONDUPE_INSERT_ERROR;

      /* lane 0 tries to claim slot with CAS, broadcast result to warp */
      uint32_t claim = 0;
      if (lane_id == 0) claim = atomicCAS(&elementstart[0], UNUSED, USED);
      claim = __shfl_sync(0xFFFFFFFF, claim, 0);

      if (claim == UNUSED) {
        /* slot claimed: write cooperatively, BUT write index 0 last.
           - lanes write indices 1..len-1 they are responsible for
           - sync
           - lane 0 publishes elementstart[0] last
        */
        uint32_t len = map->linelength;

        /* each lane writes its subset of words except index 0 */
        for (uint32_t j = 1 + lane_id; j < len; j += 32) {
          elementstart[j] = val_i[j];
        }

        /* ensure local stores visible to other lanes in warp/SM */
        __syncwarp();
        __threadfence_block();

        /* publish first word by lane 0 only (or whichever lane is responsible for index 0) */
        if (lane_id == 0) {
          elementstart[0] = val_i[0];
        }

        /* ensure the publish is observed by other lanes before leaving */
        __syncwarp();

        return NONDUPE_INSERT_SUCCESS;
      }

      if (claim == USED) {
        /* another writer is active; spin until it's done */
        while (atomicAdd(&elementstart[0], 0) == USED) { }
        /* fall through to cooperative compare */
      }

      /* cooperative compare: if all lanes agree it's equal, duplicate found */
      if (_warp32_memcmp(elementstart, val_i, map->linelength, lane_id)) {
        return idx; /* duplicate found */
      }
    }

    /* table full for this value */
    return NONDUPE_INSERT_ERROR;
  }

  /* nothing to insert in warp */
  return NONDUPE_INSERT_SUCCESS;
}