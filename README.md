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

## Requirements

CUDA 13.0 installation which should include `nvcc`.
