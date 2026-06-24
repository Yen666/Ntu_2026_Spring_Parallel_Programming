#ifndef __SHA256_HEADER__
#define __SHA256_HEADER__

#include <stddef.h>

#ifdef __cplusplus
extern "C"{
#endif  //__cplusplus

//--------------- DATA TYPES --------------
typedef unsigned int WORD;
typedef unsigned char BYTE;

typedef union _sha256_ctx{
	WORD h[8];
	BYTE b[32];
}SHA256;

//----------- FUNCTION DECLARATION --------
#ifdef __CUDACC__
__host__ __device__ void sha256_transform(SHA256 *ctx, const BYTE *msg);
__host__ __device__ void sha256(SHA256 *ctx, const BYTE *msg, size_t len);
__device__ void double_sha256_mining(SHA256 *result, const BYTE *block_header_80bytes);
__device__ void double_sha256_mining_precomputed(SHA256 *result, const WORD midstate[8], const BYTE *last16);
void sha256_midstate_cpu(WORD midstate[8], const BYTE *first64);
#else
void sha256_transform(SHA256 *ctx, const BYTE *msg);
void sha256(SHA256 *ctx, const BYTE *msg, size_t len);
void sha256_midstate_cpu(WORD midstate[8], const BYTE *first64);
#endif


#ifdef __cplusplus
}
#endif  //__cplusplus

#endif  //__SHA256_HEADER__
