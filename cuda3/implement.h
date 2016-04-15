#ifndef IMPLEMENT_H_
#define IMPLEMENT_H_

#include "blowfish.h"
#include "const.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>

__constant__ blowfish_context_t ctx;

#define BLOWFISH_GPU(x) \
	(((ctx.sbox[0][x >> 24] + ctx.sbox[1][(x >> 16) & 0xFF]) \
	^ ctx.sbox[2][(x >> 8) & 0xFF]) + ctx.sbox[3][x & 0xFF])

void version(void)
{
	printf("By Tuan Dao | EECE 5640 | CUDA #3 - Constant Memory\n");
}

__device__ void cudaBlowfishEncryptBlock(uint32_t *hi, uint32_t *lo)
{
	uint32_t i, temp;

	for(i = 0; i < 16; i++) {
		*hi ^= ctx.pbox[i];
		*lo ^= BLOWFISH_GPU(*hi);
		temp = *hi, *hi = *lo, *lo = temp;
	}
	temp = *hi, *hi = *lo, *lo = temp;

	*lo ^= ctx.pbox[16];
	*hi ^= ctx.pbox[17];
}

__device__ void cudaBlowfishEncryptBlock(blowfish_context_t *ctx, uint32_t *hi, uint32_t *lo)
{
	uint32_t i, temp;

	for(i = 0; i < 16; i++) {
		*hi ^= ctx->pbox[i];
		*lo ^= BLOWFISH_F(*hi);
		temp = *hi, *hi = *lo, *lo = temp;
	}
	temp = *hi, *hi = *lo, *lo = temp;

	*lo ^= ctx->pbox[16];
	*hi ^= ctx->pbox[17];
}

__global__ void cudaBlowfishEncryptPtr(uint32_t *ptr, size_t numblocks)
{
	size_t pos = (threadIdx.x + blockDim.x * blockIdx.x)<<1;

	// __shared__ blowfish_context_t localCtx;
	// localCtx = ctx;

	if (pos < numblocks)
	{
		uint32_t lo = ptr[pos+1];
		uint32_t hi = ptr[pos];
		cudaBlowfishEncryptBlock(&hi, &lo);
		ptr[pos+1] = lo;
		ptr[pos] = hi;
	}
}

// __device__ void cudaBlowfishDecryptBlock(blowfish_context_t *ctx, uint32_t *hi, uint32_t *lo)
// {
// 	uint32_t i, temp;

// 	for(i = 17; i > 1; i--) {
// 		*hi ^= ctx->pbox[i];
// 		*lo ^= BLOWFISH_F(*hi);
// 		temp = *hi, *hi = *lo, *lo = temp;
// 	}
// 	temp = *hi, *hi = *lo, *lo = temp;

// 	*lo ^= ctx->pbox[1];
// 	*hi ^= ctx->pbox[0];
// }

__device__ void cudaBlowfishDecryptBlock(uint32_t *hi, uint32_t *lo)
{
	uint32_t i, temp;

	for(i = 17; i > 1; i--) {
		*hi ^= ctx.pbox[i];
		*lo ^= BLOWFISH_GPU(*hi);
		temp = *hi, *hi = *lo, *lo = temp;
	}
	temp = *hi, *hi = *lo, *lo = temp;

	*lo ^= ctx.pbox[1];
	*hi ^= ctx.pbox[0];
}

__global__ void cudaBlowfishDecryptPtr(uint32_t *ptr, size_t numblocks)
{
	size_t pos = (threadIdx.x + blockDim.x * blockIdx.x)<<1;

	// __shared__ blowfish_context_t localCtx;
	// localCtx = ctx;

	if (pos < numblocks)
	{
		uint32_t lo = ptr[pos+1];
		uint32_t hi = ptr[pos];
		cudaBlowfishDecryptBlock(&hi, &lo);
		ptr[pos+1] = lo;
		ptr[pos] = hi;
	}
}

inline cudaError_t cudaCheck(cudaError_t result)
{
	if (result != cudaSuccess) 
	{
		printf("CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		exit(-1);
	}
	return result;
}

#endif
