#ifndef IMPLEMENT_H_
#define IMPLEMENT_H_

#include "blowfish.h"
#include "const.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>

void version(void)
{
	printf("By Tuan Dao | EECE 5640 | CUDA #1 - Global Memory\n");
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

__global__ void cudaBlowfishEncryptPtr(blowfish_context_t *ctx, uint32_t *ptr, size_t numblocks)
{
	size_t pos = (threadIdx.x + blockDim.x * blockIdx.x)<<1;

	if (pos < numblocks)
	{
		uint32_t lo = ptr[pos+1];
		uint32_t hi = ptr[pos];
		cudaBlowfishEncryptBlock(ctx, &hi, &lo);
		ptr[pos+1] = lo;
		ptr[pos] = hi;
	}
}

__device__ void cudaBlowfishDecryptBlock(blowfish_context_t *ctx, uint32_t *hi, uint32_t *lo)
{
	uint32_t i, temp;

	for(i = 17; i > 1; i--) {
		*hi ^= ctx->pbox[i];
		*lo ^= BLOWFISH_F(*hi);
		temp = *hi, *hi = *lo, *lo = temp;
	}
	temp = *hi, *hi = *lo, *lo = temp;

	*lo ^= ctx->pbox[1];
	*hi ^= ctx->pbox[0];
}

__global__ void cudaBlowfishDecryptPtr(blowfish_context_t *ctx, uint32_t *ptr, size_t numblocks)
{
	size_t pos = (threadIdx.x + blockDim.x * blockIdx.x)<<1;

	if (pos < numblocks)
	{
		uint32_t lo = ptr[pos+1];
		uint32_t hi = ptr[pos];
		cudaBlowfishDecryptBlock(ctx, &hi, &lo);
		ptr[pos+1] = lo;
		ptr[pos] = hi;
	}
}

#endif
