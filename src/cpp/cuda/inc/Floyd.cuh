/**
 * Copyright 2019  Microsoft Corporation.  All rights reserved.
 *
 * Please refer to the Microsoft end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
#pragma once
#ifndef FLOYD_CUH
#define FLOYD_CUH

#include <memory>
#include <cuda_runtime.h>
#include "inc/evaluator.h"

#define MAX_REGISTERS 256
#define SH_TILE_WIDTH 32
#define SH_TILE_HEIGHT 32
#define BLOCK_SIZE 16

 /**
  * CUDA handle error, if error occurs print message and exit program
 *
 * @param error: CUDA error status
 */
#define HANDLE_ERROR(error) { \
    if (error != cudaSuccess) { \
        fprintf(stderr, "%s in %s at line %d\n", \
                cudaGetErrorString(error), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} \




__device__
inline int Min(int a, int b) { return a < b ? a : b; }


// Naive Cuda
void Floyd_Warshall(const std::unique_ptr<APSPGraph>& dataHost, float* time);

// Coalesced Memory optimization
void Floyd_Warshall_COA(const std::shared_ptr<int[]>& matrix, const std::shared_ptr<int[]>& path,
	const unsigned int size, int thread_per_block, float* time);

// Shared Memory optimization
void Floyd_Warshall_Shared(const std::shared_ptr<int[]>& matrix, const std::shared_ptr<int[]>& path,
	const unsigned int size, float* time);
__global__ void cudaKernel_shared(int *matrix, int* path, int size, int k);

// Blocked Memory optimization
/**
 * Blocked implementation of Floyd Warshall algorithm in CUDA
 *
 * @param time
 * @param dataHost: unique ptr to graph data with allocated fields on host
 */
void CudaBlockedFW(const std::unique_ptr<APSPGraph>& dataHost, float* time);

#endif
