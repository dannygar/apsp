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

#include <cuda_runtime.h>

#define COA_TILE_WIDTH 32
#define SH_TILE_WIDTH 256
#define BLCK_TILE_WIDTH 32


// Naive Cuda
void Floyd_Warshall(int *matrix, int* path, unsigned int size, float* time);
__global__ void cudaKernel(int *matrix, int* path, int size, int k);

// Coalesced Memory optimization
void Floyd_Warshall_COA(int *matrix, int* path, unsigned int size, float* time);
__global__ void cudaKernel_coa(int *matrix, int* path, int size, int k);

// Shared Memory optimization
void Floyd_Warshall_Shared(int *matrix, int* path, unsigned int size, float* time);
__global__ void cudaKernel_shared(int *matrix, int* path, int size, int k);

// Blocked Memory optimization
void Floyd_Warshall_Blocked(int *matrix, int* path, unsigned int size, float* time);
__global__ void phase1(int *matrix, int* path, int size, int base);
__global__ void phase2(int *matrix, int* path, int size, int stage, int base);
__global__ void phase3(int *matrix, int* path, int size, int stage, int base);

#endif
