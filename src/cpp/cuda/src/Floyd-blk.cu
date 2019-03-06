/**
 * Copyright 2019  Microsoft Corporation.  All rights reserved.
 *
 * Please refer to the Microsoft end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 * Floyd-Warshall with blocked memory optimization
 */
#include "cuda.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "cuda/inc/Floyd.cuh"
#include "inc/test.h"


void Floyd_Warshall_Blocked(int *matrix, int* path, unsigned int size, float* time)
{
	const int stages = size / BLCK_TILE_WIDTH;

	cudaEvent_t start, stop;

	// Initialize CUDA GPU Timers
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Start CUDA Timer
	cudaEventRecord(start, nullptr);

	// allocate memory
	int *matrixOnGPU;
	int *pathOnGPU;
	cudaMalloc(reinterpret_cast<void **>(&matrixOnGPU), sizeof(int)*size*size);
	cudaMemcpy(matrixOnGPU, matrix, sizeof(int)*size*size, cudaMemcpyHostToDevice);
	cudaMalloc(reinterpret_cast<void **>(&pathOnGPU), sizeof(int)*size*size);
	cudaMemcpy(pathOnGPU, path, sizeof(int)*size*size, cudaMemcpyHostToDevice);

	// dimensions
	dim3 blockSize(BLCK_TILE_WIDTH, BLCK_TILE_WIDTH, 1);
	dim3 phase1Grid(1, 1, 1);
	dim3 phase2Grid(stages, 2, 1);
	dim3 phase3Grid(stages, stages, 1);

	// run kernel
	for (int k = 0; k < stages; ++k)
	{
		const int base = BLCK_TILE_WIDTH * k;
		phase1 << < phase1Grid, blockSize >> > (matrixOnGPU, pathOnGPU, size, base);
		phase2 << < phase2Grid, blockSize >> > (matrixOnGPU, pathOnGPU, size, k, base);
		phase3 << < phase3Grid, blockSize >> > (matrixOnGPU, pathOnGPU, size, k, base);
	}

	// get result back
	cudaMemcpy(matrix, matrixOnGPU, sizeof(int)*size*size, cudaMemcpyDeviceToHost);
	cudaMemcpy(path, pathOnGPU, sizeof(int)*size*size, cudaMemcpyDeviceToHost);


	// Stop CUDA Timer
	cudaEventRecord(stop, nullptr);
	//Synchronize GPU with CPU
	cudaEventSynchronize(stop);

	// Read the elapsed time and release memory
	cudaEventElapsedTime(*&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// free memory resources
	cudaFree(matrixOnGPU);
	cudaFree(pathOnGPU);
}




/*
 * This kernel computes the first phase (self-dependent block)
 *
 * @param matrix A pointer to the adjacency matrix
 * @param matrix A pointer to the path matrix
 * @param size   The width of the matrix
 * @param base   The base index for a block
 */
__global__ void phase1(int *matrix, int* path, int size, int base)
{
	// compute indexes
	const int v = threadIdx.y;
	const int u = threadIdx.x;

	// computes the index for a thread
	const int index = (base + v) * size + (base + u);

	// loads data from global memory to shared memory
	__shared__ int subMatrix[BLCK_TILE_WIDTH][BLCK_TILE_WIDTH];
	subMatrix[v][u] = matrix[index];
	__shared__ int subPath[BLCK_TILE_WIDTH][BLCK_TILE_WIDTH];
	subPath[v][u] = path[index];
	__syncthreads();



	// calculate shortest path
	for (int k = 0; k < BLCK_TILE_WIDTH; ++k)
	{
		// read in dependent values
		const int i0_value = subMatrix[v][u];
		const int i1_value = subMatrix[v][k];
		const int i2_value = subMatrix[k][u];

		if (i1_value != INF && i2_value != INF)
		{
			const int sum = i1_value + i2_value;
			if (i0_value == INF || sum < i0_value)
			{
				subMatrix[v][u] = sum;
				subPath[v][u] = subPath[k][u];
			}
		}
	}

	// write back to global memory
	matrix[index] = subMatrix[v][u];
	path[index] = subPath[v][u];
}

/*
 * This kernel computes the second phase (singly-dependent blocks)
 *
 * @param matrix A pointer to the adjacency matrix
 * @param matrix A pointer to the path matrix
 * @param size   The width of the matrix
 * @param stage  The current stage of the algorithm
 * @param base   The base index for a block
 */
__global__ void phase2(int *matrix, int* path, int size, int stage, int base)
{
	// computes the index for a thread
	if (blockIdx.x == stage) return;

	// compute indexes
	int v, u;
	const int v_prim = base + threadIdx.y;
	const int u_prim = base + threadIdx.x;
	if (blockIdx.y) // load for column
	{
		v = BLCK_TILE_WIDTH * blockIdx.x + threadIdx.y;
		u = u_prim;
	}
	else { // load for row
		u = BLCK_TILE_WIDTH * blockIdx.x + threadIdx.x;
		v = v_prim;
	}
	const int index = v * size + u;
	const int index_prim = v_prim * size + u_prim;

	// loads data from global memory to shared memory
	__shared__ int ownMatrix[BLCK_TILE_WIDTH][BLCK_TILE_WIDTH];
	__shared__ int primaryMatrix[BLCK_TILE_WIDTH][BLCK_TILE_WIDTH];
	ownMatrix[threadIdx.y][threadIdx.x] = matrix[index];
	primaryMatrix[threadIdx.y][threadIdx.x] = matrix[index_prim];
	__syncthreads();

	// loads data from global memory to shared memory
	__shared__ int ownPath[BLCK_TILE_WIDTH][BLCK_TILE_WIDTH];
	__shared__ int primaryPath[BLCK_TILE_WIDTH][BLCK_TILE_WIDTH];
	ownPath[threadIdx.y][threadIdx.x] = path[index];
	primaryPath[threadIdx.y][threadIdx.x] = path[index_prim];
	__syncthreads();


	// calculate shortest path
	for (int k = 0; k < BLCK_TILE_WIDTH; ++k)
	{
		// read in dependent values
		const int i0_value = ownMatrix[threadIdx.y][threadIdx.x];
		const int i1_value = ownMatrix[threadIdx.y][k];
		const int i2_value = primaryMatrix[k][threadIdx.x];

		if (i1_value != INF && i2_value != INF)
		{
			const int sum = i1_value + i2_value;
			if (i0_value == INF || sum < i0_value)
			{
				ownMatrix[threadIdx.y][threadIdx.x] = sum;
				ownPath[threadIdx.y][threadIdx.x] = primaryPath[k][threadIdx.x];
			}
		}
	}

	// write back to global memory
	matrix[index] = ownMatrix[threadIdx.y][threadIdx.x];
	path[index] = ownPath[threadIdx.y][threadIdx.x];
}


/*
 * This kernel computes the third phase (doubly-dependent blocks)
 *
 * @param matrix A pointer to the adjacency matrix
 * @param matrix A pointer to the path matrix
 * @param size   The width of the matrix
 * @param stage  The current stage of the algorithm
 * @param base   The base index for a block
 */
__global__ void phase3(int *matrix, int* path, int size, int stage, int base)
{
	// computes the index for a thread
	if (blockIdx.x == stage || blockIdx.y == stage) return;

	// compute indexes
	const int v = BLCK_TILE_WIDTH * blockIdx.y + threadIdx.y;
	const int u = BLCK_TILE_WIDTH * blockIdx.x + threadIdx.x;
	const int v_row = base + threadIdx.y;
	const int u_col = base + threadIdx.x;
	const int index = v * size + u;
	const int index_row = v_row * size + u;
	const int index_col = v * size + u_col;

	// loads data from global memory into shared memory
	__shared__ int rowMatrix[BLCK_TILE_WIDTH][BLCK_TILE_WIDTH];
	__shared__ int colMatrix[BLCK_TILE_WIDTH][BLCK_TILE_WIDTH];
	int v_u = matrix[index];
	rowMatrix[threadIdx.y][threadIdx.x] = matrix[index_row];
	colMatrix[threadIdx.y][threadIdx.x] = matrix[index_col];
	__syncthreads();

	__shared__ int rowPath[BLCK_TILE_WIDTH][BLCK_TILE_WIDTH];
	int i_j = path[index];
	rowPath[threadIdx.y][threadIdx.x] = path[index_row];
	__syncthreads();

	for (int k = 0; k < BLCK_TILE_WIDTH; ++k)
	{
		// read in dependent values
		const int i0_value = v_u;
		const int i1_value = colMatrix[threadIdx.y][k];
		const int i2_value = rowMatrix[k][threadIdx.x];

		if (i1_value != INF && i2_value != INF)
		{
			const int sum = i1_value + i2_value;
			if (i0_value == INF || sum < i0_value)
			{
				v_u = sum;
				i_j = rowPath[k][threadIdx.x];
			}
		}
	}

	// write back to global memory
	matrix[index] = v_u;
	path[index] = i_j;
}
