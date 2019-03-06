/**
 * Copyright 2019  Microsoft Corporation.  All rights reserved.
 *
 * Please refer to the Microsoft end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 * Floyd-Warshall with shared memory optimization
 */
#include "cuda.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "cuda/inc/Floyd.cuh"
#include "inc/test.h"


void Floyd_Warshall_Shared(int *matrix, int* path, unsigned int size, float* time)
{
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


	// dimension
	dim3 dimGrid(size, size / SH_TILE_WIDTH, 1);
	dim3 dimBlock(1, SH_TILE_WIDTH, 1);


	// run kernel
	for (unsigned int k = 0; k < size; ++k)
		cudaKernel_shared << < dimGrid, dimBlock >> > (matrixOnGPU, pathOnGPU, size, k);

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


__global__ void cudaKernel_shared(int *matrix, int* path, int size, int k)
{
	// compute indexes
	const int  v = blockIdx.y * SH_TILE_WIDTH + threadIdx.y;
	const int  u = blockIdx.x;

	const int  i0 = v * size + u;
	const int  i1 = v * size + k;
	const int  i2 = k * size + u;

	// read in dependent values
	const int  i0_value = matrix[i0];
	const int  i1_value = matrix[i1];
	__shared__ int i2_value; // initializer is not allowed for __shared__ variable
	if (threadIdx.y == 0) 
		i2_value = matrix[i2];

	// Synchronize to make sure that all value are current
	__syncthreads();

	// calculate Floyd-Warshall shortest path
	if (i1_value != INF && i2_value != INF)
	{
		const int sum = i1_value + i2_value;
		if (i0_value == INF || sum < i0_value)
		{
			matrix[i0] = sum;
			path[i0] = path[i2];
		}
	}


	// compute indexes
	//int i = blockIdx.y * SH_TILE_WIDTH + threadIdx.y;
	//int j = blockIdx.x;

	//int i0 = i * size + j;
	//int i1 = i * size + k;
	//int i2 = k * size + j;

	//// read in dependent values
	//int i_j_value = matrix[i0];
	//int i_k_value = matrix[i1];
	//__shared__ int k_j_value;
	//if (threadIdx.y == 0)
	//	k_j_value = matrix[i2];
	//__syncthreads();

	//// calculate shortest path
	//if (i_k_value != -1 && k_j_value != -1)
	//{
	//	int sum = i_k_value + k_j_value;
	//	if (i_j_value == -1 || sum < i_j_value)
	//	{
	//		matrix[i0] = sum;
	//		path[i0] = path[i2];
	//	}
	//}
}

