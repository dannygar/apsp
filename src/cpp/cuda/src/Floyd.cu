/**
 * Copyright 2019  Microsoft Corporation.  All rights reserved.
 *
 * Please refer to the Microsoft end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 * Basic version of Floyd-Warshall
 */
#include "cuda.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "cuda/inc/Floyd.cuh"
#include "inc/test.h"


void Floyd_Warshall(int *matrix, int* path, unsigned int size, float* time)
{
	cudaEvent_t start, stop;

	// Initialize CUDA GPU Timers
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Start CUDA Timer
	cudaEventRecord(start, 0);

	// allocate memory
	int *matrixOnGPU;
	int *pathOnGPU;
	cudaMalloc((void **)&matrixOnGPU, sizeof(int)*size*size);
	cudaMemcpy(matrixOnGPU, matrix, sizeof(int)*size*size, cudaMemcpyHostToDevice);
	cudaMalloc((void **)&pathOnGPU, sizeof(int)*size*size);
	cudaMemcpy(pathOnGPU, path, sizeof(int)*size*size, cudaMemcpyHostToDevice);

	// dimension
	dim3 dimGrid(size, size, 1);

	// run kernel
	for (unsigned int k = 0; k < size; ++k)
		cudaKernel << <dimGrid, 1 >> > (matrixOnGPU, pathOnGPU, size, k);

	// get result back
	cudaMemcpy(matrix, matrixOnGPU, sizeof(int)*size*size, cudaMemcpyDeviceToHost);
	cudaMemcpy(path, pathOnGPU, sizeof(int)*size*size, cudaMemcpyDeviceToHost);

	// Stop CUDA Timer
	cudaEventRecord(stop, 0);
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


__global__ void cudaKernel(int *matrix, int* path, int size, int k)
{
	// compute indexes
	int v = blockIdx.y;
	int u = blockIdx.x;

	int i0 = v * size + u;
	int i1 = v * size + k;
	int i2 = k * size + u;

	// read in dependent values
	int i0_value = matrix[i0];
	int i1_value = matrix[i1];
	int i2_value = matrix[i2];


	// Synchronize to make sure that all value are current
	__syncthreads();

	// calculate shortest path
	if (i1_value != INF && i2_value != INF)
	{
		const int sum = i1_value + i2_value;
		if (i0_value == INF || sum < i0_value)
		{
			matrix[i0] = sum;
			path[i0] = path[i2];
		}
	}
}

