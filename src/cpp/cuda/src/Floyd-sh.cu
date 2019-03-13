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


void Floyd_Warshall_Shared(const std::shared_ptr<int[]>& matrix, const std::shared_ptr<int[]>& path,
	const unsigned size, float* time)
{
	cudaEvent_t start, stop;
	const int memSize = sizeof(int)*size*size;

	// Initialize CUDA GPU Timers
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Start CUDA Timer
	cudaEventRecord(start, nullptr);

	// custom deleter as stateless lambda function
	const auto deleter = [&](int* ptr) { cudaFree(ptr); };

	// Allocate GPU device arrays
	std::unique_ptr<int[], decltype(deleter)> matrixOnGPU(new int[memSize], deleter);
	cudaMallocManaged(reinterpret_cast<void **>(&matrixOnGPU), memSize);
	std::unique_ptr<int[], decltype(deleter)> pathOnGPU(new int[memSize], deleter);
	cudaMallocManaged(reinterpret_cast<void **>(&pathOnGPU), memSize);

	// Copy the host data into device arrays
	cudaMemcpy(matrixOnGPU.get(), matrix.get(), memSize, cudaMemcpyHostToDevice);
	cudaMemcpy(pathOnGPU.get(), path.get(), memSize, cudaMemcpyHostToDevice);

	// It is very important to synchronize between GPU and CPU data transfers
	cudaDeviceSynchronize();

	// dimension
	dim3 thread_per_block(SH_TILE_HEIGHT, SH_TILE_WIDTH);
	dim3 num_block(static_cast<unsigned int>(ceil(1.0 * size / thread_per_block.x)),
		static_cast<unsigned int>(ceil(1.0 * size / thread_per_block.y)));


	// run kernel
	for (unsigned int k = 0; k < size; ++k)
		cudaKernel_shared <<< num_block, thread_per_block >>> (matrixOnGPU.get(), pathOnGPU.get(), size, k);

	// get result back
	cudaMemcpy(matrix.get(), matrixOnGPU.get(), memSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(path.get(), pathOnGPU.get(), memSize, cudaMemcpyDeviceToHost);

	// Stop CUDA Timer
	cudaEventRecord(stop, nullptr);
	//Synchronize GPU with CPU
	cudaEventSynchronize(stop);

	// Read the elapsed time and release memory
	cudaEventElapsedTime(time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Clean up
	cudaFree(matrixOnGPU.get());
	cudaFree(pathOnGPU.get());
	cudaDeviceReset();
}


__global__ void cudaKernel_shared(int *matrix, int* path, int size, int k)
{
	//define shared memory arrays
	__shared__ int cost_i_k[SH_TILE_HEIGHT];
	__shared__ int cost_k_j[SH_TILE_WIDTH];

	// compute indexes
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	// calculate Floyd Warshall algorithm
	if (i < size && j < size) {
		const int cost_i_j = matrix[i*size + j];
		if (i % SH_TILE_HEIGHT == 0) {
			cost_k_j[j % SH_TILE_WIDTH] = matrix[k*size + j];
		}
		if (j % SH_TILE_WIDTH == 0) {
			cost_i_k[i % SH_TILE_WIDTH] = matrix[i*size + k];
		}
		__syncthreads();

		if (cost_i_k[i % SH_TILE_HEIGHT] != INF && cost_k_j[j % SH_TILE_WIDTH] != INF) {
			const int sum = cost_i_k[i % SH_TILE_HEIGHT] + cost_k_j[j % SH_TILE_WIDTH];
			if (cost_i_j == INF || sum < cost_i_j) {
				matrix[i*size + j] = sum;
				path[i*size + j] = path[k*size + j];
			}
		}
	}
}

