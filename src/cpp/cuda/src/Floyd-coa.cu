/**
 * Copyright 2019  Microsoft Corporation.  All rights reserved.
 *
 * Please refer to the Microsoft end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 * Floyd-Warshall with coalesced memory optimization
 */
#include <memory>  // std::shared_ptr, std::unique_ptr 
#include "cuda.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <iostream>
#include <iomanip>

#include "device_launch_parameters.h"

#include "cuda/inc/Floyd.cuh"
#include "inc/test.h"


 //run CUDA kernel
static __global__
void cudaKernel_coa(int* mat, int* path, int k, int size, int segment_size)
{
	// compute indexes
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	// calculate shortest path
	if (segment_size*idx < size*size) {
		for (int offset = 0; offset < segment_size && offset + segment_size * idx < size*size; ++offset) {
			int i = (segment_size*idx + offset) / size;
			int j = segment_size * idx + offset - i * size;
			if (mat[i*size + k] != INF && mat[k*size + j] != INF) {
				const int sum = mat[i*size + k] + mat[k*size + j];
				if (mat[i*size + j] == INF || sum < mat[i*size + j]) {
					mat[i*size + j] = sum;
					path[i*size + j] = path[k*size + j];
				}
			}
		}
	}
}



// Each thread will access 'segment_size' values to improve coalescing.
// Each block now handles thread_per_block * segment_size values.
// Hence the number of blocks needed is size*size/(segment_size*thread_per_block).
void Floyd_Warshall_COA(const std::shared_ptr<int[]>& matrix, const std::shared_ptr<int[]>& path,
	const unsigned size, int thread_per_block, float* time)
{
	cudaEvent_t start, stop;

	const int memSize = sizeof(int) * size * size;

	// Calculate the threads segment/block size
	const auto segmentSize = MAX_REGISTERS / 2;


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

	// dimension
	int num_block = static_cast<int>(ceil(1.0 * size * size / (thread_per_block * segmentSize)));

	// run kernel
	for (unsigned int k = 0; k < size; ++k) {
		cudaKernel_coa <<< num_block, thread_per_block >>> (matrixOnGPU.get(), pathOnGPU.get(), k, size, segmentSize);
	}

	// It is very important to synchronize between GPU and CPU data transfers
	cudaDeviceSynchronize();

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
	cudaDeviceReset();
}




