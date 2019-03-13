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
#include <memory>  // std::shared_ptr, std::unique_ptr 
#include "cuda.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "device_launch_parameters.h"
#include <iostream>
#include <iomanip>

#include "cuda/inc/Floyd.cuh"
#include "inc/test.h"



 /**
  * Allocate memory on device and copy memory from host to device
  * @param dataHost: Reference to unique ptr to graph data with allocated fields on host
  * @param graphDevice: Pointer to array of graph with distance between vertex on device
  * @param pathDevice: Pointer to array of predecessors for a graph on device
  *
  * @return: Pitch for allocation
  */
static
size_t _cudaMoveMemoryToDevice(const std::unique_ptr<APSPGraph>& dataHost, int **graphDevice, int **pathDevice) {
	size_t height = dataHost->vertices;
	size_t width = height * sizeof(int);
	size_t pitch;

	// Allocate GPU buffers for matrix of shortest paths d(G) and predecessors p(G)
	HANDLE_ERROR(cudaMallocPitch(graphDevice, &pitch, width, height));
	HANDLE_ERROR(cudaMallocPitch(pathDevice, &pitch, width, height));

	// Copy input from host memory to GPU buffers and
	HANDLE_ERROR(cudaMemcpy2D(*graphDevice, pitch,
		dataHost->graph.get(), width, width, height, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy2D(*pathDevice, pitch,
		dataHost->path.get(), width, width, height, cudaMemcpyHostToDevice));

	return pitch;
}

/**
 * Copy memory from device to host and free device memory
 *
 * @param graphDevice: Array of graph with distance between vertex on device
 * @param pathDevice: Array of predecessors for a graph on device
 * @param dataHost: Reference to unique ptr to graph data with allocated fields on host
 * @param pitch: Pitch for allocation
 */
static
void _cudaMoveMemoryToHost(int *graphDevice, int *pathDevice, const std::unique_ptr<APSPGraph>& dataHost, size_t pitch) {
	size_t height = dataHost->vertices;
	size_t width = height * sizeof(int);

	HANDLE_ERROR(cudaMemcpy2D(dataHost->path.get(), width, pathDevice, pitch, width, height, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy2D(dataHost->graph.get(), width, graphDevice, pitch, width, height, cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaFree(pathDevice));
	HANDLE_ERROR(cudaFree(graphDevice));
}








/**
 * Naive CUDA kernel implementation algorithm Floyd Wharshall for APSP
 * check if path from vertex x -> y will be short using vertex u x -> u -> y
 * for all vertices in graph
 *
 * @param u: Index of vertex u
 * @param nvertex: Number of all vertex in graph
 * @param pitch: Length of row in memory
 * @param graph: Array of graph with distance between vertex on device
 * @param path: Array of predecessors for a graph on device
 */
static __global__
void cudaKernel(const int u, size_t pitch, const int nvertex, int* const graph, int* const path) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (y < nvertex && x < nvertex) {
		int indexYX = y * pitch + x;
		int indexUX = u * pitch + x;

		int newPath = graph[y * pitch + u] + graph[indexUX];
		int oldPath = graph[indexYX];
		if (oldPath > newPath) {
			graph[indexYX] = newPath;
			path[indexYX] = path[indexUX];
		}
	}
}




// Compute Floyd Warshall Linear optimization
void Floyd_Warshall(const std::unique_ptr<APSPGraph>& dataHost, float* time)
{
	cudaEvent_t start, stop;
	int nvertex = dataHost->vertices;
	int *graphDevice, *pathDevice;

	// Initialize CUDA GPU Timers
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Start CUDA Timer
	cudaEventRecord(start, nullptr);

	// Copy host data into the device
	size_t pitch = _cudaMoveMemoryToDevice(dataHost, &graphDevice, &pathDevice);

	// Initialize the grid and block dimensions here
	dim3 dimGrid((nvertex - 1) / BLOCK_SIZE + 1, (nvertex - 1) / BLOCK_SIZE + 1, 1);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

	// Set the preferred cache configuration for a GPU device
	cudaFuncSetCacheConfig(cudaKernel, cudaFuncCachePreferL1);

	// run kernel
	for (unsigned int k = 0; k < nvertex; ++k)
		cudaKernel << < dimGrid, dimBlock >> > (k, pitch / sizeof(int), nvertex, graphDevice, pathDevice);


	// get result back
	HANDLE_ERROR(cudaDeviceSynchronize());
	_cudaMoveMemoryToHost(graphDevice, pathDevice, dataHost, pitch);

	// Stop CUDA Timer
	cudaEventRecord(stop, nullptr);

	//Synchronize time events GPU with CPU
	cudaEventSynchronize(stop);

	// Read the elapsed time and release memory
	cudaEventElapsedTime(time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Clean up
	cudaDeviceReset();
}
