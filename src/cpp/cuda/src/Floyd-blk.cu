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
#include <cstdio>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda/inc/Floyd.cuh"
#include "inc/test.h"






/**
 * Blocked CUDA kernel implementation algorithm Floyd Wharshall for APSP
 * Dependent phase 1
 *
 * @param blockId: Index of block
 * @param nvertex: Number of all vertex in graph
 * @param pitch: Length of row in memory
 * @param graph: Array of graph with distance between vertex on device
 * @param pred: Array of predecessors for a graph on device
 */
static __global__
void _blocked_fw_dependent_ph(const int blockId, size_t pitch, const int nvertex, int* const graph, int* const pred) {
	__shared__ int cacheGraph[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ int cachePred[BLOCK_SIZE][BLOCK_SIZE];

	const int idx = threadIdx.x;
	const int idy = threadIdx.y;

	const int v1 = BLOCK_SIZE * blockId + idy;
	const int v2 = BLOCK_SIZE * blockId + idx;

	int newPred;
	int newPath;

	const int cellId = v1 * pitch + v2;
	if (v1 < nvertex && v2 < nvertex) {
		cacheGraph[idy][idx] = graph[cellId];
		cachePred[idy][idx] = pred[cellId];
		newPred = cachePred[idy][idx];
	}
	else {
		cacheGraph[idy][idx] = (INF);
		cachePred[idy][idx] = -1;
	}

	// Synchronize to make sure the all value are loaded in block
	__syncthreads();

#pragma unroll
	for (int u = 0; u < BLOCK_SIZE; ++u) {
		newPath = cacheGraph[idy][u] + cacheGraph[u][idx];

		// Synchronize before calculate new value
		__syncthreads();
		if (newPath < cacheGraph[idy][idx]) {
			cacheGraph[idy][idx] = newPath;
			newPred = cachePred[u][idx];
		}

		// Synchronize to make sure that all value are current
		__syncthreads();
		cachePred[idy][idx] = newPred;
	}

	if (v1 < nvertex && v2 < nvertex) {
		graph[cellId] = cacheGraph[idy][idx];
		pred[cellId] = cachePred[idy][idx];
	}
}

/**
 * Blocked CUDA kernel implementation algorithm Floyd Wharshall for APSP
 * Partial dependent phase 2
 *
 * @param blockId: Index of block
 * @param nvertex: Number of all vertex in graph
 * @param pitch: Length of row in memory
 * @param graph: Array of graph with distance between vertex on device
 * @param pred: Array of predecessors for a graph on device
 */
static __global__
void _blocked_fw_partial_dependent_ph(const int blockId, size_t pitch, const int nvertex, int* const graph, int* const pred) {
	if (blockIdx.x == blockId) return;

	const int idx = threadIdx.x;
	const int idy = threadIdx.y;

	int v1 = BLOCK_SIZE * blockId + idy;
	int v2 = BLOCK_SIZE * blockId + idx;

	__shared__ int cacheGraphBase[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ int cachePredBase[BLOCK_SIZE][BLOCK_SIZE];

	// Load base block for graph and predecessors
	int cellId = v1 * pitch + v2;

	if (v1 < nvertex && v2 < nvertex) {
		cacheGraphBase[idy][idx] = graph[cellId];
		cachePredBase[idy][idx] = pred[cellId];
	}
	else {
		cacheGraphBase[idy][idx] = (INF);
		cachePredBase[idy][idx] = -1;
	}

	// Load i-aligned singly dependent blocks
	if (blockIdx.y == 0) {
		v2 = BLOCK_SIZE * blockIdx.x + idx;
	}
	else {
		// Load j-aligned singly dependent blocks
		v1 = BLOCK_SIZE * blockIdx.x + idy;
	}

	__shared__ int cacheGraph[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ int cachePred[BLOCK_SIZE][BLOCK_SIZE];

	// Load current block for graph and predecessors
	int currentPath;
	int currentPred;

	cellId = v1 * pitch + v2;
	if (v1 < nvertex && v2 < nvertex) {
		currentPath = graph[cellId];
		currentPred = pred[cellId];
	}
	else {
		currentPath = (INF);
		currentPred = -1;
	}
	cacheGraph[idy][idx] = currentPath;
	cachePred[idy][idx] = currentPred;

	// Synchronize to make sure the all value are saved in cache
	__syncthreads();

	int newPath;
	// Compute i-aligned singly dependent blocks
	if (blockIdx.y == 0) {
#pragma unroll
		for (int u = 0; u < BLOCK_SIZE; ++u) {
			newPath = cacheGraphBase[idy][u] + cacheGraph[u][idx];

			if (newPath < currentPath) {
				currentPath = newPath;
				currentPred = cachePred[u][idx];
			}
			// Synchronize to make sure that all threads compare new value with old
			__syncthreads();

			// Update new values
			cacheGraph[idy][idx] = currentPath;
			cachePred[idy][idx] = currentPred;

			// Synchronize to make sure that all threads update cache
			__syncthreads();
		}
	}
	else {
		// Compute j-aligned singly dependent blocks
#pragma unroll
		for (int u = 0; u < BLOCK_SIZE; ++u) {
			newPath = cacheGraph[idy][u] + cacheGraphBase[u][idx];

			if (newPath < currentPath) {
				currentPath = newPath;
				currentPred = cachePredBase[u][idx];
			}

			// Synchronize to make sure that all threads compare new value with old
			__syncthreads();

			// Update new values
			cacheGraph[idy][idx] = currentPath;
			cachePred[idy][idx] = currentPred;

			// Synchronize to make sure that all threads update cache
			__syncthreads();
		}
	}

	if (v1 < nvertex && v2 < nvertex) {
		graph[cellId] = currentPath;
		pred[cellId] = currentPred;
	}
}

/**
 * Blocked CUDA kernel implementation algorithm Floyd Wharshall for APSP
 * Independent phase 3
 *
 * @param blockId: Index of block
 * @param nvertex: Number of all vertex in graph
 * @param pitch: Length of row in memory
 * @param graph: Array of graph with distance between vertex on device
 * @param pred: Array of predecessors for a graph on device
 */
static __global__
void _blocked_fw_independent_ph(const int blockId, size_t pitch, const int nvertex, int* const graph, int* const pred) {
	if (blockIdx.x == blockId || blockIdx.y == blockId) return;

	const int idx = threadIdx.x;
	const int idy = threadIdx.y;

	const int v1 = blockDim.y * blockIdx.y + idy;
	const int v2 = blockDim.x * blockIdx.x + idx;

	__shared__ int cacheGraphBaseRow[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ int cacheGraphBaseCol[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ int cachePredBaseRow[BLOCK_SIZE][BLOCK_SIZE];

	int v1Row = BLOCK_SIZE * blockId + idy;
	int v2Col = BLOCK_SIZE * blockId + idx;

	// Load data for block
	int cellId;
	if (v1Row < nvertex && v2 < nvertex) {
		cellId = v1Row * pitch + v2;

		cacheGraphBaseRow[idy][idx] = graph[cellId];
		cachePredBaseRow[idy][idx] = pred[cellId];
	}
	else {
		cacheGraphBaseRow[idy][idx] = (INF);
		cachePredBaseRow[idy][idx] = -1;
	}

	if (v1 < nvertex && v2Col < nvertex) {
		cellId = v1 * pitch + v2Col;
		cacheGraphBaseCol[idy][idx] = graph[cellId];
	}
	else {
		cacheGraphBaseCol[idy][idx] = (INF);
	}

	// Synchronize to make sure the all value are loaded in virtual block
	__syncthreads();

	int currentPath;
	int currentPred;
	int newPath;

	// Compute data for block
	if (v1 < nvertex && v2 < nvertex) {
		cellId = v1 * pitch + v2;
		currentPath = graph[cellId];
		currentPred = pred[cellId];

#pragma unroll
		for (int u = 0; u < BLOCK_SIZE; ++u) {
			newPath = cacheGraphBaseCol[idy][u] + cacheGraphBaseRow[u][idx];
			if (currentPath > newPath) {
				currentPath = newPath;
				currentPred = cachePredBaseRow[u][idx];
			}
		}
		graph[cellId] = currentPath;
		pred[cellId] = currentPred;
	}
}



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
  * Blocked implementation of Floyd Warshall algorithm in CUDA
  *
  * @param time
  * @param dataHost: unique ptr to graph data with allocated fields on host
  */
void CudaBlockedFW(const std::unique_ptr<APSPGraph>& dataHost, float* time) {
	int nvertex = dataHost->vertices;
	int *graphDevice, *pathDevice;

	cudaEvent_t start, stop;

	// Initialize CUDA GPU Timers
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Start CUDA Timer
	cudaEventRecord(start, nullptr);

	size_t pitch = _cudaMoveMemoryToDevice(dataHost, &graphDevice, &pathDevice);

	dim3 gridPhase1(1, 1, 1);
	dim3 gridPhase2((nvertex - 1) / BLOCK_SIZE + 1, 2, 1);
	dim3 gridPhase3((nvertex - 1) / BLOCK_SIZE + 1, (nvertex - 1) / BLOCK_SIZE + 1, 1);
	dim3 dimBlockSize(BLOCK_SIZE, BLOCK_SIZE, 1);

	int numBlock = (nvertex - 1) / BLOCK_SIZE + 1;

	for (int blockID = 0; blockID < numBlock; ++blockID) {
		// Start dependent phase
		_blocked_fw_dependent_ph << <gridPhase1, dimBlockSize >> >
			(blockID, pitch / sizeof(int), nvertex, graphDevice, pathDevice);

		// Start partially dependent phase
		_blocked_fw_partial_dependent_ph << <gridPhase2, dimBlockSize >> >
			(blockID, pitch / sizeof(int), nvertex, graphDevice, pathDevice);

		// Start independent phase
		_blocked_fw_independent_ph << <gridPhase3, dimBlockSize >> >
			(blockID, pitch / sizeof(int), nvertex, graphDevice, pathDevice);
	}

	// Check for any errors launching the kernel
	HANDLE_ERROR(cudaDeviceSynchronize());
	_cudaMoveMemoryToHost(graphDevice, pathDevice, dataHost, pitch);

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
