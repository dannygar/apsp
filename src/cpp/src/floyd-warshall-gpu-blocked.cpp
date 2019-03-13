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

#include <cstdio>
#include <climits>
#include <chrono>
#include <functional>
#include <iomanip>

#include "cuda/inc/Floyd.cuh"
#include "inc/evaluator.h"
#include "inc/floydwarshall.h"
#include "inc/utilities.h"
#include "inc/test.h"

using namespace std;
using namespace std::chrono;


///////////////////////////////////////////////////////////////////////////////////////////
// Floyd-Warshall with blocked memory optimization
///////////////////////////////////////////////////////////////////////////////////////////
float FloydWarshall::RunCudaFWBlocked(Evaluator* eval)
{
	eval->Host = { CUDA_BLOCKED_FW, "CUDA Blocked Memory Optimization" };

	// Run Floyd-Warshall with blocked memory optimization
	return dynamic_cast<FloydWarshall*>(eval)->ComputeCudaBlockedMem();
}


// --- Floyd-Warshall with blocked memory optimization
float FloydWarshall::ComputeCudaBlockedMem()
{
	float elapsedTime;


	// Run Floyd Warshall algorithm
	// Cost[] and parent[] stores shortest-Path 
	// (shortest-Cost/shortest route) information

	/* Copy the initial matrices into the compute graph */
	const auto graph = ReadData(Cost, Vertices);

	/* Compute APSP */
	CudaBlockedFW(graph, &elapsedTime);

	// Copy the results into the original arrays
	CopyResults(graph);

	return elapsedTime;
}



