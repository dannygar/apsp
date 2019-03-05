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
// Floyd-Warshall with coalesced memory optimization
///////////////////////////////////////////////////////////////////////////////////////////
float FloydWarshall::RunCudaFWCoa(Evaluator* eval)
{
	eval->Host = new Processor(CUDA_COALESCED_FW);

	// Run Floyd-Warshall with coalesced memory optimization
	return dynamic_cast<FloydWarshall*>(eval)->ComputeCudaCoalescedMem();
}

// --- Floyd-Warshall with coalesced memory optimization
float FloydWarshall::ComputeCudaCoalescedMem() const
{
	// initialize Cost[] and Path[]
	for (unsigned int v = 0; v < Vertices; v++)
	{
		for (unsigned int u = 0; u < Vertices; u++)
		{
			if (v == u)
				Path[v * Vertices + u] = 0;
			else if (Cost[v * Vertices + u] != INF)
				Path[v * Vertices + u] = v;
			else
				Path[v * Vertices + u] = -1;
		}
	}

	float elapsedTime;

	// Run Floyd Warshall algorithm
	// Cost[] and parent[] stores shortest-Path 
	// (shortest-Cost/shortest route) information
	/* Compute APSP */
	Floyd_Warshall_COA(Cost, Path, Vertices, &elapsedTime);

	return elapsedTime;
}



