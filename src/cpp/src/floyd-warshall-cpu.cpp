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
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <climits>

#include "inc/evaluator.h"
#include "inc/floydwarshall.h"
#include <iomanip>
#include "inc/test.h"
#include "inc/utilities.h"

using namespace std;


///////////////////////////////////////////////////////////////////////////////////////////
// Floyd-Warshall with naive CPU algorithm
///////////////////////////////////////////////////////////////////////////////////////////
float FloydWarshall::RunNaiveFW(Evaluator* eval)
{
	eval->Host = new Processor(NAIVE_FW);

	// Run Floyd-Warshall with naive CPU optimization
	return dynamic_cast<FloydWarshall*>(eval)->ComputeCPUNaive();
}


// --- Floyd-Warshall on CPU
float FloydWarshall::ComputeCPUNaive()
{

	// initialize Cost[] and Path[]
	// Initially Cost would be same as weight 
	// of the edge
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


	UINT wTimerRes = 0;
	bool init = InitMMTimer(wTimerRes);
	DWORD startTime = timeGetTime();

	// Run Floyd Warshall algorithm
	// Cost[] and Path[] stores shortest-Path 
	// (shortest-Cost/shortest route) information
	for (unsigned int k = 0; k < Vertices; k++)
	{
		for (unsigned int v = 0; v < Vertices; v++)
		{
			for (unsigned int u = 0; u < Vertices; u++)
			{
				int i0 = v * Vertices + u;
				int i1 = v * Vertices + k;
				int i2 = k * Vertices + u;
				// If vertex k is on the shortest Path from v to u,
				// then update the value of Cost[v][u], Path[v][u]
				if (Cost[i1] != INF && Cost[i2] != INF)
				{
					const int sum = Cost[i1] + Cost[i2];
					if (Cost[i0] == INF || sum < Cost[i0])
					{
						Cost[i0] = sum;
						Path[i0] = Path[i2];
					}
				}
			}

			// if diagonal elements become negative, the
			// graph contains a negative weight cycle
			if (Cost[v * Vertices + v] < 0)
			{
				std::cout << "Negative Weight Cycle Found!!" << std::endl;
				return 0;
			}
		}
	}

	// Measure the elapsed time
	const DWORD endTime = timeGetTime();
	float elapsedTime = static_cast<float>(endTime - startTime);
	DestroyMMTimer(wTimerRes, init);

	return elapsedTime;

}




