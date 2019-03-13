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
#include <climits>
#include <chrono>

#include "inc/evaluator.h"
#include "inc/floydwarshall.h"
#include <iomanip>
#include "inc/test.h"
#include "inc/utilities.h"

using namespace std;
using namespace std::chrono;


///////////////////////////////////////////////////////////////////////////////////////////
// Floyd-Warshall with naive CPU algorithm
///////////////////////////////////////////////////////////////////////////////////////////
float FloydWarshall::RunNaiveFW(Evaluator* eval)
{
	eval->Host = { NAIVE_FW, "CPU" };

	// Run Floyd-Warshall with naive CPU optimization
	return dynamic_cast<FloydWarshall*>(eval)->ComputeCpuNaive();
}


// --- Floyd-Warshall on CPU
float FloydWarshall::ComputeCpuNaive() const
{
	// Start measuring time
	const high_resolution_clock::time_point start = high_resolution_clock::now();

	// initialize Cost[] and Path[]
	// Initially Cost would be same as weight 
	// of the edge
	for (unsigned int v = 0; v < Vertices; v++)
	{
		for (unsigned int u = 0; u < Vertices; u++)
		{
			if (v == u)
				Path.get()[v * Vertices + u] = 0;
			else if (Cost.get()[v * Vertices + u] != (INF))
				Path.get()[v * Vertices + u] = v;
			else
				Path.get()[v * Vertices + u] = -1;
		}
	}

	// Run Floyd Warshall algorithm
	// Cost[] and Path[] stores shortest-Path 
	// (shortest-Cost/shortest route) information
	for (unsigned int k = 0; k < Vertices; k++)
	{
		for (unsigned int v = 0; v < Vertices; v++)
		{
			for (unsigned int u = 0; u < Vertices; u++)
			{
				const int  i0 = v * Vertices + u;
				const int  i1 = v * Vertices + k;
				const int  i2 = k * Vertices + u;
				// If vertex k is on the shortest Path from v to u,
				// then update the value of Cost[v][u], Path[v][u]
				if (Cost.get()[i1] != (INF) && Cost.get()[i2] != (INF))
				{
					const int sum = Cost.get()[i1] + Cost.get()[i2];
					if (Cost.get()[i0] == (INF) || sum < Cost.get()[i0])
					{
						Cost.get()[i0] = sum;
						Path.get()[i0] = Path.get()[i2];
					}
				}
			}

			// if diagonal elements become negative, the
			// graph contains a negative weight cycle
			if (Cost.get()[v * Vertices + v] < 0)
			{
				std::cout << "Negative Weight Cycle Found!!" << std::endl;
				return 0;
			}
		}
	}

	// Measure the elapsed time
	const high_resolution_clock::time_point stop = high_resolution_clock::now();
	const auto duration = duration_cast<milliseconds>(stop - start).count();
	return static_cast<float>(duration);

}




