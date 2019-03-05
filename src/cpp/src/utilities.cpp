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


#include <sstream>
#include <iostream>
#include <iostream>
#include <fstream>
#include <cassert>
#include <cstdio>
#include <iomanip>

#include <helper_timer.h>
#include <helper_math.h>

#include "inc/utilities.h"
#include "inc/evaluator.h"
#include "inc/test.h"

//Time Functions
#include <Windows.h>
#include <MMSystem.h>

#pragma comment(lib, "winmm.lib")

using namespace std;


bool InitMMTimer(UINT wTimerRes) {
	TIMECAPS tc;
	if (timeGetDevCaps(&tc, sizeof(TIMECAPS)) != TIMERR_NOERROR) { return false; }
	wTimerRes = min(max(tc.wPeriodMin, 1), tc.wPeriodMax);
	timeBeginPeriod(wTimerRes);
	return true;
}

void DestroyMMTimer(UINT wTimerRes, bool init) {
	if (init)
		timeEndPeriod(wTimerRes);
}



// Recursive Function to print path of given 
// vertex u from source vertex v
void PrintPath(int *path, int vertex, int edge, unsigned int vertices)
{
	if ( edge == INF || edge == -1 || path[vertex * vertices + edge] == vertex)
		return;

	PrintPath(path, vertex, path[vertex * vertices + edge], vertices);
	printf("%d ", path[vertex * vertices + edge]);
}



// Function to print the shortest cost with path 
// information between all pairs of vertices
void PrintMatrix(int* matrix, unsigned int vertices)
{
	{
		for (unsigned int v = 0; v < vertices; v++)
		{
			for (unsigned int u = 0; u < vertices; u++)
			{
				if (matrix[v * vertices + u] == INT_MAX)
					std::cout << setw(5) << "INF";
				else
					printf("%5d", matrix[v * vertices + u]);
			}
			std::cout << std::endl;
		}

		std::cout << std::endl;
	}
}





// Function to print the shortest cost with path 
// information between all pairs of vertices
void PrintSolution(Evaluator* eval, bool verbose)
{
	if (verbose) // print all data only if verbose switch is true
	{
		std::cout << "Cost Matrix:" << std::endl;
		unsigned int rows = 0;
		for (unsigned int v = 0; v < eval->Vertices; v++)
		{
			for (unsigned int u = 0; u < eval->Vertices; u++)
			{
				const int i0 = v * eval->Vertices + u;

				if (eval->Cost[i0] == INT_MAX)
					std::cout << setw(5) << "INF";
				else
					printf("%5d", eval->Cost[i0]);
			}
			std::cout << std::endl;
			rows++;
			if (rows > K_DISPLAY_ROWS) break;
		}

		std::cout << std::endl;
		if (rows > K_DISPLAY_ROWS)
			std::cout << "..." << std::endl;
		std::cout << std::endl;


		std::cout << "Path Matrix:" << std::endl;
		PrintMatrix(eval->Path, eval->Vertices);
	}

	if (K_DISPLAY_ROWS > 0)
	{
		// print path for each vertex
		for (unsigned int v = 0; v < eval->Vertices; v++)
		{
			unsigned int rows = 0;
			for (unsigned int u = 0; u < eval->Vertices; u++)
			{
				const int i0 = v * eval->Vertices + u;
				if (u != v && eval->Path[i0] != -1 && eval->Path[i0] != INF)
				{
					printf("Shortest Path from vertex %d to vertex %d is (%d ", v, u, v);
					std::cout << "Shortest Path from vertex " << v <<
						" to vertex " << u << " is (" << v << " ";
					PrintPath(eval->Path, v, u, eval->Vertices);
					std::cout << u << ") with the cost of: " << eval->Cost[i0] << std::endl;
				}
				rows++;
				if (rows > K_DISPLAY_ROWS) break;
			}
			if (rows > K_DISPLAY_ROWS)
			{
				std::cout << "..." << std::endl;
				break;
			}
		}
	}
}



//density will be between 0 and 100, indication the % of number of directed edges in graph
void GenerateRandomGraph(int* graph, int vertices, int density, int range) {
	//range will be the range of edge weighting of directed edges
	const int prange = (100 / density);
	for (auto v = 0; v < vertices; v++) {
		for (auto e = 0; e < vertices; e++) {
			if (v == e) {//set graph[i][i]=0
				graph[v * vertices + e] = 0;
				continue;
			}
			auto pr = rand() % prange;
			graph[v * vertices + e] = pr == 0 ? ((rand() % range) + 1) : INF;//set edge random edge weight to random value, or to INF
		}
	}
}




bool CmpArray(const int *l, const int *r, const size_t eleNum)
{
	for (unsigned int i = 0; i < eleNum; i++)
		if (l[i] != r[i])
		{
			printf("ERROR: l[%d] = %d, r[%d] = %d\n", i, l[i], i, r[i]);
			return false;
		}
	return true;
}



