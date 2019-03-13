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

using namespace std;


// Recursive Function to print path of given 
// vertex u from source vertex v
void PrintPath(const std::shared_ptr<int[]>& path, int vertex, int edge, unsigned int vertices)
{
	if ( edge == (INF) || edge == -1 || path.get()[vertex * vertices + edge] == vertex)
		return;

	PrintPath(path, vertex, path.get()[vertex * vertices + edge], vertices);
	std::cout << path.get()[vertex * vertices + edge] << " ";
}



// Function to print the shortest cost with path 
// information between all pairs of vertices
void PrintMatrix(const std::shared_ptr<int[]>& matrix, unsigned int vertices)
{
	for (unsigned int v = 0; v < vertices; v++)
	{
		for (unsigned int u = 0; u < vertices; u++)
		{
			if (matrix[v * vertices + u] == (INF))
				std::cout << setw(5) << "INF";
			else
				std::cout << setw(5) << matrix[v * vertices + u];
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;
}





// Function to print the shortest cost with path 
// information between all pairs of vertices
void PrintSolution(Evaluator* eval, bool verbose, const string& outFileName)
{
	ofstream outFile;
	const bool saveToFile = outFileName.empty();

	if (saveToFile)
	{
		outFile.open(outFileName, std::ios_base::app);

		outFile << "Cost Matrix:" << std::endl;
		unsigned int rows = 0;
		for (unsigned int v = 0; v < eval->Vertices; v++)
		{
			for (unsigned int u = 0; u < eval->Vertices; u++)
			{
				const int i0 = v * eval->Vertices + u;

				if (eval->Cost.get()[i0] == (INF))
					outFile << setw(5) << "INF";
				else
					outFile << setw(5) << eval->Cost.get()[i0];
			}
			outFile << std::endl;
			rows++;
		}

		outFile << std::endl;
	}

	if (verbose) // print all data only if verbose switch is true
	{
		std::cout << "Cost Matrix:" << std::endl;
		unsigned int rows = 0;
		for (unsigned int v = 0; v < eval->Vertices; v++)
		{
			for (unsigned int u = 0; u < eval->Vertices; u++)
			{
				const int i0 = v * eval->Vertices + u;

				if (eval->Cost.get()[i0] == (INF))
					std::cout << setw(5) << "INF";
				else
					std::cout << setw(5) << eval->Cost.get()[i0];
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
				if (u != v && eval->Path.get()[i0] != -1 && eval->Path.get()[i0] != (INF))
				{
					std::cout << "Shortest Path from vertex " << v <<
						" to vertex " << u << " is (" << v << " ";
					PrintPath(eval->Path, v, u, eval->Vertices);
					std::cout << u << ") with the cost of: " << eval->Cost.get()[i0] << std::endl;
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

	if (saveToFile)
	{
		outFile.close();
	}

}



/**
 * Print data graph (graph matrix, prep) and time
 *
 * @param graph: pointer to graph data
 * @param time: time in seconds
 * @param maxValue: maximum value in graph path
 */
void PrintDataJson(const unique_ptr<APSPGraph>& graph, int time, int maxValue) {
	// Lambda function for printMatrix -1 means no path
	ios::sync_with_stdio(false);
	auto printMatrix = [](unique_ptr<int[]>& graph, int n, int max) {
		cout << "[";
		for (int i = 0; i < n; ++i) {
			cout << "[";
			for (int j = 0; j < n; ++j) {
				if (max > graph[i * n + j])
					cout << graph[i * n + j];
				else
					cout << -1;
				if (j != n - 1) cout << ",";
			}
			if (i != n - 1)
				cout << "],\n";
			else
				cout << "]";
		}
		cout << "],\n";
	};

	cout << "{\n    \"graph\":\n";
	printMatrix(graph->graph, graph->vertices, maxValue);
	cout << "    \"predecessors\": \n";
	printMatrix(graph->path, graph->vertices, maxValue);
	cout << "    \"compute_time\": " << time << "\n}";
}



//density will be between 0 and 100, indication the % of number of directed edges in graph
void GenerateRandomGraph(const std::shared_ptr<int[]>& graph, int vertices, int density, int range) {
	//range will be the range of edge weighting of directed edges
	const int prange = (100 / density);
	for (auto v = 0; v < vertices; v++) {
		for (auto e = 0; e < vertices; e++) {
			if (v == e) {//set graph[i][i]=0
				graph.get()[v * vertices + e] = 0;
				continue;
			}
			const auto pr = rand() % prange;
			graph.get()[v * vertices + e] = pr == 0 ? ((rand() % range) + 1) : (INF);//set edge random edge weight to random value, or to INF
		}
	}
}




bool CmpArray(const std::unique_ptr<int[]> l, const std::unique_ptr<int[]> r, const size_t eleNum)
{
	for (unsigned int i = 0; i < eleNum; i++)
		if (l[i] != r[i])
		{
			std::cout << "ERROR: l[" << i << "] = " << l[i] << ", r[" << i << "] = " << r[i] << std::endl;
			return false;
		}
	return true;
}




void GenerateFixedMatrix(const std::shared_ptr<int[]>& graph)
{
	////////////////////////////////////////////////////////////
	// FOR RESULT COMPARISON TESTING ONLY!
	////////////////////////////////////////////////////////////
	int adjMatrix[K_DEF_VERTICES][K_DEF_VERTICES] =
	{
		{   0,   5, (INF),   2, (INF) },
		{ (INF),   0,   2, (INF), (INF) },
		{   3, (INF),   0, (INF),   7 },
		{ (INF), (INF),   4,   0,   1 },
		{   1,   3, (INF), (INF),   0 }
	};

	//given adjacency representation of matrix
	for (unsigned int i = 0; i < K_DEF_VERTICES; ++i)
	{
		for (unsigned int j = 0; j < K_DEF_VERTICES; ++j)
		{
			graph.get()[i * K_DEF_VERTICES + j] = adjMatrix[i][j];
		}
	}


}
