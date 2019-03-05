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
#include <functional>

#include "../inc/evaluator.h"
#include "../inc/test.h"
#include "../inc/utilities.h"
#include "../inc/floydwarshall.h"

#include <helper_math.h>
#include <helper_timer.h>

using namespace std;


//default destructor
Test::~Test() = default;



//////////////////////////////////////////
// Main Test entry
bool Test::RunTest(Evaluator* eval)
{
	pass_ = true;
	unsigned int cpu_time = 0, gpu_time = 0;
	const int memSize = sizeof(int) * eval->Vertices * eval->Vertices;
	int *graph;
	std::vector<std::function<float(Evaluator*)>> algorithms = {
		FloydWarshall::RunNaiveFW,
		FloydWarshall::RunCudaFW,
		FloydWarshall::RunCudaFWCoa,
		FloydWarshall::RunCudaFWShared,
		FloydWarshall::RunCudaFWBlocked
	};

	if (this->OutputFile.empty())
		std::cerr << "Warning: Output filename wasn't provided. The results won't be saved." << std::endl;


	// Evaluate on CPU
	std::cout << "======================================================" << std::endl;
	std::cout << "Evaluating " << eval->Name << std::endl;
	std::cout << "======================================================" << std::endl;

	if (TargetLoop)
	{
		// store original vertices to start each evalation
		const auto originalVertices = eval->Vertices;
		/////////////////////////////////////////////////////////
		// Iterate thru all algorithms and evaluate the results
		/////////////////////////////////////////////////////////
		for (auto& algorithm : algorithms)
		{
			try
			{
				eval->Vertices = originalVertices;
				auto prev_vertices = eval->Vertices;
				bool continueEvaluate = true;
				while (continueEvaluate)
				{
					// reset the graph size
					const int graphSize = sizeof(int) * eval->Vertices * eval->Vertices;

					// pointer to 2D array
					graph = static_cast<int*>(malloc(graphSize));

					// Generate a random graph
					GenerateRandomGraph(graph, eval->Vertices, eval->Density, RANGE);

					// Initialize Cost and Path arrays
					eval->InitArrays(graph);

					// Execute the algorithm
					this->elapsedTime_ = algorithm(eval);

					//Evaluate the results
					EvaluateAlgorithm(eval, this->elapsedTime_);
					//if (this->elapsedTime_ == 0 || !EvaluateAlgorithm(eval, this->elapsedTime_))
					//{
					//	std::cerr << "Error: Algorithm " << eval->Name << " " << eval->Host->Name << " has failed." << std::endl;
					//	continueEvaluate = false;
					//}

					// Save the performance metrics values
					SavePerformanceMetrics(eval, this->elapsedTime_);

					if (this->elapsedTime_ > 0 && continueEvaluate)
						//Evaluate the returning time and increment vertices 
						continueEvaluate = IncrementWithGradientDescent(eval, &prev_vertices, &eval->Vertices, this->elapsedTime_);
					else
						continueEvaluate = false;

					//Release memory
					free(graph);

					if (continueEvaluate)
						//release memory in the evaluator
						eval->ReleaseMemory();
				}
			}
			catch (runtime_error &e)
			{
				printf("runtime error (%s)\n", e.what());
				exit(EXIT_FAILURE);
			}

			//Save the results if the output parameters was provided
			if (!this->OutputFile.empty())
				SaveResults(eval);

			// Print results
			PrintSolution(eval, Verbose);

			std::cout << "================================================================================================" << std::endl << std::endl;

			//release memory in the evaluator
			eval->ReleaseMemory();

		}
	}
	else
	{
		// pointer to 2D array
		graph = static_cast<int*>(malloc(memSize));

		// Generate a random graph
		GenerateRandomGraph(graph, eval->Vertices, eval->Density, RANGE);

		////////////////////////////////////////////////////////////
		// FOR RESULT COMPARISON TESTING ONLY!
		////////////////////////////////////////////////////////////
		//int adjMatrix[4][4] =
		//{
		//	{ 		0, INF,		 -2, INF },
		//	{ 		4, 		0, 		 3, INF },
		//	{ INF, INF, 	 0,		2 },
		//	{ INF, 		-1, INF, 		0 }
		//};

		//given adjacency representation of matrix
		//memcpy(graph, adjMatrix, memSize);
		////////////////////////////////////////////////////////////
		// FOR RESULT COMPARISON TESTING ONLY!
		////////////////////////////////////////////////////////////

		if (Verbose)
		{
			std::cout << "Initial graph" << std::endl;
			for (unsigned int v = 0; v < eval->Vertices; v++)
			{
				std::cout << "[" << v << "]: ";
				for (unsigned int e = 0; e < eval->Vertices; e++)
				{
					if (graph[v * eval->Vertices + e] == INF)
						std::cout << setw(5) << "INF";
					else
						printf("%5d", graph[v * eval->Vertices + e]);
				}
				std::cout << std::endl;
			}
		}

		/////////////////////////////////////////////////////////
		// Iterate thru all algorithms and evaluate the results
		/////////////////////////////////////////////////////////
		for (auto& algorithm : algorithms)
		{
			//Initialize the compute arrays with a new graph
			eval->InitArrays(graph);

			// Execute the algorithm
			this->elapsedTime_ = algorithm(eval);

			//Evaluate the results
			if (this->elapsedTime_ > 0)
				EvaluateAlgorithm(eval, this->elapsedTime_);
			//if (this->elapsedTime_ == 0 || !EvaluateAlgorithm(eval, this->elapsedTime_))
			//{
			//	std::cerr << "Error: Algorithm " << eval->Name << " " << eval->Host->Name << " has failed." << std::endl;
			//	pass_ = false;
			//}


			if (this->elapsedTime_ > 0 && pass_)
			{
				//Save the results if the output parameters was provided
				if (!this->OutputFile.empty())
					SaveResults(eval);

				// Print results
				PrintSolution(eval, Verbose);
			}
			else
				pass_ = false;

			//Release memory
			eval->ReleaseMemory();

			std::cout << "--------------------------------------------------" << std::endl;

		}

	}


	return pass_;
}






//////////////////////////////////////////////////////////
// Evaluate the resuts from running the algorithm
bool Test::EvaluateAlgorithm(Evaluator* eval, float elapsedTime)
{
	// Display results
	const bool showInSec = elapsedTime >= 1000;
	const char* hostname = eval->Host->Name.c_str();
	const unsigned int targetTimeMs = eval->TargetTime * 1000;
	printf("Vertices: %d, %s Elapsed Time:     %.2f %s\n", eval->Vertices, hostname, (showInSec) ? elapsedTime / 1000 : elapsedTime, showInSec ? "seconds" : "milliseconds");

	// Check result
	if (elapsedTime > targetTimeMs)
	{
		printf("computed result took %.2f seconds, which does not match the expected return time of %d seconds.\n\n", elapsedTime, targetTimeMs);
		return false;
	}

	if (elapsedTime > 0.001)
	{
		if (showInSec)
			printf("Algorithm: %s, Performance = %.2f vertices/sec, Time = %.2f(sec), Processor = %s\n",
				eval->Name.c_str(), eval->Vertices * 1000 / elapsedTime, elapsedTime / 1000.0f, hostname);
		else
			printf("Algorithm: %s, Performance = %.2f vertices/sec, Time = %d(ms), Processor = %s\n",
				eval->Name.c_str(), eval->Vertices * 1000 / elapsedTime, static_cast<unsigned int>(elapsedTime), hostname);
	}
	else
		printf("Algorithm: %s, Performance ~ unlimited, Time = %.2f(ms), Processor = %s\n", eval->Name.c_str(), elapsedTime, hostname);

	std::cout << std::endl;

	return true;
}




/////////////////////////////
// Print to the output file
void Test::SaveResults(Evaluator* eval)
{
	ofstream outFile(OutputFile);
	if (outFile.is_open())
	{
		outFile << eval->Name;
		outFile << " Performance = " << eval->Vertices / elapsedTime_ << " vertices/sec, Time = ";
		if (elapsedTime_ < 1)
		{
			outFile << elapsedTime_ * 1000.0f << "(ms), Processor = " << eval->Name << endl;
		}
		else
		{
			outFile << elapsedTime_ << "(sec), Processor = " << eval->Name << endl;
		}

		for (unsigned int v = 0; v < eval->Vertices; v++)
		{
			for (unsigned int u = 0; u < eval->Vertices; u++)
			{
				if (eval->Cost[v * eval->Vertices + u] == INT_MAX)
					outFile << setw(5) << "INF";
				else
					outFile << setw(5) << eval->Cost[v * eval->Vertices + u];
			}
			outFile << endl;
		}

		outFile << endl;

		for (unsigned int v = 0; v < eval->Vertices; v++)
		{
			for (unsigned int u = 0; u < eval->Vertices; u++)
			{
				if (u != v && eval->Path[v * eval->Vertices + u] != -1)
				{
					outFile << "Shortest Path from vertex " << v <<
						" to vertex " << u << " is (" << v << " ";
					PrintPath(eval->Path, v, u, eval->Vertices);
					outFile << u << ") with the cost of: " << eval->Cost[v * eval->Vertices + u] << endl;
				}
			}
		}

		outFile.close();
	}
	else
		cout << "Unable to open file";
}




/////////////////////////////
// Save performance metrics
void Test::SavePerformanceMetrics(Evaluator* eval, const float elapsedTime)
{
	ofstream outFile;
	string csvData;
	const string metricsFile = eval->Name + "_" + eval->Host->Name + ".csv";

	ifstream file(metricsFile);
	if (IsFileEmpty(file))
		csvData = "Time,Vertices\n0,0\n";
	else
		csvData = std::to_string(elapsedTime) + "," + std::to_string(eval->Vertices) + "\n";

	outFile.open(metricsFile, std::ios_base::app);
	outFile << csvData;
	outFile.close();
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Evaluate the elapsed time correlation to the target time and use gradient descent to update the number of vertices
bool Test::IncrementWithGradientDescent(Evaluator* eval, unsigned int* v0, unsigned int* vertices, float elapsedTime)
{
	const unsigned int targetTimeMs = eval->TargetTime * 1000;

	if (elapsedTime == targetTimeMs || targetTimeMs - elapsedTime < targetTimeMs * K_ALPHA)
		return false; // the target has been reached!

	// Determine the gradient descent
	const float delta = targetTimeMs - elapsedTime;
	const auto gradientDescent = static_cast<double>(delta) / targetTimeMs;

	//save the current vertices
	*v0 = *vertices;

	//increment the new vertices with the exponential growth
	*vertices = static_cast<unsigned int>(*v0 * (1.0 + K_STEP_RATE * gradientDescent));

	return !(*vertices == *v0); // we reached the maximum gradient descent
}





bool Test::IsFileEmpty(std::ifstream& pFile)
{
	return pFile.peek() == std::ifstream::traits_type::eof();
}


