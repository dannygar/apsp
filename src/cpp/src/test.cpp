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



//////////////////////////////////////////
// Main Test entry
bool Test::RunTest(Evaluator* eval)
{
	pass_ = true;
	unsigned int cpu_time = 0, gpu_time = 0;
	const int memSize = sizeof(int) * eval->Vertices * eval->Vertices;
	std::shared_ptr<int[]> graph;
	std::vector<std::function<float(Evaluator*)>> optimizations = {
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

	if(Verbose)
		std::cout << "Init Counter: " << eval->InitCounter << std::endl;

	if (TargetLoop)
	{
		// store original vertices to start each evalation
		const auto originalVertices = eval->Vertices;
		/////////////////////////////////////////////////////////
		// Iterate thru all algorithm optimizations and evaluate the results
		/////////////////////////////////////////////////////////
		for (auto& optimization : optimizations)
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
					graph = std::shared_ptr<int[]>(new int[graphSize], std::default_delete<int[]>());

					// Generate a random graph
					GenerateRandomGraph(graph, eval->Vertices, eval->Density, RANGE);

					// Initialize Cost and Path arrays
					eval->InitArrays(graph);

					// Execute the algorithm optimization
					this->elapsedTime_ = optimization(eval);

					//Evaluate the results
					EvaluateAlgorithm(eval, this->elapsedTime_);

					// Save the performance metrics values
					SavePerformanceMetrics(eval, this->elapsedTime_);

					if (continueEvaluate)
						//Evaluate the returning time and increment vertices 
						continueEvaluate = IncrementWithGradientDescent(eval, &prev_vertices, &eval->Vertices, this->elapsedTime_);
					else
						continueEvaluate = false;

					//Release memory
					graph.reset();

					if (continueEvaluate)
						//release memory in the evaluator's iterator loop
						eval->ReleaseMemory();
				}
			}
			catch (runtime_error &e)
			{
				std::cout << "runtime error (" << e.what() << ")" << std::endl;
				//release memory in case of the exception
				graph.reset();
				eval->ReleaseMemory();
				throw e;
			}

			//Print the results if the output parameters was provided
			PrintSolution(eval, Verbose, OutputFile);

			std::cout << "================================================================================================" << std::endl << std::endl;

			//release memory in the optimizations iterator loop
			eval->ReleaseMemory();
		}
	}
	else
	{
		try
		{
			/////////////////////////////////////////////////////////
			// Iterate thru all algorithms and evaluate the results
			/////////////////////////////////////////////////////////
			for (auto& optimization : optimizations)
			{
				// pointer to 2D array
				graph = std::shared_ptr<int[]>(new int[memSize], std::default_delete<int[]>());

				if(Fixed) // use the fixed 5 vertices graph
					GenerateFixedMatrix(graph);
				else
					// Generate a random graph
					GenerateRandomGraph(graph, eval->Vertices, eval->Density, RANGE);


				if (Verbose)
				{
					std::cout << "Initial graph" << std::endl;
					for (unsigned int v = 0; v < eval->Vertices; v++)
					{
						std::cout << "[" << v << "]: ";
						for (unsigned int e = 0; e < eval->Vertices; e++)
						{
							if (graph.get()[v * eval->Vertices + e] == (INF))
								std::cout << setw(5) << "INF";
							else
								std::cout << setw(5) << graph.get()[v * eval->Vertices + e];
						}
						std::cout << std::endl;
					}
				}

				//Initialize the compute arrays with a new graph
				eval->InitArrays(graph);

				// Execute the algorithm
				this->elapsedTime_ = optimization(eval);

				//Evaluate the results
				EvaluateAlgorithm(eval, this->elapsedTime_);

				if (pass_)
				{
					//Print the results if the output parameters was provided
					PrintSolution(eval, Verbose, OutputFile);
				}
				else
					pass_ = false;

				//Release memory
				eval->ReleaseMemory();

				// Release memory
				graph.reset();

				std::cout << "--------------------------------------------------" << std::endl;
			}

		}
		catch (runtime_error &e)
		{
			std::cout << "runtime error (" << e.what() << ")" << std::endl;
			//release memory in case of the exception
			graph.reset();
			eval->ReleaseMemory();
			throw e;
		}
	}

	if (Verbose)
		std::cout << "Init Counter: " << eval->InitCounter << std::endl;

	return pass_;
}






//////////////////////////////////////////////////////////
// Evaluate the resuts from running the algorithm
bool Test::EvaluateAlgorithm(Evaluator* eval, float elapsedTime)
{
	// Display results
	const bool showInSec = elapsedTime >= 1000;
	const unsigned int targetTimeMs = eval->TargetTime * 1000;
	std::cout << setprecision(2);
	std::cout << std::fixed;
	std::cout << "Vertices: " << eval->Vertices << ", " << eval->Host.Name <<
		" Elapsed Time: " << ((showInSec) ? elapsedTime / 1000 : elapsedTime) << " "
		<< (showInSec ? "seconds" : "milliseconds") << std::endl;

	// Check result
	if (elapsedTime > targetTimeMs)
	{
		std::cout << "computed result took " << elapsedTime
			<< " seconds, which does not match the expected return time of " << targetTimeMs
			<< " seconds." 
			<< std::endl << std::endl;
		return false;
	}

	if (elapsedTime > 0.001)
	{
		if (showInSec)
			std::cout << "Algorithm: " << eval->Name 
				<< ", Performance = " << eval->Vertices * 1000 / elapsedTime 
				<< " vertices/sec, Time = " << elapsedTime / 1000.0f 
				<< "(sec), Processor = " << eval->Host.Name << std::endl;
		else
			std::cout << "Algorithm: " << eval->Name
				<< ", Performance = " << eval->Vertices * 1000 / elapsedTime
				<< " vertices/sec, Time = " << static_cast<unsigned int>(elapsedTime)
				<< "(ms), Processor = " << eval->Host.Name << std::endl;
	}
	else
		std::cout << "Algorithm: " << eval->Name
			<< ", Performance ~ infinite, Time = " << elapsedTime
			<< "(ms), Processor = " << eval->Host.Name << std::endl;

	std::cout << std::endl;

	return true;
}






/////////////////////////////
// Save performance metrics
void Test::SavePerformanceMetrics(Evaluator* eval, const float elapsedTime)
{
	ofstream outFile;
	string csvData;
	const string metricsFile = eval->Name + "_" + eval->Host.Name + ".csv";

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


