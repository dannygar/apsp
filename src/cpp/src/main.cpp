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

 ///////////////////////////////////////////////////////////////////////////////
 // Shortest Path First Algorithm: Find the shortest path between nodes in a graph
 // ========================
 //
 // This sample demonstrates various implementation techniques for how to 
 // find a shortest path between nodes in a DAG (directed acyclic graph)
 //
 // This file, main.cpp, contains the setup information to run the tests
 ///////////////////////////////////////////////////////////////////////////////


#define _CRTDBG_MAP_ALLOC
#include <cstdlib>
#include <crtdbg.h>

#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <conio.h>

// helper functions and utilities to work with CUDA
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_string.h>

// project headers
#include "../inc/test.h"
#include "inc/floydwarshall.h"


// Forward declarations
void ShowHelp(const int argc, const char **argv);
template <typename T>
void runTest(int argc, const char **argv);

int main(int argc, char **argv)
{
	using std::invalid_argument;
	using std::string;

	// Cause automatic call to display a memory leak report when the app exits
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);

	
	// If help flag is set, display help and exit immediately
	if (checkCmdLineFlag(argc, const_cast<const char **>(argv), "help"))
	{
		printf("Displaying help on console\n");
		ShowHelp(argc, (const char **)argv);
		exit(EXIT_SUCCESS);
	}

	// Check the value for GPU run
	try
	{
		char *value;
		bool isExitPrompt = true;

		if (getCmdLineArgumentString(argc, const_cast<const char **>(argv), "noprompt", &value))
		{
			isExitPrompt = false;
		}


		//cudaError_t cudaResult = cudaSuccess;
		//int deviceCount = 0;

		//check for the processor type (GPU enabled)
		//if (getCmdLineArgumentString(argc, const_cast<const char **>(argv), "processor", &value))
		//{
		//	// Check requested alg is valid
		//	std::string gpu(value);

		//	if (gpu.compare("gpu") == 0 || gpu.compare("\"gpu\"") == 0)
		//	{
		//		// Get number of available devices
		//		cudaResult = cudaGetDeviceCount(&deviceCount);

		//		if (cudaResult != cudaSuccess)
		//		{
		//			printf("could not get GPU Device count (%s).\n", cudaGetErrorString(cudaResult));
		//			throw invalid_argument("cudaGetDeviceCount");
		//		}

		//		IsGpu = true;
		//	}
		//	else if (gpu.compare("cpu") == 0 || gpu.compare("\"cpu\"") == 0)
		//	{
		//		IsGpu = false;
		//	}
		//	else
		//	{
		//		printf("specified processor (%s) is invalid, must be \"cpu\" or \"gpu\".\n", value);
		//		throw invalid_argument("processor");
		//	}
		//}


		// check algorithm
		if (getCmdLineArgumentString(argc, const_cast<const char **>(argv), "algorithm", &value))
		{
			// Check requested alg is valid
			string alg(value);

			if (alg.compare("floyd") == 0 || alg.compare("\"Floyd\"") == 0)
			{
				//if (IsGpu)
				//	runTest<FloydWarshallGPU>(argc, const_cast<const char **>(argv));
				//else
					runTest<FloydWarshall>(argc, const_cast<const char **>(argv));
			}
			else
			{
				printf("specified algorithm (%s) is not implemented.\n", value);
				throw invalid_argument("algorithm");
			}
		}

		if (isExitPrompt)
		{
			std::string s;
			std::cout << "Press any key to exit...";
			_getch();
		}
	}
	catch (invalid_argument &e)
	{
		printf("invalid command line argument (%s)\n", e.what());
		exit(EXIT_FAILURE);
	}

	// Finish
	exit(EXIT_SUCCESS);
}



template <typename T>
void runTest(int argc, const char **argv)
{
	using std::invalid_argument;
	using std::runtime_error;
	

	try
	{
		auto test = new Test();
		int deviceCount = 0;
		unsigned int vertices;
		unsigned int density;
		unsigned int targetTime = TARGET_TIME;

		char *value = 0;

		if (getCmdLineArgumentString(argc, argv, "vertices", &value))
		{
			vertices = static_cast<unsigned int>(atoi(value));
			std::cout << "number of vertices = " << vertices << std::endl;

			if (vertices < 1)
			{
				printf("specified number of vertices (%d) is invalid, must be greater than 0.\n", vertices);
				throw invalid_argument("sims");
			}
		}
		else
		{
			vertices = 0; // will be incremented gradually until the target time is met
			test->TargetLoop = true;
		}

		if (getCmdLineArgumentString(argc, argv, "density", &value))
		{
			density = static_cast<unsigned int>(atoi(value));
			printf("edge density = %d\n", density);

			if (density < K_DENSITY_MIN || density > K_DENSITY_MAX)
			{
				printf("specified edges density (%d) is invalid, must be between %d and %d.\n", density, K_DENSITY_MIN, K_DENSITY_MAX);
				throw invalid_argument("sims");
			}
		}
		else
		{
			density = K_DENSITY_DEF;
		}


		if (getCmdLineArgumentString(argc, argv, "output", &value))
		{
			test->OutputFile = value;
			std::cout << "Output file = " << test->OutputFile << std::endl;
		}
		else
		{
			test->OutputFile = "";
		}

		if (getCmdLineArgumentString(argc, argv, "verbose", &value))
		{
			test->Verbose = true;
			std::cout << "verbose output = true" << std::endl;
		}


		//Obtain the target time of the execution
		if (getCmdLineArgumentString(argc, argv, "target", &value))
		{
			targetTime = static_cast<unsigned int>(atoi(value));

			if (targetTime < 1)
			{
				printf("invalid target time specified on command line (target time must be provided in seconds and greater than 0).\n");
				throw invalid_argument("Target");
			}
		}


		// Get number of available devices
		const cudaError_t cudaResult = cudaGetDeviceCount(&deviceCount);
		if (cudaResult == cudaSuccess)
		{
			test->IsGpu = true;
		}
		else
		{
			printf("could not get Device count (%s). GPU tests will be skipped.\n", cudaGetErrorString(cudaResult));
			test->IsGpu = false;
		}


		if (test->IsGpu)
		{
			if (getCmdLineArgumentString(argc, argv, "device", &value))
			{
				test->Device = (int)atoi(value);

				if (test->Device >= deviceCount)
				{
					printf("invalid target Device specified on command line (Device %d does not exist).\n", test->Device);
					throw invalid_argument("Device");
				}
			}
			else
			{
				// This will pick the best possible CUDA capable Device, otherwise
				// override the Device ID based on input provided at the command line
				test->Device = findCudaDevice(argc, static_cast<const char **>(argv));
			}
		}

		

		//Instantiate the Algorithm evaluator
		auto eval = new T(vertices, density, test->Device, targetTime);

		// Execute evaluation
		test->RunTest(eval);
	}
	catch (invalid_argument &e)
	{
		printf("invalid command line argument (%s)\n", e.what());
		exit(EXIT_FAILURE);
	}
	catch (runtime_error &e)
	{
		printf("runtime error (%s)\n", e.what());
		exit(EXIT_FAILURE);
	}
}



void ShowHelp(int argc, const char **argv)
{
	using std::cout;
	using std::endl;
	using std::left;
	using std::setw;

	if (argc > 0)
	{
		cout << endl << argv[0] << endl;
	}

	cout << endl << "Syntax:" << endl;
	cout << left;
	cout << "    " << setw(20) << "--algorithm=<name>" << "Specify the algorithm to be evaluated" << endl;
	cout << "    " << setw(20) << "--vertices=<N>" << "Specify number of vertices for the auto-generated graph to execute" << endl;
	cout << "    " << setw(20) << "--density=<N>" << "Specify density from 0 to 100 of the edges in the auto-generated graph to execute" << endl;
	cout << "    " << setw(20) << "--target=<N>" << "Specify target time (in seconds) for the test to run" << endl;
	cout << "    " << setw(20) << "--device=<N>" << "Specify GPU Device to use for execution" << endl;
	cout << "    " << setw(20) << "--output=<filename>" << "Specify the output file name for the results" << endl;
	cout << "    " << setw(20) << "--verbose" << "Turn verbose output on" << endl;
	cout << endl;
	cout << "    " << setw(20) << "--noprompt" << "Skip prompt before exit" << endl;
	cout << endl;
}
