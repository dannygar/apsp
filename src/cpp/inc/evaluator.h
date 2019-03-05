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

#ifndef EVALUATOR_H
#define EVALUATOR_H

#include <cstdlib>
#include <cstdio>
#include <string>
#include "algorithm.h"

using namespace std;

// Defaults are arbitrary to give sensible runtime
#define K_VERTICES_MIN	500
#define K_DENSITY_MIN	0
#define K_DENSITY_MAX	100
#define K_DENSITY_DEF	25


class Evaluator
{
public:
	virtual ~Evaluator()
	{
		free(Cost);
		free(Path);
	};

	Evaluator(unsigned int vertices, unsigned int density, unsigned int device, unsigned int targetTime)
	{
		Vertices = vertices == 0 ? K_VERTICES_MIN : vertices;
		Density = density;
		Device = device;
		TargetTime = targetTime;
	}

	//Array initialization
	void InitArrays(int* graph)
	{
		const int memSize = sizeof(int) * Vertices * Vertices;
		Cost = static_cast<int*>(malloc(memSize));
		Path = static_cast<int*>(malloc(memSize));
		memcpy(Cost, graph, memSize);
		memcpy(Path, graph, memSize);

		for (unsigned int i = 0; i < Vertices; ++i)
		{
			Path[i * Vertices + i] = -1;
		}
	}

	void ReleaseMemory() const
	{
		free(Cost);
		free(Path);
	}

	string	Name{};
	Processor* Host{};

	virtual float ComputeCPUNaive() { return 0; };
	virtual float ComputeCudaNaive() { return 0; };

	

	// cost[] and parent[] stores shortest-path 
	// (shortest-cost/shortest route) information
	int*	Cost{};
	int*	Path{};

	// Vertices Count
	unsigned int	Vertices{};
	// Density Count
	unsigned int	Density{};
	// Device
	unsigned int	Device{};
	//Target time
	unsigned int	TargetTime{};


};


#endif

