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
#include <iostream>

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
		if (InitCounter > 0)
			ReleaseMemory();
	};

	Evaluator(unsigned int vertices, unsigned int density, unsigned int device, unsigned int targetTime, std::string name)
	{
		Vertices = vertices == 0 ? K_VERTICES_MIN : vertices;
		Density = density;
		Device = device;
		TargetTime = targetTime;
		Name = name;
		InitCounter = 0;
	}

	Evaluator(const Evaluator& other) :
		Name(other.Name), 
		Vertices(other.Vertices), 
		Density(other.Density), 
		Device(other.Device), 
		TargetTime(other.TargetTime),
		Host(other.Host)
	{
		Cost = new int[other.Vertices * other.Vertices * sizeof(int)];
		*Cost = *other.Cost;

		Path = new int[other.Vertices * other.Vertices * sizeof(int)];
		*Path = *other.Path;

	}

	//Array initialization
	void InitArrays(int* graph)
	{
		InitCounter++;
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

	void ReleaseMemory()
	{
		if (InitCounter > 0)
		{
			free(Cost);
			free(Path);
			InitCounter--;
		}
	}

	string	Name{};
	Processor Host;


	// cost[] and parent[] stores shortest-path 
	// (shortest-cost/shortest route) information
	// TODO: Replaced with the smart pointers
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


	int				InitCounter{};

};


#endif

