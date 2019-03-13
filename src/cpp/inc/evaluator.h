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
#pragma once

#ifndef EVALUATOR_H
#define EVALUATOR_H

#include <string>
#include "algorithm.h"

using namespace std;

// Defaults are arbitrary to give sensible runtime
#define K_VERTICES_MIN	500
#define K_DENSITY_MIN	0
#define K_DENSITY_MAX	100
#define K_DENSITY_DEF	25

/* Maximum distance value for path form
 * vertex x to vertex y means no path v1 -> v2.
 * This value should be MAX_INT / 2 - 1
 * because we should be able to compare path v1 -> v2 with
 * path v1 -> u -> v2 so adding to value of paths v1 -> u and
 * u -> v2 should be smaller than maximum int value
 */
#define	INF					(1 << 30 - 1)

/* Default structure for graph */
struct APSPGraph {
	unsigned int vertices; // number of vertex in graph
	std::unique_ptr<int[]> path; // predecessors matrix
	std::unique_ptr<int[]> graph; // graph matrix

	/* Constructor for init fields */
	APSPGraph(int size) : vertices(size) {
		const int memSize = sizeof(int) * vertices * vertices;
		path = std::unique_ptr<int[]>(new int[memSize]);
		graph = std::unique_ptr<int[]>(new int[memSize]);
	}
};


class Evaluator
{
public:
	virtual ~Evaluator();

	Evaluator(unsigned int vertices, unsigned int density, unsigned int device, unsigned int targetTime,
	          std::string name, int threads);

	Evaluator(const Evaluator& other);

	//Array initialization
	void InitArrays(std::shared_ptr<int[]> graph);

	void ReleaseMemory();

	void CopyResults(const unique_ptr<APSPGraph>& result);

	static unique_ptr<APSPGraph> ReadData(const std::shared_ptr<int[]>& matrix, unsigned int maxValue);

	string	Name{};
	Processor Host;


	// cost[] and parent[] stores shortest-path 
	// (shortest-cost/shortest route) information
	std::shared_ptr<int[]>	Cost{};
	std::shared_ptr<int[]>	Path{};

	// Vertices Count
	unsigned int	Vertices{};
	// Density Count
	unsigned int	Density{};
	// Device
	unsigned int	Device{};
	//Target time
	unsigned int	TargetTime{};

	// Array initialization counter
	int				InitCounter{};

	// Threads per block
	int				Threads{};
};


#endif

