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
#include <cstdlib>
#include <string>
#include <iostream>

#include "inc/evaluator.h"


Evaluator::Evaluator(unsigned vertices, unsigned density, unsigned device, 
	unsigned targetTime, std::string name, int threads)
{
	Vertices = vertices == 0 ? K_VERTICES_MIN : vertices;
	Density = density;
	Device = device;
	TargetTime = targetTime;
	Name = name;
	InitCounter = 0;
	Threads = threads;
}

Evaluator::Evaluator(const Evaluator& other):
	Name(other.Name),
	Host(other.Host),
	Vertices(other.Vertices),
	Density(other.Density),
	Device(other.Device),
	TargetTime(other.TargetTime),
	Threads(other.Threads)
{
	const int memSize = other.Vertices * other.Vertices * sizeof(int);
	std::unique_ptr<int[]> p1(new int[memSize]);
	Cost = std::move(p1);
	for (unsigned int i = 0; i < Vertices; ++i)
	{
		Cost.get()[i * Vertices + i] = other.Cost.get()[i * Vertices + i];
	}

	std::unique_ptr<int[]> p2(new int[memSize]);
	Path = std::move(p2);
	for (unsigned int i = 0; i < Vertices; ++i)
	{
		Path.get()[i * Vertices + i] = other.Path.get()[i * Vertices + i];
	}
}


Evaluator::~Evaluator()
{
	if (InitCounter > 0)
		ReleaseMemory();
}


void Evaluator::InitArrays(std::shared_ptr<int[]> graph)
{
	InitCounter++;
	const int memSize = sizeof(int) * Vertices * Vertices;
	//std:unique_ptr<int[]> p1(new int[memSize]);
	Cost = std::shared_ptr<int[]>(new int[memSize], std::default_delete<int[]>());
	Cost = std::move(graph);
	// Cost = std::move(graph);

	//std:unique_ptr<int[]> p2(new int[memSize]);
	//Path = std::move(p2);
	Path = std::shared_ptr<int[]>(new int[memSize], std::default_delete<int[]>());
	for (unsigned int i = 0; i < Vertices; ++i)
	{
		Path.get()[i * Vertices + i] = -1;
	}
}

void Evaluator::ReleaseMemory()
{
	if (InitCounter > 0)
	{
		Cost.reset();
		Path.reset();
		InitCounter--;
	}
}


/*
 * Copy the results matrices into the instance arrays
 */
void Evaluator::CopyResults(const unique_ptr<APSPGraph>& result)
{
	//Release memory first
	ReleaseMemory();

	//Copy the results into the new pointers
	Cost = std::move(result->graph);
	Path = std::move(result->path);
}



/**
 * Read data from input
 *
 * @param: max value for edges in input graph
 * @result: unique ptr to graph data with allocated fields
 */
unique_ptr<APSPGraph> Evaluator::ReadData(const std::shared_ptr<int[]>& matrix, unsigned int vertices)
{
	unique_ptr<APSPGraph> data = std::make_unique<APSPGraph>(vertices);

	fill_n(data->path.get(), vertices * vertices, 0);
	fill_n(data->graph.get(), vertices * vertices, (INF));

	/* Load data from  the input graph*/
	for (unsigned int i = 0; i < vertices; ++i)
	{
		for (unsigned int j = 0; j < vertices; ++j)
		{
			data->graph[i * vertices + j] = matrix.get()[i * vertices + j];
			data->path[i * vertices + j] = i;
		}
	}

	/* Path from vertex v to vertex v is 0 */
	for (unsigned int i = 0; i < vertices; ++i) {
		data->graph[i * vertices + i] = 0;
		data->path[i * vertices + i] = 0;
	}
	return data;
}

