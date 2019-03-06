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

#ifndef FLOYDWARSHALL_H
#define FLOYDWARSHALL_H

#include "evaluator.h"
using namespace std;


class FloydWarshall final : public Evaluator
{
public:
	/**
	 * Constructor
	 * \param vertices
	 * \param density
	 * \param device
	 * \param targetTime
	 */
	FloydWarshall(unsigned int vertices, unsigned int density, unsigned int device, unsigned int targetTime) :
		Evaluator(vertices, density, device, targetTime, "Floyd-Warshall")
	{
	}

	// Static pointers to the evaluator's entry points
	static float RunNaiveFW(Evaluator* eval);
	static float RunCudaFW(Evaluator* eval);
	static float RunCudaFWCoa(Evaluator* eval);
	static float RunCudaFWShared(Evaluator* eval);
	static float RunCudaFWBlocked(Evaluator* eval);

	/* APSP API to compute all pairs shortest paths in graph,
	 * init graph matrix should be point by graph in data, results will be
	 * store in prep (predecessors) and in graph (value for shortest paths)
	 */
	float ComputeCpuNaive() const;
	float ComputeCudaNaive() const;
	float ComputeCudaCoalescedMem() const;
	float ComputeCudaSharedMem() const;
	float ComputeCudaBlockedMem() const;


};

#endif
