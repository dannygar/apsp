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

#ifndef ALGORITHM_H
#define ALGORITHM_H
#include <string>

/* ASPS algorithms types */
typedef enum {
	NAIVE_FW = 0,
	CUDA_NAIVE_FW = 1,
	CUDA_COALESCED_FW = 2,
	CUDA_SHARED_MEM_FW = 3,
	CUDA_BLOCKED_FW = 4
} APSPAlgorithm;



class Processor final
{
public:

	explicit Processor(APSPAlgorithm type)
	{
		Type = type;
		if (Type == NAIVE_FW)
			Name = "CPU";
		else if (Type == CUDA_NAIVE_FW)
			Name = "CUDA Linear Optimization";
		else if (Type == CUDA_COALESCED_FW)
			Name = "CUDA Coalesced Memory Optimization";
		else if (Type == CUDA_SHARED_MEM_FW)
			Name = "CUDA Shared Memory Optimization";
		else if (Type == CUDA_BLOCKED_FW)
			Name = "CUDA Blocked Memory Optimization";
		else
			Name = "Unknown";
	}

	APSPAlgorithm		Type{};
	std::string			Name{};
};



#endif
