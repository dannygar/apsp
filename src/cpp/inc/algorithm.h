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



struct Processor final
{
	APSPAlgorithm		Type{};
	std::string			Name{};
};



#endif
