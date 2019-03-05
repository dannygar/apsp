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

#ifndef TEST_H
#define TEST_H
#include <string>
#include "evaluator.h"



class Test final
{
private:
	bool	pass_;
	float	elapsedTime_{};
	int		algorithmCount_;

	public:
		Test() : pass_(false) {};

		~Test();

		int				Device{};
		bool			IsGpu{};
		bool			Verbose{};
		bool			TargetLoop{};
		std::string		OutputFile{};


		bool RunTest(Evaluator* eval);
		void SaveResults(Evaluator* eval);

		static bool EvaluateAlgorithm(Evaluator* eval, float elapsedTime);
		static void SavePerformanceMetrics(Evaluator* eval, const float elapsedTime);
		static bool IncrementWithGradientDescent(Evaluator* eval, unsigned int* v0, unsigned int* vertices, float elapsedTime);
		static bool IsFileEmpty(std::ifstream& pFile);
};

// Defaults are arbitrary to give sensible runtime
#define TARGET_TIME			2 //seconds
#define	INF					INT_MAX
#define RANGE				100
#define K_STEP_RATE			0.20	// incremental step rate 
#define K_ALPHA				0.02	// the boundary limits of the vertices increment
#define K_DISPLAY_ROWS		5		// number of rows to skip when displaying the results, if verbose flag is not present


#endif