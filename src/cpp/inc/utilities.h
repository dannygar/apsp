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


#ifndef UTILITIES_H
#define UTILITIES_H
#include <cstdlib>
#include <cstdio>
#include <string>
#include <helper_math.h>
#include <helper_timer.h>


#include "inc/evaluator.h"
/// Generate a random matrix.
//
// Parameters:
// int *mat - pointer to the generated matrix. mat should have been 
//            allocated before calling this function.
// const int N - number of vertices.
void GenerateRandomGraph(int *graph, int vertices, int density, int range);

bool InitMMTimer(UINT wTimerRes);
void DestroyMMTimer(UINT wTimerRes, bool init);
void PrintMatrix(int* matrix, unsigned int vertices);
void PrintSolution(Evaluator* eval, bool verbose);
void PrintPath(int *path, int vertex, int edge, unsigned int vertices);

/// Compare the content of two integer arrays. Return true if they are
// exactly the same; otherwise return false.
//
// Parameters:
// const int *l, const int *r - the two integer arrays to be compared.
// const int eleNum - the length of the two matrices.
bool CmpArray(const int *l, const int *r, const size_t eleNum);


#endif
