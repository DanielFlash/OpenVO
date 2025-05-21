/*Copyright (c) <2024> <OOO "ORIS">
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.*/
#pragma once
#include <vector>
#include <algorithm>
#include <functional>

#include "TypeVOext.h"


/// <summary>
/// Class implementing LMEDS filter
/// </summary>
class LMEDS
{
private:
	Matrix<> bestTransform;
	double bestError = 0;
	bool firstError = false;
	
	int maxIterations;
	int countRandPoints;
public:
	LMEDS() = default;

	LMEDS(int maxIterations, int countRandPoints)
	{
		set(maxIterations, countRandPoints);
	}

	/// <summary>
	/// Set filter parameters
	/// </summary>
	/// <param name="maxIterations">Maximum number of iterations</param>
	/// <param name="countRandPoints">Number of pairs for calculating transformation matrices</param>
	void set(int maxIterations, int countRandPoints)
	{
		this->maxIterations = maxIterations;
		if (countRandPoints < 3)
			this->countRandPoints = 3;
		else
			this->countRandPoints = countRandPoints;
	}

	/// <summary>
	/// Finding the best transformation matrix
	/// </summary>
	/// <param name="src">Unsorted vector of source keypoints</param>
	/// <param name="dst">Unsorted vector of target keypoints</param>
	/// <param name="matches">Match vector</param>
	/// <param name="getTransform">Method for calculating the transformation matrix</param>
	/// <param name="getDelta">Method for calculating the error value of the resulting transformation matrix</param>
	/// <returns>Matrix best transformation matrix. Zero if an error occurred</returns>
	Matrix<> calc(std::vector<Point> src, std::vector<Point> dst, 
		FindTranformMatrixFunc getTransform, CalcDeltaTranformedPointFunc getDelta)
	{
		firstError = false;
		UniqueCombinationGenerator generatePairs(src.size(), countRandPoints);
		size_t countMaxCombination = generatePairs.countMaxCombination();
		if (maxIterations > countMaxCombination)
			maxIterations = countMaxCombination;

		std::vector<Point> randSrc(countRandPoints), randDst(countRandPoints);
		for (int iteration = 0; iteration < maxIterations; iteration++)
		{
			///Generate countRandPoints random matches
			std::vector<int> randMatch = generatePairs.generate();
			if (randMatch.size() == 0)
			{
				std::cout << "There are no more unique combinations left" << std::endl;
				break;
			}

			//We get the coordinates of random points
			size_t iter = 0;
			for (int numberMatch : randMatch)
			{
				randSrc[iter] = src[numberMatch];
				randDst[iter] = dst[numberMatch];
				iter++;
			}

			///We calculate the transformation matrix
			Matrix<> transformMatrix = getTransform(randSrc, randDst);
			if (transformMatrix.sizeRow() == 0)
			{
				std::cout << "Degenerate system of linear equations" << std::endl;
				continue;
			}

			///We calculate the delta by applying the transformation matrix to the original image
			std::vector<double> delta = getDelta(src, dst, transformMatrix);
			if (delta.size() == 0)
			{
				std::cout << "Error calculate delta points after apply affine" << std::endl;
				continue;
			}

			double medianError = 0;
			///We apply the filtration method

			std::sort(delta.begin(), delta.end());
			if (delta.size() % 2 == 0)
				medianError = delta[delta.size() / 2];
			else
				medianError = delta[delta.size() % 2];

			//Compare with the best result
			if (!firstError || medianError < bestError)
			{
				firstError = true;
				bestError = medianError;
				bestTransform = transformMatrix;
			}
		}

		return bestTransform;
	}

	/// <summary>
	/// Finding the best transformation matrix
	/// </summary>
	/// <param name="src">Unsorted vector of source keypoints</param>
	/// <param name="dst">Unsorted vector of target keypoints</param>
	/// <param name="matches">Match vector</param>
	/// <param name="getTransform">Method for calculating the transformation matrix</param>
	/// <param name="getDelta">Method for calculating the error value of the resulting transformation matrix</param>
	/// <returns>Matrix best transformation matrix. Zero if an error occurred</returns>
	Matrix<> calc(std::vector<Point> src, std::vector<Point> dst, std::vector<Match> matches,
		FindTranformMatrixFunc getTransform, CalcDeltaTranformedPointNoSortFunc getDelta)
	{
		firstError = false;
		UniqueCombinationGenerator generatePairs(matches.size(), countRandPoints);
		size_t countMaxCombination = generatePairs.countMaxCombination();
		if (maxIterations > countMaxCombination)
			maxIterations = countMaxCombination;

		Points randSrc(countRandPoints), randDst (countRandPoints);
		for (int iteration = 0; iteration < maxIterations; iteration++)
		{
			///Generate countRandPoints random matches
			std::vector<int> randMatch = generatePairs.generate();
			//We get the coordinates of random points
			size_t iter = 0;
			for (int numberMatch : randMatch)
			{
				randSrc[iter] = src[matches[numberMatch].src];
				randDst[iter] = dst[matches[numberMatch].dst];
				iter++;
			}

			//We calculate the transformation matrix
			Matrix<> transformMatrix = getTransform(randSrc, randDst);
			if (transformMatrix.sizeRow() == 0)
			{
				std::cout << "Degenerate system of linear equations" << std::endl;
				continue;
			}

			//We calculate the delta by applying the transformation matrix to the original image
			std::vector<double> delta = getDelta(src, dst, matches, transformMatrix);
			if (delta.size() == 0)
			{
				std::cout << "Error calculate delta points after apply affine" << std::endl;
				continue;
			}

			double medianError = 0;
			//We apply the filtration method

			std::sort(delta.begin(), delta.end());
			if (delta.size() % 2 == 0)
				medianError = delta[delta.size() / 2];
			else
				medianError = delta[delta.size() % 2];

			//Compare with the best result
			if (!firstError || medianError < bestError)
			{
				firstError = true;
				bestError = medianError;
				bestTransform = transformMatrix;
			}
		}

		return bestTransform;
	}
};
