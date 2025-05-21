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
#include <algorithm>
#include <functional>
#include "cmath"

#include "TypeVOext.h"
#include "GeneratePair.h"
#include "ExtendMatrix.h"


/// <summary>
/// RANSAC filter class
/// </summary>
class RANSAC
{
private:
	Matrix<> bestTransform;
	int bestInlines = -1;

	double threshold;
	int maxIterations;
	int countRandPoints;

public:
	RANSAC() = default;

	RANSAC(double threshold, int maxIteration, int countRandPoints)
	{
		set(threshold, maxIteration, countRandPoints);
	}

	/// <summary>
	/// Set filter parameters
	/// </summary>
	/// <param name="threshold">Maximum allowed reprojection error for treating a point pair as a source</param>
	/// <param name="maxIterations">Maximum number of iterations</param>
	/// <param name="countRandPoints">Number of keypoint pairs to calculate the transformation matrix</param>
	void set(double threshold, int maxIterations, int countRandPoints)
	{
		this->threshold = threshold;
		this->maxIterations = maxIterations;
		if (countRandPoints < 3)
			this->countRandPoints = 3;
		else
			this->countRandPoints = countRandPoints;
	}

	/// <summary>
	/// Calculate the best transformation matrix given two unsorted keypoint vectors and a Match vector
	/// </summary>
	/// <param name="src">Unsorted source keypoint vector</param>
	/// <param name="dst">Unsorted target keypoint vector</param>
	/// <param name="matches">Match vector</param>
	/// <param name="getTransform">Transformation matrix calculation method</param>
	/// <param name="getDelta">Transformation matrix error calculation method</param>
	/// <returns>Matrix best transformation matrix. Zero if an error occurred</returns>
	Matrix<> calc(std::vector<Point> src, std::vector<Point> dst,
		FindTranformMatrixFunc getTransform, CalcDeltaTranformedPointFunc getDelta)
	{
		UniqueCombinationGenerator generatePairs(src.size(), countRandPoints);
		size_t countMaxCombination = generatePairs.countMaxCombination();
		if (maxIterations > countMaxCombination)
			maxIterations = countMaxCombination;

		std::vector<Point> randSrc(countRandPoints), randDst(countRandPoints);
		for (int iteration = 0; iteration < maxIterations; iteration++)
		{
			//Generate countRandPoints of random matches
			std::vector<int> randMatch = generatePairs.generate();
			if (randMatch.size() == 0)
			{
				return Matrix<>(0, 0);
			}

			//Get coordinates of random points
			size_t iter = 0;
			Matches forTest;
			for (int numberMatch : randMatch)
			{
				randSrc[iter] = src[numberMatch];
				randDst[iter] = dst[numberMatch];
				iter++;
			}

			//Calculate the transformation matrix
			Matrix<double> transformMatrix = getTransform(randSrc, randDst);
			if (transformMatrix.sizeRow() == 0)
			{
				std::cout << "Вырожденная СЛАУ" << std::endl;
				continue;
			}
			
			//Calculate the delta by applying the transformation matrix to the original image
			std::vector<double> delta = getDelta(src, dst, transformMatrix);
			if (delta.size() == 0)
			{
				std::cout << "Error calculate delta points after apply homography" << std::endl;
				continue;
			}

			int inlineCount = 0;
			//We apply the filtering method
			for (int i = 0; i < delta.size(); i++)
			{
				if (delta[i] < threshold)
					inlineCount++;
			}

			if (inlineCount > bestInlines)
			{
				bestInlines = inlineCount;
				bestTransform = transformMatrix;
			}
		}

		return bestTransform;
	}

	/// <summary>
	/// Calculate the best transformation matrix given two unsorted keypoint vectors and a Match vector
	/// </summary>
	/// <param name="src">Unsorted source keypoint vector</param>
	/// <param name="dst">Unsorted target keypoint vector</param>
	/// <param name="matches">Match vector</param>
	/// <param name="getTransform">Transformation matrix calculation method</param>
	/// <param name="getDelta">Transformation matrix error calculation method</param>
	/// <returns>Matrix best transformation matrix. Zero if an error occurred</returns>
	Matrix<> calc(std::vector<Point> src, std::vector<Point> dst, std::vector<Match> matches,
		FindTranformMatrixFunc getTransform, CalcDeltaTranformedPointNoSortFunc getDelta)
	{
		UniqueCombinationGenerator generatePairs(matches.size(), countRandPoints);
		size_t countMaxCombination = generatePairs.countMaxCombination();
		if (maxIterations > countMaxCombination)
			maxIterations = countMaxCombination;

		std::vector<Point> randSrc(countRandPoints), randDst(countRandPoints);
		for (int iteration = 0; iteration < maxIterations; iteration++)
		{
			//Generate countRandPoints of random matches
			std::vector<int> randMatch = generatePairs.generate();

			//Get coordinates of random points
			size_t iter = 0;
			for (int numberMatch : randMatch)
			{
				randSrc[iter] = src[matches[numberMatch].src];
				randDst[iter] = dst[matches[numberMatch].dst];
				iter++;
			}

			//Calculate the transformation matrix
			Matrix<> transformMatrix = getTransform(randSrc, randDst);
			if (transformMatrix.sizeRow() == 0)
			{
				std::cout << "Вырожденная СЛАУ" << std::endl;
				continue;
			}

			//Calculate the delta by applying the transformation matrix to the original image
			std::vector<double> delta = getDelta(src, dst, matches, transformMatrix);
			if (delta.size() == 0)
			{
				std::cout << "Error calculate delta points after apply homography" << std::endl;
				continue;
			}

			int inlineCount = 0;
			//We apply the filtering method
			for (int i = 0; i < delta.size(); i++)
			{
				if (delta[i] < threshold)
					inlineCount++;
			}

			if (inlineCount > bestInlines)
			{
				bestInlines = inlineCount;
				bestTransform = transformMatrix;
			}
		}

		return bestTransform;
	}
};