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
#include <iostream>

#include <opencv2/opencv.hpp>

#include "TypeVOext.h"
#include "MathFilters.h"
#include "SolverLinearEquations.h"
#include "math_test.h"

using namespace std;

namespace affine_func
{
	enum class TypeInterp { NEAREST };

	/// <summary>
	/// Applying an affine matrix to an image.
	/// </summary>
	/// <param name="srcImg">Image matrix</param>
	/// <param name="dstImg">Image matrix to store the result in</param>
	/// <param name="affine">Affine matrix</param>
	/// <param name="interpolation">Interpolation method</param>
	/// <returns>bool</returns>
	bool apply(Matrix<> srcImg, Matrix<>& dstImg, Matrix<> affine, TypeInterp interpolation = TypeInterp::NEAREST)
	{
		size_t sizeY = srcImg.sizeRow();
		size_t sizeX = srcImg.sizeColumn();

		dstImg.resize(sizeY, sizeX);

		Tensor<> transformCoordinate(sizeY, sizeX, 2);
		//image.exportImage(img);

		///Get pixel coordinates after applying affine transformations
		for (size_t i = 0; i < sizeY; i++)
		{
			for (size_t j = 0; j < sizeX; j++)
			{
				transformCoordinate[0][i][j] = affine[0][0] * j + affine[0][1] * i + affine[0][2];
				transformCoordinate[1][i][j] = affine[1][0] * j + affine[1][1] * i + affine[1][2];
			}
		}

		///We apply interpolation
		if (interpolation == TypeInterp::NEAREST)
		{
			for (size_t i = 0; i < sizeY; i++)
			{
				for (size_t j = 0; j < sizeX; j++)
				{
					transformCoordinate[0][i][j] = round(transformCoordinate[0][i][j]);
					transformCoordinate[1][i][j] = round(transformCoordinate[1][i][j]);
				}
			}
		}

		///Form the target image by combining the new coordinates from the tensor and the
		/// intensity value from the original image
		cv::Mat imageTest(sizeY, sizeX, CV_8UC3, cv::Scalar(0, 0, 0));

		for (size_t i = 0; i < sizeY; i++)
		{
			for (size_t j = 0; j < sizeX; j++)
			{
				int row = transformCoordinate[1][i][j];
				int column = transformCoordinate[0][i][j];

				if (row < sizeY && column < sizeX)
				{
					imageTest.at<cv::Vec3b>(row, column) = cv::Vec3b(255, 255, 255);
					dstImg[row][column] = srcImg[i][j]; // We write down the original pixel intensity
				}
			}
		}

		cv::imshow("Image with a modified pixel", imageTest);
		// Waiting for a key to be pressed
		cv::waitKey(0);

		return true;
	}

	/// <summary>
	/// Applying an affine matrix to a vector of keypoints.
	/// </summary>
	/// <param name="points">A vector of keypoints</param>
	/// <param name="affine">Affine matrix</param>
	/// <returns>A vector of points projected according to the affine matrix</returns>
	vector<Point> apply(vector<Point> points, Matrix<> affine)
	{
		vector<Point> result;
		for (int i = 0; i < points.size(); i++)
		{
			Point point;
			point.x = affine[0][0] * points[i].x + affine[0][1] * points[i].y + affine[0][2];
			point.y = affine[1][0] * points[i].x + affine[1][1] * points[i].y + affine[1][2];

			if (std::isinf(point.x) || std::isinf(point.y))
				return vector<Point>();

			result.push_back(point);
		}

		return result;
	}

	/// <summary>
	/// To test the obtained matrix of affine transformations
	/// </summary>
	/// <param name="img"></param>
	/// <param name="affine"></param>
	void apply(cv::Mat img, Matrix<> affine)
	{
		cv::Mat cv_affine = MatrixToMat(affine);
		cv::Mat transformed_image;
		cv::warpAffine(img, transformed_image, cv_affine, img.size());
		cv::Mat combined;
		cv::hconcat(img, transformed_image, combined);
		cv::imshow("Image comparison ", combined);
		cv::waitKey(0);
	}
}

/// <summary>
/// Forming a matrix equation for SLAE
/// A * p = b
/// </summary>
/// <param name="src">Initial key points</param>
/// <param name="dst">Target key points</param>
/// <returns>The equation formed</returns>
pair<vector<vector<double>>, vector<double>> createMatrixEquationAffine(vector<Point> src,
	vector<Point> dst)
{
	size_t points = src.size();
	size_t rows = points * 2, cols = 6;
	vector<vector<double>> A(rows, vector<double>(cols));
	vector<double> b(rows);

	for (int i = 0; i < points; ++i)
	{
		double x1 = src[i].x;
		double y1 = src[i].y;

		double x2 = dst[i].x;
		double y2 = dst[i].y;

		// Equation for x'
		int r1 = static_cast<int>(2 * i);
		A[r1][0] = x1;
		A[r1][1] = y1;
		A[r1][2] = 1;
		A[r1][3] = 0;
		A[r1][4] = 0;
		A[r1][5] = 0;
		b[r1] = x2;

		// Equation for y'
		int r2 = r1 + 1;
		A[r2][0] = 0;
		A[r2][1] = 0;
		A[r2][2] = 0;
		A[r2][3] = x1;
		A[r2][4] = y1;
		A[r2][5] = 1;
		b[r2] = y2;
	}

	return pair<vector<vector<double>>, vector<double>>{A, b};
}

/// <summary>
/// Calculation of the matrix of affine transformations
/// </summary>
/// <param name="matrixEquation">Equation</param>
/// <returns>Matrix affine transformations<</returns>
Matrix<> calcAffine(pair<vector<vector<double>>, vector<double>> matrixEquation)
{
	///Solving SLAU
	LuSolver solver;
	vector<double> x = solver.solve(matrixEquation.first, matrixEquation.second);
	// Checking if the system has a solution
	if (x.size() == 0)
	{
		std::cerr << "The system of equations is degenerate and has no unique solution.." << std::endl;
		return Matrix<>(0, 0);
	}

	Matrix<> affineMatrix(2, 3);
	affineMatrix[0][0] = x[0]; // a
	affineMatrix[0][1] = x[1]; // b
	affineMatrix[0][2] = x[2]; // tx
	affineMatrix[1][0] = x[3]; // c
	affineMatrix[1][1] = x[4]; // d
	affineMatrix[1][2] = x[5]; //ty

	return affineMatrix;
}

/// <summary>
/// Calculation of the matrix of affine transformations for a large number of pairs of key points
/// </summary>
/// <param name="matrixEquation">Equation</param>
/// <returns>Matrix affine transformations</returns>
Matrix<> calcSingularAffine(pair<vector<vector<double>>, vector<double>> matrixEquation)
{
	size_t rows = matrixEquation.first.size(), cols = 6;
	SvdSolver solver;
	vector<double> x = solver.solveDirectLinear(rows, cols, matrixEquation);
	Matrix<> affineMatrix(2, 3);
	affineMatrix[0][0] = x[0]; // a11
	affineMatrix[0][1] = x[1]; // a12
	affineMatrix[0][2] = x[2]; // a13
	affineMatrix[1][0] = x[3]; // a21
	affineMatrix[1][1] = x[4]; // a22
	affineMatrix[1][2] = x[5]; // a23

	return affineMatrix;
}

/// <summary>
/// Obtaining an affine matrix by solving a SLAE
/// </summary>
/// <param name="src">Initial key points</param>
/// <param name="dst">Target key points</param>
/// <returns>Matrix affine transformations</returns>
Matrix<> findMatrixAffine(vector<Point> src, vector<Point> dst)
{
	pair<vector<vector<double>>, vector<double>> matrixEquation =
		createMatrixEquationAffine(src, dst);
	if (src.size() == 3)
		return calcAffine(matrixEquation);
	else
		return calcSingularAffine(matrixEquation);
}

/// <summary>
/// Calculate the coordinate delta between the projection and target keypoint vectors.
/// </summary>
/// <param name="src">Projected keypoint vector</param>
/// <param name="dst">Target keypoint vector</param>
/// <param name="affine">Affine matrix</param>
/// <returns>vector<double> of coordinate delta. Zero vector if infinite point coordinates are found after projection</returns>
vector<double> calcDeltaPointAffine(std::vector<Point> src, std::vector<Point> dst, Matrix<> affine)
{
	std::vector<Point> projection = affine_func::apply(src, affine);
	if (projection.size() == 0)
		return vector<double>();

	vector<double> vectorDelta;
	for (size_t i = 0; i < src.size(); i++)
	{
		vectorDelta.push_back(pow((projection[i].x - dst[i].x), 2) +
			pow((projection[i].y - dst[i].y), 2));
	}

	return vectorDelta;
}

/// <summary>
/// Calculate the coordinate delta between the projection and target keypoint vectors.
/// </summary>
/// <param name="src">Projected keypoint vector</param>
/// <param name="dst">Target keypoint vector</param>
/// <param name="affine">Affine matrix</param>
/// <returns>vector<double> of coordinate delta. Zero vector if infinite point coordinates are found after projection</returns>
vector<double> calcDeltaMatchAffine(std::vector<Point> src, std::vector<Point> dst,
	std::vector<Match> matches, Matrix<> affine)
{
	std::vector<Point> projection = affine_func::apply(src, affine);
	if (projection.size() == 0)
		return vector<double>();

	vector<double> vectorDelta;
	for (size_t i = 0; i < matches.size(); i++)
	{
		vectorDelta.push_back(pow((projection[matches[i].src].x - dst[matches[i].dst].x), 2) +
			pow((projection[matches[i].src].y - dst[matches[i].dst].y), 2));
	}

	return vectorDelta;
}

/// <summary>
/// Calculate the matrix of affine transformations from two unordered keypoint vectors and the matches vector
/// </summary>
/// <param name="src">The keypoint vector of image 1</param>
/// <param name="dst">The keypoint vector of image 2</param>
/// <param name="matches">The Match vector of keypoints between images 1 and 2</param>
/// <param name="method">The filter method</param>
/// <param name="testData">Then delete</param>
/// <returns>Matrix of affine transformations. Zero if an error occurred</returns>
Matrix<> estimateAffinepartial2D(std::vector<Point> src, std::vector<Point> dst,
	Filters method, double threshold, int maxIteration = 1000, int countRandomPoints = 3)
{
	if (src.size() < 3 || src.size() != dst.size())
	{
		return Matrix<>(0, 0);
	}

	if (threshold <= 0)
		threshold = 0.85;

	if (maxIteration <= 0)
		maxIteration = 1000;

	if (countRandomPoints < 3)
		countRandomPoints = 3;

	Matrix<> affine;
	if (method == Filters::RANSAC)
	{
		RANSAC ransac(threshold, maxIteration, countRandomPoints);
		affine = ransac.calc(src, dst, findMatrixAffine, calcDeltaPointAffine);
	}
	else if (method == Filters::LMEDS)
	{
		LMEDS lmeds(maxIteration, countRandomPoints);
		affine = lmeds.calc(src, dst, findMatrixAffine, calcDeltaPointAffine);
	}

	return affine;
}

/// <summary>
/// Calculate the matrix of affine transformations from two unordered keypoint vectors and the matches vector
/// </summary>
/// <param name="src">The keypoint vector of image 1</param>
/// <param name="dst">The keypoint vector of image 2</param>
/// <param name="matches">The Match vector of keypoints between images 1 and 2</param>
/// <param name="method">The filter method</param>
/// <param name="testData">Then delete</param>
/// <returns>The matrix of affine transformations. Zero if an error occurred</returns>
Matrix<> estimateAffinepartial2D(vector<Point> src, vector<Point> dst, vector<Match> matches, Filters method, double threshold, int maxIteration = 1000, int countRandomPoints = 3)
{
	if (src.size() < 3 || src.size() != dst.size() || src.size() != matches.size())
	{
		return Matrix<>(0, 0);
	}

	if (threshold <= 0)
		threshold = 0.85;

	if (maxIteration <= 0)
		maxIteration = 1000;

	if (countRandomPoints < 3)
		countRandomPoints = 3;

	Matrix<> affine;
	if (method == Filters::RANSAC)
	{
		RANSAC ransac(threshold, maxIteration, countRandomPoints);
		affine = ransac.calc(src, dst, matches, findMatrixAffine, calcDeltaMatchAffine);
	}
	else if (method == Filters::LMEDS)
	{
		LMEDS lmeds(maxIteration, countRandomPoints);
		affine = lmeds.calc(src, dst, matches, findMatrixAffine, calcDeltaMatchAffine);
	}

	return affine;
}