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
#include <chrono>

#include <opencv2/opencv.hpp>

#include "TypeVOext.h"
#include "MathFilters.h"
#include "SolverLinearEquations.h"
#include "math_test.h"


namespace homography_func
{
	/// <summary>
	/// Applying a homography matrix to a keypoint vector
	/// </summary>
	/// <param name="points">Keypoint vector</param>
	/// <param name="H">Homography matrix</param>
	/// <returns>vector<Point>. Zero if the coordinates of the projected point are infinity</returns>
	vector<Point> apply(vector<Point> points, Matrix<> H)
	{
		vector<Point> result;

		for (int32_t i = 0; i < points.size(); i++)
		{
			Point point;
			point.x = (H[0][0] * points[i].x + H[0][1] * points[i].y + H[0][2]) /
				(H[2][0] * points[i].x + H[2][1] * points[i].y + H[2][2]);

			point.y = (H[1][0] * points[i].x + H[1][1] * points[i].y + H[1][2]) /
				(H[2][0] * points[i].x + H[2][1] * points[i].y + H[2][2]);

			if (std::isinf(point.x) || std::isinf(point.y))
				return vector<Point>();

			result.push_back(point);
		}

		return result;
	}

	/// <summary>
	/// For testing the obtained homography matrix
	/// </summary>
	/// <param name="img"></param>
	/// <param name="H"></param>
	/// <param name="additionName"></param>
	void apply(cv::Mat img, Matrix<> H, std::string additionName = "")
	{
		cv::Mat cv_H = MatrixToMat(H);
		cv::Mat transformed_image;
		cv::warpPerspective(img, transformed_image, cv_H, img.size());

		cv::Mat combined;
		cv::hconcat(img, transformed_image, combined);
		cv::imshow("Image comparison. " + additionName, combined);
		cv::waitKey(0);
	}
}

/// <summary>
/// Generate matrix equation for SLAE
/// Equation type Ax=b
/// </summary>
/// <param name="src">Vector of key points on the source image</param>
/// <param name="dst">Vector of key points on the target image</param>
/// <returns>A pair of containers. 1) Matrix of equation Ax. 2) Vector b</returns>
pair<vector<vector<double>>, vector<double>> createMatrixEquation(vector<Point>& src,
	vector<Point>& dst)
{
	size_t size = src.size();
	vector<vector<double>> A;
	vector<double> b;
	///Formation of matrix A
	for (int i = 0; i < size; i++)
	{
		double x = src[i].x;
		double y = src[i].y;
		double _x = dst[i].x;
		double _y = dst[i].y;

		vector<double> row_A1(8);
		row_A1[0] = x;
		row_A1[1] = y;
		row_A1[2] = 1;
		row_A1[3] = 0;
		row_A1[4] = 0;
		row_A1[5] = 0;
		row_A1[6] = -x * _x;
		row_A1[7] = -y * _x;
		b.push_back(_x);
		A.push_back(row_A1);

		vector<double> row_A2(8);
		row_A2[0] = 0;
		row_A2[1] = 0;
		row_A2[2] = 0;
		row_A2[3] = x;
		row_A2[4] = y;
		row_A2[5] = 1;
		row_A2[6] = -x * _y;
		row_A2[7] = -y * _y;
		b.push_back(_y);
		A.push_back(row_A2);
	}

	return pair<vector<vector<double>>, vector<double>>{A, b};
}

/// <summary>
/// Generate matrix equation for SLAE
/// Equation type Ah=0
/// </summary>
/// <param name="src">Vector of key points on the source image</param>
/// <param name="dst">Vector of key points on the target image</param>
/// <returns>Single-row matrix of equation Ah</returns>
vector<double> createMatrixEquationSingular(vector<Point>& src, vector<Point>& dst)
{
	size_t rows = 2 * src.size();
	size_t cols = 9;
	vector<double> A(rows * cols, 0.0);

	for (int i = 0; i < src.size(); i++)
	{
		double x1 = src[i].x;
		double y1 = src[i].y;
		double x2 = dst[i].x;
		double y2 = dst[i].y;

		int i0 = 2 * i;     // even line
		A[i0 * 9 + 0] = x1;
		A[i0 * 9 + 1] = y1;
		A[i0 * 9 + 2] = 1.0;
		A[i0 * 9 + 3] = 0.0;
		A[i0 * 9 + 4] = 0.0;
		A[i0 * 9 + 5] = 0.0;
		A[i0 * 9 + 6] = -x2 * x1;
		A[i0 * 9 + 7] = -x2 * y1;
		A[i0 * 9 + 8] = -x2;

		int i1 = 2 * i + 1; // odd
		A[i1 * 9 + 0] = 0.0;
		A[i1 * 9 + 1] = 0.0;
		A[i1 * 9 + 2] = 0.0;
		A[i1 * 9 + 3] = x1;
		A[i1 * 9 + 4] = y1;
		A[i1 * 9 + 5] = 1.0;
		A[i1 * 9 + 6] = -y2 * x1;
		A[i1 * 9 + 7] = -y2 * y1;
		A[i1 * 9 + 8] = -y2;
	}

	return A;
}

/// <summary>
/// Solve SLAE with 4 unknowns using LU method
/// </summary>
/// <param name="src">Vector of key points on the source image</param>
/// <param name="dst">Vector of key points on the target image</param>
/// <returns>Homography matrix. Empty if calculation error occurred</returns>
Matrix<> calc(vector<Point>& src, vector<Point>& dst)
{
	pair<vector<vector<double>>, vector<double>> equation = createMatrixEquation(src, dst);
	///SLAU solution
	LuSolver solver;
	vector<double> x = solver.solve(equation.first, equation.second);
	// Check if the system has a solution
	if (x.size() == 0)
	{
		cout << "The system of equations is degenerate and has no unique solution." << std::endl;
		return Matrix<>(0, 0);
	}

	///Formation of the homography matrix
	Matrix<> H(3, 3);
	for (int i = 0; i < 8; i++)
	{
		H[i / 3][i % 3] = x[i];
	}
	H[2][2] = 1.0;

	return H;
}

/// <summary>
/// Solve SLAE with > 4 unknowns using SVD
/// </summary>
/// <param name="src">Vector of key points on the source image</param>
/// <param name="dst">Vector of key points on the target image</param>
/// <returns>Matrix<double>Homography matrix. Empty if a calculation error occurred</returns>
Matrix<> calcSingular(vector<Point> src, vector<Point> dst)
{
	size_t countRowEquation = src.size() * 2;
	vector<double> equation = createMatrixEquationSingular(src, dst);
	SvdSolver solver;
	vector<double> x = solver.solveHomogeneous(countRowEquation, 9, equation);
	// Checking if the system has a solution
	if (x.size() == 0)
	{
		std::cout << "The system of equations is degenerate and has no unique solution.." << std::endl;
		return Matrix<>(0, 0);
	}

	///Formation of the homography matrix
	Matrix<> H(3, 3);
	for (int i = 0; i < 9; i++)
	{
		H[i / 3][i % 3] = x[i] / x[8];
	}

	return H;
}

/// <summary>
/// Function for calculating the homography matrix from a set of point pairs
/// </summary>
/// <param name="src">Key points on the source image</param>
/// <param name="dst">Vector of key points on the target image</param>
/// <returns>Matrix<double> Homography matrix. Empty if a calculation error occurred</returns>
Matrix<double> findMatrix(vector<Point> src, vector<Point> dst)
{
	if (src.size() == 4)
		return calc(src, dst);
	else
		return calcSingular(src, dst);
}

/// <summary>
/// Calculate the delta between the coordinates of the projection and target keypoints
/// </summary>
/// <param name="src">Vector of projected keypoints</param>
/// <param name="dst">Vector of target keypoints</param>
/// <param name="H">homography matrix</param>
/// <returns></returns>
vector<double> calcDeltaPoint(Points src, Points dst, Matrix<> H)
{
	Points projection = homography_func::apply(src, H);
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
/// Calculate the delta between the coordinates of the projection and target keypoints
/// </summary>
/// <param name="src">Vector of projected keypoints</param>
/// <param name="dst">Vector of target keypoints</param>
/// <param name="H">homography matrix</param>
/// <returns></returns>
vector<double> calcDeltaMatch(Points src, Points dst, Matches matches, Matrix<> H)
{
	Points projection = homography_func::apply(src, H);
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
/// Method for calculating homography from two vectors of unordered sets of keypoints
/// and a Match vector
/// </summary>
/// <param name="src">Vector of keypoints on the source image</param>
/// <param name="dst">Vector of keypoints on the target image</param>
/// <param name="matches">Match vector</param>
/// <param name="method">Filtering method</param>
/// <param name="testData"></param>
/// <returns>Homography matrix. Null if no error occurred</returns>
Matrix<> findHomography(vector<Point> src, vector<Point> dst, Filters method,
	double threshold, int maxIteration = 1000, int countRandomPoints = 4)
{
	if (src.size() < 4 || src.size() != dst.size())
	{
		throw "Count points less min need for homography";
		return Matrix<>(0, 0);
	}

	if (threshold <= 0)
		threshold = 0.85;

	if (maxIteration <= 0)
		maxIteration = 1000;

	if (countRandomPoints < 4)
		countRandomPoints = 4;

	Matrix<> H;
	if (method == Filters::RANSAC)
	{
		RANSAC ransac(threshold, maxIteration, countRandomPoints);
		H = ransac.calc(src, dst, findMatrix, calcDeltaPoint);
	}
	else if (method == Filters::LMEDS)
	{
		LMEDS lmeds(maxIteration, countRandomPoints);
		H = lmeds.calc(src, dst, findMatrix, calcDeltaPoint);
	}

	return H;
}


/// <summary>
/// Homography calculation method based on two vectors of unordered keypoint sets and a Match vector
/// </summary>
/// <param name="src">Vector of keypoints on the source image</param>
/// <param name="dst">Vector of keypoints on the target image</param>
/// <param name="matches">Vector of matches of detected keypoints</param>
/// <param name="method">Filtration method</param>
/// <param name="threshold">Maximum permissible reprojection error when searching for the homography matrix</param>
/// <param name="maxIteration">Maximum number of iterations of searching for the homography matrix</param>
/// <param name="countRandomPoints">Number of used keypoint pairs in one iteration</param>
/// <returns></returns>
Matrix<> findHomography(vector<Point> src, vector<Point> dst, vector<Match> matches,
	Filters method, double threshold, int maxIteration = 1000, int countRandomPoints = 4)
{
	if (src.size() < 4 || src.size() != dst.size() || src.size() != matches.size())
		return Matrix<>(0, 0);

	if (threshold <= 0)
		threshold = 0.85;

	if (maxIteration <= 0)
		maxIteration = 1000;

	if (countRandomPoints < 4)
		countRandomPoints = 4;

	Matrix<> H;

	if (method == Filters::RANSAC)
	{
		RANSAC ransac(threshold, maxIteration, countRandomPoints);
		H = ransac.calc(src, dst, matches, findMatrix, calcDeltaMatch);
	}
	else if (method == Filters::LMEDS)
	{
		LMEDS lmeds(maxIteration, countRandomPoints);
		H = lmeds.calc(src, dst, matches, findMatrix, calcDeltaMatch);
	}

	return H;
}
