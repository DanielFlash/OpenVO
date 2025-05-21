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
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>

#include <opencv2/opencv.hpp> 

#include "ExtendMatrix.h"
#include "TypeVOext.h"
#include "GeneratePair.h"

using namespace std;

static const double EPSILON = 1e-12;

struct DataMatches
{
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    std::vector<cv::DMatch> matches;
};

static cv::Mat getDescriptionPoints(string nameFile)
{
    cv::Mat img1 = cv::imread(nameFile, cv::IMREAD_GRAYSCALE);

    // Initializing the ORB detector object
    cv::Ptr<cv::ORB> orb = cv::ORB::create();

    // Finding Keypoints and Descriptors
    std::vector<cv::KeyPoint> keypoints1;
    cv::Mat descriptors1;
    orb->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);

    return descriptors1;
}

static vector<Point> convertCvKeypointToPoint(std::vector<cv::KeyPoint> keypoints)
{
    vector<Point> points;
    for (int i = 0; i < keypoints.size(); i++)
    {
        Point point;
        point.x = keypoints[i].pt.x;
        point.y = keypoints[i].pt.y;
        point.w = 1.0;
        points.push_back(point);
    }

    return points;
}

void writeToFile(string absolutePath, Points& points)
{
    std::ofstream outFile(absolutePath, std::ios::binary);
    if (!outFile) {
        std::cerr << "Failed to open file for writing.\n";
        return;
    }

    // You can write the vector size first if needed when reading.
    size_t count = points.size();
    outFile.write(reinterpret_cast<const char*>(&count), sizeof(count));

    // Writing an array of structures to a file
    outFile.write(
        reinterpret_cast<const char*>(points.data()), points.size() * sizeof(Point));

    outFile.close();
    std::cout << "Data successfully written to points.bin file" << std::endl;
    return;
}

static DataMatches getMatchesSort(vector<Point>& src, vector<Point>& dst, Matches& v_matches)
{
    DataMatches testData;
    cv::Mat img1 = cv::imread("image/transform/image1.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread("image/transform/image2.jpg", cv::IMREAD_GRAYSCALE);

    // Initializing the ORB detector object
    cv::Ptr<cv::ORB> orb = cv::ORB::create();

    // Finding Keypoints and Descriptors
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    orb->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
    orb->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);


    // Matching Descriptors Using BFMatcher
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    //Sort matches by distance
    std::sort(matches.begin(), matches.end(), [](const cv::DMatch& a, const cv::DMatch& b) {
        return a.distance < b.distance;
        });

    // We leave only the best matches
    const int numGoodMatches = 25;
    matches.resize(numGoodMatches);

    //std::vector<cv::KeyPoint> bestKeypoints1, bestKeypoints2;
    //// We collect indices of key points participating in the best matches
    //std::set<int> keypoints1_indices, keypoints2_indices;
    //for (const auto& match : matches) {
    //    keypoints1_indices.insert(match.queryIdx);
    //    keypoints2_indices.insert(match.trainIdx);
    //}

    //// Filter key points, leaving only those involved in the best matches
    //for (int i = 0; i < keypoints1.size(); ++i) {
    //    if (keypoints1_indices.count(i) > 0) {
    //        bestKeypoints1.push_back(keypoints1[i]);
    //    }
    //}
    //for (int i = 0; i < keypoints2.size(); ++i) {
    //    if (keypoints2_indices.count(i) > 0) {
    //        bestKeypoints2.push_back(keypoints2[i]);
    //    }
    //}

    testData.keypoints1 = keypoints1;
    testData.keypoints2 = keypoints2;
    testData.matches = matches;

    //Matches v_matches;
    for (int i = 0; i < matches.size(); i++)
    {
        Match match;
        match.src = matches[i].queryIdx;
        match.dst = matches[i].trainIdx;
        match.distance = matches[i].distance;
        v_matches.push_back(match);
    }

    src = convertCvKeypointToPoint(keypoints1);
    dst = convertCvKeypointToPoint(keypoints2);

    Points nSrc, nDst;
    for (int i = 0; i < matches.size(); i++)
    {
        nSrc.push_back(src[matches[i].queryIdx]);
        nDst.push_back(dst[matches[i].trainIdx]);
    }

    src = nSrc;
    dst = nDst;

    //writeToFile("srcPoints.bin", src);
    //writeToFile("dstPoints.bin", dst);

    //std::vector<cv::DMatch> treeMatch;
    //treeMatch.push_back(matches[0]);
    //treeMatch.push_back(matches[1]);
    //treeMatch.push_back(matches[2]);

    cv::Mat img_matches;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches, cv::Scalar::all(-1),
        cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    ////// Display the image with key points
    cv::imshow("Good Matches", img_matches);
    cv::waitKey(0);

    return testData;
}

static DataMatches getMatches(vector<Point> &src, vector<Point> &dst, Matches &v_matches)
{
    DataMatches testData;
    cv::Mat img1 = cv::imread("image1.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread("image2.jpg", cv::IMREAD_GRAYSCALE);

    // 
    cv::Ptr<cv::ORB> orb = cv::ORB::create();

    // Finding Keypoints and Descriptors
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    orb->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
    orb->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);

    

    // Matching Descriptors Using BFMatcher
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    // Sort matches by distance
    std::sort(matches.begin(), matches.end(), [](const cv::DMatch& a, const cv::DMatch& b) {
        return a.distance < b.distance;
        });

    // We leave only the best matches
    const int numGoodMatches = 50;
    //matches.resize(numGoodMatches);

    //std::vector<cv::KeyPoint> bestKeypoints1, bestKeypoints2;
    //// We collect indices of key points participating in the best matches
    //std::set<int> keypoints1_indices, keypoints2_indices;
    //for (const auto& match : matches) {
    //    keypoints1_indices.insert(match.queryIdx);
    //    keypoints2_indices.insert(match.trainIdx);
    //}

    //// Filter key points, leaving only those involved in the best matches
    //for (int i = 0; i < keypoints1.size(); ++i) {
    //    if (keypoints1_indices.count(i) > 0) {
    //        bestKeypoints1.push_back(keypoints1[i]);
    //    }
    //}
    //for (int i = 0; i < keypoints2.size(); ++i) {
    //    if (keypoints2_indices.count(i) > 0) {
    //        bestKeypoints2.push_back(keypoints2[i]);
    //    }
    //}

    testData.keypoints1 = keypoints1;
    testData.keypoints2 = keypoints2;
    testData.matches = matches;

    //Matches v_matches;
    for (int i = 0; i < matches.size(); i++)
    {
        Match match;
        match.src = matches[i].queryIdx;
        match.dst = matches[i].trainIdx;
        match.distance = matches[i].distance;
        v_matches.push_back(match);
    }

    src = convertCvKeypointToPoint(keypoints1);
    dst = convertCvKeypointToPoint(keypoints2);
    
 
    //std::vector<cv::DMatch> treeMatch;
    //treeMatch.push_back(matches[0]);
    //treeMatch.push_back(matches[1]);
    //treeMatch.push_back(matches[2]);

    cv::Mat img_matches;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches, cv::Scalar::all(-1),
        cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    ////// Display the image with key points
    cv::imshow("Good Matches", img_matches);
    cv::waitKey(0);

    return testData;
}

static void showKeypoints(DataMatches data, Matches matches)
{
    cv::Mat img1 = cv::imread("image1.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread("image2.jpg", cv::IMREAD_GRAYSCALE);

    std::vector<cv::DMatch> treeMatch;
    for (Match match : matches)
    {
        for (int i = 0; i < data.matches.size(); i++)
        {
            if (match.src == data.matches[i].queryIdx && match.dst == data.matches[i].trainIdx)
                treeMatch.push_back(data.matches[i]);
        }
    }
    
    cv::Mat img_matches;
    cv::drawMatches(img1, data.keypoints1, img2, data.keypoints2, treeMatch, img_matches, cv::Scalar::all(-1),
        cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    //// Display the image with key points
    cv::imshow("RandMatches", img_matches);
    cv::waitKey(0);
}

static void getDescription(Descriptions &desc1, Descriptions &desc2)
{
    Description d1;
    d1.numbers = { 0,2,0,0.25 };
    Description d2;
    d2.numbers = { 1,5,10,0.45 };

    Description d11;
    d11.numbers = { 6,1,6,2.25 };
    Description d22;
    d22.numbers = { 0,3,6,1.45 };

    Description d111;
    d111.numbers = { 1,1,1,1.25 };
    Description d222;
    d222.numbers = { 2,3,7,2.45 };

    desc1.push_back(d1);
    desc1.push_back(d11);
    desc1.push_back(d111);

    desc2.push_back(d2);
    desc2.push_back(d22);
    desc2.push_back(d222);
}

static cv::Mat MatrixToMat(vector<vector<double>> src)
{
    cv::Mat dst(src.size(), src[0].size(), CV_64F);
    for (int i = 0; i < src.size(); i++)
    {
        for (int j = 0; j < src[0].size(); j++)
        {
            dst.at<double>(i, j) = src[i][j];
        }
    }

    return dst;
}


static cv::Mat MatrixToMat(Matrix<> src)
{
    cv::Mat dst(src.sizeRow(), src.sizeColumn(), CV_64F);
    for (int i = 0; i < src.sizeRow(); i++)
    {
        for (int j = 0; j < src.sizeColumn(); j++)
        {
            dst.at<double>(i, j) = src[i][j];
        }
    }

    return dst;
}

static cv::Mat loadImage(string nameFile)
{
    cv::Mat image = cv::imread(nameFile);
    if (image.empty()) {
        std::cerr << "Failed to load image!" << std::endl;
    }

    return image;
}

class Math
{
private:

public:
    

    //Function to perform full pivoted LU factorization
    bool luDecompositionFullPivot(const vector<vector<double>>& A_input,
        vector<vector<double>>& L,
        vector<vector<double>>& U,
        vector<int>& row_perm,
        vector<int>& col_perm) {
        int n = A_input.size();
        vector<vector<double>> A = A_input; // Copy matrix A

        // Initialization of permutations
        row_perm.resize(n);
        col_perm.resize(n);
        for (int i = 0; i < n; ++i) {
            row_perm[i] = i;
            col_perm[i] = i;
        }

        L.assign(n, vector<double>(n, 0.0));
        U.assign(n, vector<double>(n, 0.0));

        for (int k = 0; k < n; ++k) {
            // Finding the maximum element for a complete permutation
            double max_val = 0.0;
            int max_row = k, max_col = k;
            for (int i = k; i < n; ++i) {
                for (int j = k; j < n; ++j) {
                    if (fabs(A[i][j]) > fabs(max_val)) {
                        max_val = A[i][j];
                        max_row = i;
                        max_col = j;
                    }
                }
            }

            // Test for degeneracy
            if (fabs(max_val) < EPSILON) {
                cerr << "Matrix is singular or nearly singular." << endl;
                return false;
            }

            // Row permutation
            swap(A[k], A[max_row]);
            swap(row_perm[k], row_perm[max_row]);
            // Rearrange columns
            for (int i = 0; i < n; ++i) {
                swap(A[i][k], A[i][max_col]);
            }
            swap(col_perm[k], col_perm[max_col]);

            // Filling L and U
            U[k][k] = A[k][k];
            for (int i = k + 1; i < n; ++i) {
                L[i][k] = A[i][k] / U[k][k];
                U[k][i] = A[k][i];
            }
            L[k][k] = 1.0;

            // Updating the rest of the matrix A
            for (int i = k + 1; i < n; ++i) {
                for (int j = k + 1; j < n; ++j) {
                    A[i][j] -= L[i][k] * U[k][j];
                }
            }
        }
        return true;
    }

    // Function for solving the system Ax = b using LU decomposition
    vector<double> solveWithLU(const vector<vector<double>>& L,
        const vector<vector<double>>& U,
        const vector<int>& row_perm,
        const vector<int>& col_perm,
        const vector<double>& b) {
        int n = L.size();
        vector<double> x(n);
        vector<double> y(n);
        vector<double> b_permuted(n);

        // Apply row permutation to vector b
        for (int i = 0; i < n; ++i) {
            b_permuted[i] = b[row_perm[i]];
        }

        // Direct move (solve L y = b_permuted)
        for (int i = 0; i < n; ++i) {
            y[i] = b_permuted[i];
            for (int j = 0; j < i; ++j) {
                y[i] -= L[i][j] * y[j];
            }
            // Since L[i][i] = 1, we do not need to divide by the diagonal element.
        }

        // Reverse move (we solve U z = y)
        vector<double> z(n);
        for (int i = n - 1; i >= 0; --i) {
            z[i] = y[i];
            for (int j = i + 1; j < n; ++j) {
                z[i] -= U[i][j] * z[j];
            }
            z[i] /= U[i][i];
        }

        // Apply the inverse column permutation to the vector z to obtain x
        for (int i = 0; i < n; ++i) {
            x[col_perm[i]] = z[i];
        }

        return x;
    }

    // Function for outputting a matrix
    void printMatrix(const vector<vector<double>>& A) {
        for (const auto& row : A) {
            for (double elem : row) {
                cout << elem << "\t";
            }
            cout << endl;
        }
    }

    // Function to output a vector
    void printVector(const vector<double>& v) {
        for (double elem : v) {
            cout << elem << "\t";
        }
        cout << endl;
    }


};