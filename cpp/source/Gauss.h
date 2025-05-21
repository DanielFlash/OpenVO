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
#include <iostream>
#include <algorithm>
#include <cmath>

#include <opencv2/opencv.hpp>

#include "ExtendMatrix.h"

namespace private_gauss
{
    inline int clamp(int val, int minVal, int maxVal)
    {
        return std::max(minVal, std::min(val, maxVal));
    }

    Vector<float> generateGaussianKernel1D(int kSize, float sigma) 
    {
        Vector<float> kernel(kSize);
        int center = kSize / 2;
        float sum = 0.f;

        for (int i = 0; i < kSize; ++i) {
            int x = i - center;
            float val = std::exp(-(x * x) / (2 * sigma * sigma));
            kernel[i] = val;
            sum += val;
        }

        for (int i = 0; i < kSize; i++)
            kernel[i] /= sum;

        return kernel;
    }

    float calcSigmaFromKsize(int kSize)
    {
        return 0.3f * ((kSize - 1) * 0.5f - 1.0f) + 0.8f;
    }
}
  
#pragma region Gauss
enum BorderType 
{
    BORDER_CONSTANT, //Filled with a fixed value (default 0)
    BORDER_REPLICATE, //Edges are repeated
    BORDER_REFLECT, //Mirror reflection
    BORDER_REFLECT_101, //Like REFLECT, but does not repeat the edge pixel
    BORDER_WRAP, //Edges "wrap" (periodically)
    IGNORE //Ignore edges
};

struct Color
{
    uchar b;
    uchar g;
    uchar r;

    Color()
    {
        b = 0;
        g = 0;
        r = 0;
    }

    Color(int blue, int green, int red)
    {
        b = static_cast<uchar>(private_gauss::clamp(0, blue, 255));
        g = static_cast<uchar>(private_gauss::clamp(0, green, 255));
        r = static_cast<uchar>(private_gauss::clamp(0, red, 255));
    }
};

int getBorderIndex(int p, int size, BorderType borderType) 
{
    switch (borderType) 
    {
    case BORDER_CONSTANT:
        if (p < 0 || p >= size) return -1;
        return p;
    case BORDER_REPLICATE:
        return std::max(0, std::min(p, size - 1));
    case BORDER_REFLECT:
        if (p < 0) return -p - 1;
        if (p >= size) return 2 * size - p - 1;
        return p;
    case BORDER_REFLECT_101:
        if (p < 0) return -p;
        if (p >= size) return 2 * size - p - 2;
        return p;
    case BORDER_WRAP:
        return (p + size) % size;
    default:
        return -1;
    }
}

cv::Mat ñopyMakeBorder(const cv::Mat& src, int top, int bottom, int left, int right,
    BorderType borderType, Color color = Color())
{
    if (src.type() != CV_8UC1 && src.type() != CV_8UC3)
        throw "Invalid image format!";

    int rows = src.rows;
    int cols = src.cols;
    int channels = src.channels();

    int newRows = rows + top + bottom;
    int newCols = cols + left + right;

    cv::Mat dst(newRows, newCols, src.type());

    for (int y = 0; y < newRows; y++)
    {
        for (int x = 0; x < newCols; x++)
        {
            int srcY = getBorderIndex(y - top, rows, borderType);
            int srcX = getBorderIndex(x - left, cols, borderType);

            if (srcY == -1 || srcX == -1)
            {
                if (channels == 1)
                    dst.at<uchar>(y, x) = color.b;
                else
                    dst.at<cv::Vec3b>(y, x) = cv::Vec3b(color.b, color.g, color.r);
            }
            else
            {
                if (channels == 1)
                    dst.at<uchar>(y, x) = src.at<uchar>(srcY, srcX);
                else
                    dst.at<cv::Vec3b>(y, x) = src.at<cv::Vec3b>(srcY, srcX);
            }
        }
    }

    return dst;
}

/// <summary>
/// Applying a Gaussian filter to an image
/// </summary>
/// <param name="inputImage">Input image as cv::Mat (image format CV_8UC1 or CV_8UC3)</param>
/// <param name="kSizeX">Gaussian kernel size along the x-axis. Should be positive, non-zero, odd</param>
/// <param name="kSizeY">Gaussian kernel size along the y-axis. Should be positive, non-zero, odd</param>
/// <param name="sigmaX">Standard deviation of the Gaussian kernel in the x-direction. Should be a positive number. If zero, it is calculated automatically;</param>
/// <param name="sigmaY">Standard deviation of the Gaussian kernel in the y-direction. Must be a positive number. If zero, calculated automatically;</param>
/// <param name="borderType">Image border handling flag</param>
/// <returns>A copy of the input image as cv::Mat </returns>
cv::Mat applyGaussian(cv::Mat& inputImage, int kSizeX, int kSizeY, float sigmaX, float sigmaY, BorderType borderType)
{
    if (inputImage.type() != CV_8UC1 && inputImage.type() != CV_8UC3)
        throw "Invalid image format!";

    if (inputImage.empty())
        throw "Empty image!";

    if (kSizeX == 0 || kSizeX % 2 != 1)
        kSizeX++;

    if (kSizeY == 0 || kSizeY % 2 != 1)
        kSizeY++;

    if (sigmaX <= 0)
        sigmaX = private_gauss::calcSigmaFromKsize(kSizeX);
        
    if (sigmaY <= 0)
        sigmaY = private_gauss::calcSigmaFromKsize(kSizeY);

    int channels = inputImage.channels();
    //We get the core
    Vector<float> kernelX = private_gauss::generateGaussianKernel1D(kSizeX, sigmaX);
    int offsetX = kSizeX / 2;
    Vector<float> kernelY = private_gauss::generateGaussianKernel1D(kSizeY, sigmaY);
    int offsetY = kSizeY / 2;

    cv::Mat temp = cv::Mat::zeros(inputImage.size(), inputImage.type());
    cv::Mat dst = cv::Mat::zeros(inputImage.size(), inputImage.type());

    //Performing a horizontal convolution
    for (int y = 0; y < inputImage.rows; y++)
    {
        uchar* srcRow = inputImage.ptr<uchar>(y);
        uchar* tempRow = temp.ptr<uchar>(y);

        for (int x = 0; x < inputImage.cols; x++)
        {
            vector<float> sum(channels, 0.f);
            for (int k = -offsetX; k <= offsetX; k++)
            {
                int xk = getBorderIndex(x + k, inputImage.cols, borderType);
                if (xk == -1)
                    continue;

                uchar* pixel = &srcRow[xk * channels];
                for (short channel = 0; channel < channels; channel++)
                    sum[channel] += pixel[channel] * kernelX[k + offsetX];
            }

            uchar* out = &tempRow[x * channels];
            for (short channel = 0; channel < channels; channel++)
                out[channel] = static_cast<uchar>(sum[channel]);
        }
    }
    //We perform a vertical convolution
    for (int y = 0; y < inputImage.rows; y++)
    {
        uchar* dstRow = dst.ptr<uchar>(y);
        for (int x = 0; x < inputImage.cols; x++)
        {
            vector<float> sum(channels, 0.0f);
            for (int k = -offsetY; k <= offsetY; k++)
            {
                int yk = getBorderIndex(y + k, inputImage.rows, borderType);
                if (yk == -1)
                    continue;

                const uchar* pixel = &temp.ptr<uchar>(yk)[x * channels];
                for (short channel = 0; channel < channels; channel++)
                    sum[channel] += pixel[channel] * kernelY[k + offsetY];
            }

            uchar* out = &dstRow[x * channels];
            for (short channel = 0; channel < channels; channel++)
                out[channel] = static_cast<uchar>(sum[channel]);
        }
    }

    return dst;
}

/// <summary>
/// box filter via integral image
/// </summary>
/// <param name="src">Input image as cv::Mat (image format CV_8UC1 or CV_8UC3)</param>
/// <param name="kSizeX">Gaussian kernel size along the x-axis. Should be positive, non-zero, odd.</param>
/// <param name="kSizeY">Gaussian kernel size along the y-axis. Should be positive, non-zero, odd.</param>
/// <param name="borderType">Flag of image border processing method</param>
/// <returns>Copy of input image as cv::Mat</returns>
cv::Mat boxFilterIntegral(cv::Mat inputImage, int kSizeX, int kSizeY, BorderType borderType)
{
    if (inputImage.type() != CV_8UC1 && inputImage.type() != CV_8UC3)
        throw "Invalid image format!";

    if (inputImage.empty())
        throw "Empty image!";

    if (kSizeX == 0 || kSizeX % 2 != 1)
        kSizeX++;

    if (kSizeY == 0 || kSizeY % 2 != 1)
        kSizeY++;

    int channels = inputImage.channels();
    int halfX = kSizeX / 2;
    int halfY = kSizeY / 2;
    int normFactor = kSizeX * kSizeY;
    if (borderType != BorderType::IGNORE)
        inputImage = ñopyMakeBorder(inputImage, halfY, halfY, halfX, halfX, borderType, Color(0, 0, 0));

    cv::Mat dst = cv::Mat::zeros(inputImage.size(), inputImage.type());
    std::vector<cv::Mat> srcChannels(channels);
    cv::split(inputImage, srcChannels);

    std::vector<cv::Mat> dstChannels(channels);

    for (int channel = 0; channel < channels; channel++)
    {
        cv::Mat integralImg;
        cv::integral(srcChannels[channel], integralImg, CV_32S);

        cv::Mat result = cv::Mat::zeros(inputImage.size(), CV_8UC1);

        for (int y = 0; y < inputImage.rows; y++)
        {
            for (int x = 0; x < inputImage.cols; x++)
            {
                int x1 = std::max(x - halfX, 0);
                int y1 = std::max(y - halfY, 0);
                int x2 = std::min(x + halfX + 1, inputImage.cols);
                int y2 = std::min(y + halfY + 1, inputImage.rows);

                int A = integralImg.at<int>(y1, x1);
                int B = integralImg.at<int>(y1, x2);
                int C = integralImg.at<int>(y2, x1);
                int D = integralImg.at<int>(y2, x2);

                int sum = D - B - C + A;
                int area = (x2 - x1) * (y2 - y1);

                result.at<uchar>(y, x) = static_cast<uchar>(sum / area);
            }
        }

        dstChannels[channel] = result;
    }

    cv::merge(dstChannels, dst);

    return dst;
}

#pragma endregion


