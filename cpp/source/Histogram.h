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

#pragma region Histogram

/// <summary>
/// Improved histogram equalization, due to contrast gain limitation and adaptive image block matching
/// </summary>
/// <param name="inputImage">Input image as cv::Mat (image format CV_8UC1 or CV_8UC3)</param>
/// <param name="tileSizeX">Image correction block size along the x-axis, must be greater than zero</param>
/// <param name="tileSizeY">Image correction block size along the y-axis, must be greater than zero</param>
/// <param name="relativeClipLimit">Relative (depending on the block size) limit on the height (maximum value) of the histogram in each block, must be greater than zero, usually from 2.0 to 4.0</param>
/// <returns>A copy of the input image as cv::Mat</returns>
cv::Mat applyHistogramCLAHE(const cv::Mat& inputImage, int tileSizeX, int tileSizeY,
    float relativeClipLimit = 4.0f)
{
    cv::Mat ycrcb; std::vector<cv::Mat> channels;
    cv::Mat input;

    if(inputImage.empty())
        throw "Empty image!";

    if (tileSizeX == 0 || inputImage.cols > tileSizeX)
        tileSizeX = inputImage.cols;
    
    if (tileSizeY == 0 || inputImage.rows > tileSizeY)
        tileSizeY = inputImage.rows;

    if (inputImage.type() == CV_8UC3) //If the image is in color
    {
        //Convert the image to YCrCb
        cv::cvtColor(inputImage, ycrcb, cv::COLOR_BGR2YCrCb);
        //Let's separate the channels
        cv::split(ycrcb, channels);
        input = channels[0]; //Brightness
    }
    else if (inputImage.type() == CV_8UC1)
    {
        input = inputImage;
    }
    else
        throw "Invalid image format!";

    const int width = input.cols;
    const int height = input.rows;
    const int tileX = (width + tileSizeX - 1) / tileSizeX;
    const int tileY = (height + tileSizeY - 1) / tileSizeY;

    std::vector<std::vector<std::vector<uchar>>> lut(
        tileY, std::vector<std::vector<uchar>>(tileX, std::vector<uchar>(256)));

    //1) Let's build a LUT (histogram + limiting + normalization) for each block
    for (int ty = 0; ty < tileY; ++ty)
    {
        for (int tx = 0; tx < tileX; ++tx)
        {
            int x0 = tx * tileSizeX;
            int y0 = ty * tileSizeY;
            int x1 = std::min(x0 + tileSizeX, width);
            int y1 = std::min(y0 + tileSizeY, height);
            int area = (x1 - x0) * (y1 - y0);
            int clipLimit = std::max(1, static_cast<int>(relativeClipLimit * area / 256));

            //Histogram
            int hist[256] = { 0 };
            for (int y = y0; y < y1; ++y)
            {
                const uchar* row = input.ptr<uchar>(y);
                for (int x = x0; x < x1; ++x)
                {
                    hist[row[x]]++;
                }
            }

            //Limitation
            int excess = 0;
            for (int i = 0; i < 256; ++i)
            {
                if (hist[i] > clipLimit)
                {
                    excess += hist[i] - clipLimit;
                    hist[i] = clipLimit;
                }
            }

            //We distribute excess
            int bonus = excess / 256;
            for (int i = 0; i < 256; ++i)
            {
                hist[i] += bonus;
            }

            //Cumulative histogram
            int cdf[256] = { 0 };
            cdf[0] = hist[0];
            for (int i = 1; i < 256; ++i)
            {
                cdf[i] = cdf[i - 1] + hist[i];
            }

            //Normalize LUT
            for (int i = 0; i < 256; ++i)
            {
                lut[ty][tx][i] = cv::saturate_cast<uchar>((float)cdf[i] * 255 / area);
            }
        }
    }

    //2) Create an output image and apply the interpolated value
    cv::Mat output = input.clone();
    cv::Mat result;
    for (int y = 0; y < height; ++y)
    {
        int ty = y / tileSizeY;
        float dy = (float)(y % tileSizeY) / tileSizeY;
        int ty1 = std::min(ty + 1, tileY - 1);
        const uchar* row = input.ptr<uchar>(y);
        uchar* outRow = output.ptr<uchar>(y);

        for (int x = 0; x < width; ++x)
        {
            int tx = x / tileSizeX;
            float dx = (float)(x % tileSizeX) / tileSizeX;
            int tx1 = std::min(tx + 1, tileX - 1);

            uchar pix = row[x];

            //Bilinear interpolation between LUTs
            float v00 = lut[ty][tx][pix];
            float v10 = lut[ty][tx1][pix];
            float v01 = lut[ty1][tx][pix];
            float v11 = lut[ty1][tx1][pix];
            float val = (1 - dy) * ((1 - dx) * v00 + dx * v10) +
                dy * ((1 - dx) * v01 + dx * v11);

            outRow[x] = cv::saturate_cast<uchar>(val);
        }
    }

    if (inputImage.type() == CV_8UC3) //If the image is in color
    {
        channels[0] = output;
        //Let's merge the channels and return them to BGR
        cv::merge(channels, ycrcb);
        cv::cvtColor(ycrcb, result, cv::COLOR_YCrCb2BGR);
    }
    else
    {
        result = output;
    }

    return result;
}

/// <summary>
/// Apply histogram color correction to the image.
/// </summary>
/// <param name="input">Input image as cv::Mat (image format CV_8UC1 or CV_8UC3).</param>
/// <returns>Copy of input image as cv::Mat</returns>
cv::Mat applyHistogram(const cv::Mat& inputImage)
{
    std::vector<cv::Mat> channels;
    cv::Mat ycrcb;
    cv::Mat input;

    if (inputImage.empty())
        throw "Empty image!";

    if (inputImage.type() == CV_8UC3)
    {
        cv::cvtColor(inputImage, ycrcb, cv::COLOR_BGR2YCrCb);
        //Let's separate the channels
        cv::split(ycrcb, channels);
        input = channels[0]; //Y is brightness
    }
    else if (inputImage.type() == CV_8UC1)
    {
        input = inputImage;
    }
    else
        throw "Invalid image format!";

    //1) Calculate the histogram
    int hist[256] = { 0 };
    for (int y = 0; y < input.rows; ++y)
    {
        for (int x = 0; x < input.cols; ++x)
        {
            uchar pixel = input.at<uchar>(y, x);
            hist[pixel]++;
        }
    }

    //2) Calculate the cumulative histogram (CDF)
    int cdf[256] = { 0 };
    cdf[0] = hist[0];
    for (int i = 1; i < 256; ++i)
    {
        cdf[i] = cdf[i - 1] + hist[i];
    }

    //3) CDF Normalization
    int cdf_min = 0;
    for (int i = 0; i < 256; ++i)
    {
        if (cdf[i] != 0)
        {
            cdf_min = cdf[i];
            break;
        }
    }

    int total_pixels = input.rows * input.cols;
    uchar lut[256] = { 0 };
    for (int i = 0; i < 256; ++i)
    {
        lut[i] = cv::saturate_cast<uchar>(255.0 * (cdf[i] - cdf_min) /
            (total_pixels - cdf_min));
    }

    // 4) Apply LUT to image
    cv::Mat output;
    for (int y = 0; y < input.rows; ++y)
    {
        uchar* row = input.ptr<uchar>(y);
        for (int x = 0; x < input.cols; ++x)
        {
            row[x] = lut[row[x]];
        }
    }

    if (inputImage.type() == CV_8UC3)
    {
        //Let's merge the channels and return them to BGR
        cv::merge(channels, ycrcb);
        cv::cvtColor(ycrcb, output, cv::COLOR_YCrCb2BGR);
    }
    else
    {
        output = input;
    }

    return output;
}
#pragma endregion Histogram