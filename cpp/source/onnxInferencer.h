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
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include "baseInferencer.h"

class OnnxInference : public BaseInference
{
    /// <summary>
    /// .onnx detector class
    /// </summary>
private:
    /// <summary>
    /// Method to convert frame to square frame
    /// </summary>
    /// <param name="source">input frame</param>
    /// <returns>square frame</returns>
    cv::Mat formatToSquare(const cv::Mat& source);

    std::string modelPath{};
    const std::map<int, std::string>& classes;
    bool cudaEnabled{};
    cv::Size2f modelShape{};

    float modelScoreThreshold{ 0.45 };
    float modelNMSThreshold{ 0.50 };
    int modelMaxDet{ 100 };

    bool letterBoxForSquare = true;

    cv::dnn::Net net;

public:
    /// <summary>
    /// Class initialization
    /// </summary>
    /// <param name="onnxModelPath">onnx model file</param>
    /// <param name="modelClasses">txt file with object labels</param>
    /// <param name="imgW">input image width</param>
    /// <param name="imgH">input image height</param>
    /// <param name="runWithCuda">device: CPU or GPU</param>
    /// <param name="scoreThresh">model score threshold</param>
    /// <param name="nmsThresh">model IoU threshold</param>
    /// <param name="maxDet">max detections per image</param>
    OnnxInference(const std::string& onnxModelPath, const std::map<int, std::string>& modelClasses, const int imgW = 640, const int imgH = 640,
        const bool& runWithCuda = true, const float scoreThresh = 0.45, const float nmsThresh = 0.50, const int maxDet = 100);

    /// <summary>
    /// Method to load model
    /// </summary>
    void loadOnnxNetwork();

    /// <summary>
    /// Method to run inference
    /// </summary>
    /// <param name="input">input frame</param>
    /// <returns>detection list</returns>
    virtual std::vector<Detection> runInference(const cv::Mat& input) override final;
};
