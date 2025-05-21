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
#include <torch/torch.h>
#include <torch/script.h>
#include "baseInferencer.h"

using torch::indexing::Slice;
using torch::indexing::None;

class TorchInference : public BaseInference
{
    /// <summary>
    /// .torchscript detector class
    /// </summary>
private:
    /// <summary>
    /// Method to convert frame to square frame
    /// </summary>
    /// <param name="source">input frame</param>
    /// <returns>square frame</returns>
    cv::Mat formatToSquare(const cv::Mat& source);

    /// <summary>
    /// Method to calculate frame scale alteration
    /// </summary>
    /// <param name="image">input frame</param>
    /// <returns>scale</returns>
    float generateScale(const cv::Mat& image);

    /// <summary>
    /// Method to convert xywh box into xyxy box
    /// </summary>
    /// <param name="x">box tensor</param>
    /// <returns>converted box tensor</returns>
    torch::Tensor xywh2xyxy(const torch::Tensor& x);

    /// <summary>
    /// Method to convert boxes to the initial frame scale
    /// </summary>
    /// <param name="img1_shape">scaled frame</param>
    /// <param name="boxes">boxes</param>
    /// <param name="img0_shape">initial frame</param>
    /// <returns>converted boxes</returns>
    torch::Tensor scaleBoxes(const std::vector<int>& img1_shape, torch::Tensor& boxes, const std::vector<int>& img0_shape);

    /// <summary>
    /// Method to wrap NMS algorithm
    /// </summary>
    /// <param name="prediction">detections</param>
    /// <returns>filtered detections</returns>
    torch::Tensor nonMaxSuppression(torch::Tensor& prediction);

    /// <summary>
    /// Method to implement NMS algorithm
    /// </summary>
    /// <param name="bboxes">detected boxes</param>
    /// <param name="scores">predicted scores</param>
    /// <param name="iou_threshold">IoU threshold</param>
    /// <returns>filtered detections</returns>
    torch::Tensor nms(const torch::Tensor& bboxes, const torch::Tensor& scores, float iou_threshold);

    std::string modelPath{};
    const std::map<int, std::string>& classes;
    bool cudaEnabled{};
    cv::Size2f modelShape{};

    float modelScoreThreshold{ 0.45 };
    float modelNMSThreshold{ 0.50 };
    int modelMaxDet{ 100 };

    bool letterBoxForSquare = true;

    torch::jit::script::Module net;

public:
    /// <summary>
    /// Class initialization
    /// </summary>
    /// <param name="torchModelPath">torch model file</param>
    /// <param name="modelClasses">txt file with object labels</param>
    /// <param name="imgW">input image width</param>
    /// <param name="imgH">input image height</param>
    /// <param name="runWithCuda">device: CPU or GPU</param>
    /// <param name="scoreThresh">model score threshold</param>
    /// <param name="nmsThresh">model IoU threshold</param>
    /// <param name="maxDet">max detections per image</param>
    TorchInference(const std::string& torchModelPath, const std::map<int, std::string>& modelClasses, const int imgW = 640, const int imgH = 640,
        const bool& runWithCuda = true, const float scoreThresh = 0.45, const float nmsThresh = 0.50, const int maxDet = 100);

    /// <summary>
    /// Method to load model
    /// </summary>
    void loadTorchNetwork();

    /// <summary>
    /// Method to run inference
    /// </summary>
    /// <param name="input">input frame</param>
    /// <returns>detection list</returns>
    virtual std::vector<Detection> runInference(const cv::Mat& input) override final;
};
