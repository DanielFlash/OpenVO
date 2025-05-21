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
#include <fstream>
#include <map>
#include <torch/script.h>
#include "onnxInferencer.h"
#include "torchInferencer.h"

class Detector
{
	/// <summary>
	/// Wrap-class for detection implementation
	/// </summary>
private:
	const char* m_labelsFile{};
	const char* m_modelPath{};
	std::map<int, std::string> m_labels{};

	bool m_cudaEnabled{};
	int m_imgW{};
	int m_imgH{};
	bool m_isOnnxModel{};
	float m_scoreThresh{ 0.45 };
	float m_nmsThresh{ 0.50 };
	int m_maxDet{ 100 };

	BaseInference* inferencer = nullptr;
	OnnxInference onnxInferencer;
	TorchInference torchInferencer;

public:
	/// <summary>
	/// Class initialization
	/// </summary>
	/// <param name="labelsFile">txt file with object labels</param>
	/// <param name="modelPath">NN model file</param>
	/// <param name="cudaEnabled">device: CPU or GPU</param>
	/// <param name="imgW">input image width</param>
	/// <param name="imgH">input image height</param>
	/// <param name="scoreThresh">model score threshold</param>
	/// <param name="nmsThresh">model IoU threshold</param>
	/// <param name="maxDet">max detections per image</param>
	Detector(const char* labelsFile, const char* modelPath, bool cudaEnabled = true, int imgW = 640, int imgH = 640,
		const float scoreThresh = 0.45, const float nmsThresh = 0.50, const int maxDet = 100);

	/// <summary>
	/// Method to read image
	/// </summary>
	/// <param name="inputFile">image file</param>
	/// <returns>loaded image</returns>
	cv::Mat readImage(const char* inputFile);

	/// <summary>
	/// Method to detect objects
	/// </summary>
	/// <param name="image">input image</param>
	/// <returns>detection list</returns>
	std::vector<Detection> detect(const cv::Mat& image);
};
