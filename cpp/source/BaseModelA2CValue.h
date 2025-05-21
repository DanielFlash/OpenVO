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
#include <torch/torch.h>
#include <vector>

class BaseModelA2CValue : public torch::nn::Module
{
	/// <summary>
	/// Base A2C NN Value model class
	/// </summary>
public:
	std::vector<int64_t> m_hiddenSizes;
	int64_t m_inputSize;

	std::vector<torch::nn::Linear> m_fcLayers{};

	/// <summary>
	/// Class initialization
	/// </summary>
	/// <param name="inputSize">input layer size</param>
	/// <param name="hiddenSizes">list of hidden layer sizes</param>
	BaseModelA2CValue(const int64_t inputSize, const std::vector<int64_t>& hiddenSizes);

	/// <summary>
	/// Model feedforward method 
	/// </summary>
	/// <param name="x">input tensor</param>
	/// <returns>predicted result</returns>
	torch::Tensor forward(torch::Tensor x);
};

