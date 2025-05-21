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
#include "BaseModelDQN.h"

class BaseTrainerDQN
{
	/// <summary>
	/// Base DQN trainer class
	/// </summary>
public:
	const double m_gamma;
	torch::optim::Adam* m_optimizerAdam;
	BaseModelDQN* m_Model;
	torch::Device* m_device;
	bool m_cudaEnabled;

	/// <summary>
	/// Class initialization
	/// </summary>
	/// <param name="model">NN model</param>
	/// <param name="gamma">discount coefficient</param>
	/// <param name="optimizerAdam">optimizer</param>
	/// <param name="device">device: CPU or GPU</param>
	BaseTrainerDQN(BaseModelDQN* model, const double gamma, torch::optim::Adam* optimizerAdam, torch::Device* device);

	/// <summary>
	/// Train iteration method
	/// </summary>
	/// <param name="state">state</param>
	/// <param name="action">action</param>
	/// <param name="reward">reward</param>
	/// <param name="nextState">nest state</param>
	/// <param name="done">whether episode is done</param>
	void trainStep(std::vector<std::vector<int>> state, std::vector<std::vector<int>> action,
		std::vector<std::vector<double>> reward, std::vector<std::vector<int>> nextState, std::vector<bool> done);
};
