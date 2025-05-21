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
#include "BaseModelA2CActor.h"
#include "BaseModelA2CValue.h"

class BaseTrainerA2C
{
	/// <summary>
	/// Base A2C trainer class
	/// </summary>
public:
	torch::optim::Adam* m_optimizerAdamActor;
	torch::optim::Adam* m_optimizerAdamValue;
	BaseModelA2CActor* m_ActorModel;
	BaseModelA2CValue* m_ValueModel;
	torch::Device* m_device;
	bool m_cudaEnabled;

	/// <summary>
	/// Class initialization
	/// </summary>
	/// <param name="actorModel">NN actor model</param>
	/// <param name="valueModel">NN value model</param>
	/// <param name="optimizerAdamActor">actor optimizer</param>
	/// <param name="optimizerAdamValue">value optimizer</param>
	/// <param name="device">device: CPU or GPU</param>
	BaseTrainerA2C(BaseModelA2CActor* actorModel, BaseModelA2CValue* valueModel, torch::optim::Adam* optimizerAdamActor,
		torch::optim::Adam* optimizerAdamValue, torch::Device* device);

	/// <summary>
	/// Train iteration method
	/// </summary>
	/// <param name="state">state</param>
	/// <param name="action">action</param>
	/// <param name="reward">reward (is cummulative rewards here)</param>
	void trainStep(std::vector<std::vector<int>> state, std::vector<std::vector<int>> action,
		std::vector<std::vector<double>> reward);
};
