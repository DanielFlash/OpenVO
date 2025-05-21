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
#include "BaseModelPPOActor.h"
#include "BaseModelPPOValue.h"

class BaseTrainerPPO
{
	/// <summary>
	/// Base PPO trainer class
	/// </summary>
public:
	double m_klCoeff{};
	double m_vCoeff{};
	torch::optim::Adam* m_optimizerAdam;
	BaseModelPPOActor* m_ActorModel;
	BaseModelPPOValue* m_ValueModel;
	torch::Device* m_device;
	bool m_cudaEnabled;

	/// <summary>
	/// Class initialization
	/// </summary>
	/// <param name="actorModel">NN actor model</param>
	/// <param name="valueModel">NN value model</param>
	/// <param name="klCoeff">KL-divergence error coefficient</param>
	/// <param name="vCoeff">critic error coefficient</param>
	/// <param name="optimizerAdam">optimizer</param>
	/// <param name="device">device: CPU or GPU</param>
	BaseTrainerPPO(BaseModelPPOActor* actorModel, BaseModelPPOValue* valueModel, const double klCoeff, const double vCoeff,
		torch::optim::Adam* optimizerAdam, torch::Device* device);

	/// <summary>
	/// Train iteration method
	/// </summary>
	/// <param name="state">state</param>
	/// <param name="action">action</param>
	/// <param name="logits">logits</param>
	/// <param name="logProbs">log-probs</param>
	/// <param name="reward">reward (is cummulative rewards here)</param>
	void trainStep(std::vector<std::vector<int>> state, std::vector<std::vector<int>> action, std::vector<std::vector<float>> logits,
		std::vector<std::vector<float>> logProbs, std::vector<std::vector<double>> reward);
};