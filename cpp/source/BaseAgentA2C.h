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
#include "BaseTrainerA2C.h"
#include "TypeVO.h"
#include <deque>
#include <random>
#include <algorithm>
#include <iterator>

class BaseAgentA2C
{
	/// <summary>
	/// Base A2C agent class
	/// </summary>
public:
	int m_nEpisode{ 0 };

	BaseModelA2CActor* m_ActorModel;
	BaseModelA2CValue* m_ValueModel;
	BaseTrainerA2C* m_Trainer;

	/// <summary>
	/// Class initialization
	/// </summary>
	/// <param name="actorModel">NN actor model</param>
	/// <param name="valueModel">NN value model</param>
	/// <param name="trainer">trainer module</param>
	BaseAgentA2C(BaseModelA2CActor* actorModel, BaseModelA2CValue* valueModel, BaseTrainerA2C* trainer);

	/// <summary>
	/// Method to train an episode
	/// </summary>
	/// <param name="memory">episode memory</param>
	void trainEpisode(std::vector<MemoryCell> memory);

	/// <summary>
	/// Method to choose agent action
	/// </summary>
	/// <param name="state">environment state</param>
	/// <returns>chosen action</returns>
	std::vector<int> act(std::vector<int> state);
};

