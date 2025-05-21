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
#include "BaseTrainerSARSA.h"
#include "TypeVO.h"
#include <deque>
#include <random>
#include <algorithm>
#include <iterator>

class BaseAgentSARSA
{
	/// <summary>
	/// Base SARSA agent class
	/// </summary>
public:
	int m_nEpisode{ 0 };
	int m_epsilon{ 0 };
	int m_maxMemory{};
	int m_batchSize{};
	int m_randCoef{};
	int m_randRange{};
	std::deque<MemoryCell> m_memory{};

	BaseModelSARSA* m_Model;
	BaseTrainerSARSA* m_Trainer;

	/// <summary>
	/// Class initialization
	/// </summary>
	/// <param name="maxMemory">max memory size</param>
	/// <param name="batchSize">batch size</param>
	/// <param name="randCoef">random actions threshold</param>
	/// <param name="randRange">random action ranage; if rand(0, randRange) < (randCoef - num_episode) => create a random action</param>
	/// <param name="model">NN model</param>
	/// <param name="trainer">trainer module</param>
	BaseAgentSARSA(const int maxMemory, const int batchSize, const int randCoef, const int randRange, BaseModelSARSA* model, BaseTrainerSARSA* trainer);

	/// <summary>
	/// Method to remember memory cell
	/// </summary>
	/// <param name="memoryCell">memory cell</param>
	void remember(MemoryCell memoryCell);

	/// <summary>
	/// Method to train short-term memory
	/// </summary>
	/// <param name="memoryCell">memory cell</param>
	void trainShortMemory(MemoryCell memoryCell);

	/// <summary>
	/// Method to train long-term memory
	/// </summary>
	void trainLongMemory();

	/// <summary>
	/// Method to choose agent action
	/// </summary>
	/// <param name="state">environment state</param>
	/// <returns>chosen action</returns>
	std::vector<int> act(std::vector<int> state);
};