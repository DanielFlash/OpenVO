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
#include "TypeVO.h"

class BaseEnvironment
{
	/// <summary>
	/// Base class for environment implementation
	/// </summary>
public:
	int m_episodeIteration{ 0 };

	/// <summary>
	/// Class initialization
	/// </summary>
	BaseEnvironment();

	/// <summary>
	/// Method to define obstacles
	/// </summary>
	void defineObstacles();

	/// <summary>
	/// Method to place a goal
	/// </summary>
	void placeGoal();

	/// <summary>
	/// Method to reset an episode
	/// </summary>
	void reset();

	/// <summary>
	/// Method to get an environment state
	/// </summary>
	/// <returns>environment state</returns>
	std::vector<int> getState();

	/// <summary>
	/// Method to make agent action steps
	/// </summary>
	/// <param name="action">chosen agent action</param>
	void move(std::vector<int> action);

	/// <summary>
	/// Method to calculate a reward for an applied action
	/// </summary>
	/// <returns>calculated reward</returns>
	double setReward();

	/// <summary>
	/// Method to check collisions
	/// </summary>
	/// <returns>whether a collision took place</returns>
	bool isCollision();

	/// <summary>
	/// Method to check if goal is reached
	/// </summary>
	/// <returns>whether a goal was reached</returns>
	bool reachGoal();

	/// <summary>
	/// Method to make an action
	/// </summary>
	/// <param name="action">chosen agent action</param>
	/// <returns>action result</returns>
	ActionResult makeAction(std::vector<int> action);
};

