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
#include <string>
#include <vector>

struct SurfaceImgData {
	/// <summary>
	/// Struct for unlabeled satellite map image data. 1 image - 1 instance
	/// <param name="imgName">image name</param>
	/// <param name="imgW">image width</param>
	/// <param name="imgH">image height</param>
	/// <param name="imgTopLeftX">top left X image coordinate</param>
	/// <param name="imgTopLeftY">top left Y image coordinate</param>
	/// <param name="imgBotRightX">bot right X image coordinate</param>
	/// <param name="imgBotRightY">bot right X image coordinate</param>
	/// </summary>
	std::string imgName;
	int imgW;
	int imgH;
	double imgTopLeftX;
	double imgTopLeftY;
	double imgBotRightX;
	double imgBotRightY;
};

struct SurfaceObjData {
	/// <summary>
	/// Struct for labeled satellite map image objects. 1 object - 1 instance	
	/// <param name="imgName">image name</param>
	/// <param name="imgW">image width</param>
	/// <param name="imgH">image height</param>
	/// <param name="imgTopLeftX">top left X image coordinate</param>
	/// <param name="imgTopLeftY">top left Y image coordinate</param>
	/// <param name="imgBotRightX">bot right X image coordinate</param>
	/// <param name="imgBotRightY">bot right X image coordinate</param>
	/// <param name="objLabel">object label</param>
	/// <param name="bbX">boundary box top left X coordinate</param>
	/// <param name="bbY">boundary box top left Y coordinate</param>
	/// <param name="bbW">boundary box width</param>
	/// <param name="bbH">boundary box height</param>
	/// </summary>
	std::string imgName;
	int imgW;
	int imgH;
	double imgTopLeftX;
	double imgTopLeftY;
	double imgBotRightX;
	double imgBotRightY;
	int objLabel;
	int bbX;
	int bbY;
	int bbW;
	int bbH;
};

struct SurfaceData {
	/// <summary>
	/// Struct for processed satellite map image objects. 1 object - 1 instance
	/// <param name="imgName">image name</param>
	/// <param name="imgW">image width</param>
	/// <param name="imgH">image height</param>
	/// <param name="imgTopLeftX">top left X image coordinate</param>
	/// <param name="imgTopLeftY">top left Y image coordinate</param>
	/// <param name="imgBotRightX">bot right X image coordinate</param>
	/// <param name="imgBotRightY">bot right X image coordinate</param>
	/// <param name="objId">object ID</param>
	/// <param name="objLabel">object label</param>
	/// <param name="bbX">boundary box top left X coordinate</param>
	/// <param name="bbY">boundary box top left Y coordinate</param>
	/// <param name="bbW">boundary box width</param>
	/// <param name="bbH">boundary box height</param>
	/// <param name="objCoordX">object center X coordinate</param>
	/// <param name="objCoordY">object center Y coordinate</param>
	/// <param name="mappedTo">ID of mapped local detected object</param>
	/// </summary>
	std::string imgName;
	int imgW;
	int imgH;
	double imgTopLeftX;
	double imgTopLeftY;
	double imgBotRightX;
	double imgBotRightY;
	int objId;
	int objLabel;
	int bbX;
	int bbY;
	int bbW;
	int bbH;
	double objCoordX;
	double objCoordY;
	int mappedTo{ -1 };
};

struct LocalData {
	/// <summary>
	/// Struct for local detected objects. 1 object - 1 instance
	/// <param name="overlapLevel">number of object overlap by its detection on different frames</param>
	/// <param name="mappedTo">ID of mapped processed satellite map image object</param>
	/// <param name="objId">object ID</param>
	/// <param name="objLabel">object label</param>
	/// <param name="objCoordX">object center X coordinate</param>
	/// <param name="objCoordY">object center Y coordinate</param>
	/// </summary>
	int overlapLevel{ 0 };
	int mappedTo{ -1 };
	int objId{ -1 };
	int objLabel{ -1 };
	double objCoordX;
	double objCoordY;
};

struct MapEdges {
	/// <summary>
	/// Struct for found satellite map extreme vertices
	/// <param name="topLeftX">top left X map coordinate</param>
	/// <param name="topLeftY">top left Y map coordinate</param>
	/// <param name="botRightX">bot right X map coordinate</param>
	/// <param name="botRightY">bot right Y map coordinate</param>
	/// </summary>
	double topLeftX;
	double topLeftY;
	double botRightX;
	double botRightY;
};

struct Detection {
	/// <summary>
	/// Struct for detections
	/// <param name="class_id">label id</param>
	/// <param name="className">label name</param>
	/// <param name="confidence">confidence</param>
	/// <param name="x">boundary box top left X coordinate</param>
	/// <param name="y">boundary box top left Y coordinate</param>
	/// <param name="w">boundary box width</param>
	/// <param name="h">boundary box height</param>
	/// </summary>
	int class_id{ 0 };
	std::string className{};
	float confidence{ 0.0 };
	int x{ 0 };
	int y{ 0 };
	int w{ 0 };
	int h{ 0 };
};

struct ObjectDist {
	/// <summary>
	/// Struct for distance between local detected and satellite map objects
	/// <param name="localData">local detected object reference</param>
	/// <param name="surfaceData">satellite map object reference</param>
	/// <param name="dist">distance</param>
	/// <param name="deltaX">OX delta from matching algorithm</param>
	/// <param name="deltaY">OY delta from matching algorithm</param>
	/// </summary>
	LocalData* localData = nullptr;
	SurfaceData* surfaceData = nullptr;
	double dist;
	double deltaX;
	double deltaY;
};

struct MemoryCell {
	/// <summary>
	/// Struct for memory cell for reinforcement learning algorithms
	/// <param name="state">current environment state</param>
	/// <param name="action">current agent action</param>
	/// <param name="nextState">next environment state</param>
	/// <param name="nextAction">next agent action</param>
	/// <param name="reward">agent reward</param>
	/// <param name="done">whether episode is done</param>
	/// </summary>
	std::vector<int> state{};
	std::vector<int> action{};
	std::vector<int> nextState{};
	std::vector<int> nextAction{};
	double reward;
	bool done;
};

struct ActionResult {
	/// <summary>
	/// Struct for action result of agent in reinforcement learning algorithms
	/// <param name="reward">agent reward</param>
	/// <param name="done">whether episode is done</param>
	/// </summary>
	double reward;
	bool done;
};

struct PPOMemoryCell {
	/// <summary>
	/// Struct for memory cell for PPO reinforcement learning algorithm
	/// <param name="state">current environment state</param>
	/// <param name="action">current agent action</param>
	/// <param name="logits">logits</param>
	/// <param name="logProbs">log-probs</param>
	/// <param name="reward">agent reward</param>
	/// </summary>
	std::vector<int> state{};
	std::vector<int> action{};
	std::vector<float> logits{};
	std::vector<float> logProbs{};
	double reward;
};
