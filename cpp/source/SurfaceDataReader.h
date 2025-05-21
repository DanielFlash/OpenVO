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
#include <iostream>
#include <sstream>
#include "TypeVO.h"

class SurfaceDataReader
{
	/// <summary>
	/// Class to load satellite map data
	/// </summary>
private:
	const char* m_inputFile{};
	const char* m_imgFolder{};
	const char* m_outputFile{};

public:
	/// <summary>
	/// Class initialization
	/// </summary>
	/// <param name="inputFile">txt file with saved satellite map data</param>
	/// <param name="imgFolder">image folder</param>
	/// <param name="outputFile">txt file with saved processed satellite map image objects</param>
	SurfaceDataReader(const char* inputFile, const char* imgFolder, const char* outputFile);

	/// <summary>
	/// Method to read processed satellite map image objects 
	/// </summary>
	/// <returns>processed satellite map images objects list</returns>
	std::vector<SurfaceData> readProcessedData();

	/// <summary>
	/// Method to read labeled satellite map images objects
	/// </summary>
	/// <returns>satellite map images objects</returns>
	std::vector<SurfaceObjData> readRawLabeledData();

	/// <summary>
	/// Method to read unlabeled satellite map images objects
	/// </summary>
	/// <returns>unlabeled satellite map images objects</returns>
	std::vector<SurfaceImgData> readRawData();
};
