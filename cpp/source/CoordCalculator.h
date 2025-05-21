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
#include <filesystem>
#include "Detector.h"
#include "TypeOVO.h"

class CoordCalculator
{
	/// <summary>
	/// Class to repform object coodinates calculation
	/// </summary>
public:
	/// <summary>
	/// Method to process labeled satellite map images and object coordinates calculation
	/// </summary>
	/// <param name="surfaceObjDataList">satellite map images objects list</param>
	/// <returns>processed satellite map images objects list</returns>
	std::vector<SurfaceData> calcObjCoords(const std::vector<SurfaceObjData>& surfaceObjDataList);

	/// <summary>
	/// Method to process unlabeled satellite map images, detect objects and object coordinates calculation
	/// </summary>
	/// <param name="surfaceImgDataList">satellite map images data list</param>
	/// <param name="detector">detector</param>
	/// <param name="imgFolder">image folder</param>
	/// <returns>processed satellite map images objects list</returns>
	std::vector<SurfaceData> detectAndCalcObjCoords(const std::vector<SurfaceImgData>& surfaceImgDataList, Detector& detector, const char* imgFolder);

	/// <summary>
	/// Method to calculate coordinates of satellite map extreme vertices
	/// </summary>
	/// <param name="surfaceDataList">processed satellite map images objects list</param>
	/// <returns>calculated coordinates</returns>
	MapEdges calcMapEdges(const std::vector<SurfaceData>& surfaceDataList);

	/// <summary>
	/// Method to calculate coordinates of local detected objects
	/// </summary>
	/// <param name="detections">detection list</param>
	/// <param name="imgShape">image shape</param>
	/// <param name="cam_params">camera parameters</param>
	/// <param name="curr_angles">current angles</param>
	/// <param name="curr_offset">current offset</param>
	/// <param name="meter_in_pixel">number of meters in pixel</param>
	/// <returns>processed local detected objects list</returns>
	std::vector<LocalData> calcLocalObjCoords(const std::vector<Detection>& detections, const cv::Size2f& imgShape,
		Camera_params& cam_params, Pos_angle& curr_angles, Pos_d3& curr_offset, Pos_f2& meter_in_pixel);
};
