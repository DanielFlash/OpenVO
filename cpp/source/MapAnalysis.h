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
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "SurfaceDataReader.h"
#include "SurfaceDataWriter.h"
#include "CoordCalculator.h"
#include "Detector.h"

class MapAnalysis
{
	/// <summary>
	/// Class for navigation correction
	/// </summary>
private:
	const char* m_inputFile;
	const char* m_imgFolder;
	const char* m_outputFile;
	const char* m_labelsFile;

	const char* m_globalModelPath;
	bool m_glCudaEnabled{};
	int m_glImgW{};
	int m_glImgH{};
	float m_glScoreThresh{ 0.45 };
	float m_glNmsThresh{ 0.50 };
	int m_glMaxDet{ 100 };

	const char* m_modelPath;
	bool m_cudaEnabled{};
	int m_imgW{};
	int m_imgH{};
	float m_scoreThresh{ 0.45 };
	float m_nmsThresh{ 0.50 };
	int m_maxDet{ 100 };

	SurfaceDataReader m_surfaceDataReader;
	SurfaceDataWriter m_surfaceDataWriter;
	CoordCalculator m_coordCalculator;
	Detector m_detector;
	Detector m_glDetector;

	std::vector<SurfaceImgData> m_surfaceImgDataList{};
	std::vector<SurfaceObjData> m_surfaceObjDataList{};
	std::vector<SurfaceData> m_surfaceDataList{};
	std::vector<LocalData> m_localDataList{};
	MapEdges m_mapEdges{};

	/// <summary>
	/// Method to calculate minimal bias between masks of object's areas
	/// </summary>
	/// <param name="localMinX">min X coordinate of local area</param>
	/// <param name="localMaxY">max Y coordinate of local area</param>
	/// <param name="globalMinX">min X coordinate of satellite area</param>
	/// <param name="globalMaxY">max Y coordinate of satellite area</param>
	/// <param name="lw">local area width</param>
	/// <param name="lh">local area height</param>
	/// <param name="gw">satellite area width</param>
	/// <param name="gh">satellite area height</param>
	/// <param name="label">class label</param>
	/// <param name="scale">meters per pixel</param>
	/// <param name="matchDelta">object match threshold in masks</param>
	/// <param name="properObjects">filtered local detected objects</param>
	/// <param name="properSurfaceObj">filtered satellite map objects</param>
	/// <returns>calculated bias for OX and OY</returns>
	std::vector<double> calcDeltas(const double localMinX, const double localMaxY, const double globalMinX, const double globalMaxY,
		const double lw, const double lh, const double gw, const double gh, const int label, const double scale, const int matchDelta,
		const std::vector<LocalData*> properObjects, const std::vector<SurfaceData*> properSurfaceObj);

	/// <summary>
	/// Method to calculate minimal distance between objects from different areas
	/// </summary>
	/// <param name="deltas"calculated bias</param>
	/// <param name="label">class label</param>
	/// <param name="properObjects">filtered local detected objects</param>
	/// <param name="properSurfaceObj">filtered satellite map objects</param>
	/// <returns>calculated distance</returns>
	std::vector<ObjectDist> calcObjDist(const std::vector<double>& deltas, const int label,
		const std::vector<LocalData*> properObjects, const std::vector<SurfaceData*> properSurfaceObj);

	/// <summary>
	/// Method to match local and satellite objects
	/// </summary>
	/// <param name="properSurfaceObj">best matches list</param>
	void mapObjects(std::vector<ObjectDist>* bestCandidates);

	/// <summary>
	/// Method to update local objects coordinates
	/// </summary>
	/// <param name="deltaX">calculated bias for OX</param>
	/// <param name="deltaY">calculated bias for OY</param>
	void updateLocalDataCoord(const double deltaX, const double deltaY);

public:
	/// <summary>
	/// Class initialization
	/// </summary>
	/// <param name="inputFile">txt file with saved satellite map data</param>
	/// <param name="imgFolder">image folder</param>
	/// <param name="outputFile">txt file with saved processed satellite map image objects</param>
	/// <param name="labelsFile">txt file with object labels</param>
	/// <param name="modelPath">NN model file for local object detection</param>
	/// <param name="cudaEnabled">device: CPU or GPU; for local model</param>
	/// <param name="imgW">input image width; for local model</param>
	/// <param name="imgH">input image height; for local model</param>
	/// <param name="scThres">model score threshold; for local model</param>
	/// <param name="nmsThres">model IoU threshold; for local model</param>
	/// <param name="maxD">max detections per image; for local model</param>
	/// <param name="glModelPath">NN model file for satellite map object detection</param>
	/// <param name="glCudaEnabled">device: CPU or GPU; for satellite map model</param>
	/// <param name="glImgW">input image width; for satellite map model</param>
	/// <param name="glImgH">input image height; for satellite map model</param>
	/// <param name="glScThres">model score threshold; for satellite map model</param>
	/// <param name="glNmsThres">model IoU threshold; for satellite map model</param>
	/// <param name="glMaxD">max detections per image; for satellite map model</param>
	MapAnalysis(const char* inputFile, const char* imgFolder, const char* outputFile, const char* labelsFile,
		const char* modelPath, bool cudaEnabled, int imgW, int imgH, float scThres, float nmsThres, int maxD,
		const char* glModelPath, bool glCudaEnabled, int glImgW, int glImgH, float glScThres, float glNmsThres, int glMaxD);

	/// <summary>
	/// Class initialization
	/// </summary>
	/// <param name="inputFile">txt file with saved satellite map data</param>
	/// <param name="imgFolder">image folder</param>
	/// <param name="outputFile">txt file with saved processed satellite map image objects</param>
	/// <param name="labelsFile">txt file with object labels</param>
	/// <param name="modelPath">NN model file for local object detection</param>
	/// <param name="cudaEnabled">device: CPU or GPU; for local model</param>
	/// <param name="imgW">input image width; for local model</param>
	/// <param name="imgH">input image height; for local model</param>
	/// <param name="scThres">model score threshold; for local model</param>
	/// <param name="nmsThres">model IoU threshold; for local model</param>
	/// <param name="maxD">max detections per image; for local model</param>
	MapAnalysis(const char* inputFile, const char* imgFolder, const char* outputFile, const char* labelsFile,
		const char* modelPath, bool cudaEnabled, int imgW, int imgH, float scThres, float nmsThres, int maxD);

	// Global map processing inner steps

	/// <summary>
	/// Method to read unlabeled satellite map images objects
	/// </summary>
	void loadRawData();

	/// <summary>
	/// Method to read labeled satellite map images objects
	/// </summary>
	void loadRawLabeledData();

	/// <summary>
	/// Method to read processed satellite map image objects
	/// </summary>
	void loadProcessedData();

	/// <summary>
	/// Method to save processed satellite map data
	/// </summary>
	void saveProcessedData();

	/// <summary>
	/// Method to process satellite map data and calculate objects coordinates
	/// </summary>
	void processRawData();

	// Global map processing steps (either)

	/// <summary>
	/// Method to read satellite map data and calculate objects coordinates
	/// </summary>
	void calculateMapObjects();

	/// <summary>
	/// Method to read processed satellite map data
	/// </summary>
	void loadMapObjects();

	// Position verification

	/// <summary>
	/// Method to calculate coordinates of satellite map extreme vertices
	/// </summary>
	void calcMapEdges();

	/// <summary>
	/// Method to check position inside satellite map
	/// </summary>
	/// <param name="currX">current position X</param>
	/// <param name="currY">current position Y</param>
	/// <param name="FOVX">OX camera FOV</param>
	/// <param name="FOVY">OY camera FOV</param>
	/// <param name="deltaFOV">FOV delta</param>
	/// <returns>true, if position is inside satellite map; false otherwise</returns>
	bool locationVerification(const double currX, const double currY, const double FOVX, const double FOVY, const double deltaFOV);

	// Local obj processing inner steps

	/// <summary>
	/// Method to detect object
	/// </summary>
	/// <param name="image">input image</param>
	/// <returns>detection list</returns>
	std::vector<Detection> objectDetection(const cv::Mat& image);

	/// <summary>
	/// Method to calculate local detected object's coordinates
	/// </summary>
	/// <param name="detections">detection list</param>
	/// <param name="imgShape">input image shape</param>
	/// <param name="cam_params">camera parameters</param>
	/// <param name="curr_angles">current angles</param>
	/// <param name="curr_offset">current offset</param>
	/// <param name="meter_in_pixel">number of meters in pixel</param>
	/// <returns>processed local objects list</returns>
	std::vector<LocalData> objectCoordProc(const std::vector<Detection>& detections, const cv::Size2f& imgShape, Camera_params& cam_params, Pos_angle& curr_angles, Pos_d3& curr_offset, Pos_f2& meter_in_pixel);

	/// <summary>
	/// Method for local objects verification: adds new objects or updates existing ones
	/// </summary>
	/// <param name="localDataList">local detected objects list</param>
	/// <param name="identityDelta">threshold for object reidentification</param>
	void objectVerification(std::vector<LocalData>& localDataList, const int identityDelta);

	// Local obj processing steps

	/// <summary>
	/// Method to local object detection and processing
	/// </summary>
	/// <param name="image">input image</param>
	/// <param name="identityDelta">threshold for object reidentification</param>
	/// <param name="cam_params">camera parameters</param>
	/// <param name="curr_angles">current angles</param>
	/// <param name="curr_offset">current offset</param>
	/// <param name="meter_in_pixel">number of meters in pixel</param>
	void calculateLocalObjects(const cv::Mat& image, const int identityDelta, Camera_params& cam_params, Pos_angle& curr_angles, Pos_d3& curr_offset, Pos_f2& meter_in_pixel);

	// Object matching

	/// <summary>
	/// Method for local and satellite objects matching, bias calculation and local object's coordinates update
	/// </summary>
	/// <param name="currX">current position X</param>
	/// <param name="currY">current position Y</param>
	/// <param name="FOVX">OX camera FOV</param>
	/// <param name="FOVY">OY camera FOV</param>
	/// <param name="deltaFOV">FOV delta</param>
	/// <param name="deltaOffset">extra width for local and satellite map's masks comparison</param>
	/// <param name="matchDelta">threshold for objects matching on masks</param>
	/// <param name="confOverlap">minimal local object reidentification level (number of overlap)</param>
	/// <param name="objPerClass">minimal number of objects per class for performing masks matching</param>
	/// <param name="scale">number of meters in pixel</param>
	std::vector<double> objectMatcher(const double currX, const double currY, const double FOVX, const double FOVY, const double deltaFOV,
		const double deltaOffset, const int matchDelta, const int confOverlap, const int objPerClass, const double scale);
};
