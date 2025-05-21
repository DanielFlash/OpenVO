#include "MapAnalysis.h"

MapAnalysis::MapAnalysis(
	const char* inputFile, const char* imgFolder, const char* outputFile, const char* labelsFile,
	const char* modelPath, bool cudaEnabled, int imgW, int imgH, float scThres, float nmsThres, int maxD,
	const char* glModelPath, bool glCudaEnabled, int glImgW, int glImgH, float glScThres, float glNmsThres, int glMaxD
)
	: m_inputFile{ inputFile }, m_imgFolder{ imgFolder }, m_outputFile{ outputFile }, m_labelsFile{ labelsFile },
	m_modelPath{ modelPath }, m_cudaEnabled{ cudaEnabled }, m_imgW{ imgW }, m_imgH{ imgH }, m_scoreThresh{ scThres },
	m_nmsThresh{ nmsThres }, m_maxDet{ maxD }, m_globalModelPath{ glModelPath }, m_glCudaEnabled{ glCudaEnabled },
	m_glImgW{ glImgW }, m_glImgH{ glImgH }, m_glScoreThresh{ glScThres }, m_glNmsThresh{ glNmsThres }, m_glMaxDet{ glMaxD }, 
	m_surfaceDataReader(inputFile, m_imgFolder, m_outputFile), m_surfaceDataWriter(m_outputFile), m_detector(m_labelsFile, m_modelPath, m_cudaEnabled, m_imgW, m_imgH, m_scoreThresh, m_nmsThresh, m_maxDet), 
	m_glDetector(m_labelsFile, m_globalModelPath, m_glCudaEnabled, m_glImgW, m_glImgH, m_glScoreThresh, m_glNmsThresh, m_glMaxDet) {

	m_coordCalculator = CoordCalculator();
}

MapAnalysis::MapAnalysis(
	const char* inputFile, const char* imgFolder, const char* outputFile, const char* labelsFile,
	const char* modelPath, bool cudaEnabled, int imgW, int imgH, float scThres, float nmsThres, int maxD
)
	: m_inputFile{ inputFile }, m_imgFolder{ imgFolder }, m_outputFile{ outputFile }, m_labelsFile{ labelsFile },
	m_modelPath{ modelPath }, m_cudaEnabled{ cudaEnabled }, m_imgW{ imgW }, m_imgH{ imgH }, m_scoreThresh{ scThres },
	m_nmsThresh{ nmsThres }, m_maxDet{ maxD }, m_globalModelPath{ nullptr }, m_surfaceDataReader(inputFile, m_imgFolder, m_outputFile), 
	m_surfaceDataWriter(m_outputFile), m_detector(m_labelsFile, m_modelPath, m_cudaEnabled, m_imgW, m_imgH, m_scoreThresh, m_nmsThresh, m_maxDet), m_glDetector(m_detector) {

	m_coordCalculator = CoordCalculator();
}

void MapAnalysis::loadRawData() {
	m_surfaceImgDataList = m_surfaceDataReader.readRawData();
}

void MapAnalysis::loadRawLabeledData() {
	m_surfaceObjDataList = m_surfaceDataReader.readRawLabeledData();
}

void MapAnalysis::loadProcessedData() {
	m_surfaceDataList = m_surfaceDataReader.readProcessedData();
}

void MapAnalysis::saveProcessedData() {
	m_surfaceDataWriter.writeData(m_surfaceDataList);
}

void MapAnalysis::processRawData() {
	if (m_imgFolder == nullptr) {
		m_surfaceDataList = m_coordCalculator.calcObjCoords(m_surfaceObjDataList);
	}
	else {
		m_surfaceDataList = m_coordCalculator.detectAndCalcObjCoords(m_surfaceImgDataList, m_glDetector, m_imgFolder);
	}
}

void MapAnalysis::calculateMapObjects() {
	if (m_imgFolder == nullptr) {
		loadRawLabeledData();
	}
	else {
		loadRawData();
	}
	processRawData();
	saveProcessedData();
}

void MapAnalysis::loadMapObjects() {
	loadProcessedData();
}

void MapAnalysis::calcMapEdges() {
	m_mapEdges = m_coordCalculator.calcMapEdges(m_surfaceDataList);
}

bool MapAnalysis::locationVerification(const double currX, const double currY, const double FOVX, const double FOVY, const double deltaFOV) {
	if ((currX - FOVX - deltaFOV > m_mapEdges.topLeftX) && (currX + FOVX + deltaFOV < m_mapEdges.botRightX) &&
		(currY - FOVY - deltaFOV < m_mapEdges.topLeftY) && (currY + FOVY + deltaFOV > m_mapEdges.botRightY)) {
		return true;
	}
	else {
		return false;
	}
}

std::vector<Detection> MapAnalysis::objectDetection(const cv::Mat& image) {
	std::vector<Detection> output = m_detector.detect(image);
	return output;
}

std::vector<LocalData> MapAnalysis::objectCoordProc(const std::vector<Detection>& detections, const cv::Size2f& imgShape, Camera_params& cam_params, Pos_angle& curr_angles, Pos_d3& curr_offset, Pos_f2& meter_in_pixel) {
	std::vector<LocalData> localDataList = m_coordCalculator.calcLocalObjCoords(detections, imgShape, cam_params, curr_angles, curr_offset, meter_in_pixel);
	return localDataList;
}

void MapAnalysis::objectVerification(std::vector<LocalData>& localDataList, const int identityDelta) {
	for (LocalData& localData : localDataList) {
		int minBias = 0;
		LocalData* bestCandidate = nullptr;
		int maxId = -1;
		for (LocalData& m_localData : m_localDataList) {
			if ((m_localData.objCoordX - identityDelta < localData.objCoordX) && (localData.objCoordX < m_localData.objCoordX + identityDelta) &&
				(m_localData.objCoordY - identityDelta < localData.objCoordY) && (localData.objCoordY < m_localData.objCoordY + identityDelta)) {
				float deltaX = std::abs(m_localData.objCoordX - localData.objCoordX);
				float deltaY = std::abs(m_localData.objCoordY - localData.objCoordY);
				if (((bestCandidate == nullptr) || (deltaX + deltaY < minBias)) && (m_localData.objLabel == localData.objLabel)) {
					minBias = deltaX + deltaY;
					bestCandidate = &m_localData;
				}
			}

			if (m_localData.objId > maxId) {
				maxId = m_localData.objId;
			}
		}

		if (bestCandidate != nullptr) {
			bestCandidate->objCoordX = (bestCandidate->objCoordX + localData.objCoordX) / 2;
			bestCandidate->objCoordY = (bestCandidate->objCoordY + localData.objCoordY) / 2;
			bestCandidate->overlapLevel += 1;
		}
		else {
			localData.objId = maxId + 1;
			m_localDataList.push_back(localData);
		}
	}
}

void MapAnalysis::calculateLocalObjects(const cv::Mat& image, const int identityDelta, Camera_params& cam_params, Pos_angle& curr_angles, Pos_d3& curr_offset, Pos_f2& meter_in_pixel) {
	cv::Size2f imgShape{};
	imgShape.height = image.size().height;
	imgShape.width = image.size().width;
	std::vector<Detection> output = objectDetection(image);
	std::vector<LocalData> localDataList = objectCoordProc(output, imgShape, cam_params, curr_angles, curr_offset, meter_in_pixel);
	objectVerification(localDataList, identityDelta);
}

std::vector<double> MapAnalysis::objectMatcher(const double currX, const double currY, const double FOVX, const double FOVY,
	const double deltaFOV, const double deltaOffset, const int matchDelta, const int confOverlap,
	const int objPerClass, const double scale) {

	double localMinX = currX - FOVX - deltaFOV;
	double localMaxX = currX + FOVX + deltaFOV;
	double localMinY = currY - FOVY - deltaFOV;
	double localMaxY = currY + FOVY + deltaFOV;
	double globalMinX = localMinX - deltaOffset;
	double globalMaxX = localMaxX + deltaOffset;
	double globalMinY = localMinY - deltaOffset;
	double globalMaxY = localMaxY + deltaOffset;

	std::vector<LocalData*> confidentObjects;
	for (LocalData& localData : m_localDataList) {
		if (localData.overlapLevel >= confOverlap) {
			confidentObjects.push_back(&localData);
		}
	}
	
	if (confidentObjects.size() >= objPerClass) {
		std::vector<LocalData*> properObjects;
		for (LocalData* localData : confidentObjects) {
			if ((localData->objCoordX > localMinX) && (localData->objCoordX < localMaxX) &&
				(localData->objCoordY > localMinY) && (localData->objCoordY < localMaxY)) {
				properObjects.push_back(localData);
			}
		}
		
		if (properObjects.size() >= objPerClass) {
			std::map<int, int> objectClasses;
			std::vector<int> objClKeys;
			std::vector<SurfaceData*> properSurfaceObj;
			for (SurfaceData& surfaceData : m_surfaceDataList) {
				if ((surfaceData.objCoordX > globalMinX) && (surfaceData.objCoordX < globalMaxX) &&
					(surfaceData.objCoordY > globalMinY) && (surfaceData.objCoordY < globalMaxY)) {
					properSurfaceObj.push_back(&surfaceData);
					if (objectClasses.find(surfaceData.objLabel) == objectClasses.end()) {
						objectClasses[surfaceData.objLabel] = 0;
						objClKeys.push_back(surfaceData.objLabel);
					}
				}
			}
			
			if (properSurfaceObj.size() == 0) {
				return std::vector<double> { 0., 0.};
			}

			for (LocalData* localData : properObjects) {
				if (objectClasses.find(localData->objLabel) != objectClasses.end()) {
					objectClasses[localData->objLabel] += 1;
				}
			}

			double lw = std::round((localMaxX - localMinX) / scale);
			double lh = std::round((localMaxY - localMinY) / scale);
			double gw = std::round((globalMaxX - globalMinX) / scale);
			double gh = std::round((globalMaxY - globalMinY) / scale);

			std::map<int, std::vector<ObjectDist>> deltasPerClass;
			std::vector<int> deltasPClKeys;
			for (const int key : objClKeys) {
				if (objectClasses[key] >= objPerClass) {
					std::vector<double> deltas = calcDeltas(localMinX, localMaxY, globalMinX, globalMaxY, lw, lh, gw, gh,
						key, scale, matchDelta, properObjects, properSurfaceObj);

					if (deltas[0] == 0. && deltas[1] == 0.) {
						return std::vector<double> { 0., 0.};
					}

					std::vector<ObjectDist> objectDistList = calcObjDist(deltas, key, properObjects, properSurfaceObj);
					if (objectDistList.size() != 0) {
						deltasPerClass[key] = objectDistList;
						deltasPClKeys.push_back(key);
					}
				}
			}

			if (deltasPerClass.size() == 0) {
				return std::vector<double> { 0., 0.};
			}

			std::vector<ObjectDist>* bestCandidates = nullptr;
			double minDist = -1;
			for (int key : deltasPClKeys) {
				if (minDist == -1) {
					minDist = deltasPerClass[key][0].dist;
					bestCandidates = &deltasPerClass[key];
				}
				else if (deltasPerClass[key][0].dist < minDist) {
					minDist = deltasPerClass[key][0].dist;
					bestCandidates = &deltasPerClass[key];
				}
			}

			int objPerClassPassed = 0;
			for (const ObjectDist& objectDist : *bestCandidates) {
				if (objectDist.dist < matchDelta) {
					objPerClassPassed += 1;
					if (objPerClassPassed >= objPerClass) {
						break;
					}
				}
			}

			if (objPerClassPassed >= objPerClass) {
				double deltaX = (*bestCandidates)[0].deltaX;
				double deltaY = (*bestCandidates)[0].deltaY;

				mapObjects(bestCandidates);

				updateLocalDataCoord(deltaX, deltaY);

				return std::vector<double> { deltaX, deltaY };
			}

			else {
				return std::vector<double> { 0., 0.};
			}
		}

		else {
			return std::vector<double> { 0., 0.};
		}
	}

	else {
		return std::vector<double> { 0., 0.};
	}
}

std::vector<double> MapAnalysis::calcDeltas(const double localMinX, const double localMaxY, const double globalMinX, const double globalMaxY,
	const double lw, const double lh, const double gw, const double gh, const int label, const double scale, const int matchDelta,
	const std::vector<LocalData*> properObjects, const std::vector<SurfaceData*> properSurfaceObj) {

	cv::Mat localMap = cv::Mat(lh, lw, CV_32F, 0.0);
	cv::Mat globalMap = cv::Mat(gh, gw, CV_32F, 0.0);

	for (LocalData* localData : properObjects) {
		if (localData->objLabel == label) {
			int xNorm = std::round((localData->objCoordX - localMinX) / scale);
			int yNorm = std::round((localMaxY - localData->objCoordY) / scale);

			if (xNorm == lw) {
				xNorm -= 1;
			}
			else if (xNorm < 0) {
				xNorm = 0;
			}
			if (yNorm == lh) {
				yNorm -= 1;
			}
			else if (yNorm < 0) {
				yNorm = 0;
			}

			localMap.at<int>(yNorm, xNorm) = 1;
		}
	}

	if (cv::countNonZero(localMap) < 1) {
		return std::vector<double> { 0.0, 0.0 };
	}

	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(matchDelta, matchDelta));

	cv::Mat dilatedLocalMap;
	cv::dilate(localMap, dilatedLocalMap, kernel);

	for (SurfaceData* surfaceData : properSurfaceObj) {
		if (surfaceData->objLabel == label) {
			int xNorm = std::round((surfaceData->objCoordX - globalMinX) / scale);
			int yNorm = std::round((globalMaxY - surfaceData->objCoordY) / scale);

			if (xNorm == gw) {
				xNorm -= 1;
			}
			else if (xNorm < 0) {
				xNorm = 0;
			}
			if (yNorm == gh) {
				yNorm -= 1;
			}
			else if (yNorm < 0) {
				yNorm = 0;
			}

			localMap.at<int>(yNorm, xNorm) = 1;
		}
	}

	cv::Mat dilatedGlobalMap;
	cv::dilate(globalMap, dilatedGlobalMap, kernel);

	cv::Mat res(dilatedGlobalMap.rows - dilatedLocalMap.rows + 1, dilatedGlobalMap.cols - dilatedLocalMap.cols + 1, CV_32FC1);
	cv::matchTemplate(dilatedGlobalMap, dilatedLocalMap, res, cv::TM_CCOEFF);
	cv::normalize(res, res, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
	double minVal;
	double maxVal;
	cv::Point minLoc;
	cv::Point maxLoc;
	minMaxLoc(res, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());
	cv::Point matchLoc = maxLoc;

	double deltaX = matchLoc.x * scale;
	double deltaY = matchLoc.y * scale;
	deltaX = globalMinX - localMinX + deltaX;
	deltaY = globalMaxY - localMaxY + deltaY;

	return std::vector<double> { deltaX, deltaY };
}

std::vector<ObjectDist> MapAnalysis::calcObjDist(const std::vector<double>& deltas, const int label,
	const std::vector<LocalData*> properObjects, const std::vector<SurfaceData*> properSurfaceObj) {

	std::vector<ObjectDist> objectDistList;
	for (LocalData* localData : properObjects) {
		if (localData->objLabel == label) {
			for (SurfaceData* surfaceData : properSurfaceObj) {
				if (surfaceData->objLabel == label) {
					double distX = surfaceData->objCoordX - (localData->objCoordX + deltas[0]);
					double distY = surfaceData->objCoordY - (localData->objCoordY + deltas[1]);
					double dist = std::sqrt(distX * distX + distY * distY);
					ObjectDist objectDist;
					objectDist.dist = dist;
					objectDist.deltaX = deltas[0];
					objectDist.deltaY = deltas[1];
					objectDist.localData = localData;
					objectDist.surfaceData = surfaceData;
					objectDistList.push_back(objectDist);
				}
			}
		}
	}

	std::sort(objectDistList.begin(), objectDistList.end(), [](ObjectDist a, ObjectDist b) {
		return a.dist < b.dist;
		});

	return objectDistList;
}

void MapAnalysis::mapObjects(std::vector<ObjectDist>* bestCandidates) {
	for (const ObjectDist& objectDist : *bestCandidates) {
		if ((objectDist.localData->mappedTo == -1) && (objectDist.surfaceData->mappedTo == -1)) {
			objectDist.localData->mappedTo = objectDist.surfaceData->objId;
			objectDist.surfaceData->mappedTo = objectDist.localData->objId;
		}
	}
}

void MapAnalysis::updateLocalDataCoord(const double deltaX, const double deltaY) {
	for (LocalData& localData : m_localDataList) {
		localData.objCoordX += deltaX;
		localData.objCoordY += deltaY;
	}
}
