#include "CoordCalculator.h"

std::vector<SurfaceData> CoordCalculator::calcObjCoords(const std::vector<SurfaceObjData>& surfaceObjDataList) {
	std::vector<SurfaceData> surfaceDataList;
	int i = 0;
	for (SurfaceObjData surfaceObjData : surfaceObjDataList) {
		int centralPointX = surfaceObjData.bbX + std::round(surfaceObjData.bbW / 2);
		int centralPointY = surfaceObjData.bbY + std::round(surfaceObjData.bbH / 2);
		double relCoordX = centralPointX * (surfaceObjData.imgBotRightX - surfaceObjData.imgTopLeftX) / surfaceObjData.imgW;
		double relCoordY = centralPointY * (surfaceObjData.imgBotRightY - surfaceObjData.imgTopLeftY) / surfaceObjData.imgH;
		double coordX = surfaceObjData.imgTopLeftX + relCoordX;
		double coordY = surfaceObjData.imgTopLeftY + relCoordY;

		SurfaceData surfaceData;
		surfaceData.imgName = surfaceObjData.imgName;
		surfaceData.imgW = surfaceObjData.imgW;
		surfaceData.imgH = surfaceObjData.imgH;
		surfaceData.imgTopLeftX = surfaceObjData.imgTopLeftX;
		surfaceData.imgTopLeftY = surfaceObjData.imgTopLeftY;
		surfaceData.imgBotRightX = surfaceObjData.imgBotRightX;
		surfaceData.imgBotRightY = surfaceObjData.imgBotRightY;
		surfaceData.objId = i;
		surfaceData.objLabel = surfaceObjData.objLabel;
		surfaceData.bbX = surfaceObjData.bbX;
		surfaceData.bbY = surfaceObjData.bbY;
		surfaceData.bbW = surfaceObjData.bbW;
		surfaceData.bbH = surfaceObjData.bbH;
		surfaceData.objCoordX = coordX;
		surfaceData.objCoordY = coordY;

		surfaceDataList.push_back(surfaceData);

		++i;
	}

	return surfaceDataList;
}

std::vector<SurfaceData> CoordCalculator::detectAndCalcObjCoords(const std::vector<SurfaceImgData>& surfaceImgDataList, Detector& detector, const char* imgFolder) {
	std::vector<SurfaceData> surfaceDataList;
	int i = 0;
	namespace fs = std::filesystem;
	for (SurfaceImgData surfaceImgData : surfaceImgDataList) {
		fs::path folderPath(imgFolder);
		fs::path imagePath = folderPath.append(surfaceImgData.imgName);
		if (fs::exists(imagePath)) {
			const char* str_imagePath(imagePath.u8string().c_str());
			cv::Mat frame = detector.readImage(str_imagePath);
			std::vector<Detection> output = detector.detect(frame);

			for (int i = 0; i < output.size(); ++i)
			{
				Detection detection = output[i];

				int centralPointX = detection.x + std::round(detection.w / 2);
				int centralPointY = detection.y + std::round(detection.h / 2);
				double relCoordX = centralPointX * (surfaceImgData.imgBotRightX - surfaceImgData.imgTopLeftX) / surfaceImgData.imgW;
				double relCoordY = centralPointY * (surfaceImgData.imgBotRightY - surfaceImgData.imgTopLeftY) / surfaceImgData.imgH;
				double coordX = surfaceImgData.imgTopLeftX + relCoordX;
				double coordY = surfaceImgData.imgTopLeftY + relCoordY;

				SurfaceData surfaceData;
				surfaceData.imgName = surfaceImgData.imgName;
				surfaceData.imgW = surfaceImgData.imgW;
				surfaceData.imgH = surfaceImgData.imgH;
				surfaceData.imgTopLeftX = surfaceImgData.imgTopLeftX;
				surfaceData.imgTopLeftY = surfaceImgData.imgTopLeftY;
				surfaceData.imgBotRightX = surfaceImgData.imgBotRightX;
				surfaceData.imgBotRightY = surfaceImgData.imgBotRightY;
				surfaceData.objId = i;
				surfaceData.objLabel = detection.class_id;
				surfaceData.bbX = detection.x;
				surfaceData.bbY = detection.y;
				surfaceData.bbW = detection.w;
				surfaceData.bbH = detection.h;
				surfaceData.objCoordX = coordX;
				surfaceData.objCoordY = coordY;

				surfaceDataList.push_back(surfaceData);
			}
		}
	}

	return surfaceDataList;
}

MapEdges CoordCalculator::calcMapEdges(const std::vector<SurfaceData>& surfaceDataList) {
	MapEdges mapEdges;
	mapEdges.topLeftX = surfaceDataList[0].imgTopLeftX;
	mapEdges.topLeftY = surfaceDataList[0].imgTopLeftY;
	mapEdges.botRightX = surfaceDataList[0].imgBotRightX;
	mapEdges.botRightY = surfaceDataList[0].imgBotRightY;
	for (SurfaceData surfaceData : surfaceDataList) {
		if (mapEdges.topLeftX > surfaceData.imgTopLeftX) {
			mapEdges.topLeftX = surfaceData.imgTopLeftX;
		}
		if (mapEdges.topLeftY < surfaceData.imgTopLeftY) {
			mapEdges.topLeftY = surfaceData.imgTopLeftY;
		}
		if (mapEdges.botRightX < surfaceData.imgBotRightX) {
			mapEdges.botRightX = surfaceData.imgBotRightX;
		}
		if (mapEdges.botRightY > surfaceData.imgBotRightY) {
			mapEdges.botRightY = surfaceData.imgBotRightY;
		}
	}

	return mapEdges;
}

std::vector<LocalData> CoordCalculator::calcLocalObjCoords(const std::vector<Detection>& detections, const cv::Size2f& imgShape,
	Camera_params& cam_params, Pos_angle& curr_angles, Pos_d3& curr_offset, Pos_f2& meter_in_pixel) {
	std::vector<LocalData> localDataList;
	for (Detection detection : detections) {
		int centralPointX = detection.x + std::round(detection.w / 2);
		int centralPointY = detection.y + std::round(detection.h / 2);

		Pos_f2 scale;
		Pos_f2 offset;
		Pos_f2 vs;
		scale.x = cam_params.resolution.x / imgShape.width;
		scale.y = cam_params.resolution.y / imgShape.height;
		offset.x = centralPointX * scale.x;
		offset.y = centralPointY * scale.y;
		vs.x = (offset.x - cam_params.resolution.x / 2) * meter_in_pixel.x;
		vs.y = (offset.y - cam_params.resolution.y / 2) * meter_in_pixel.y;
		double objCoordX = curr_offset.x + (vs.x * cosf(curr_angles.yaw) - vs.y * sinf(curr_angles.yaw));
		double objCoordY = curr_offset.x + (vs.x * sinf(curr_angles.yaw) + vs.y * cosf(curr_angles.yaw));

		LocalData localdata;
		localdata.objLabel = detection.class_id;
		localdata.objCoordX = objCoordX;
		localdata.objCoordY = objCoordY;

		localDataList.push_back(localdata);
	}

	return localDataList;
}
