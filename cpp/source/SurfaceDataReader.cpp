#include "SurfaceDataReader.h"

SurfaceDataReader::SurfaceDataReader(const char* inputFile, const char* imgFolder, const char* outputFile)
	: m_inputFile{ inputFile }, m_imgFolder{ imgFolder }, m_outputFile{ outputFile } {}


std::vector<SurfaceData> SurfaceDataReader::readProcessedData() {
	std::ifstream file(SurfaceDataReader::m_outputFile);

	std::vector<SurfaceData> surfaceDataList;

	if (file.is_open()) {
		std::string line;

		while (getline(file, line)) {
			std::stringstream ss(line);
			SurfaceData surfaceData;
			int i = 0;

			while (ss.good()) {
				std::string substr;
				getline(ss, substr, ',');
				switch (i) {
				case 0:
					surfaceData.imgName = substr;
					break;
				case 1:
					surfaceData.imgW = std::stoi(substr);
					break;
				case 2:
					surfaceData.imgH = std::stoi(substr);
					break;
				case 3:
					surfaceData.imgTopLeftX = std::stod(substr);
					break;
				case 4:
					surfaceData.imgTopLeftY = std::stod(substr);
					break;
				case 5:
					surfaceData.imgBotRightX = std::stod(substr);
					break;
				case 6:
					surfaceData.imgBotRightY = std::stod(substr);
					break;
				case 7:
					surfaceData.objId = std::stoi(substr);
					break;
				case 8:
					surfaceData.objLabel = std::stoi(substr);
					break;
				case 9:
					surfaceData.bbX = std::stoi(substr);
					break;
				case 10:
					surfaceData.bbY = std::stoi(substr);
					break;
				case 11:
					surfaceData.bbW = std::stoi(substr);
					break;
				case 12:
					surfaceData.bbH = std::stoi(substr);
					break;
				case 13:
					surfaceData.objCoordX = std::stod(substr);
					break;
				case 14:
					surfaceData.objCoordY = std::stod(substr);
					break;
				default:
					break;
				}

				++i;
			}

			surfaceDataList.push_back(surfaceData);
		}
	}

	file.close();

	return surfaceDataList;
}

std::vector<SurfaceObjData> SurfaceDataReader::readRawLabeledData() {
	std::ifstream file(SurfaceDataReader::m_inputFile);

	std::vector<SurfaceObjData> surfaceObjDataList;

	if (file.is_open()) {
		std::string line;

		while (getline(file, line)) {
			std::stringstream ss(line);
			SurfaceObjData surfaceObjData;
			int i = 0;

			while (ss.good()) {
				std::string substr;
				getline(ss, substr, ',');
				switch (i) {
				case 0:
					surfaceObjData.imgName = substr;
					break;
				case 1:
					surfaceObjData.imgW = std::stoi(substr);
					break;
				case 2:
					surfaceObjData.imgH = std::stoi(substr);
					break;
				case 3:
					surfaceObjData.imgTopLeftX = std::stod(substr);
					break;
				case 4:
					surfaceObjData.imgTopLeftY = std::stod(substr);
					break;
				case 5:
					surfaceObjData.imgBotRightX = std::stod(substr);
					break;
				case 6:
					surfaceObjData.imgBotRightY = std::stod(substr);
					break;
				case 7:
					surfaceObjData.objLabel = std::stoi(substr);
					break;
				case 8:
					surfaceObjData.bbX = std::stoi(substr);
					break;
				case 9:
					surfaceObjData.bbY = std::stoi(substr);
					break;
				case 10:
					surfaceObjData.bbW = std::stoi(substr);
					break;
				case 11:
					surfaceObjData.bbH = std::stoi(substr);
					break;
				default:
					break;
				}

				++i;
			}

			surfaceObjDataList.push_back(surfaceObjData);
		}
	}

	file.close();

	return surfaceObjDataList;
}

std::vector<SurfaceImgData> SurfaceDataReader::readRawData() {
	std::ifstream file(SurfaceDataReader::m_inputFile);

	std::vector<SurfaceImgData> surfaceImgDataList;

	if (file.is_open()) {
		std::string line;

		while (getline(file, line)) {
			std::stringstream ss(line);
			SurfaceImgData surfaceImgData;
			int i = 0;

			while (ss.good()) {
				std::string substr;
				getline(ss, substr, ',');
				switch (i) {
				case 0:
					surfaceImgData.imgName = substr;
					break;
				case 1:
					surfaceImgData.imgW = std::stoi(substr);
					break;
				case 2:
					surfaceImgData.imgH = std::stoi(substr);
					break;
				case 3:
					surfaceImgData.imgTopLeftX = std::stod(substr);
					break;
				case 4:
					surfaceImgData.imgTopLeftY = std::stod(substr);
					break;
				case 5:
					surfaceImgData.imgBotRightX = std::stod(substr);
					break;
				case 6:
					surfaceImgData.imgBotRightY = std::stod(substr);
					break;
				default:
					break;
				}

				++i;
			}

			surfaceImgDataList.push_back(surfaceImgData);
		}
	}

	file.close();

	return surfaceImgDataList;
}