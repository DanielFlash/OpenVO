#include "SurfaceDataWriter.h"

SurfaceDataWriter::SurfaceDataWriter(const char* outputFile) : m_outputFile{ outputFile } {}

void SurfaceDataWriter::writeData(const std::vector<SurfaceData>& surfaceDataList) {
	std::ofstream file(SurfaceDataWriter::m_outputFile);

	if (file.is_open()) {

		for (SurfaceData surfaceData : surfaceDataList) {
			std::string line = surfaceData.imgName + ", " +
				std::to_string(surfaceData.imgW) + ", " + std::to_string(surfaceData.imgH) + ", " +
				std::to_string(surfaceData.imgTopLeftX) + ", " + std::to_string(surfaceData.imgTopLeftY) + ", " +
				std::to_string(surfaceData.imgBotRightX) + ", " + std::to_string(surfaceData.imgBotRightY) + ", " +
				std::to_string(surfaceData.objId) + ", " + std::to_string(surfaceData.objLabel) + ", " +
				std::to_string(surfaceData.bbX) + ", " + std::to_string(surfaceData.bbY) + ", " +
				std::to_string(surfaceData.bbW) + ", " + std::to_string(surfaceData.bbH) + ", " +
				std::to_string(surfaceData.objCoordX) + ", " + std::to_string(surfaceData.objCoordY);

			file << line << std::endl;
		}

	}

	file.close();
}
