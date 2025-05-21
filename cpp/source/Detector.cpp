#include "Detector.h"

Detector::Detector(const char* labelsFile, const char* modelPath, bool cudaEnabled, int imgW, int imgH,
	const float scoreThresh, const float nmsThresh, const int maxDet)
	: m_labelsFile{ labelsFile }, m_modelPath{ modelPath }, m_cudaEnabled{ cudaEnabled }, m_imgW{ imgW }, m_imgH{ imgH },
	m_scoreThresh{ scoreThresh }, m_nmsThresh{ nmsThresh }, m_maxDet{ maxDet },
	onnxInferencer{ OnnxInference(m_modelPath, m_labels, m_imgW, m_imgH, m_cudaEnabled, m_scoreThresh, m_nmsThresh, m_maxDet) },
	torchInferencer{ TorchInference(m_modelPath, m_labels, m_imgW, m_imgH, m_cudaEnabled, m_scoreThresh, m_nmsThresh, m_maxDet) } {
	std::ifstream file(Detector::m_labelsFile);

	if (file.is_open()) {
		std::string line;

		while (getline(file, line)) {
			std::stringstream ss(line);
			int i = 0;
			int label;
			std::string labelName;
			while (ss.good()) {
				std::string substr;
				getline(ss, substr, ',');
				switch (i) {
				case 0:
					label = stoi(substr);
					break;
				case 1:
					labelName = substr;
					break;
				default:
					break;
				}

				++i;
			}

			m_labels[label] = labelName;
		}
	}

	file.close();

	std::string mPath = m_modelPath;
	if (mPath.substr(mPath.find_last_of(".") + 1) == "onnx") {
		m_isOnnxModel = true;
		onnxInferencer.loadOnnxNetwork();
		inferencer = &onnxInferencer;
	}
	else {
		m_isOnnxModel = false;
		torchInferencer.loadTorchNetwork();
		inferencer = &torchInferencer;
	}
}


cv::Mat Detector::readImage(const char* inputFile) {
	cv::Mat image = cv::imread(inputFile, cv::IMREAD_COLOR);
	if (image.empty()) {
		std::cout << "Image File: " << inputFile << " Is Not Found" << std::endl;
	}

	return image;
}

std::vector<Detection> Detector::detect(const cv::Mat& image) {
	return inferencer->runInference(image);
}