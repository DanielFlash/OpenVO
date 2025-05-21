#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <chrono>

#include "MapAnalysis.h"
#include "BaseAgentDQN.h"
#include "BaseAgentSARSA.h"
#include "BaseAgentPG.h"
#include "BaseAgentA2C.h"
#include "BaseAgentPPO.h"


class Examples {
public:
    void detectionExample(const std::string& projectBasePath, const std::string& modelName, const std::string& labels, const std::string& testImg, const bool runOnGPU) {
        
        auto start = std::chrono::high_resolution_clock::now();
        
        Detector detector((projectBasePath + labels).c_str(), (projectBasePath + modelName).c_str(), runOnGPU, 640, 640);
        cv::Mat frame = detector.readImage((projectBasePath + testImg).c_str());
        std::vector<Detection> output = detector.detect(frame);

        int detections = output.size();
        std::cout << "Number of detections:" << detections << std::endl;

        for (int i = 0; i < detections; ++i)
        {
            Detection detection = output[i];

            std::cout << detection.class_id << " " << detection.className << " " << detection.confidence << " " << detection.x << " " << detection.y << " " << detection.w << " " << detection.h << " " << std::endl;

            cv::Rect box{ detection.x, detection.y, detection.w, detection.h };

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int> dis(100, 255);
            cv::Scalar color = cv::Scalar(dis(gen), dis(gen), dis(gen));

            // Detection box
            cv::rectangle(frame, box, color, 2);

            // Detection box text
            std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
            cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
            cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

            cv::rectangle(frame, textBox, color, cv::FILLED);
            cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
        }

        float scale = 0.8;
        cv::resize(frame, frame, cv::Size(frame.cols * scale, frame.rows * scale));
        
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Duration: " << duration.count() << std::endl;
        
        cv::imshow("Inference", frame);

        cv::waitKey(-1);
    }

    void surfaceDataProcExample(const std::string& inputFile1, const std::string& inputFile2, const std::string& imgFolder, const std::string& outputFile) {

        auto start = std::chrono::high_resolution_clock::now();

        SurfaceDataWriter surfaceDataWriter(outputFile.c_str());
        std::vector<SurfaceData> surfaceDataList;
        for (int i = 0; i < 2; i++) {
            SurfaceData surfaceData;
            surfaceData.imgName = "test.jpg";
            surfaceData.imgW = 512;
            surfaceData.imgH = 512;
            surfaceData.imgTopLeftX = 12.3;
            surfaceData.imgTopLeftY = 12.5;
            surfaceData.imgBotRightX = 53.23;
            surfaceData.imgBotRightY = 53.09;
            surfaceData.objId = i;
            surfaceData.objLabel = i + 10;
            surfaceData.bbX = 44;
            surfaceData.bbY = 55;
            surfaceData.bbW = 30;
            surfaceData.bbH = 35;
            surfaceData.objCoordX = 123;
            surfaceData.objCoordY = 567;
            surfaceDataList.push_back(surfaceData);
        }
        surfaceDataWriter.writeData(surfaceDataList);

        SurfaceDataReader surfaceDataReader(inputFile1.c_str(), imgFolder.c_str(), outputFile.c_str());
        std::vector<SurfaceData> surfaceDataList2;
        surfaceDataList2 = surfaceDataReader.readProcessedData();
        std::cout << "read processed data:" << std::endl;
        for (SurfaceData surfaceData : surfaceDataList2) {
            std::cout << surfaceData.imgName << ", " << surfaceData.objId << std::endl;
        }
        std::vector<SurfaceObjData> surfaceObjDataList;
        surfaceObjDataList = surfaceDataReader.readRawLabeledData();
        std::cout << "read obj data:" << std::endl;
        for (SurfaceObjData surfaceObjData : surfaceObjDataList) {
            std::cout << surfaceObjData.imgName << ", " << surfaceObjData.objLabel << std::endl;
        }

        CoordCalculator coordCalculator;
        std::vector<SurfaceData> surfaceDataList3;
        surfaceDataList3 = coordCalculator.calcObjCoords(surfaceObjDataList);
        std::cout << "calc obj coords:" << std::endl;
        for (SurfaceData surfaceData : surfaceDataList3) {
            std::cout << surfaceData.imgName << ", " << surfaceData.objId << std::endl;
        }

        MapEdges mapEdges;
        mapEdges = coordCalculator.calcMapEdges(surfaceDataList3);
        std::cout << "map edges: " << mapEdges.botRightX << ", " << mapEdges.botRightY << ", " << mapEdges.topLeftX << ", " << mapEdges.topLeftY << std::endl;

        SurfaceDataReader surfaceDataReader2(inputFile2.c_str(), imgFolder.c_str(), outputFile.c_str());
        std::vector<SurfaceImgData> surfaceImgDataList;
        surfaceImgDataList = surfaceDataReader2.readRawData();
        std::cout << "read img data:" << std::endl;
        for (SurfaceImgData surfaceImgData : surfaceImgDataList) {
            std::cout << surfaceImgData.imgName << ", " << surfaceImgData.imgBotRightX << std::endl;
        }

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Duration: " << duration.count() << std::endl;
    }

    void detectionAndCalcExample(const std::string& projectBasePath, const std::string& modelName, const std::string& labels, const std::string& testImg, const bool runOnGPU, const std::string& inputFile, const std::string& imgFolder, const std::string& outputFile) {

        auto start = std::chrono::high_resolution_clock::now();

        Detector detector((projectBasePath + labels).c_str(), (projectBasePath + modelName).c_str(), runOnGPU, 640, 640);
        SurfaceDataReader surfaceDataReader(inputFile.c_str(), imgFolder.c_str(), outputFile.c_str());
        std::vector<SurfaceImgData> surfaceImgDataList;
        surfaceImgDataList = surfaceDataReader.readRawData();
        CoordCalculator coordCalculator;
        std::vector<SurfaceData> surfaceDataList;
        surfaceDataList = coordCalculator.detectAndCalcObjCoords(surfaceImgDataList, detector, imgFolder.c_str());
        for (SurfaceData surfaceData : surfaceDataList) {
            std::cout << surfaceData.imgName << ", " << surfaceData.objId << std::endl;
        }

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Duration: " << duration.count() << std::endl;
    }

    void mapAnalysisExample(const std::string& projectBasePath, const std::string& modelName, const std::string& labels, const std::string& testImg, const bool runOnGPU, const std::string& inputFile, const std::string& imgFolder, const std::string& outputFile) {

        auto start = std::chrono::high_resolution_clock::now();

        MapAnalysis mapAnalysis(inputFile.c_str(), imgFolder.c_str(), outputFile.c_str(), (projectBasePath + labels).c_str(), (projectBasePath + modelName).c_str(), runOnGPU, 640, 640, 0.45, 0.50, 100, (projectBasePath + modelName).c_str(), runOnGPU, 640, 640, 0.45, 0.50, 100);
        mapAnalysis.calculateMapObjects();
        mapAnalysis.calcMapEdges();
        bool res = mapAnalysis.locationVerification(15.6, 14.5, 0.84, 0.58, 0.2);
        std::cout << res << std::endl;

        Detector detector((projectBasePath + labels).c_str(), (projectBasePath + modelName).c_str(), runOnGPU, 640, 640);
        cv::Mat frame = detector.readImage((projectBasePath + testImg).c_str());
        // mapAnalysis.calculateLocalObjects(frame, 10, ); // Add your parameters
        // mapAnalysis.calculateLocalObjects(frame, 10, ); // Do it twice to increase objects overlap (like if it were two frames)
        
        auto deltas = mapAnalysis.objectMatcher(15.6, 14.5, 400, 400, 0.2, 2, 100, 1, 1, 1);
        std::cout << deltas << std::endl;

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Duration: " << duration.count() << std::endl;
    }
    
    void trajectoryPrediction() {

        auto start = std::chrono::high_resolution_clock::now();

        const int maxMemory = 100000;
        const int batchSize = 1000;
        const int randCoef = 60;
        const int randRange = 200;
        const int64_t inputSize = 4;
        const std::vector<int64_t> hiddenSizes{ 32, 16 };
        const int64_t numClasses = 4;
        const double lr = 0.001;
        const double gamma = 0.9;
        torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        std::cout << device << std::endl;

        BaseModelDQN model(inputSize, hiddenSizes, numClasses);

        torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(lr));
       
        BaseTrainerDQN trainer(&model, gamma, &optimizer, &device);
        BaseAgentDQN agent(maxMemory, batchSize, randCoef, randRange, &model, &trainer);

        std::vector<int> oldState{ 1, 0, 0, 1 };
        std::vector<int> newState{ 0, 1, 0, 1 };

        std::vector<int> finalMove = agent.act(oldState);        
        std::vector<int> newMove = agent.act(newState);
        std::cout << finalMove << ", " << newMove << std::endl;

        MemoryCell cell;
        cell.state = oldState;
        cell.action = finalMove;
        cell.nextState = newState;
        cell.reward = 2.3;
        cell.done = false;

        std::cout << "Short memory training..." << std::endl;
        agent.trainShortMemory(cell);

        std::cout << "Long memory training..." << std::endl;
        agent.remember(cell);
        agent.remember(cell);
        agent.remember(cell);
        agent.trainLongMemory();

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Duration: " << duration.count() << std::endl;
    }
};

int main() {
    std::string projectBasePath = "D:/Work/Start/volib/volib";
    std::string modelName = "/best.onnx";
    std::string labels = "/labels.txt";
    std::string testImg = "/test_image3.jpg";
    bool runOnGPU = true;

    std::string inputFile1 = "D:/Work/Start/volib/volib/input_tmp1.txt";
    std::string inputFile2 = "D:/Work/Start/volib/volib/input_tmp2.txt";
    std::string imgFolder = "D:/Work/Start/volib/volib";
    std::string outputFile1 = "D:/Work/Start/volib/volib/output_tmp1.txt";
    std::string outputFile2 = "D:/Work/Start/volib/volib/output_tmp2.txt";

    Examples example{};
    
    example.detectionExample(projectBasePath, modelName, labels, testImg, runOnGPU);
    example.surfaceDataProcExample(inputFile1, inputFile2, imgFolder, outputFile1);
    example.detectionAndCalcExample(projectBasePath, modelName, labels, testImg, runOnGPU, inputFile2, imgFolder, outputFile1);
    example.mapAnalysisExample(projectBasePath, modelName, labels, testImg, runOnGPU, inputFile2, imgFolder, outputFile2);
    example.trajectoryPrediction();
	
	return 0;
}
