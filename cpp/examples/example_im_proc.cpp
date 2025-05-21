#include <iostream>
#include <cstdlib>     // для _wputenv
#include <cwchar>      // для wchar_t
#include "MathFilters.h"
#include "TransformationImage.h"
#include "math_test.h"
#include <chrono>

int main()
{
    std::cout << "Hello World!\n";

    //ВЫЗОВ knnMatch
    KnnMatch knn;
    Descriptions desc1, desc2;
    getDescription(desc1, desc2);
    Matches matchesTest = knn.find(desc1, desc2, 2);

    //Вызов цветовой коррекции по гистограмме
    cv::Mat img2 = cv::imread("image/2.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat correctOpencv = applyHistogramOpencv(img2);
    cv::imshow("OpenCv hist", correctOpencv);
    cv::waitKey(0);
    cv::Mat correctSelf = applyHistogram(img2);
    cv::imshow("Self hist", correctSelf);
    cv::waitKey(0);
    cv::Mat correctSelfCLAHE = applyHistogramCLAHE(img2, 32, 32, 4.0f);
    cv::imshow("Self hist CLAHE", correctSelfCLAHE);
    cv::waitKey(0);


    //Вызов фильтрации по Гауссу
    cv::Mat img2 = cv::imread("image/1.jpg");
    cv::Mat correctOpencv = applyGaussianOpencv(img2, 7, 1.5, 1.5, cv::BORDER_CONSTANT);
    cv::imshow("OpenCv gaus", correctOpencv);
    cv::waitKey(0);
    cv::Mat correctSelf = applyGaussian(img2, 7, 7,1.5, BORDER_CONSTANT);
    cv::imshow("Self gaus", correctSelf);
    cv::waitKey(0);
    cv::Mat correctSelfBox = boxFilterIntegral(img2, 7, 7, BORDER_CONSTANT);
    cv::imshow("Self gaus box", correctSelfBox);
    cv::waitKey(0);
    

    Points src, dst;
    Matches matches;
    getMatchesSort(src, dst, matches);


	//ВЫЗОВ АФФИНКИ И ГОМОГРАФИИ
    Matrix<> affineMatrix = estimateAffinepartial2D(src, dst, Filters::RANSAC, 0.5, 100,5);
    Matrix<> H = findHomography(src, dst, Filters::RANSAC, 0.5);
}
