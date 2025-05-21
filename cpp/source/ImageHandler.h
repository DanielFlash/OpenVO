#pragma once

//Разработка на будущее
//struct Size
//{
//    size_t rows;
//    size_t cols;
//};
//
//class Image
//{
//private:
//    enum class Type { GRAY, COLOR };
//
//    //Хранение данных
//public:
//    Type type;
//    Tensor<uint8_t> data;
//
//#pragma region Work Opencv
//    /// <summary>
//    /// Из opencv в свое
//    /// Серое изображение
//    /// </summary>
//    /// <param name="image"></param>
//    void setOpencvGray(const cv::Mat& image)
//    {
//        type = Type::GRAY;
//        data.resize(image.cols, image.rows, 1);
//        for (size_t y = 0; y < image.rows; y++)
//        {
//            const uchar* row = image.ptr<uchar>(y);
//            for (size_t x = 0; x < image.cols; x++)
//            {
//                data[0][y][x] = row[x];
//            }
//        }
//    }
//
//    /// <summary>
//    /// Из opencv в свое
//    /// Цветное изображение
//    /// </summary>
//    /// <param name="image"></param>
//    void setOpencvColor(const cv::Mat& image)
//    {
//        type = Type::COLOR;
//        data.resize(image.rows, image.cols, 3);
//        for (size_t y = 0; y < image.rows; y++)
//        {
//            const cv::Vec3b* row = image.ptr<cv::Vec3b>(y);
//            for (size_t x = 0; x < image.cols; x++)
//            {
//                data[0][y][x] = row[x][0];
//                data[1][y][x] = row[x][1];
//                data[2][y][x] = row[x][2];
//            }
//        }
//    }
//
//    /// <summary>
//    /// Из своего в opencv изображение
//    /// Серое изображение
//    /// </summary>
//    /// <returns></returns>
//    cv::Mat getOpencvGray()
//    {
//        cv::Mat mat(data.rows(), data.cols(), CV_8UC1);
//
//        for (int y = 0; y < data.rows(); ++y)
//        {
//            uchar* row = mat.ptr<uchar>(y);
//            for (size_t x = 0; x < data.cols(); x++)
//            {
//                row[0] = data[0][y][x];
//            }
//        }
//
//        return mat;
//    }
//
//    /// <summary>
//    /// Из своего в opencv
//    /// Цветное изображение
//    /// </summary>
//    /// <returns></returns>
//    cv::Mat getOpencvColor()
//    {
//        cv::Mat mat(data.rows(), data.cols(), CV_8UC3);
//
//        for (int y = 0; y < data.rows(); ++y)
//        {
//            cv::Vec3b* row = mat.ptr<cv::Vec3b>(y);
//            for (size_t x = 0; x < data.cols(); x++)
//            {
//                row[0][0] = data[0][y][x];
//                row[1][0] = data[1][y][x];
//                row[2][0] = data[2][y][x];
//            }
//        }
//
//        return mat;
//    }
//
//#pragma endregion
//
//
//    //Публичные методы класса
//public:
//    Image() = default;
//
//    Image(size_t height, size_t width, size_t depth)
//    {
//        data.resize(height, width, depth);
//    }
//
//    Image(Size size, size_t depth)
//    {
//        data.resize(size.rows, size.cols, depth);
//    }
//
//    Image(const cv::Mat& image)
//    {
//        switch (image.type())
//        {
//        case CV_8UC1:
//            setOpencvGray(image);
//            break;
//        case CV_8UC3:
//            setOpencvColor(image);
//            break;
//        default:
//            std::cout << "Error convert opencv image. Type image is not supported";
//            break;
//        }
//    }
//
//    size_t colors()
//    {
//        return data.depth();
//    }
//
//    size_t height()
//    {
//        return data.rows();
//    }
//
//    size_t width()
//    {
//        return data.cols();
//    }
//
//    Size size()
//    {
//        return Size{ data.rows(), data.cols() };
//    }
//
//    Matrix<uint8_t> operator[](size_t pos)
//    {
//        return data[pos];
//    }
//
//#pragma region Work Opencv
//    cv::Mat exportToOpencv()
//    {
//        switch (type)
//        {
//        case Type::GRAY:
//            return getOpencvGray();
//            break;
//        case Type::COLOR:
//            return getOpencvColor();
//            break;
//        default:
//            std::cout << "Error convert opencv image. Type image is not supported";
//            return cv::Mat();
//            break;
//        }
//    }
//
//#pragma endregion
//
//};