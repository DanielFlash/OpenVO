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

#ifdef VIDEOPROCESSOROVO_EXPORTS
#define VIDEOPROCESSOROVO_API __declspec(dllexport)
#else
#define VIDEOPROCESSOROVO_API __declspec(dllimport)
#endif


#include "coreOVO.h"

using namespace cv;

/// <summary>
/// Detect and compute keypoints. Algotithm get from OpenCV. 
/// </summary>
/// <param name="frame1">Mat first frame, matrice from OpenCV lib</param>
/// <param name="keypoints_object">Out vector KeyPoints vector like Opencv method</param>
/// <param name="desc">Descrtiption mat first frame</param>
/// <param name="minHessian"></param>
/// <returns>Compute keypoints</returns>
bool keypoints_check(Mat frame1, std::vector<KeyPoint>& keypoints_object, Mat& desc, Ptr<ORB> detector);

class VideoProcessorOVO
{

public:
	/// <summary>
	/// VideoProcessor class witch get and send to nex 
	/// </summary>
	/// <param name="p"> Camera_params </param>
	/// <param name="filename">string filename</param>
	/// <param name="SOURCE_FLAG">short flag, will see on TypeOVO.h </param>
	/// <param name = "maxPoints"> int number max points witch search on frame</param>
	VideoProcessorOVO(Camera_params &p, String filename ,short SOURCE_FLAG = OVO_ANGLES_FROM_VIDEO, int maxPoints = 100);
	/// <summary>
	/// VideoProcessor class witch get and send to nex 
	/// </summary>
	/// <param name="p"> Camera_params </param>
	/// <param name="index"> index camera for backline Opencv  </param>
	/// <param name="apiReference">Pos_i2 (GSTREAMER, MJPEG or other)</param>
	/// <param name="filename">string filename</param>
	/// <param name="SOURCE_FLAG">short flag, will see on TypeOVO.h </param>
	/// <param name = "maxPoints"> int number max points witch search on frame</param>
	VideoProcessorOVO(Camera_params &p, int index, int apiReference, Pos_i2 custom_shape, short SOURCE_FLAG = OVO_ANGLES_FROM_VIDEO, int maxPoints = 100);

	~VideoProcessorOVO();
	
	/// <summary>
	/// Function for get frame from device
	/// </summary>
	/// <returns>frame cv::Mat </returns>
	cv::Mat getFrame();
	/// <summary>
	/// Function for set custom shape for frame
	/// </summary>
	/// <param name="x">int width</param>
	/// <param name="y">int hight</param>
	void setCustomShape(int x, int y);
	/// <summary>
	/// Function for set custom shape from struct Pos_i2
	/// </summary>
	/// <param name="nshape">Pos_i2 struct</param>
	void setCustomShape(Pos_i2 nshape);
	/// <summary>
	/// Function for set special data for calculate position in coordinate system for one iteration
	/// </summary>
	/// <param name="alt">float altitude object from telemetry info</param>
	/// <param name="nshape">Pos_i2 float Pitch angle </param>
	/// <param name="nshape">Pos_i2 float Yaw angle</param>
	/// <param name="nshape">Pos_i2 float Row angle</param>
	bool setDataForOneIteration(float alt, float pitch, float yaw, float row);
	/// <summary>
	/// Function for set altitude witch actual for 1 iteration, wanna use before send frame to VideoProcessorOVO class
	/// </summary>
	/// <param name="alt"> float alitude </param>
	bool setDataForOneIteration(float alt);
	/// <summary>
	///  Function for grab frame and data witch set in functions SetDataForOneIteration
	/// </summary>
	bool grabFrameAndData();
	/// <summary>
	/// Function calculate affine matrix, you can get result from VideoProcessorOVO->affine_params
	/// </summary>
	bool searchAffineMatrix();
	/// <summary>
	/// Function for set custom angle before calculate affine matrix
	/// </summary>
	void customSetAngles(float pitch, float yaw, float row);
	/// <summary>
	/// Function for reset telemtry and feature searcher class
	/// </summary>
	void reset();
	/// <summary>
	/// Function for checking frame and resize, resize params set in VideoProcessorOVO class
	/// </summary>
	/// <param name="fr"> cv::Mat matrix frame with BGR colormap
	cv::Mat checkAndResizeFrame(cv::Mat fr);
	/// <summary>
	/// Getter affine info, y can change ratio_thresh for up scale params
	/// </summary>
	Affine_params getAffineInfo(int method = OVO_RANSAC, float ratio_thresh = 0.8f);

	Trajectory* trajectory;

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);

protected:
	
	/***********************************************************************************************************/
	/****************************  Opencv 4.x varibales for feature matching algoritm **************************/
	/***********************************************************************************************************/


	cv::VideoCapture* cap;
	cv::Mat frame;
	cv::Mat prev_frame;
	Ptr<ORB> detector;
	std::vector<KeyPoint> keypoints1, keypoints2;
	cv::Mat descriptor1, descriptor2;

	/****************************************************END*****************************************************/

	/// <summary>
	/// Struct for affine params, check TypeOVO.h
	/// </summary>
	Affine_params affine_params;
	/// <summary>
	/// Struct for camera params, check TypeOVO.h
	/// </summary>
	Camera_params params;
	/// <summary>
	/// Struct with angle, , check TypeOVO.h
	/// </summary>
	Pos_angle angles;
	/// <summary>
	/// Altitude
	/// </summary>
	float alt;
	/// <summary>
	/// if True , function calcute affine use custom angle 
	/// </summary>
	bool angles_from_stream;
	/// <summary>
	/// shape frame to resize
	/// </summary>
	Pos_i2 custom_shape;
	/// <summary>
	/// shape input frame
	/// </summary>
	Pos_i2 shape;

private:



};