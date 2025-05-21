/*Copyright (c) <2024> <OOO "ORIS">
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.*/
#include "VideoProcessorOVO.h"


VideoProcessorOVO::VideoProcessorOVO(Camera_params &p, int index, int apiReference, Pos_i2 cs,short SOURCE_FLAG, int maxPoints) {
	params = p;
	cap = new VideoCapture(index, apiReference);
	angles_from_stream = SOURCE_FLAG;
	custom_shape.x = cs.x;
	custom_shape.y = cs.y;
	frame = cv::Mat::zeros(cv::Size(0, 0), 0);
	prev_frame = cv::Mat::zeros(cv::Size(0, 0), 0);
	detector = ORB::create(maxPoints);
	//Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);
	trajectory = new Trajectory(params);
}



VideoProcessorOVO::VideoProcessorOVO(Camera_params &p, String filename,short SOURCE_FLAG, int maxPoints) {
	params = p;
	cap = new VideoCapture(filename);
	angles_from_stream = SOURCE_FLAG;
	trajectory = new Trajectory(params);
	detector = ORB::create(maxPoints);
	//Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);
}

void  VideoProcessorOVO::reset() {
	frame = cv::Mat::zeros(cv::Size(0, 0), 0);
	prev_frame = cv::Mat::zeros(cv::Size(0, 0), 0);
}

VideoProcessorOVO::~VideoProcessorOVO() {
	cap->release();
	cap->~VideoCapture();
	trajectory->~Trajectory();

}

cv::Mat VideoProcessorOVO::getFrame() {
	return frame;
}


bool VideoProcessorOVO::setDataForOneIteration(float h, float pitch, float yaw, float roll) {
	if (h > 0) {
		alt = h;
		return true;
	}
	else {
		return false;
	}
	customSetAngles(pitch, yaw, roll);
}
cv::Mat VideoProcessorOVO::checkAndResizeFrame(cv::Mat fr) {
	int h, w;
	w = fr.cols;
	h = fr.rows;
	cv::Rect rr;
	rr.x = w / 2 - custom_shape.x / 2;
	rr.y = h / 2 - custom_shape.y / 2;
	rr.height = custom_shape.y;
	rr.width = custom_shape.x;
	return fr(rr);
}

void VideoProcessorOVO::setCustomShape(int x, int y) {
	custom_shape.x = x;
	custom_shape.y = y;
}

void VideoProcessorOVO::setCustomShape(Pos_i2 nshape) {
	custom_shape = nshape;
}


bool VideoProcessorOVO::setDataForOneIteration(float h) {
	if (h > 0) {
		alt = h;
		return true;
	}
	else {
		return false;
	}

}

void VideoProcessorOVO::customSetAngles(float pitch, float yaw, float roll) {
	angles.pitch = pitch;
	angles.yaw = yaw;
	angles.roll = roll;
}

bool VideoProcessorOVO::grabFrameAndData() {
	Mat frame_b;
	if (cap->isOpened()) {
		cap->read(frame_b);
		frame_b = checkAndResizeFrame(frame_b);
		if (frame_b.empty()) {
			return false;
		}

	}
	else {
		return false;
	}
	cvtColor(frame_b, frame, COLOR_BGR2GRAY);

	if (keypoints1.empty()) {
		prev_frame = frame;
		keypoints_check(prev_frame, keypoints1, descriptor1, detector);
	}
	else {
		keypoints_check(frame, keypoints2, descriptor2, detector);
		bool res = searchAffineMatrix();
		if (res) {
			trajectory->updateDataFromAffineMatrix(affine_params, alt);
		}
		prev_frame = frame;
		keypoints1 = keypoints2;
		descriptor1 = descriptor2;
		return res;

	}

}

bool VideoProcessorOVO::searchAffineMatrix() {
	try {
		affine_params = getAffineInfo( 8, 0.8f);
		return true;
	}
	catch (cv::Exception& ex) {
		return false;
	}
}



Affine_params VideoProcessorOVO::getAffineInfo( int method, float ratio_thresh) {
 //TODO add methods(RANSAC, LMEDS)
	Affine_params target{};
	std::vector<KeyPoint> keypoints_object, keypoints_scene;
	std::vector<std::vector<DMatch> > knn_matches;
	matcher->knnMatch(descriptor1, descriptor2, knn_matches, 4);
	std::vector<DMatch> good_matches;
	for (size_t i = 0; i < knn_matches.size(); i++)
	{
		if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
		{
			good_matches.push_back(knn_matches[i][0]);
		}
	}
	std::vector<cv::Point2f> kpts1, kpts2;
	std::vector<uchar> inliners;
	for (int i = 0; i < good_matches.size(); i++) {
		kpts1.push_back(keypoints1[good_matches[i].queryIdx].pt);
		kpts2.push_back(keypoints2[good_matches[i].trainIdx].pt);
	}
	cv::Mat resultAffine;
	resultAffine = cv::estimateAffinePartial2D(kpts1, kpts2, inliners, method);
	Mat1f mm(resultAffine);
	target.tx = mm(0, 2);
	target.ty = mm(1, 2);
	float cs, sn, scale = 0.0;
	cs = mm(0, 0);
	sn = mm(1, 0);
	target.angle = atan(sn / cs);
	target.scale = cs / cos(target.angle);
	return target;
}


bool keypoints_check(Mat frame1,std::vector<KeyPoint> &keypoints_object, Mat &desc, Ptr<ORB> detector ) {
	try {
		detector->detectAndCompute(frame1, noArray(), keypoints_object, desc);
		return true;
	}
	catch (cv::Exception& ex) {
		return false;
	}
}