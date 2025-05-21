#include "coreOVO.h"
#include "VideoProcessorOVO.h"
#include <stdio.h>
#include <opencv2/highgui.hpp>
#include <ctime>
#include <stdlib.h>

using namespace cv;

void main() {
	Camera_params param;
	param.fov = 86;
	Pos_i2 res;
	res.set(1920, 1280);
	param.resolution = res;
	param.type = 0;
	VideoProcessorOVO VPO = VideoProcessorOVO(param, "test.mp4", 1,100);
	int k = 0;
	float h = 300;
	Pos_d3 pos;
	VPO.setCustomShape(400, 400);
	int start = 0;
	int end = 0;
	int resul = 0;
	while (k != 27) {
		start = clock();
		VPO.setDataForOneIteration(h);


		if (VPO.grabFrameAndData()) {
			end = clock();
			resul = end - start;
			std::cout << " \n Full tick:" << resul;
			pos = VPO.trajectory->get_curr_pos();
			printf("\n position x,y,z:%lf, %lf, %lf", pos.x, pos.y, pos.z);

		}
		imshow("Capture", VPO.getFrame() );

		k = waitKey(1);
	}
	
}
