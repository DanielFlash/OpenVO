/*Copyright (c) <2024> <OOO "ORIS">
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.*/
#include "TrajectoryOVO.h"

/*Trajectory::Trajectory() {
	curr_pos = Pos_d3{};
	curr_offset = Pos_f2{};
	curr_angles = Pos_angle{};
	prev_angles = Pos_angle{};
	meter_in_pixel = Pos_f2{};
	start_pos = Pos_WGS84{};
	video_real_size = Pos_i2{};
	video_custom_size = Pos_i2{};
	Pos_f3 init_speed{};
	Pos_angle init_aspeed{};
	init_speed.x = 30;
	init_speed.y = 30;
	init_speed.z = 10;
	init_aspeed.pitch = 0.1;
	init_aspeed.roll = 0.1;
	init_aspeed.yaw = 0.1;
	max_angle_speed = init_aspeed;
	max_speed = init_speed;
	Affine_params init_params{};
	init_params.angle = 0;
	init_params.scale = 0;
	init_params.tx = 0;
	init_params.ty = 0;
	curr_affine_parameters = init_params;
	dt = 33; // msec
	type_Kallman = 0;
	k_inert = 0;
	time_since_obj_comparsion = 0;
	flag_inertc = false;
	flag_filter_kallman = false;

}*/

Trajectory::Trajectory(Camera_params &pr) {
	cam_params = pr;
	curr_pos = Pos_d3{};
	curr_offset = Pos_f2{};
	curr_angles = Pos_angle{};
	prev_angles = Pos_angle{};
	meter_in_pixel = Pos_f2{};
	start_pos = Pos_WGS84{};
	video_real_size = Pos_i2{};
	video_custom_size = Pos_i2{};
	Pos_f3 init_speed{};
	Pos_angle init_aspeed{};
	init_speed.x = 30;
	init_speed.y = 30;
	init_speed.z = 10;
	init_aspeed.pitch = 0.1;
	init_aspeed.roll = 0.1;
	init_aspeed.yaw = 0.1;
	max_angle_speed = init_aspeed;
	max_speed = init_speed;
	Affine_params init_params{};
	init_params.angle = 0;
	init_params.scale = 0;
	init_params.tx = 0;
	init_params.ty = 0;
	curr_affine_parameters = init_params;
	dt = 33; // msec
	type_Kallman = 0;
	k_inert = 0;
	time_since_obj_comparsion = 0;
	flag_inertc = false;
	flag_filter_kallman = false;

}
Pos_d3 Trajectory::get_curr_pos() {
	return curr_pos;
}

Pos_angle Trajectory::get_curr_angles() {

	return curr_angles;

}

Pos_f2 Trajectory::GetInterestPointsCoordinates(Pos_f2 pos) {
	Pos_f2 result;
	result.x = curr_offset.x * cosf(curr_angles.yaw) - curr_offset.y * sinf(curr_angles.yaw);
	result.y = curr_offset.x * sinf(curr_angles.yaw) + curr_offset.y * cosf(curr_angles.yaw);
	result.x = curr_pos.x + result.x;
	result.y = curr_pos.y + result.y;

	return result;

}

Pos_f2 Trajectory::GetInterestPointsCoordinates(float x, float y) {
	Pos_f2 result;
	result.x = x * cosf(curr_angles.yaw) - y * sinf(curr_angles.yaw);
	result.y = y * sinf(curr_angles.yaw) + x * cosf(curr_angles.yaw);
	result.x = curr_pos.x + result.x;
	result.y = curr_pos.y + result.y;

	return result;
}

Pos_f2 Trajectory::PositionFromDecartToLatLong(Pos_f2 pos) {
	Pos_f2 result;
	Pos_f2 current;
	result.x = degreeToRad * start_pos.latitude;
	result.y = degreeToRad * start_pos.longitude;
	result.y = result.y + (pos.y / (R_earth * cosf(result.x)));
	result.x = result.x + (pos.x / R_earth);

	return result;
}

Pos_f2 Trajectory::getLocalPosition(Pos_f2 offset_obj, Pos_i2 resolution) {
	Pos_f2 scale;
	Pos_f2 offset;
	Pos_f2 vs;
	Pos_f2 result;
	scale.x = cam_params.resolution.x / resolution.x;
	scale.y = cam_params.resolution.y / resolution.y;
	offset.x = offset_obj.x * scale.x;
	offset.y = offset_obj.y * scale.y;
	vs.x = (offset.x - cam_params.resolution.x / 2) * meter_in_pixel.x;
	vs.y = (offset.y - cam_params.resolution.y / 2) * meter_in_pixel.y;
	result.x = curr_offset.x + (vs.x * cosf(curr_angles.yaw) - vs.y * sinf(curr_angles.yaw));
	result.y = curr_offset.x + (vs.x * sinf(curr_angles.yaw) + vs.y * cosf(curr_angles.yaw));

	return result;
}

Pos_f2 Trajectory::getLocalPosition(Pos_f2 offset_obj, int width, int higth) {
	Pos_f2 scale;
	Pos_f2 offset;
	Pos_f2 vs;
	Pos_f2 result;
	scale.x = cam_params.resolution.x / width;
	scale.y = cam_params.resolution.y / higth;
	offset.x = offset_obj.x * scale.x;
	offset.y = offset_obj.y * scale.y;
	vs.x = (offset.x - cam_params.resolution.x / 2) * meter_in_pixel.x;
	vs.y = (offset.y - cam_params.resolution.y / 2) * meter_in_pixel.y;
	result.x = curr_offset.x + (vs.x * cosf(curr_angles.yaw) - vs.y * sinf(curr_angles.yaw));
	result.y = curr_offset.x + (vs.x * sinf(curr_angles.yaw) + vs.y * cosf(curr_angles.yaw));

	return result;
}

Pos_f2 Trajectory::PositionFromDecartToLatLong(float x, float y) {
	Pos_f2 result;
	Pos_f2 current;
	result.x = degreeToRad * start_pos.latitude;
	result.y = degreeToRad * start_pos.longitude;
	result.y = result.y + (y / (R_earth * cosf(result.x)));
	result.x = result.x + (x / R_earth);

	return result;
}

bool Trajectory::updateDataFromAffineMatrix(float** matrix2d, float alt)
{
	if (matrix2d[0][2] != 0 || matrix2d[1][2] != 0 || matrix2d[0][0] != 0 || matrix2d[0][1] != 0) {
		curr_affine_parameters.angle = atanf(-matrix2d[0][1] / matrix2d[0][0]);
		curr_affine_parameters.scale = matrix2d[0][0] / cosf(curr_affine_parameters.angle);
		curr_affine_parameters.tx = matrix2d[0][2];
		curr_affine_parameters.ty = matrix2d[1][2];
		curr_pos.z = alt;
		return calculatePosition();
	}
	else return false;
}

bool Trajectory::updateDataFromAffineMatrix(Affine_params &params, float alt)
{
	curr_affine_parameters = params;
	curr_pos.z = alt;
	return calculatePosition();
}

bool Trajectory::calculatePosition() {
	float x, y;
	float new_x, new_y;
	Pos_f2 map_scale = mapScale(cam_params, curr_pos.z);
	if (map_scale.x != NAN && map_scale.y != NAN) {
		x = curr_affine_parameters.tx * map_scale.x;
		y = curr_affine_parameters.ty * map_scale.y;
		new_x = x * cos(curr_angles.yaw) - y * sin(curr_angles.yaw);
		new_y = x * sin(curr_angles.yaw) + y * cos(curr_angles.yaw);
		curr_pos.x += new_x;
		curr_pos.y += new_y;

		return true;
	}
	return false;
}


bool Trajectory::updateDataFromPixelPoint(int x, int y, float angle, float alt, float scale=1)
{
	if (x != 0 || y != 0 ||angle != 0) {
		curr_affine_parameters.angle = angle;
		curr_affine_parameters.scale = scale;
		curr_affine_parameters.tx = x;
		curr_affine_parameters.ty = y;
		curr_pos.z = alt;
		return true;
	}
	else return false;
}

Pos_f2 mapScale(Camera_params params, float alt) {
	Pos_f2 result;
	if (params.fovx == 0 && params.fovy == 0) {
		result.x = (1 / tanf(params.fov)) * sqrtf(pow(params.resolution.x / 2, 2) + pow(params.resolution.y, 2));
		result.x = alt / result.x;
		result.y = result.x;
		return result;
	}
	else if (params.fov != 0) {
		result.x = (1 / tanf(params.fovx)) * sqrtf(pow(params.resolution.x / 2, 2) + pow(params.resolution.y, 2));
		result.x = alt / result.x;
		result.y = (1 / tanf(params.fovy)) * sqrtf(pow(params.resolution.x / 2, 2) + pow(params.resolution.y, 2));
		result.y = alt / result.y;
		return result;
	}
	result.x = NAN;
	result.y = NAN;
	return result;
}

Pos_f2 mapScale(Camera_params params, int width, int height, float alt){
	Pos_f2 result;
	if (params.fovx == 0 && params.fovy == 0) {
		result.x = (1 / tanf(params.fov)) * sqrtf(pow(width / 2, 2) + pow(height, 2));
		result.x = alt / result.x;
		result.y = result.x;
		return result;
	}
	else if (params.fov != 0) {
		result.x = (1 / tanf(params.fovx)) * sqrtf(pow(width / 2, 2) + pow(height, 2));
		result.x = alt / result.x;
		result.y = (1 / tanf(params.fovy)) * sqrtf(pow(width / 2, 2) + pow(height, 2));
		result.y = alt / result.y;
		return result;
	}
	result.x = NAN;
	result.y = NAN;
	return result;

}

void mapScale(Camera_params params, Pos_i2 shape, float alt, int accuracy){
	// for next version lib
}

void mapScale(Camera_params params, int width, int height,float alt, int accuracy) {
	// for next version lib
}

Pos_f2 rotatePointForCoordinateSystem(float x, float y, float localAngle) {
	Pos_f2 result;
	result.x = x * cosf(localAngle) - y * sinf(localAngle);
	result.y = x * sinf(localAngle) + y * cosf(localAngle);
	return result;
}

