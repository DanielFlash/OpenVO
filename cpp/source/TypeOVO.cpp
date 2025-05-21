#include "typeOVO.h"

Pos_d2::Pos_d2() {
	x = 0;
	y = 0;
}

void Pos_d2::set(double x1, double y1) {
	x = x1;
	y = y1;
}

Pos_f2::Pos_f2() {
	x = 0;
	y = 0;
}

void Pos_f2::set(float x1, float y1) {
	x = x1;
	y = y1;
}

Pos_i2::Pos_i2() {
	x = 0;
	y = 0;
}

void Pos_i2::set(int x1, int y1) {
	x = x1;
	y = y1;
}

Camera_params::Camera_params() {
	fov = 0;
	fovx = 0;
	fovy = 0;
	type = 0;
	resolution = Pos_i2();
}

void Camera_params::set(float fov1, float fovx1 , float fovy1 , int type1, Pos_i2 res)
{
	fov = fov1;
	fovx = fovx1;
	fovy = fovy1;
	type = type1;
	resolution = res;
}