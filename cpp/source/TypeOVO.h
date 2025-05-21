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

#ifdef TYPEOVO_EXPORTS
#define TYPEOVO_API __declspec(dllexport)
#else
#define TYPEOVO_API __declspec(dllimport)
#endif



#define dm_pi 3.14159265358979
#define dm_sm_a 6378137.0
#define dm_sm_b 6356752.314
#define dm_sm_EccSquared 6.69437999013e-03
#define UTMScaleFactor 0.9996
#define R_earth 6378137.0
#define degreeToRad  0.01745329252

#define OVO_ACCURACY_INT 0
#define OVO_ACCURACY_FLOAT 1
#define OVO_ACCURACY_DOUBLE 2
#define OVO_TO_OPENCV 1
#define OVO_ANGLES_FROM_SOURCE 0
#define OVO_ANGLES_FROM_VIDEO 1
#define OPENCV_TO_OVO 0
#define OVO_RANSAC 8
#define OVO_LMEDS 9

/// <summary>
/// Struct for wotk with 3D position object, double elements
/// </summary>
typedef struct Pos_d3 {
	double x;
	double y;
	double z;
};

/// <summary>
/// Struct for wotk with 3D position object, float elements
/// </summary>
typedef struct Pos_f3 {
	float x;
	float y;
	float z;
};

/// <summary>
/// Struct for wotk with 3D position object, int elements
/// </summary>
typedef struct Pos_i3 {
	int x;
	int y;
	int z;
};

/// <summary>
/// Struct for wotk with 2D position object, double elements
/// </summary>
typedef struct Pos_d2 {
	double x;
	double y;
	Pos_d2();
	void set(double x1, double y1);
};

/// <summary>
/// Struct for wotk with 2D position object, float elements
/// </summary>
typedef struct Pos_f2 {
	float x;
	float y;
	Pos_f2();
	void set(float x1, float y1);

};

/// <summary>
/// Struct for wotk with 2D position object or size frame, int elements
/// </summary>
typedef struct Pos_i2 {
	int x;
	int y;
	Pos_i2();
	void set(int x1, int y1);

};

/// <summary>
/// Struct camera params got info about fov, resolution and type cameras
/// </summary>
typedef struct Camera_params {
	float fov;
	float fovx;
	float fovy;
	int type;
	Pos_i2 resolution;
	Camera_params();
	void set(float fov1 = 0, float fovx1 = 0, float fovy1 = 0, int type1 = 0, Pos_i2 res = Pos_i2());
};

/// <summary>
/// Position angle params
/// </summary>
typedef struct Pos_angle {
	float pitch;
	float roll;
	float yaw;
};

/// <summary>
/// Position WGS84
/// </summary>
typedef struct Pos_WGS84 {
	float latitude;
	float longitude;
	int force_zone_number;
	int force_zone_letter;
};

/// <summary>
/// Struct like a affine matrix
/// </summary>
typedef struct Affine_params {
	int tx;
	int ty;
	float scale;
	float angle;
};