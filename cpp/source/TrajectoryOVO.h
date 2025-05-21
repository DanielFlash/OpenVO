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

#ifdef TRAJECTORYOVO_EXPORTS
#define TRAJECTORYOVO_API __declspec(dllexport)
#else
#define TRAJECTORYOVO_API __declspec(dllimport)
#endif

#include "coreOVO.h"

/// <summary>
/// A function that calculates the pixel scale to meters based on the camera's hardware characteristics and shooting height
/// </summary>
/// <param name="params">Camera parameters in the form of a structure</param>
/// <param name="alt">height in meters</param>
/// <returns>Pos_f2.x (float), Posf2.y (float) coefficient for the corresponding axes</returns>
Pos_f2 mapScale(Camera_params params, float alt);
/// <summary>
///  A function that calculates the pixel scale to meters based on the camera's hardware characteristics and shooting height for 
/// custom size frame
/// </summary>
/// <param name="params">Camera parameters in the form of a structure</param>
/// <param name="width"> width frame</param>
/// <param name="height">height frame</param>
/// <param name="alt">height in meters</param>
/// <returns>Pos_f2.x (float), Posf2.y (float) coefficient for the corresponding axes</returns>
Pos_f2 mapScale(Camera_params params, int width, int height, float alt);
/// <summary>
/// A function that calculates the pixel scale to meters based on the camera's hardware characteristics and shooting height 
/// for custom size frame and custom accuracy
/// </summary>
/// <param name="params">Camera parameters in the form of a structure</param>
/// <param name="shape">width and height frame</param>
/// <param name="alt">height in meters</param>
/// <param name="accuracy">accuracy define parametr check typeOVO</param>
void mapScale(Camera_params params, Pos_i2 shape, float alt, int accuracy);
/// <summary>
/// A function that calculates the pixel scale to meters based on the camera's hardware characteristics and shooting height for 
/// custom size frame and custom accuracy
/// </summary>
/// <param name="params"></param>
/// <param name="width"></param>
/// <param name="height"></param>
/// <param name="alt"></param>
/// <param name="accuracy"></param>
void mapScale(Camera_params params, int width, int height, float alt, int accuracy);
/// <summary>
/// The function of translating the coordinates of a point into the coordinate system of the observer
/// </summary>
/// <param name="x"> coordinate x for meters</param>
/// <param name="y"> coordinate y for meters</param>
/// <param name="localAngle"></param>
/// <returns>new position in observer coordinate system</returns>
Pos_f2 rotatePointForCoordinateSystem(float x, float y, float localAngle);


/// <summary>
/// OVO class::The trajectory includes methods for calculating the position in the local coordinate system, on a scale of meter to pixel, 
/// functions for calculating points on the frame in coordinates in meters, as well as accounting for the rotation of the device
/// </summary>
class Trajectory
{
public:
	//Trajectory();
	Trajectory(Camera_params& pr);
	/// <summary>
	/// The function of converting pixel coordinates to coordinates in pixel from the calculation point ( start )
	/// </summary>
	/// <param name="">the pixel position of the point to translate</param>
	/// <returns>position in pixel</returns>
	Pos_f2 GetInterestPointsCoordinates(Pos_f2);
	/// <summary>
	/// The function of converting pixel coordinates to coordinates in pixel from the calculation point ( start )
	/// </summary>
	/// <param name="x"></param>
	/// <param name="y"></param>
	/// <returns>position in pixel</returns>
	Pos_f2 GetInterestPointsCoordinates(float x, float y);
	/// <summary>
	/// The function of converting pixel coordinates to coordinates in WGS84
	/// </summary>
	/// <param name="">the pixel position of the point to translate x and y</param>
	/// <returns>WGS84 lat and long</returns>
	Pos_f2 PositionFromDecartToLatLong(Pos_f2);
	/// <summary>
	/// The function of converting pixel coordinates to coordinates in WGS84
	/// </summary>
	/// <param name="x">the pixel position of the point to translate Ox</param>
	/// <param name="y">the pixel position of the point to translate Oy</param>
	/// <returns>WGS84 lat and long</returns>
	Pos_f2 PositionFromDecartToLatLong(float x, float y);
	/// <summary>
	/// Converting pixel coordinates of a point to coordinates in meters
	/// </summary>
	/// <param name="">Pixel points coordinate</param>
	/// <param name="resolution">resolution frame</param>
	/// <returns>struct with 2 varibalse float type in meter from local coordinate system </returns>
	Pos_f2 getLocalPosition(Pos_f2, Pos_i2 resolution);
	/// <summary>
	/// Converting pixel coordinates of a point to coordinates in meters
	/// </summary>
	/// <param name="">Pixel points coordinate</param>
	/// <param name="wigth"></param>
	/// <param name="higth"></param>
	/// <returns>struct with 2 varibalse float type in meter from local coordinate system </returns>
	Pos_f2 getLocalPosition(Pos_f2, int wigth, int higth);
	/// <summary>
	/// The function of updating data by calculating the offset from the matrix of affine transformations 
	/// between successive frames, taking into account camera parameters such as FoV, resolution
	/// </summary>
	/// <param name="matrix2d">2D affine transformation (4 degrees of freedom) matrix or empty matrix if transformation could not be estimated.</param>
	/// <param name="alt">the height of the observer </param>
	/// <returns>bool param, if the update was successful returned true</returns>
	bool updateDataFromAffineMatrix(float** matrix2d, float alt);
	/// <summary>
	/// The function of updating data by calculating the offset from the matrix of affine transformations 
	/// between successive frames, taking into account camera parameters such as FoV, resolution
	/// </summary>
	/// <param name="params">OVO type parametrs about frame from affine matrix</param>
	/// <param name="alt">the height of the observer </param>
	/// <returns>bool param, if the update was successful returned true</returns>
	bool updateDataFromAffineMatrix(Affine_params& params, float alt);
	/// <summary>
	/// The function of updating data by calculating the offset from the custom parametrs
	/// between successive frames, taking into account camera parameters such as FoV, resolution
	/// </summary>
	/// <param name="x"> delta on Ox between two frames</param>
	/// <param name="y"> delta on Oy between two frames</param>
	/// <param name="angle"> rotate observer between two frames on Oz </param>
	/// <param name="alt">the height of the observer </param>
	/// <param name="scale">scale between two frames</param>
	/// <returns>bool param, if the update was successful returned true</returns>
	bool updateDataFromPixelPoint(int x, int y, float angle, float alt, float scale);
	/// <summary>
	/// the function calculates the position based on the data stored in the class returns True if the condition is successful
	/// </summary>
	/// <returns>True if succes update</returns>
	bool calculatePosition();
	/// <summary>
	/// GET function current position
	/// </summary>
	/// <returns> current position</returns>
	Pos_d3 get_curr_pos();
	/// <summary>
	/// GET function current angles
	/// </summary>
	/// <returns>current angles</returns>
	Pos_angle get_curr_angles();

protected:
	/// <summary>
	/// Camera params
	/// </summary>
	Camera_params cam_params;

	/// <summary>
	/// current position observer
	/// </summary>
	Pos_d3 curr_pos;
	/// <summary>
	/// current offset observer
	/// </summary>
	Pos_f2 curr_offset;
	/// <summary>
	/// current angles observer
	/// </summary>
	Pos_angle curr_angles;
	/// <summary>
	/// position angles before last update
	/// </summary>
	Pos_angle prev_angles;
	/// <summary>
	///  pixel scale to meters
	/// </summary>
	Pos_f2 meter_in_pixel;
	/// <summary>
	/// start position in WGS 84 
	/// </summary>
	Pos_WGS84 start_pos;
	/// <summary>
	/// real size h and w 
	/// </summary>
	Pos_i2 video_real_size;
	/// <summary>
	/// custom size h and w 
	/// </summary>
	Pos_i2 video_custom_size;
	/// <summary>
	/// max speed changing angle - use when created dlc inertial filter
	/// </summary>
	Pos_angle max_angle_speed;
	/// <summary>
	/// max speed - use when created dlc inertial filter
	/// </summary>
	Pos_f3 max_speed;
	/// <summary>
	/// affine parametrs about last pair frames
	/// </summary>
	Affine_params curr_affine_parameters;
	/// <summary>
	/// delta t - use when created dls inertial filter
	/// </summary>
	int dt; // msec
	/// <summary>
	/// type Kallman filter first(1),second(2) and third(3) order
	/// </summary>
	int type_Kallman;
	/// <summary>
	/// inertial scale 
	/// </summary>
	float k_inert;
	/// <summary>
	/// 
	/// </summary>
	float time_since_obj_comparsion;
	/// <summary>
	/// on or off inertial dls ( future version ) 
	/// </summary>
	bool flag_inertc;
	/// <summary>
	/// on or off kallman filtration coordinates with dls ( future version ) 
	/// </summary>
	bool flag_filter_kallman;

private:

	

};

