import math
import numpy as np  # For potential matrix operations if needed, though not heavily used in original
from .ovo_types import (Pos_d3, Pos_f2, Pos_i2, CameraParams, PosAngle,
                       PosWGS84, AffineParams, R_EARTH, DEGREE_TO_RAD, Pos_f3)


# --- Helper free functions (originally in TrajectoryOVO.h/cpp) ---
def map_scale_py(params: CameraParams, alt: float) -> Pos_f2:
    result = Pos_f2(x=math.nan, y=math.nan)
    if alt <= 0:
        return result

    # This logic is directly from C++ `mapScale`
    # Ensure math.pow, math.sqrt, math.tan are used
    # Note: C++ pow(base, exp) can take float/double, Python math.pow always float.
    # Original C++ uses params.resolution.x / 2 and params.resolution.y for sqrt term.
    term_x_sq = math.pow(params.resolution.x / 2.0, 2.0)
    term_y_sq = math.pow(params.resolution.y, 2.0)  # Original used full y

    common_sqrt_term = math.sqrt(term_x_sq + term_y_sq)
    if common_sqrt_term == 0: return result  # Avoid division by zero

    if params.fovx == 0.0 and params.fovy == 0.0 and params.fov != 0.0:
        tan_fov = math.tan(params.fov)  # Assuming fov is in radians
        if tan_fov == 0.0: return result
        temp_calc = (1.0 / tan_fov) * common_sqrt_term
        if temp_calc == 0.0: return result
        result.x = alt / temp_calc
        result.y = result.x
        return result
    # C++ had 'else if (params.fov != 0)' for the fovx/fovy part.
    # This implies fovx/fovy are used IF params.fov is also non-zero.
    # Let's assume fovx/fovy take precedence if non-zero.
    elif params.fovx != 0.0:  # If fovx is specified, use it
        tan_fovx = math.tan(params.fovx)
        if tan_fovx == 0.0:
            result.x = math.nan
        else:
            temp_calc_x = (1.0 / tan_fovx) * common_sqrt_term
            if temp_calc_x == 0.0:
                result.x = math.nan
            else:
                result.x = alt / temp_calc_x

        if params.fovy != 0.0:
            tan_fovy = math.tan(params.fovy)
            if tan_fovy == 0.0:
                result.y = math.nan
            else:
                temp_calc_y = (1.0 / tan_fovy) * common_sqrt_term
                if temp_calc_y == 0.0:
                    result.y = math.nan
                else:
                    result.y = alt / temp_calc_y
        else:  # If fovy is not specified, assume square pixels based on fovx calculation
            result.y = result.x
        return result
    return result


def map_scale_wh_py(params: CameraParams, width: int, height: int, alt: float) -> Pos_f2:
    # Create a temporary CameraParams with overridden resolution
    temp_params = CameraParams(
        fov=params.fov, fovx=params.fovx, fovy=params.fovy, type=params.type,
        resolution=Pos_i2(width, height)
    )
    return map_scale_py(temp_params, alt)


def rotate_point_for_coordinate_system_py(x: float, y: float, local_angle: float) -> Pos_f2:
    res_x = x * math.cos(local_angle) - y * math.sin(local_angle)
    res_y = x * math.sin(local_angle) + y * math.cos(local_angle)
    return Pos_f2(res_x, res_y)


class Trajectory:
    def __init__(self, pr: CameraParams):
        self.cam_params: CameraParams = pr
        self.curr_pos: Pos_d3 = Pos_d3()
        self.curr_offset: Pos_f2 = Pos_f2()
        self.curr_angles: PosAngle = PosAngle()
        self.prev_angles: PosAngle = PosAngle()
        self.meter_in_pixel: Pos_f2 = Pos_f2()
        self.start_pos: PosWGS84 = PosWGS84()
        self.video_real_size: Pos_i2 = Pos_i2()
        self.video_custom_size: Pos_i2 = Pos_i2()

        init_speed = Pos_f3(x=30.0, y=30.0, z=10.0)
        self.max_speed: Pos_f3 = init_speed

        init_aspeed = PosAngle(pitch=0.1, roll=0.1, yaw=0.1)
        self.max_angle_speed: PosAngle = init_aspeed

        self.curr_affine_parameters: AffineParams = AffineParams()
        self.dt: int = 33  # msec
        self.type_Kallman: int = 0
        self.k_inert: float = 0.0
        self.time_since_obj_comparsion: float = 0.0  # C++ was float
        self.flag_inertc: bool = False
        self.flag_filter_kallman: bool = False

    def get_curr_pos(self) -> Pos_d3:
        return self.curr_pos

    def get_curr_angles(self) -> PosAngle:
        return self.curr_angles

    def get_interest_points_coordinates(self, pos_or_x, y_coord=None) -> Pos_f2:
        # Handles both (Pos_f2 pos) and (float x, float y) overloads
        if y_coord is None and isinstance(pos_or_x, Pos_f2):
            # C++ logic: result.x = curr_offset.x * cosf(curr_angles.yaw) - curr_offset.y * sinf(curr_angles.yaw);
            # This uses self.curr_offset, not the passed 'pos_or_x'. This seems like a bug in C++ or misunderstanding.
            # The (float x, float y) version uses x and y arguments.
            # Assuming the (Pos_f2 pos) version should use pos_or_x.x and pos_or_x.y for consistency:
            px = pos_or_x.x
            py = pos_or_x.y
        elif y_coord is not None:
            px = float(pos_or_x)
            py = float(y_coord)
        else:
            raise TypeError("Invalid arguments for get_interest_points_coordinates")

        # Common logic from C++ (float x, float y) version:
        # result.x = x * cosf(curr_angles.yaw) - y * sinf(curr_angles.yaw);
        # result.y = y * sinf(curr_angles.yaw) + x * cosf(curr_angles.yaw);
        # The C++ (float x, float y) has: result.y = y * sinf(curr_angles.yaw) + x * cosf(curr_angles.yaw);
        # Using the C++ version's formula:
        res_x_rotated = px * math.cos(self.curr_angles.yaw) - py * math.sin(self.curr_angles.yaw)
        res_y_rotated = py * math.sin(self.curr_angles.yaw) + px * math.cos(
            self.curr_angles.yaw)  # C++ version's specific y rotation

        final_x = self.curr_pos.x + res_x_rotated
        final_y = self.curr_pos.y + res_y_rotated
        return Pos_f2(x=float(final_x), y=float(final_y))

    def position_from_decart_to_lat_long(self, pos_or_x, y_coord=None) -> Pos_f2:
        if y_coord is None and isinstance(pos_or_x, Pos_f2):
            decart_x = pos_or_x.x
            decart_y = pos_or_x.y
        elif y_coord is not None:
            decart_x = float(pos_or_x)
            decart_y = float(y_coord)
        else:
            raise TypeError("Invalid arguments")

        res_lat_rad = DEGREE_TO_RAD * self.start_pos.latitude
        res_lon_rad = DEGREE_TO_RAD * self.start_pos.longitude

        cos_lat = math.cos(res_lat_rad)
        if cos_lat == 0:  # Avoid division by zero
            return Pos_f2(x=math.nan, y=math.nan)

        res_lon_rad = res_lon_rad + (decart_y / (R_EARTH * cos_lat))
        res_lat_rad = res_lat_rad + (decart_x / R_EARTH)
        return Pos_f2(x=res_lat_rad, y=res_lon_rad)  # Returns radians

    def get_local_position(self, offset_obj: Pos_f2, resolution_or_width, height=None) -> Pos_f2:
        if height is None and isinstance(resolution_or_width, Pos_i2):
            img_res_x = resolution_or_width.x
            img_res_y = resolution_or_width.y
        elif height is not None:
            img_res_x = int(resolution_or_width)
            img_res_y = int(height)
        else:
            raise TypeError("Invalid arguments for get_local_position")

        if img_res_x == 0 or img_res_y == 0 or \
                self.cam_params.resolution.x == 0 or self.cam_params.resolution.y == 0:
            return Pos_f2(x=math.nan, y=math.nan)

        scale_x = float(self.cam_params.resolution.x) / img_res_x
        scale_y = float(self.cam_params.resolution.y) / img_res_y

        offset_scaled_x = offset_obj.x * scale_x
        offset_scaled_y = offset_obj.y * scale_y

        # vs = (offset - cam_center_px) * meter_per_pixel
        vs_x = (offset_scaled_x - self.cam_params.resolution.x / 2.0) * self.meter_in_pixel.x
        vs_y = (offset_scaled_y - self.cam_params.resolution.y / 2.0) * self.meter_in_pixel.y

        # Rotate vs by yaw and add to curr_offset
        # C++: result.x = curr_offset.x + (vs.x * cosf(curr_angles.yaw) - vs.y * sinf(curr_angles.yaw));
        # C++: result.y = curr_offset.x + (vs.x * sinf(curr_angles.yaw) + vs.y * cosf(curr_angles.yaw)); (Typo, should be curr_offset.y)
        rotated_vs_x = vs_x * math.cos(self.curr_angles.yaw) - vs_y * math.sin(self.curr_angles.yaw)
        rotated_vs_y = vs_x * math.sin(self.curr_angles.yaw) + vs_y * math.cos(self.curr_angles.yaw)

        res_x = self.curr_offset.x + rotated_vs_x
        res_y = self.curr_offset.y + rotated_vs_y  # Corrected from curr_offset.x
        return Pos_f2(x=res_x, y=res_y)

    def update_data_from_affine_matrix(self, matrix_or_params, alt: float) -> bool:
        if isinstance(matrix_or_params, AffineParams):
            self.curr_affine_parameters = matrix_or_params
            self.curr_pos.z = alt
            return self.calculate_position()
        elif isinstance(matrix_or_params, list):  # Assuming list of 2 lists for matrix2d
            # This was float** matrix2d in C++
            # Python equivalent: [[m00, m01, m02], [m10, m11, m12]]
            m = matrix_or_params
            if not (len(m) == 2 and len(m[0]) == 3 and len(m[1]) == 3):
                # print("Warning: Affine matrix in list form has incorrect dimensions.")
                return False

            # C++ condition: matrix2d[0][2]!=0 || matrix2d[1][2]!=0 || matrix2d[0][0]!=0 || matrix2d[0][1]!=0
            if m[0][2] != 0 or m[1][2] != 0 or m[0][0] != 0 or m[0][1] != 0:
                m00, m01, m02 = m[0]
                m10, m11, m12 = m[1]  # m11 not used in original C++ for angle/scale

                # angle = atanf(-matrix2d[0][1] / matrix2d[0][0]);
                # scale = matrix2d[0][0] / cosf(angle);
                if m00 == 0:  # Avoid division by zero
                    # Handle case, e.g., if angle is +/- 90 degrees
                    if m01 != 0:
                        self.curr_affine_parameters.angle = math.copysign(math.pi / 2, -m01)  # +/- pi/2
                        self.curr_affine_parameters.scale = abs(m01)  # scale = abs(-sin(angle)*scale) = abs(m01)
                    else:  # Matrix is likely degenerate
                        return False
                else:
                    self.curr_affine_parameters.angle = math.atan2(-m01, m00)  # More robust
                    cos_angle = math.cos(self.curr_affine_parameters.angle)
                    if cos_angle == 0:  # Should not happen if m00 was non-zero & atan2 is used
                        self.curr_affine_parameters.scale = math.sqrt(m00 ** 2 + m01 ** 2)
                    else:
                        self.curr_affine_parameters.scale = m00 / cos_angle

                # More robust scale and angle:
                # self.curr_affine_parameters.angle = math.atan2(m10, m00) # if using [s*c, -s*s; s*s, s*c] structure
                # self.curr_affine_parameters.scale = math.sqrt(m00**2 + m10**2)

                self.curr_affine_parameters.tx = int(m02)  # tx, ty are int in C++ Affine_params
                self.curr_affine_parameters.ty = int(m12)
                self.curr_pos.z = alt
                return self.calculate_position()
            else:
                return False
        else:
            raise TypeError("Invalid argument for update_data_from_affine_matrix")

    def calculate_position(self) -> bool:
        map_s = map_scale_py(self.cam_params, float(self.curr_pos.z))  # curr_pos.z is double

        if math.isnan(map_s.x) or math.isnan(map_s.y):
            return False

        # curr_affine_parameters.tx/ty are int
        x_offset_map = float(self.curr_affine_parameters.tx) * map_s.x
        y_offset_map = float(self.curr_affine_parameters.ty) * map_s.y

        # C++ uses cos/sin (double) for curr_angles.yaw (float). Python math.cos/sin take float.
        new_x_rotated = x_offset_map * math.cos(self.curr_angles.yaw) - \
                        y_offset_map * math.sin(self.curr_angles.yaw)
        new_y_rotated = x_offset_map * math.sin(self.curr_angles.yaw) + \
                        y_offset_map * math.cos(self.curr_angles.yaw)

        self.curr_pos.x += new_x_rotated
        self.curr_pos.y += new_y_rotated
        return True

    def update_data_from_pixel_point(self, x: int, y: int, angle: float, alt: float, scale: float = 1.0) -> bool:
        # C++: if (x != 0 || y != 0 ||angle != 0)
        if x != 0 or y != 0 or angle != 0.0:  # Compare float angle to 0.0
            self.curr_affine_parameters.angle = angle
            self.curr_affine_parameters.scale = scale
            self.curr_affine_parameters.tx = x  # tx, ty are int
            self.curr_affine_parameters.ty = y
            self.curr_pos.z = alt
            # Original C++ returned true here without calling calculatePosition.
            # This seems inconsistent if other update methods call it.
            # Assuming calculatePosition should be called:
            return self.calculate_position()
            # return True # If calculatePosition is not intended
        else:
            return False