from __future__ import annotations

import cv2
import numpy as np
import math
from .ovo_types import CameraParams, Pos_i2, AffineParams, PosAngle, OVO_RANSAC, OVO_ANGLES_FROM_VIDEO, \
    OVO_ANGLES_FROM_SOURCE
from .trajectory_ovo import Trajectory  # Assuming Trajectory class is in trajectory_ovo.py


def keypoints_check_py(frame_gray: np.ndarray, detector: cv2.ORB) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    """Detects keypoints and computes descriptors."""
    if frame_gray is None or frame_gray.size == 0 or detector is None:
        return [], None
    try:
        kps, des = detector.detectAndCompute(frame_gray, None)
        return kps, des
    except cv2.error as e:
        # print(f"OpenCV error in keypoints_check_py: {e}")
        return [], None


class VideoProcessorOVO:
    def __init__(self, p_params: CameraParams, source_info,  # source_info can be filename (str) or index (int)
                 api_reference_or_source_flag,  # apiReference (int) or SOURCE_FLAG (short/int)
                 custom_shape_or_max_points,  # Pos_i2 or int
                 source_flag_if_cam_index: int = OVO_ANGLES_FROM_VIDEO,  # Only if source_info is cam_index
                 max_points_if_cam_index: int = 100):

        self.params: CameraParams = p_params
        self.trajectory: Trajectory = Trajectory(self.params)
        self.custom_shape: Pos_i2 = Pos_i2(0, 0)  # Default
        self.angles_from_stream: bool = False  # Default
        max_points = 100  # Default

        if isinstance(source_info, str):  # Constructor with filename
            self.cap = cv2.VideoCapture(source_info)
            self.angles_from_stream = (api_reference_or_source_flag != 0)  # This was SOURCE_FLAG
            max_points = int(custom_shape_or_max_points)  # This was maxPoints
        elif isinstance(source_info, int):  # Constructor with camera index
            self.cap = cv2.VideoCapture(source_info, api_reference_or_source_flag)  # apiReference
            self.custom_shape = custom_shape_or_max_points  # This was cs (Pos_i2)
            self.angles_from_stream = (source_flag_if_cam_index != 0)
            max_points = int(max_points_if_cam_index)
        else:
            raise TypeError("Invalid source_info for VideoProcessorOVO constructor")

        if not self.cap.isOpened():
            raise IOError(f"Cannot open video source: {source_info}")

        self.frame: np.ndarray | None = None  # Current processed gray frame
        self.prev_frame: np.ndarray | None = None  # Previous processed gray frame

        self.detector: cv2.ORB = cv2.ORB_create(nfeatures=max_points)
        # NORM_HAMMING for ORB descriptors
        self.matcher: cv2.BFMatcher = cv2.BFMatcher_create(cv2.NORM_HAMMING,
                                                           crossCheck=False)  # crossCheck=False for knnMatch

        self.keypoints1: list[cv2.KeyPoint] = []
        self.descriptor1: np.ndarray | None = None
        self.keypoints2: list[cv2.KeyPoint] = []
        self.descriptor2: np.ndarray | None = None

        self.affine_params: AffineParams = AffineParams()  # Internal storage
        self.angles: PosAngle = PosAngle()
        self.alt: float = 0.0

    def __del__(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def reset(self):
        self.frame = None
        self.prev_frame = None
        self.keypoints1 = []
        self.descriptor1 = None
        self.keypoints2 = []  # Not strictly needed to clear here, but for consistency
        self.descriptor2 = None
        # self.trajectory.reset_state() # If trajectory needs a reset method

    def get_frame(self) -> np.ndarray | None:
        # In C++, this returned the processed gray frame.
        # If you want the BGR frame, capture it before converting.
        return self.frame

    def set_custom_shape(self, x_or_pos_i2, y: int | None = None):
        if y is not None and isinstance(x_or_pos_i2, int):
            self.custom_shape.set(x_or_pos_i2, y)
        elif isinstance(x_or_pos_i2, Pos_i2):
            self.custom_shape = x_or_pos_i2
        else:
            raise TypeError("Invalid arguments for set_custom_shape")

    def set_data_for_one_iteration(self, alt_val: float, pitch: float | None = None,
                                   yaw: float | None = None, roll: float | None = None) -> bool:
        h_is_valid = False
        if alt_val > 0:
            self.alt = alt_val
            h_is_valid = True

        if pitch is not None and yaw is not None and roll is not None:
            self.custom_set_angles(pitch, yaw, roll)
        return h_is_valid

    def custom_set_angles(self, pitch: float, yaw: float, roll: float):
        self.angles.pitch = pitch
        self.angles.yaw = yaw
        self.angles.roll = roll
        if self.angles_from_stream == OVO_ANGLES_FROM_SOURCE:  # C++ used SOURCE_FLAG for this check
            self.trajectory.curr_angles = self.angles

    def _check_and_resize_frame(self, input_frame: np.ndarray) -> np.ndarray | None:
        if input_frame is None or input_frame.size == 0:
            return None
        if self.custom_shape.x <= 0 or self.custom_shape.y <= 0:
            return input_frame  # No custom shape

        h_orig, w_orig = input_frame.shape[:2]

        if self.custom_shape.x > w_orig or self.custom_shape.y > h_orig:
            print("Warning: Custom shape larger than frame. Returning original.")
            return input_frame

        start_x = w_orig // 2 - self.custom_shape.x // 2
        start_y = h_orig // 2 - self.custom_shape.y // 2
        end_x = start_x + self.custom_shape.x
        end_y = start_y + self.custom_shape.y

        # Ensure bounds are valid (integer division might make them slightly off for odd sizes)
        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(w_orig, end_x)
        end_y = min(h_orig, end_y)

        if start_x >= end_x or start_y >= end_y:  # ROI is invalid
            return input_frame

        return input_frame[start_y:end_y, start_x:end_x]

    def get_affine_info(self, method_flag: int = OVO_RANSAC, ratio_thresh: float = 0.8) -> AffineParams:
        target = AffineParams()  # Default values

        if self.descriptor1 is None or self.descriptor2 is None or \
                len(self.keypoints1) == 0 or len(self.keypoints2) == 0:
            return target

        # KNN match
        knn_matches = self.matcher.knnMatch(self.descriptor1, self.descriptor2, k=2)

        good_matches = []
        for m, n in knn_matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

        if len(good_matches) < 3:  # Need at least 3 for affine
            return target

        # Extract location of good matches
        pts1 = np.float32([self.keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([self.keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # cv2.estimateAffinePartial2D returns a 2x3 matrix (float64) or None
        # method: cv2.RANSAC or cv2.LMEDS
        cv_method = cv2.RANSAC if method_flag == OVO_RANSAC else cv2.LMEDS

        affine_matrix, inliers_mask = cv2.estimateAffinePartial2D(pts1, pts2, method=cv_method)

        if affine_matrix is None:
            return target

        # Affine_params in C++ has int tx, ty
        target.tx = int(affine_matrix[0, 2])
        target.ty = int(affine_matrix[1, 2])

        # Extract scale and angle
        # m00 = scale*cos(angle), m10 = scale*sin(angle)
        m00 = affine_matrix[0, 0]
        m10 = affine_matrix[1, 0]
        # m01 = -scale*sin(angle), m11 = scale*cos(angle)

        # Original C++ logic:
        # cs = mm(0, 0); sn = mm(1, 0);
        # target.angle = atan(sn / cs); target.scale = cs / cos(target.angle);
        cs = m00
        sn = m10

        if cs == 0.0 and sn == 0.0:
            target.angle = 0.0
            target.scale = 0.0
        else:
            target.angle = math.atan2(sn, cs)  # More robust
            cos_angle = math.cos(target.angle)
            if abs(cos_angle) > 1e-6:  # Avoid division by zero
                target.scale = cs / cos_angle
            elif abs(math.sin(target.angle)) > 1e-6:  # Fallback if cos is near zero
                target.scale = sn / math.sin(target.angle)
            else:  # Both sin and cos are zero - degenerate
                target.scale = math.sqrt(cs ** 2 + sn ** 2)  # Should be 0 if cs and sn are 0

        return target

    def _search_affine_matrix(self) -> bool:
        try:
            # get_affine_info returns AffineParams
            calculated_params = self.get_affine_info(method_flag=OVO_RANSAC, ratio_thresh=0.8)
            # Check if params are valid (e.g., non-zero scale)
            if calculated_params.scale != 0.0:  # Basic check
                self.affine_params = calculated_params
                return True
            return False
        except cv2.error as e:
            print(f"OpenCV error in _search_affine_matrix: {e}")
            return False
        except Exception as e:
            print(f"General error in _search_affine_matrix: {e}")
            return False

    def grab_frame_and_data(self) -> bool:
        if not self.cap or not self.cap.isOpened():
            # print("Cap not opened")
            return False

        ret, bgr_frame = self.cap.read()
        if not ret or bgr_frame is None:
            # print("Failed to grab frame")
            return False

        bgr_frame_resized = self._check_and_resize_frame(bgr_frame)
        if bgr_frame_resized is None:
            # print("Resized frame is None")
            return False

        current_gray_frame = cv2.cvtColor(bgr_frame_resized, cv2.COLOR_BGR2GRAY)
        self.frame = current_gray_frame  # Store the processed gray frame

        op_successful = False
        if self.prev_frame is None or not self.keypoints1:
            # First frame or after reset
            self.keypoints1, self.descriptor1 = keypoints_check_py(current_gray_frame, self.detector)
            if self.keypoints1:
                op_successful = True
        else:
            self.keypoints2, self.descriptor2 = keypoints_check_py(current_gray_frame, self.detector)
            if self.keypoints2:
                if self._search_affine_matrix():  # This updates self.affine_params
                    self.trajectory.update_data_from_affine_matrix(self.affine_params, self.alt)
                    op_successful = True

            # Prepare for next iteration
            self.keypoints1 = self.keypoints2
            self.descriptor1 = self.descriptor2
            # self.keypoints2 and self.descriptor2 will be overwritten in the next call

        self.prev_frame = current_gray_frame.copy()  # Important to copy
        return op_successful