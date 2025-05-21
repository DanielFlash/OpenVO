import cv2
import numpy as np
import math
import copy
from typing import List, Tuple, Dict, Optional
from .file_io import SurfaceDataReaderPy, SurfaceDataWriterPy
from .detector import DetectorPy
from .coord_calculator import CoordCalculatorPy
from .data_types import (SurfaceImgData, SurfaceObjData, SurfaceData, LocalData, MapEdges, Detection, ObjectDist,
                        CameraParams, PosAngle, Pos_f2, Pos_d3, Pos_i2)


class MapAnalysis:
    """
    Class for navigation correction
    """

    def __init__(self,
                 input_file: str,
                 img_folder: Optional[str],  # Can be None
                 output_file: str,
                 labels_file: str,
                 model_path: str,  # Local model
                 cuda_enabled: bool,
                 img_w: int,
                 img_h: int,
                 sc_thres: float,
                 nms_thres: float,
                 max_d: int,
                 gl_model_path: Optional[str] = None,  # Global model, optional
                 gl_cuda_enabled: Optional[bool] = None,
                 gl_img_w: Optional[int] = None,
                 gl_img_h: Optional[int] = None,
                 gl_sc_thres: Optional[float] = None,
                 gl_nms_thres: Optional[float] = None,
                 gl_max_d: Optional[int] = None):

        self.input_file: str = input_file
        self.img_folder: Optional[str] = img_folder
        self.output_file: str = output_file
        self.labels_file: str = labels_file

        # Local detector parameters
        self.model_path: str = model_path
        self.cuda_enabled: bool = cuda_enabled
        self.img_w: int = img_w
        self.img_h: int = img_h
        self.score_thresh: float = sc_thres
        self.nms_thresh: float = nms_thres
        self.max_det: int = max_d

        # Global detector parameters
        self.gl_model_path: Optional[str] = gl_model_path
        self.gl_cuda_enabled: bool = gl_cuda_enabled if gl_cuda_enabled is not None else cuda_enabled  # Default to local
        self.gl_img_w: int = gl_img_w if gl_img_w is not None else img_w
        self.gl_img_h: int = gl_img_h if gl_img_h is not None else img_h
        self.gl_score_thresh: float = gl_sc_thres if gl_sc_thres is not None else sc_thres
        self.gl_nms_thresh: float = gl_nms_thres if gl_nms_thres is not None else nms_thres
        self.gl_max_det: int = gl_max_d if gl_max_d is not None else max_d

        # Initialize components (using stubs for now)
        self.surface_data_reader: SurfaceDataReaderPy = SurfaceDataReaderPy(self.input_file, self.img_folder,
                                                                                self.output_file)
        self.surface_data_writer: SurfaceDataWriterPy = SurfaceDataWriterPy(self.output_file)
        self.coord_calculator: CoordCalculatorPy = CoordCalculatorPy()

        self.detector: DetectorPy = DetectorPy(self.labels_file, self.model_path, self.cuda_enabled,
                                                   self.img_w, self.img_h, self.score_thresh,
                                                   self.nms_thresh, self.max_det)

        if self.gl_model_path:
            self.gl_detector: DetectorPy = DetectorPy(self.labels_file, self.gl_model_path, self.gl_cuda_enabled,
                                                          self.gl_img_w, self.gl_img_h, self.gl_score_thresh,
                                                          self.gl_nms_thresh, self.gl_max_det)
        else:
            # C++: m_glDetector(m_detector) - copy constructor
            # Python: Create a new instance or copy. Using copy.deepcopy for safety if Detector has internal state.
            print("Global detector model path not provided, copying local detector settings for global detector.")
            self.gl_detector = copy.deepcopy(self.detector)

        self.surface_img_data_list: List[SurfaceImgData] = []
        self.surface_obj_data_list: List[SurfaceObjData] = []
        self.surface_data_list: List[SurfaceData] = []
        self.local_data_list: List[LocalData] = []
        self.map_edges: MapEdges = MapEdges()

    def _calc_deltas(self, local_min_x: float, local_max_y: float, global_min_x: float, global_max_y: float,
                     lw: float, lh: float, gw: float, gh: float, label: int, scale: float, match_delta: int,
                     proper_objects: List[LocalData], proper_surface_obj: List[SurfaceData]) -> Tuple[float, float]:

        # Ensure dimensions are integers for creating cv2.Mat
        lh_int, lw_int = int(round(lh)), int(round(lw))
        gh_int, gw_int = int(round(gh)), int(round(gw))

        if lh_int <= 0 or lw_int <= 0 or gh_int <= 0 or gw_int <= 0:
            # print("Warning: Invalid dimensions for local/global maps in _calc_deltas.")
            return 0.0, 0.0

        local_map = np.zeros((lh_int, lw_int), dtype=np.float32)
        global_map = np.zeros((gh_int, gw_int), dtype=np.float32)

        for local_data in proper_objects:
            if local_data.objLabel == label:
                x_norm = int(round((local_data.objCoordX - local_min_x) / scale))
                y_norm = int(round((local_max_y - local_data.objCoordY) / scale))  # Y is inverted

                x_norm = np.clip(x_norm, 0, lw_int - 1)
                y_norm = np.clip(y_norm, 0, lh_int - 1)
                local_map[y_norm, x_norm] = 1.0  # Use 1.0 for float matrix

        if cv2.countNonZero(local_map) < 1:
            return 0.0, 0.0

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (match_delta, match_delta))
        dilated_local_map = cv2.dilate(local_map, kernel)

        for surface_data in proper_surface_obj:
            if surface_data.objLabel == label:
                x_norm = int(round((surface_data.objCoordX - global_min_x) / scale))
                y_norm = int(round((global_max_y - surface_data.objCoordY) / scale))  # Y is inverted

                x_norm = np.clip(x_norm, 0, gw_int - 1)
                y_norm = np.clip(y_norm, 0, gh_int - 1)
                global_map[y_norm, x_norm] = 1.0  # Use 1.0 for float matrix

        if cv2.countNonZero(global_map) < 1:  # Check if global map also has points
            return 0.0, 0.0

        dilated_global_map = cv2.dilate(global_map, kernel)

        if dilated_local_map.shape[0] > dilated_global_map.shape[0] or \
                dilated_local_map.shape[1] > dilated_global_map.shape[1]:
            # print("Warning: Local map larger than global map in _calc_deltas after dilation.")
            return 0.0, 0.0

        res = cv2.matchTemplate(dilated_global_map, dilated_local_map, cv2.TM_CCOEFF)
        cv2.normalize(res, res, 0, 1, cv2.NORM_MINMAX, -1)
        _, _, _, max_loc = cv2.minMaxLoc(res)  # We only need max_loc for TM_CCOEFF

        match_loc = max_loc  # Point object

        delta_x = match_loc[0] * scale  # match_loc.x
        delta_y = match_loc[1] * scale  # match_loc.y
        delta_x = global_min_x - local_min_x + delta_x
        # Y-axis is typically inverted in image coordinates vs some world coordinates.
        # The C++ code: deltaY = globalMaxY - localMaxY + deltaY;
        # This needs careful check depending on coordinate system conventions.
        # Assuming match_loc[1] (y in image template matching) corresponds to downward displacement.
        # If localMaxY and globalMaxY are "top" Y values (larger means higher up):
        # A positive match_loc[1] means the local_map pattern was found "lower" in the global_map.
        # If Y in world coords increases upwards:
        # delta_y_map = global_max_y_map_origin - local_max_y_map_origin - match_loc[1]*scale
        # The C++ logic `deltaY = globalMaxY - localMaxY + deltaY` (where the last deltaY is `matchLoc.y * scale`)
        # suggests `matchLoc.y` is an offset in the "global map pixel space" (Y down).
        # If globalMaxY and localMaxY are the "upper Y bounds" in world space (Y up),
        # and matchLoc.y is pixels downwards in the global_map image representation:
        # World_Y_local_origin_in_Global = Global_World_Y_of_GlobalMapImageOrigin - matchLoc.y * scale
        # World_Y_local_origin = local_max_y (top of local view in world)
        # World_Y_global_image_origin = global_max_y (top of global view in world)
        # local_max_y + delta_y_world = World_Y_local_origin_in_Global
        # delta_y_world = global_max_y - matchLoc.y * scale - local_max_y
        # This matches the C++ if positive Y is up in world coords.
        delta_y = (global_max_y - (match_loc[1] * scale)) - local_max_y

        return delta_x, delta_y

    def _calc_obj_dist(self, deltas: Tuple[float, float], label: int,
                       proper_objects: List[LocalData], proper_surface_obj: List[SurfaceData]) -> List[ObjectDist]:
        object_dist_list: List[ObjectDist] = []
        delta_x_val, delta_y_val = deltas

        for local_data in proper_objects:
            if local_data.objLabel == label:
                for surface_data in proper_surface_obj:
                    if surface_data.objLabel == label:
                        dist_x = surface_data.objCoordX - (local_data.objCoordX + delta_x_val)
                        dist_y = surface_data.objCoordY - (local_data.objCoordY + delta_y_val)
                        dist = math.sqrt(dist_x ** 2 + dist_y ** 2)

                        # Create new instances, don't pass references to lists being iterated if they are modified
                        obj_dist = ObjectDist(localData=local_data,  # Storing reference to original object
                                              surfaceData=surface_data,  # Storing reference
                                              dist=dist,
                                              deltaX=delta_x_val,
                                              deltaY=delta_y_val)
                        object_dist_list.append(obj_dist)

        object_dist_list.sort(key=lambda od: od.dist)
        return object_dist_list

    def _map_objects(self, best_candidates: List[ObjectDist]):
        for object_dist in best_candidates:
            # Ensure objects exist and haven't been mapped
            if object_dist.localData and object_dist.surfaceData:
                if object_dist.localData.mappedTo == -1 and object_dist.surfaceData.mappedTo == -1:
                    object_dist.localData.mappedTo = object_dist.surfaceData.objId
                    object_dist.surfaceData.mappedTo = object_dist.localData.objId

    def _update_local_data_coord(self, delta_x: float, delta_y: float):
        for local_data in self.local_data_list:
            local_data.objCoordX += delta_x
            local_data.objCoordY += delta_y

    # --- Public methods ---
    def load_raw_data(self):
        self.surface_img_data_list = self.surface_data_reader.read_raw_data()

    def load_raw_labeled_data(self):
        self.surface_obj_data_list = self.surface_data_reader.read_raw_labeled_data()

    def load_processed_data(self):
        self.surface_data_list = self.surface_data_reader.read_processed_data()

    def save_processed_data(self):
        self.surface_data_writer.write_data(self.surface_data_list)

    def process_raw_data(self):
        if self.img_folder is None:  # Corresponds to C++ m_imgFolder == nullptr
            self.surface_data_list = self.coord_calculator.calc_obj_coords(self.surface_obj_data_list)
        else:
            self.surface_data_list = self.coord_calculator.detect_and_calc_obj_coords(
                self.surface_img_data_list, self.gl_detector, self.img_folder
            )

    def calculate_map_objects(self):
        if self.img_folder is None:
            self.load_raw_labeled_data()
        else:
            self.load_raw_data()
        self.process_raw_data()
        self.save_processed_data()

    def load_map_objects(self):
        self.load_processed_data()

    def calc_map_edges(self):
        self.map_edges = self.coord_calculator.calc_map_edges(self.surface_data_list)

    def location_verification(self, curr_x: float, curr_y: float, fov_x: float, fov_y: float, delta_fov: float) -> bool:
        # Ensure map_edges has been calculated
        if self.map_edges.topLeftX is None:  # Or some other indicator it's not set
            print("Warning: Map edges not calculated. Call calc_map_edges() first.")
            # Potentially calculate it here if not done, or return False
            # self.calc_map_edges() # if desired
            # if self.map_edges.top_left_x is None: # check again
            return False

        return (curr_x - fov_x - delta_fov > self.map_edges.topLeftX and
                curr_x + fov_x + delta_fov < self.map_edges.botRightX and
                curr_y - fov_y - delta_fov < self.map_edges.topLeftY and  # Assuming top_left_y is max Y
                curr_y + fov_y + delta_fov > self.map_edges.botRightY)  # Assuming bot_right_y is min Y
        # Check Y coordinate conventions carefully

    def object_detection(self, image: np.ndarray) -> List[Detection]:
        return self.detector.detect(image)

    def object_coord_proc(self, detections: List[Detection], img_shape: Tuple[int, int],  # (height, width)
                          cam_params: CameraParams, curr_angles: PosAngle,
                          curr_offset: Pos_d3, meter_in_pixel: Pos_f2) -> List[LocalData]:
        return self.coord_calculator.calc_local_obj_coords(detections, img_shape, cam_params,
                                                           curr_angles, curr_offset, meter_in_pixel)

    def object_verification(self, current_frame_local_data: List[LocalData], identity_delta: int):
        max_id = -1
        for ld_existing in self.local_data_list:
            if ld_existing.objId > max_id:
                max_id = ld_existing.objId

        for local_data_new in current_frame_local_data:
            min_bias_sum = float('inf')
            best_candidate_existing: Optional[LocalData] = None

            for ld_existing in self.local_data_list:
                # Check proximity
                if (ld_existing.objCoordX - identity_delta < local_data_new.objCoordX < ld_existing.objCoordX + identity_delta and
                        ld_existing.objCoordY - identity_delta < local_data_new.objCoordY < ld_existing.objCoordY + identity_delta):

                    # Check label match
                    if ld_existing.objLabel == local_data_new.objLabel:
                        delta_x_abs = abs(ld_existing.objCoordX - local_data_new.objCoordX)
                        delta_y_abs = abs(ld_existing.objCoordY - local_data_new.objCoordY)
                        current_bias_sum = delta_x_abs + delta_y_abs

                        if current_bias_sum < min_bias_sum:
                            min_bias_sum = current_bias_sum
                            best_candidate_existing = ld_existing

            if best_candidate_existing:
                # Update existing object
                best_candidate_existing.objCoordX = (best_candidate_existing.objCoordX + local_data_new.objCoordX) / 2
                best_candidate_existing.objCoordY = (best_candidate_existing.objCoordY + local_data_new.objCoordY) / 2
                best_candidate_existing.overlapLevel += 1
            else:
                # Add as new object
                max_id += 1
                local_data_new.obj_id = max_id
                # local_data_new.overlap_level is already 0 by default from LocalData definition if not set elsewhere
                self.local_data_list.append(local_data_new)

    def calculate_local_objects(self, image: np.ndarray, identity_delta: int,
                                cam_params: CameraParams, curr_angles: PosAngle,
                                curr_offset: Pos_d3, meter_in_pixel: Pos_f2):
        img_shape = (image.shape[0], image.shape[1])  # height, width
        detections = self.object_detection(image)
        current_frame_local_data = self.object_coord_proc(detections, img_shape, cam_params,
                                                          curr_angles, curr_offset, meter_in_pixel)
        self.object_verification(current_frame_local_data, identity_delta)

    def object_matcher(self, curr_x: float, curr_y: float, fov_x: float, fov_y: float,
                       delta_fov: float, delta_offset: float, match_delta: int,
                       conf_overlap: int, obj_per_class_thresh: int, scale: float) -> Tuple[float, float]:

        local_min_x = curr_x - fov_x - delta_fov
        local_max_x = curr_x + fov_x + delta_fov
        local_min_y = curr_y - fov_y - delta_fov  # Lower Y bound in world
        local_max_y = curr_y + fov_y + delta_fov  # Upper Y bound in world

        global_min_x = local_min_x - delta_offset
        global_max_x = local_max_x + delta_offset
        global_min_y = local_min_y - delta_offset
        global_max_y = local_max_y + delta_offset

        confident_objects = [ld for ld in self.local_data_list if ld.overlapLevel >= conf_overlap]

        if len(confident_objects) < obj_per_class_thresh:  # Initial check, maybe not objects of a single class yet
            return 0.0, 0.0

        proper_local_objects = [
            ld for ld in confident_objects
            if local_min_x < ld.objCoordX < local_max_x and local_min_y < ld.objCoordY < local_max_y
        ]

        if len(proper_local_objects) < obj_per_class_thresh:  # Check again after spatial filtering
            return 0.0, 0.0

        object_classes_count: Dict[int, int] = {}  # Counts common objects between local and global for a class
        # Get satellite objects in the global ROI
        proper_surface_obj = [
            sd for sd in self.surface_data_list
            if global_min_x < sd.objCoordX < global_max_x and global_min_y < sd.objCoordY < global_max_y
        ]

        if not proper_surface_obj:
            return 0.0, 0.0

        # Populate object_classes_count with classes present in proper_surface_obj
        # And count how many local objects of that class are in proper_local_objects
        surface_obj_labels_in_roi = {sd.objLabel for sd in proper_surface_obj}

        for local_data in proper_local_objects:
            if local_data.objLabel in surface_obj_labels_in_roi:
                object_classes_count[local_data.objLabel] = object_classes_count.get(local_data.objLabel, 0) + 1

        # Filtered map for labels that have enough objects in local view and are present in global view
        labels_to_process = [
            label for label, count in object_classes_count.items() if count >= obj_per_class_thresh
        ]

        if not labels_to_process:
            return 0.0, 0.0

        lw = round((local_max_x - local_min_x) / scale)
        lh = round((local_max_y - local_min_y) / scale)  # Height in world units / scale
        gw = round((global_max_x - global_min_x) / scale)
        gh = round((global_max_y - global_min_y) / scale)

        deltas_per_class: Dict[int, List[ObjectDist]] = {}

        for label in labels_to_process:
            # Filter proper_local_objects and proper_surface_obj for current label
            current_label_proper_local = [p_obj for p_obj in proper_local_objects if p_obj.objLabel == label]
            current_label_proper_surface = [ps_obj for ps_obj in proper_surface_obj if ps_obj.objLabel == label]

            if not current_label_proper_local or not current_label_proper_surface:
                continue

            deltas_tuple = self._calc_deltas(local_min_x, local_max_y, global_min_x, global_max_y,
                                             lw, lh, gw, gh, label, scale, match_delta,
                                             current_label_proper_local, current_label_proper_surface)

            if deltas_tuple == (0.0, 0.0) and not (
                    lw == 0 and lh == 0 and gw == 0 and gh == 0):  # Avoid false positive if maps were empty
                # Potentially skip this class or return early if this implies failure for a critical class
                # The C++ code returns [0,0] which stops further processing for this object_matcher call
                # print(f"Warning: _calc_deltas returned (0,0) for label {label}. This might indicate an issue.")
                # Let's allow continuing with other classes unless this is the only class
                # For now, mimic C++ and return if any class fails to give non-zero delta
                # However, the C++ code has a specific check: if (deltas[0] == 0. && deltas[1] == 0.) return {0,0}
                # This seems too strict. A (0,0) delta could be valid.
                # The original C++ logic seems to return [0,0] from object_matcher if _calc_deltas returns [0,0]
                # This check will be inside the loop, so it will return if the *first* processed class yields 0,0 delta
                # This behavior is suspicious, but I will replicate it.
                # return 0.0, 0.0 # Replicating the early exit from C++ if a class yields (0,0) delta from calc_deltas.
                # This should be reconsidered in a real application.
                # If calc_deltas returns 0,0 because maps are empty, that's handled by countNonZero.
                # If it's due to perfect alignment or no match, it's different.
                # The original C++ if (deltas[0] == 0. && deltas[1] == 0.) return std::vector<double> { 0., 0.};
                # This return is from object_matcher, not just skipping the class.
                pass  # Let's assume (0,0) is a possible valid delta, unless maps were empty.

            object_dist_list = self._calc_obj_dist(deltas_tuple, label, current_label_proper_local,
                                                   current_label_proper_surface)
            if object_dist_list:
                deltas_per_class[label] = object_dist_list

        if not deltas_per_class:
            return 0.0, 0.0

        best_candidates_list: Optional[List[ObjectDist]] = None
        min_overall_dist = float('inf')

        for label_key in deltas_per_class:
            if deltas_per_class[label_key]:  # If list is not empty
                # The first element is the one with the minimum distance for that class
                current_min_dist_for_class = deltas_per_class[label_key][0].dist
                if current_min_dist_for_class < min_overall_dist:
                    min_overall_dist = current_min_dist_for_class
                    best_candidates_list = deltas_per_class[label_key]

        if not best_candidates_list:
            return 0.0, 0.0

        obj_per_class_passed = 0
        for obj_dist in best_candidates_list:
            if obj_dist.dist < match_delta:  # Using match_delta as distance threshold too
                obj_per_class_passed += 1
                if obj_per_class_passed >= obj_per_class_thresh:  # This condition is on count for the *best class*
                    break

        if obj_per_class_passed >= obj_per_class_thresh:
            final_delta_x = best_candidates_list[0].deltaX
            final_delta_y = best_candidates_list[0].deltaY

            self._map_objects(best_candidates_list)
            self._update_local_data_coord(final_delta_x, final_delta_y)
            return final_delta_x, final_delta_y
        else:
            return 0.0, 0.0
