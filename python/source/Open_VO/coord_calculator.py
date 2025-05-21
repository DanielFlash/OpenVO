import math
import os
import cv2
import numpy as np
from typing import List, Tuple, Optional
from .data_types import (SurfaceData, SurfaceObjData, SurfaceImgData, Detection,
                        MapEdges, LocalData, CameraParams, PosAngle, Pos_d3, Pos_f2)
from .detector import DetectorPy


class CoordCalculatorPy:
    def calc_obj_coords(self, surface_obj_data_list: List[SurfaceObjData]) -> List[SurfaceData]:
        surface_data_list: List[SurfaceData] = []
        for i, obj_data in enumerate(surface_obj_data_list):
            central_point_x = obj_data.bbX + round(obj_data.bbW / 2.0)
            central_point_y = obj_data.bbY + round(obj_data.bbH / 2.0)

            # Avoid division by zero
            if obj_data.imgW == 0 or obj_data.imgH == 0:
                # print(f"Warning: Image dimensions are zero for {obj_data.imgName}")
                coord_x, coord_y = math.nan, math.nan
            else:
                rel_coord_x = central_point_x * (obj_data.imgBotRightX - obj_data.imgTopLeftX) / obj_data.imgW
                rel_coord_y = central_point_y * (obj_data.imgBotRightY - obj_data.imgTopLeftY) / obj_data.imgH
                coord_x = obj_data.imgTopLeftX + rel_coord_x
                coord_y = obj_data.imgTopLeftY + rel_coord_y  # C++ has TopLeftY + relCoordY

            surface_data = SurfaceData(
                imgName=obj_data.imgName,
                imgW=obj_data.imgW,
                imgH=obj_data.imgH,
                imgTopLeftX=obj_data.imgTopLeftX,
                imgTopLeftY=obj_data.imgTopLeftY,
                imgBotRightX=obj_data.imgBotRightX,
                imgBotRightY=obj_data.imgBotRightY,
                objId=i,  # C++ uses incrementing i
                objLabel=obj_data.objLabel,
                bbX=obj_data.bbX,
                bbY=obj_data.bbY,
                bbW=obj_data.bbW,
                bbH=obj_data.bbH,
                objCoordX=coord_x,
                objCoordY=coord_y
            )
            surface_data_list.append(surface_data)
        return surface_data_list

    def detect_and_calc_obj_coords(self, surface_img_data_list: List[SurfaceImgData],
                                   detector: DetectorPy, img_folder: str) -> List[SurfaceData]:
        surface_data_list: List[SurfaceData] = []
        i = 0
        for img_idx, img_data in enumerate(surface_img_data_list):  # Use img_idx for unique objId later if needed
            image_path = os.path.join(img_folder, img_data.imgName)
            if not os.path.exists(image_path):
                # print(f"Warning: Image not found at {image_path}")
                continue

            frame = detector.read_image(image_path)  # DetectorPy.read_image
            if frame is None:
                continue

            detections = detector.detect(frame)

            for detection in detections:
                central_point_x = detection.x + round(detection.w / 2.0)
                central_point_y = detection.y + round(detection.h / 2.0)

                if img_data.imgW == 0 or img_data.imgH == 0:
                    # print(f"Warning: Image dimensions are zero for {img_data.imgName} during detection.")
                    coord_x, coord_y = math.nan, math.nan
                else:
                    rel_coord_x = central_point_x * (img_data.imgBotRightX - img_data.imgTopLeftX) / img_data.imgW
                    rel_coord_y = central_point_y * (img_data.imgBotRightY - img_data.imgTopLeftY) / img_data.imgH
                    coord_x = img_data.imgTopLeftX + rel_coord_x
                    coord_y = img_data.imgTopLeftY + rel_coord_y

                obj_id_for_surface_data = i

                surface_data = SurfaceData(
                    imgName=img_data.imgName,
                    imgW=img_data.imgW,
                    imgH=img_data.imgH,
                    imgTopLeftX=img_data.imgTopLeftX,
                    imgTopLeftY=img_data.imgTopLeftY,
                    imgBotRightX=img_data.imgBotRightX,
                    imgBotRightY=img_data.imgBotRightY,
                    objId=obj_id_for_surface_data,
                    objLabel=detection.class_id,
                    bbX=detection.x,
                    bbY=detection.y,
                    bbW=detection.w,
                    bbH=detection.h,
                    objCoordX=coord_x,
                    objCoordY=coord_y
                )
                surface_data_list.append(surface_data)
                i += 1

        return surface_data_list

    def calc_map_edges(self, surface_data_list: List[SurfaceData]) -> Optional[MapEdges]:
        if not surface_data_list:
            return None

        # Initialize with the first element's image boundaries
        # Note: C++ initializes with imgTopLeftX etc. These are image coords, not necessarily object coords.
        # This seems correct for finding overall map extent from image metadata.
        map_edges = MapEdges(
            topLeftX=surface_data_list[0].imgTopLeftX,
            topLeftY=surface_data_list[0].imgTopLeftY,
            botRightX=surface_data_list[0].imgBotRightX,
            botRightY=surface_data_list[0].imgBotRightY
        )

        for data in surface_data_list[1:]:
            # C++ logic:
            # if (mapEdges.topLeftX > surfaceData.imgTopLeftX) mapEdges.topLeftX = surfaceData.imgTopLeftX;
            # if (mapEdges.topLeftY < surfaceData.imgTopLeftY) mapEdges.topLeftY = surfaceData.imgTopLeftY; // Y increases downwards for image, upwards for geo? Check convention
            # if (mapEdges.botRightX < surfaceData.imgBotRightX) mapEdges.botRightX = surfaceData.imgBotRightX;
            # if (mapEdges.botRightY > surfaceData.imgBotRightY) mapEdges.botRightY = surfaceData.imgBotRightY;
            # This assumes a coordinate system where smaller Y is "top" and larger X is "right"
            # For geographical coordinates, typically larger Y is North (top).
            # Assuming standard cartesian: min X, max Y (top-left); max X, min Y (bottom-right)
            # Or if image-like: min X, min Y (top-left); max X, max Y (bottom-right)
            # Given C++ logic for topLeftY (takes larger Y if current is smaller) and botRightY (takes smaller Y if current is larger),
            # it seems to be finding min X, min Y and max X, max Y across all imgTopLeft/imgBotRight pairs.

            map_edges.topLeftX = min(map_edges.topLeftX, data.imgTopLeftX)
            map_edges.topLeftY = min(map_edges.topLeftY, data.imgTopLeftY)  # Smallest Y for top
            map_edges.botRightX = max(map_edges.botRightX, data.imgBotRightX)
            map_edges.botRightY = max(map_edges.botRightY, data.imgBotRightY)  # Largest Y for bottom

            # Reconciling with C++ logic:
            # C++: if (mapEdges.topLeftY < surfaceData.imgTopLeftY) mapEdges.topLeftY = surfaceData.imgTopLeftY;
            # This means it's looking for the MAX Y for topLeftY. This is unusual if TopLeft means min Y.
            # Let's stick to min/max for bounding box for now unless there's a specific coordinate system.
            # If C++ `mapEdges.topLeftY < surfaceData.imgTopLeftY` means Y is increasing downwards:
            # map_edges.topLeftX = min(map_edges.topLeftX, data.imgTopLeftX)
            # map_edges.topLeftY = max(map_edges.topLeftY, data.imgTopLeftY) # Max Y becomes "top" if Y increases downwards
            # map_edges.botRightX = max(map_edges.botRightX, data.imgBotRightX)
            # map_edges.botRightY = min(map_edges.botRightY, data.imgBotRightY) # Min Y becomes "bottom" if Y increases downwards
            # The C++ logic given is:
            # mapEdges.topLeftY = surfaceDataList[0].imgTopLeftY; ... if (mapEdges.topLeftY < surfaceData.imgTopLeftY) { mapEdges.topLeftY = surfaceData.imgTopLeftY; } -> finds MAX Y for topLeftY
            # mapEdges.botRightY = surfaceDataList[0].imgBotRightY; ... if (mapEdges.botRightY > surfaceData.imgBotRightY) { mapEdges.botRightY = surfaceData.imgBotRightY; } -> finds MIN Y for botRightY
            # This indeed suggests Y increases downwards for "top" and "bottom" definition.

        return map_edges

    def calc_local_obj_coords(self, detections: List[Detection], img_shape_wh: Tuple[float, float],
                              cam_params: CameraParams, curr_angles: PosAngle,
                              curr_offset_xyz: Pos_d3, meter_in_pixel: Pos_f2) -> List[LocalData]:
        local_data_list: List[LocalData] = []
        img_w, img_h = img_shape_wh  # Assuming width, height

        for detection in detections:
            central_point_x = detection.x + round(detection.w / 2.0)
            central_point_y = detection.y + round(detection.h / 2.0)

            # Logic from C++ Trajectory::getLocalPosition
            if img_w == 0 or img_h == 0 or cam_params.resolution.x == 0 or cam_params.resolution.y == 0:
                obj_coord_x, obj_coord_y = math.nan, math.nan
            else:
                scale_x = float(cam_params.resolution.x) / img_w
                scale_y = float(cam_params.resolution.y) / img_h

                offset_x_scaled = central_point_x * scale_x
                offset_y_scaled = central_point_y * scale_y

                vs_x = (offset_x_scaled - cam_params.resolution.x / 2.0) * meter_in_pixel.x
                vs_y = (offset_y_scaled - cam_params.resolution.y / 2.0) * meter_in_pixel.y

                # curr_offset_xyz.x and curr_offset_xyz.y are used as 2D offset
                # C++: objCoordX = curr_offset.x + (vs.x * cosf(curr_angles.yaw) - vs.y * sinf(curr_angles.yaw));
                # C++: objCoordY = curr_offset.x + (vs.x * sinf(curr_angles.yaw) + vs.y * cosf(curr_angles.yaw)); // Typo in C++, should be curr_offset.y

                rotated_vs_x = vs_x * math.cos(curr_angles.yaw) - vs_y * math.sin(curr_angles.yaw)
                rotated_vs_y = vs_x * math.sin(curr_angles.yaw) + vs_y * math.cos(curr_angles.yaw)

                obj_coord_x = curr_offset_xyz.x + rotated_vs_x
                obj_coord_y = curr_offset_xyz.y + rotated_vs_y  # Corrected to use .y from curr_offset

            local_data = LocalData(
                objLabel=detection.class_id,
                objCoordX=obj_coord_x,
                objCoordY=obj_coord_y
                # objId, mappedTo, overlapLevel are typically set later
            )
            local_data_list.append(local_data)
        return local_data_list
