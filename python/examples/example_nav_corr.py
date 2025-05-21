import cv2
import os
import numpy as np
# from map_analysis import (MapAnalysis, Detection, DetectorPy, CameraParams, PosAngle, Pos_i2, Pos_d3, Pos_f2,
#                           SurfaceData, LocalData)
from Open_VO import (MapAnalysis, Detection, DetectorPy, CameraParams, PosAngle, Pos_i2, Pos_d3, Pos_f2,
                          SurfaceData, LocalData)


def detection_example():
    model_path = os.path.join(os.path.dirname(__file__), 'best.torchscript')
    # model_path = os.path.join(os.path.dirname(__file__), 'best.onnx')
    label_path = os.path.join(os.path.dirname(__file__), 'labels.txt')
    img_path = os.path.join(os.path.dirname(__file__), 'test_image3.jpg')
    detector = DetectorPy(label_path, model_path, True, 640, 640)
    frame = detector.read_image(img_path)
    output = detector.detect(frame)

    for det in output:
        print(det.class_id, det.className, det.confidence, det.x, det.y, det.w, det.h)
        frame = cv2.rectangle(frame, (det.x, det.y), (det.x + det.w, det.y + det.h), (0, 255, 0), 2)

    cv2.imshow("Result", frame)
    cv2.waitKey(0)


def map_analysis_example():
    print("Initializing MapAnalysis...")
    # Example with global model
    map_analyzer_full = MapAnalysis(
        input_file="input_tmp2.txt",
        img_folder="./",
        output_file="output_tmp2.txt",
        labels_file="labels.txt",
        model_path="best.torchscript",  # Or .pt, .onnx etc.
        cuda_enabled=True,
        img_w=640, img_h=640,
        sc_thres=0.45, nms_thres=0.50, max_d=100,
        gl_model_path="best.torchscript",
        gl_cuda_enabled=True,
        gl_img_w=1280, gl_img_h=1280,
        gl_sc_thres=0.3, gl_nms_thres=0.45, gl_max_d=300
    )
    print("\nMapAnalysis with global model initialized.")
    map_analyzer_full.calculate_map_objects()  # Process satellite map
    map_analyzer_full.calc_map_edges()

    # Simulate local object processing
    print("\nSimulating local object processing...")
    detector = DetectorPy("labels.txt", "best.torchscript", True, 640, 640)
    frame = detector.read_image(os.path.join(os.path.dirname(__file__), 'test_image3.jpg'))

    cam_p = CameraParams(fov=60.0, resolution=Pos_i2(x=640, y=480))
    angles = PosAngle(pitch=0.1, roll=0.0, yaw=0.5)
    offset = Pos_d3(x=1.0, y=2.0, z=1.5)
    m_in_pix = Pos_f2(x=0.1, y=0.1)

    map_analyzer_full.calculate_local_objects(frame, identity_delta=10,
                                              cam_params=cam_p, curr_angles=angles,
                                              curr_offset=offset, meter_in_pixel=m_in_pix)
    print(f"Local objects after processing: {len(map_analyzer_full.local_data_list)}")
    if map_analyzer_full.local_data_list:
        print(f"First local object: {map_analyzer_full.local_data_list[0]}")

    # Simulate object matching
    print("\nSimulating object matching...")
    # Ensure surface_data_list is populated for matching
    if not map_analyzer_full.surface_data_list:
        # Add some dummy surface data if calculate_map_objects didn't run or returned empty
        map_analyzer_full.surface_data_list.append(
            SurfaceData(objId=100, objLabel=0, objCoordX=55.0, objCoordY=55.0)
        )
        map_analyzer_full.surface_data_list.append(
            SurfaceData(objId=101, objLabel=0, objCoordX=150.0, objCoordY=160.0)
        )
        map_analyzer_full.calc_map_edges()  # Recalculate if data changed

    # Ensure local_data_list is populated for matching
    if not map_analyzer_full.local_data_list:
        map_analyzer_full.local_data_list.append(
            LocalData(objId=1, objLabel=0, objCoordX=50.0, objCoordY=50.0, overlapLevel=5)
        )

    deltas_found = map_analyzer_full.object_matcher(
        curr_x=50.0, curr_y=50.0, fov_x=100.0, fov_y=100.0, delta_fov=10.0,
        delta_offset=20.0, match_delta=15, conf_overlap=2, obj_per_class_thresh=1, scale=0.1
    )
    print(f"Object matcher found deltas: {deltas_found}")
    if map_analyzer_full.local_data_list:
        print(f"First local object after matching: {map_analyzer_full.local_data_list[0]}")

    print("\nVerification of location")
    is_inside = map_analyzer_full.location_verification(curr_x=10.0, curr_y=10.0, fov_x=5.0, fov_y=5.0, delta_fov=1.0)
    print(f"Is (10,10) inside map? {is_inside}")


def main():
    # detection_example()
    map_analysis_example()


if __name__ == '__main__':
    main()
