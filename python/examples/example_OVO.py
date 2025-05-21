import cv2
import time
import math # For math.nan if needed
# from ovo_types import CameraParams, Pos_i2
# from video_processor_ovo import VideoProcessorOVO
# from ovo_constants import OVO_ANGLES_FROM_VIDEO # or OVO_ANGLES_FROM_SOURCE
from Open_VO import CameraParams, Pos_i2, VideoProcessorOVO, OVO_ANGLES_FROM_VIDEO


def main():
    cam_p = CameraParams()
    cam_p.fov = 86 # Assuming 86 was degrees, convert to radians if map_scale expects it
    # Check how map_scale_py uses fov. If it's directly in tan(fov), it should be radians.
    # If fov in C++ Camera_params was degrees, then the C++ mapScale would convert.
    # Python's math.tan expects radians.

    res = Pos_i2(1920, 1280)
    cam_p.resolution = res
    cam_p.type = 0 # Example type

    video_file = "test.mp4"  # Make sure this file exists

    try:
        # VideoProcessorOVO(Camera_params &p, String filename,short SOURCE_FLAG, int maxPoints)
        vpo = VideoProcessorOVO(
            p_params=cam_p,
            source_info=video_file,
            api_reference_or_source_flag=OVO_ANGLES_FROM_VIDEO, # SOURCE_FLAG
            custom_shape_or_max_points=100 # maxPoints
        )
    except IOError as e:
        print(f"Error initializing VideoProcessor: {e}")
        return

    vpo.set_custom_shape(640, 640)
    h_altitude = 300.0 # meters
    k = 0

    print("Starting video processing loop. Press ESC to quit.")

    while k != 27: # 27 is ASCII for ESC
        start_time = time.perf_counter()

        vpo.set_data_for_one_iteration(h_altitude) # Only setting altitude

        if vpo.grab_frame_and_data():
            end_time = time.perf_counter()
            processing_time_ms = (end_time - start_time) * 1000
            print(f"Full tick: {processing_time_ms:.2f} ms")

            pos = vpo.trajectory.get_curr_pos()
            print(f"Position x,y,z: {pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f}")

            current_display_frame = vpo.get_frame() # This is the processed gray frame
            if current_display_frame is not None:
                cv2.imshow("Capture OVO Python", vpo.frame)
            else:
                print("No frame to display from VPO.")
        else:
            # print("grab_frame_and_data failed or no new data.")
            # If video ends, cap.read() will return False, bgr_frame will be None
            # Check if cap is still good
            if not vpo.cap.isOpened() or vpo.cap.read()[0] == False : # Heuristic check if stream ended
                print("Video stream ended or cannot read frame.")
                break


        k = cv2.waitKey(1) & 0xFF # Wait for 1 ms

    cv2.destroyAllWindows()
    if vpo.cap.isOpened():
        vpo.cap.release()
    print("Processing finished.")

if __name__ == "__main__":
    main()