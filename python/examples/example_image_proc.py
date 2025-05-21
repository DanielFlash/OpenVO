from typing import List, Dict

import cv2
import numpy as np
import math  # For math.pi, math.cos, math.sin

# Assume all translated classes and functions are in this module
# import combined_imgproc as ips
from Open_VO import (apply_histogram_clahe_py, apply_histogram_py, Description, Point, KnnMatchPy, Match,
                     find_homography_py_custom_ransac, UniqueCombinationGeneratorPy,
                     estimate_affine_partial2d_py_custom_ransac)


def create_sample_images_for_transform():
    """Creates two simple images for transformation tests if they don't exist."""
    img1_path = "test_image1.png"
    img2_path = "test_image2_transformed.png"

    if not (cv2.imread(img1_path) is not None and cv2.imread(img2_path) is not None):
        print("Creating sample images for transformation tests...")
        # Image 1: A square with text
        img1 = np.zeros((300, 300, 3), dtype=np.uint8) + 200  # Light gray background
        cv2.rectangle(img1, (50, 50), (150, 150), (0, 0, 255), -1)  # Red square
        cv2.putText(img1, "SRC", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(img1_path, img1)

        # Image 2: Image 1 transformed (e.g., perspective)
        pts1 = np.float32([[50, 50], [150, 50], [50, 150], [150, 150]])
        pts2 = np.float32([[30, 60], [170, 40], [70, 180], [190, 160]])
        matrix_h = cv2.getPerspectiveTransform(pts1, pts2)
        img2 = cv2.warpPerspective(img1, matrix_h, (300, 300))
        cv2.putText(img2, "DST", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)  # Black text
        cv2.imwrite(img2_path, img2)
        print(f"Sample images created: {img1_path}, {img2_path}")
    return img1_path, img2_path


def main():
    print("--- Python Example Usage for Translated Image Processing Suite ---")

    # Create or load sample images
    img1_path, img2_path_transformed = create_sample_images_for_transform()
    image1_bgr = cv2.imread(img1_path)
    image2_bgr_transformed = cv2.imread(img2_path_transformed)

    if image1_bgr is None:
        print(f"Error: Could not load {img1_path}. Please ensure it exists.")
        return
    if image2_bgr_transformed is None:
        print(f"Error: Could not load {img2_path_transformed}. Please ensure it exists.")
        return

    image1_gray = cv2.cvtColor(image1_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Histogram Equalization
    print("\n--- Testing Histogram Functions ---")
    try:
        # Custom CLAHE
        clahe_result_custom = apply_histogram_clahe_py(image1_bgr.copy(), tile_size_x=80, tile_size_y=80,
                                                           relative_clip_limit=3.0)
        # cv2.imshow("Original", image1_bgr)
        # cv2.imshow("Custom CLAHE", clahe_result_custom)
        print("Custom CLAHE applied.")

        # OpenCV CLAHE for comparison
        clahe_cv = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(image1_bgr.shape[1] // 80, image1_bgr.shape[0] // 80))
        ycrcb_img = cv2.cvtColor(image1_bgr, cv2.COLOR_BGR2YCrCb)
        ycrcb_img[:, :, 0] = clahe_cv.apply(ycrcb_img[:, :, 0])
        clahe_result_cv = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
        # cv2.imshow("OpenCV CLAHE", clahe_result_cv)
        print("OpenCV CLAHE applied for comparison.")

        # Global Histogram Equalization
        hist_eq_result_custom = apply_histogram_py(image1_bgr.copy())
        # cv2.imshow("Custom Global HistEq", hist_eq_result_custom)
        print("Custom Global Histogram Equalization applied.")
        # cv2.waitKey(0)
    except Exception as e:
        print(f"Error during histogram tests: {e}")

    # 2. Feature Matching and Homography/Affine
    print("\n--- Testing Feature Matching & Transformations ---")
    try:
        # Initialize ORB detector (from OpenCV)
        orb = cv2.ORB_create(nfeatures=500)

        # Find keypoints and descriptors
        kp1_cv, des1_cv = orb.detectAndCompute(image1_bgr, None)
        kp2_cv, des2_cv = orb.detectAndCompute(image2_bgr_transformed, None)

        if des1_cv is None or des2_cv is None or len(kp1_cv) < 4 or len(kp2_cv) < 4:
            print("Not enough keypoints found for matching. Skipping transformation tests.")
        else:
            print(f"Found {len(kp1_cv)} keypoints in image1, {len(kp2_cv)} in image2.")

            # Convert to custom Description and Point types for custom KnnMatch
            des1_custom = Description.list_from_cv_descriptors(des1_cv)
            des2_custom = Description.list_from_cv_descriptors(des2_cv)
            kp1_custom = Point.list_from_cv_keypoints(kp1_cv)
            kp2_custom = Point.list_from_cv_keypoints(kp2_cv)

            # --- Custom KnnMatch ---
            knn_matcher_custom = KnnMatchPy()
            # knnMatch in C++ returns all k neighbors for each query descriptor.
            # The C++ KnnMatch.find returns a flat list of Matches.
            raw_matches_custom = knn_matcher_custom.find(des1_custom, des2_custom, k_neighbors=2)
            print(f"Custom KnnMatcher found {len(raw_matches_custom)} raw candidate pairs (k=2).")

            # Apply Lowe's ratio test to custom KNN matches
            good_matches_custom_idx: List[Match] = []
            # Group raw_matches_custom by source index to apply ratio test
            matches_by_src_idx: Dict[int, List[Match]] = {}
            for m in raw_matches_custom:
                if m.src not in matches_by_src_idx:
                    matches_by_src_idx[m.src] = []
                matches_by_src_idx[m.src].append(m)

            for src_idx in matches_by_src_idx:
                src_matches = sorted(matches_by_src_idx[src_idx], key=lambda x: x.distance)
                if len(src_matches) >= 2 and src_matches[0].distance < 0.75 * src_matches[1].distance:
                    good_matches_custom_idx.append(src_matches[0])
            print(f"Good matches after ratio test (custom KNN): {len(good_matches_custom_idx)}")

            # --- OpenCV BFMatcher for comparison ---
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)  # Use NORM_HAMMING for ORB
            knn_matches_cv = bf.knnMatch(des1_cv, des2_cv, k=2)
            good_matches_cv: List[cv2.DMatch] = []
            for m_grp in knn_matches_cv:
                if len(m_grp) == 2 and m_grp[0].distance < 0.75 * m_grp[1].distance:
                    good_matches_cv.append(m_grp[0])
            print(f"Good matches after ratio test (OpenCV KNN): {len(good_matches_cv)}")

            # Select which set of matches to use (e.g., OpenCV's for robustness)
            # For demonstrating custom RANSAC, we need lists of corresponding Point objects
            if len(good_matches_cv) >= 4:  # Need at least 4 for homography
                src_pts_for_transform = [Point.from_cv_keypoint(kp1_cv[m.queryIdx]) for m in good_matches_cv]
                dst_pts_for_transform = [Point.from_cv_keypoint(kp2_cv[m.trainIdx]) for m in good_matches_cv]

                # --- Homography using Custom RANSAC ---
                print("\nCalculating Homography with Custom RANSAC...")
                # Note: Threshold for RANSAC is typically in pixels.
                # The C++ code had threshold = 0.85 for findHomography, which might be too small
                # if point coordinates are large. Adjust as needed.
                H_custom_ransac = find_homography_py_custom_ransac(
                    src_pts_for_transform, dst_pts_for_transform,
                    method_filter="RANSAC", threshold=5.0, max_iterations=1000,
                    num_random_points_homography=4
                    # matches_list=ips.Match.list_from_cv_dmatches(good_matches_cv) # If passing all points and matches
                )
                if H_custom_ransac is not None:
                    print("Homography Matrix (Custom RANSAC):\n", H_custom_ransac)
                    # Warp image1 using custom H
                    img1_warped_custom_h = cv2.warpPerspective(image1_bgr, H_custom_ransac,
                                                               (image1_bgr.shape[1], image1_bgr.shape[0]))
                    # cv2.imshow("Image1 Warped (Custom RANSAC H)", img1_warped_custom_h)
                else:
                    print("Custom RANSAC Homography failed.")

                # --- Homography using OpenCV RANSAC ---
                src_np = np.float32([kp1_cv[m.queryIdx].pt for m in good_matches_cv]).reshape(-1, 1, 2)
                dst_np = np.float32([kp2_cv[m.trainIdx].pt for m in good_matches_cv]).reshape(-1, 1, 2)
                H_cv_ransac, mask_cv_h = cv2.findHomography(src_np, dst_np, cv2.RANSAC, 5.0)
                if H_cv_ransac is not None:
                    print("Homography Matrix (OpenCV RANSAC):\n", H_cv_ransac)
                    img1_warped_cv_h = cv2.warpPerspective(image1_bgr, H_cv_ransac,
                                                           (image1_bgr.shape[1], image1_bgr.shape[0]))
                    # cv2.imshow("Image1 Warped (CV RANSAC H)", img1_warped_cv_h)
                else:
                    print("OpenCV RANSAC Homography failed.")

                # --- Affine using Custom RANSAC ---
                print("\nCalculating Affine with Custom RANSAC...")
                # Affine needs at least 3 points
                if len(src_pts_for_transform) >= 3:
                    A_custom_ransac = estimate_affine_partial2d_py_custom_ransac(
                        src_pts_for_transform, dst_pts_for_transform,
                        method_filter="RANSAC", threshold=3.0, max_iterations=1000,
                        num_random_points_affine=3
                    )
                    if A_custom_ransac is not None:
                        print("Affine Matrix (Custom RANSAC):\n", A_custom_ransac)
                        img1_warped_custom_a = cv2.warpAffine(image1_bgr, A_custom_ransac,
                                                              (image1_bgr.shape[1], image1_bgr.shape[0]))
                        # cv2.imshow("Image1 Warped (Custom RANSAC A)", img1_warped_custom_a)
                    else:
                        print("Custom RANSAC Affine failed.")

                    # --- Affine using OpenCV RANSAC ---
                    # cv2.estimateAffine2D needs full affine, estimateAffinePartial2D for similarity+translation
                    A_cv_ransac, mask_cv_a = cv2.estimateAffinePartial2D(src_np, dst_np, method=cv2.RANSAC,
                                                                         ransacReprojThreshold=3.0)
                    if A_cv_ransac is not None:
                        print("Affine Matrix (OpenCV RANSAC - Partial):\n", A_cv_ransac)
                        img1_warped_cv_a = cv2.warpAffine(image1_bgr, A_cv_ransac,
                                                          (image1_bgr.shape[1], image1_bgr.shape[0]))
                        # cv2.imshow("Image1 Warped (CV RANSAC A)", img1_warped_cv_a)
                    else:
                        print("OpenCV RANSAC Affine (Partial) failed.")
                else:
                    print("Not enough good matches for Affine RANSAC.")
            else:
                print("Not enough good matches to calculate transformations.")
            # cv2.waitKey(0)
    except Exception as e:
        import traceback
        print(f"Error during feature matching/transformation tests: {e}")
        traceback.print_exc()

    # 3. Test UniqueCombinationGenerator
    print("\n--- Testing UniqueCombinationGenerator ---")
    try:
        gen = UniqueCombinationGeneratorPy(max_value_exclusive=10, combination_size=3)
        print(f"Max combinations for (10, 3): {gen.count_max_combination()}")
        for _ in range(5):  # Generate a few
            combo = gen.generate()
            if combo:
                print(f"Generated combo: {combo}")
            else:
                print("No more unique combinations.")
                break
    except Exception as e:
        print(f"Error in UniqueCombinationGenerator test: {e}")

    cv2.destroyAllWindows()
    print("\n--- Example Usage Finished ---")


if __name__ == "__main__":
    main()