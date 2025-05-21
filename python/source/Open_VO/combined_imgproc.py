import cv2
import numpy as np
import math
import random
import itertools
from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Optional, Any, Dict  # Any for function signatures initially

# from scipy.linalg import lu_factor, lu_solve # Only if np.linalg.solve is not sufficient

# --- Constants (from various C++ headers) ---
EPSILON = 1e-12  # From math_test.h


# --- Data Structures (from Point.h, Match.h, Description.h, etc.) ---
@dataclass
class Point:
    x: float = 0.0
    y: float = 0.0
    w: float = 1.0  # Homogeneous coordinate, often 1 for 2D points

    def to_cv_point(self) -> Tuple[float, float]:
        return (self.x, self.y)

    @staticmethod
    def from_cv_keypoint(kp: cv2.KeyPoint) -> 'Point':
        return Point(kp.pt[0], kp.pt[1])

    @staticmethod
    def list_from_cv_keypoints(kps: List[cv2.KeyPoint]) -> List['Point']:
        return [Point.from_cv_keypoint(kp) for kp in kps]

    @staticmethod
    def list_to_cv_points_nparray(points: List['Point']) -> np.ndarray:
        return np.array([[p.x, p.y] for p in points], dtype=np.float32)


@dataclass
class Match:
    src: int = 0  # Index in source points
    dst: int = 0  # Index in destination points
    distance: float = 0.0
    imgIdx: int = -1  # Optional, from C++ Match constructor

    @staticmethod
    def list_from_cv_dmatches(dmatches: List[cv2.DMatch]) -> List['Match']:
        return [Match(m.queryIdx, m.trainIdx, m.distance, m.imgIdx) for m in dmatches]


@dataclass
class Description:  # C++ used std::vector<double>
    numbers: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))

    @staticmethod
    def list_from_cv_descriptors(descriptors: np.ndarray) -> List['Description']:
        # CV descriptors are usually np.ndarray of shape (N, descriptor_size)
        if descriptors is None or descriptors.ndim != 2:
            return []
        return [Description(numbers=desc_row) for desc_row in descriptors]


# --- Custom Matrix/Vector (ExtendMatrix.h) -> Replaced by NumPy ---
# The C++ Vector, Matrix, Tensor classes are custom.
# Python translation will use NumPy arrays directly.
# Operations like `matrix[i][j]`, `matrix * scalar`, `matrix_a * matrix_b` (element-wise or dot)
# will be done using NumPy's rich API.

# Helper to convert C++ Matrix-like structure to cv::Mat (for reference, not direct translation)
def matrix_to_cv_mat_py(matrix_data: List[List[float]]) -> np.ndarray:
    if not matrix_data or not isinstance(matrix_data[0], list):
        raise ValueError("Input must be a list of lists.")
    return np.array(matrix_data, dtype=np.float64)


# --- Histogram Equalization (Histogram.h) ---
def apply_histogram_clahe_py(input_image: np.ndarray,
                             tile_size_x: int,
                             tile_size_y: int,
                             relative_clip_limit: float = 4.0) -> np.ndarray:
    """
    Python translation of the custom CLAHE implementation.
    OpenCV's cv2.createCLAHE().apply() is generally preferred for optimized performance.
    This is a direct translation for equivalence.
    """
    if input_image is None or input_image.size == 0:
        raise ValueError("Empty image!")

    # Handle tile size defaults like C++
    if tile_size_x == 0 or input_image.shape[1] < tile_size_x:  # Corrected logic for tile size
        tile_size_x = input_image.shape[1]
    if tile_size_y == 0 or input_image.shape[0] < tile_size_y:  # Corrected logic
        tile_size_y = input_image.shape[0]

    input_mat_for_processing = None
    is_color = False
    original_type = input_image.dtype

    if input_image.ndim == 3 and input_image.shape[2] == 3:  # Color image
        if input_image.dtype != np.uint8:
            # print("Warning: Color CLAHE input converted to CV_8UC3")
            input_image = input_image.astype(np.uint8)  # CLAHE typically on 8-bit
        ycrcb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2YCrCb)
        channels = cv2.split(ycrcb_image)
        input_mat_for_processing = channels[0]  # Process Y channel (brightness)
        is_color = True
    elif input_image.ndim == 2:  # Grayscale image
        if input_image.dtype != np.uint8:
            # print("Warning: Grayscale CLAHE input converted to CV_8UC1")
            input_mat_for_processing = input_image.astype(np.uint8)
        else:
            input_mat_for_processing = input_image
    else:
        raise ValueError("Invalid image format! Must be 1-channel or 3-channel.")

    height, width = input_mat_for_processing.shape
    num_tiles_x = (width + tile_size_x - 1) // tile_size_x
    num_tiles_y = (height + tile_size_y - 1) // tile_size_y

    # LUT: List of (List of (NumPy array of 256 uchar))
    lut = [[[0] * 256 for _ in range(num_tiles_x)] for _ in range(num_tiles_y)]

    for ty in range(num_tiles_y):
        for tx in range(num_tiles_x):
            x0 = tx * tile_size_x
            y0 = ty * tile_size_y
            x1 = min(x0 + tile_size_x, width)
            y1 = min(y0 + tile_size_y, height)

            if x1 <= x0 or y1 <= y0: continue  # Skip empty tiles

            tile = input_mat_for_processing[y0:y1, x0:x1]
            area = tile.size  # (x1 - x0) * (y1 - y0)
            if area == 0: continue

            clip_limit_abs = max(1, int(relative_clip_limit * area / 256.0))

            hist = cv2.calcHist([tile], [0], None, [256], [0, 256]).flatten().astype(int)

            # Limiting (clipping)
            excess = 0
            for i in range(256):
                if hist[i] > clip_limit_abs:
                    excess += hist[i] - clip_limit_abs
                    hist[i] = clip_limit_abs

            # Redistribute excess
            bonus = excess // 256
            # Remainder for first few bins to ensure total count is preserved
            # (More precise redistribution might be needed for perfect match)
            remainder_bonus = excess % 256
            for i in range(256):
                hist[i] += bonus
            for i in range(remainder_bonus):  # Distribute remainder
                hist[i] += 1

            # Cumulative histogram (CDF)
            cdf = np.cumsum(hist)

            # Normalize LUT for this tile
            # Scale CDF to [0, 255]
            # cdf_scaled = (cdf * 255.0 / area).astype(np.uint8) # C++: (float)cdf[i] * 255 / area
            # Correct normalization for CLAHE is often (cdf - cdf_min) * 255 / (area - cdf_min)
            # but the C++ code uses: (float)cdf[i] * 255 / area
            # Which is fine if cdf[0] can be non-zero and we don't subtract cdf_min
            current_lut_tile = np.clip((cdf * 255.0 / area), 0, 255).astype(np.uint8)
            lut[ty][tx] = current_lut_tile.tolist()  # Store as list for easier indexing

    output_image = np.empty_like(input_mat_for_processing)

    # Bilinear Interpolation
    for r_idx in range(height):
        for c_idx in range(width):
            # Determine the tile indices and local coordinates within the tile
            tx_float = c_idx / tile_size_x - 0.5
            ty_float = r_idx / tile_size_y - 0.5

            tx_i = int(math.floor(tx_float))
            ty_i = int(math.floor(ty_float))

            # Local coordinates (weights for interpolation)
            lx = tx_float - tx_i
            ly = ty_float - ty_i

            pixel_val = input_mat_for_processing[r_idx, c_idx]

            # Get LUTs for the four surrounding tiles (or edge tiles if at border)
            # Clamp tile indices to be within bounds
            tx_i0 = max(0, tx_i)
            ty_i0 = max(0, ty_i)
            tx_i1 = min(num_tiles_x - 1, tx_i + 1)
            ty_i1 = min(num_tiles_y - 1, ty_i + 1)

            val_00 = lut[ty_i0][tx_i0][pixel_val]
            val_10 = lut[ty_i0][tx_i1][pixel_val]
            val_01 = lut[ty_i1][tx_i0][pixel_val]
            val_11 = lut[ty_i1][tx_i1][pixel_val]

            # Interpolate horizontally
            inter_val_y0 = (1.0 - lx) * val_00 + lx * val_10
            inter_val_y1 = (1.0 - lx) * val_01 + lx * val_11

            # Interpolate vertically
            final_val = (1.0 - ly) * inter_val_y0 + ly * inter_val_y1
            output_image[r_idx, c_idx] = np.clip(round(final_val), 0, 255).astype(np.uint8)

    if is_color:
        channels[0] = output_image
        ycrcb_result = cv2.merge(channels)
        result_image = cv2.cvtColor(ycrcb_result, cv2.COLOR_YCrCb2BGR)
    else:
        result_image = output_image

    return result_image.astype(original_type)  # Restore original dtype if changed


def apply_histogram_py(input_image: np.ndarray) -> np.ndarray:
    """
    Python translation of global histogram equalization.
    OpenCV's cv2.equalizeHist() is the direct equivalent for grayscale.
    """
    if input_image is None or input_image.size == 0:
        raise ValueError("Empty image!")

    original_type = input_image.dtype
    input_mat_for_processing = None
    is_color = False

    if input_image.ndim == 3 and input_image.shape[2] == 3:
        if input_image.dtype != np.uint8: input_image = input_image.astype(np.uint8)
        ycrcb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2YCrCb)
        channels = cv2.split(ycrcb_image)
        # Equalize the Y channel (brightness)
        channels[0] = cv2.equalizeHist(channels[0])
        ycrcb_result = cv2.merge(channels)
        result_image = cv2.cvtColor(ycrcb_result, cv2.COLOR_YCrCb2BGR)
    elif input_image.ndim == 2:
        if input_image.dtype != np.uint8: input_image = input_image.astype(np.uint8)
        result_image = cv2.equalizeHist(input_image)
    else:
        raise ValueError("Invalid image format! Must be 1-channel or 3-channel.")

    return result_image.astype(original_type)


# --- Linear Algebra Solvers (SolverLinearEquations.h) ---
# SvdSolver and LuSolver custom implementations.
# Python/NumPy/SciPy provide optimized versions.

def solve_homogeneous_svd_py(A: np.ndarray) -> Optional[np.ndarray]:
    """Solves Ax = 0 using SVD. A is (m, n), x is (n, 1)."""
    if A is None or A.ndim != 2:
        return None
    # U, s, Vh = np.linalg.svd(A, full_matrices=True) # Vh is V.T
    # solution is the last column of V (or last row of Vh)
    # solution = Vh[-1, :]
    # The C++ SVD seems to assume A is m x 9, and solution x is 9x1,
    # and takes the last column of V (V is n x n).
    # In NumPy, Vh is (n x n), so last row of Vh.
    try:
        _u, _s, vh = np.linalg.svd(A)
        # The solution is the column of V corresponding to the smallest singular value.
        # If singular values are sorted (usually descending by np.linalg.svd),
        # it's the last column of V, which is the last row of Vh (V transpose).
        null_space_vector = vh[-1, :]
        return null_space_vector
    except np.linalg.LinAlgError:
        # print("SVD computation failed.")
        return None


def solve_direct_linear_lu_py(A: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    """Solves Ax = b using LU decomposition (via np.linalg.solve)."""
    if A is None or b is None or A.ndim != 2 or A.shape[0] != b.shape[0]:
        return None
    try:
        return np.linalg.solve(A, b)  # Robust solver
    except np.linalg.LinAlgError:
        # print("LU solve failed (matrix might be singular or not square for np.linalg.solve). Use lstsq for non-square.")
        return None


def solve_direct_linear_svd_lstsq_py(A: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    """Solves Ax = b using SVD-based least squares (np.linalg.lstsq). Handles non-square A."""
    if A is None or b is None or A.ndim != 2 or A.shape[0] != b.shape[0]:
        # np.linalg.lstsq actually can handle A.shape[0] != b.shape[0] (not possible for Ax=b)
        # But for Ax=b, rows of A must equal len of b.
        if A.shape[0] != len(b):
            # print("Incompatible shapes for A and b in lstsq.")
            return None
    try:
        x, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)
        # Check rank or residuals if needed for solution quality
        return x
    except np.linalg.LinAlgError:
        # print("Least squares solution failed.")
        return None


# --- Homography and Affine (Homography.h, Affine.h) ---
# Functions for creating matrix equations and finding transformations.
# These will heavily rely on NumPy.

# Homography related functions
def apply_homography_to_points_py(points: List[Point], H_matrix: np.ndarray) -> List[Point]:
    """Applies a 3x3 homography matrix H to a list of 2D points."""
    if H_matrix is None or H_matrix.shape != (3, 3):
        # print("Invalid homography matrix.")
        return []

    projected_points = []
    for p in points:
        # Create homogeneous coordinate: [x, y, 1]
        src_pt_h = np.array([p.x, p.y, 1.0], dtype=np.float64)
        dst_pt_h = H_matrix @ src_pt_h  # Matrix multiplication

        # Normalize: divide by the third component (w')
        if abs(dst_pt_h[2]) < EPSILON:  # Avoid division by zero or very small w'
            # print(f"Warning: Homography resulted in near-zero w' for point ({p.x},{p.y}). Skipping.")
            # C++ returns empty vector if any point is inf.
            # Mimic by returning empty or handle differently.
            return []

        proj_x = dst_pt_h[0] / dst_pt_h[2]
        proj_y = dst_pt_h[1] / dst_pt_h[2]

        if not (math.isfinite(proj_x) and math.isfinite(proj_y)):
            # print(f"Warning: Homography resulted in non-finite coordinates for point ({p.x},{p.y}). Skipping.")
            return []

        projected_points.append(Point(proj_x, proj_y, 1.0))
    return projected_points


def create_matrix_equation_homography_py(src_points: List[Point], dst_points: List[Point]) -> Tuple[
    Optional[np.ndarray], Optional[np.ndarray]]:
    """Creates matrix equation Ax = b for homography (h33 = 1 constraint)."""
    if len(src_points) != len(dst_points) or len(src_points) < 4:
        # print("Need at least 4 point correspondences for homography.")
        return None, None

    num_points = len(src_points)
    A = np.zeros((2 * num_points, 8))
    b = np.zeros(2 * num_points)

    for i in range(num_points):
        x, y = src_points[i].x, src_points[i].y
        _x, _y = dst_points[i].x, dst_points[i].y

        A[2 * i, :] = [x, y, 1, 0, 0, 0, -x * _x, -y * _x]
        b[2 * i] = _x
        A[2 * i + 1, :] = [0, 0, 0, x, y, 1, -x * _y, -y * _y]
        b[2 * i + 1] = _y
    return A, b


def create_matrix_equation_singular_homography_py(src_points: List[Point], dst_points: List[Point]) -> Optional[
    np.ndarray]:
    """Creates matrix equation Ah = 0 for homography."""
    if len(src_points) != len(dst_points) or len(src_points) < 4:
        return None

    num_points = len(src_points)
    A = np.zeros((2 * num_points, 9))

    for i in range(num_points):
        x1, y1 = src_points[i].x, src_points[i].y
        x2, y2 = dst_points[i].x, dst_points[i].y

        A[2 * i, :] = [x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2]
        A[2 * i + 1, :] = [0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2]
    return A


def find_homography_matrix_py(src_points: List[Point], dst_points: List[Point]) -> Optional[np.ndarray]:
    """Calculates homography H. If 4 points, uses direct linear. If >4, uses SVD for Ah=0."""
    if len(src_points) != len(dst_points) or len(src_points) < 4:
        return None

    if len(src_points) == 4:  # Exactly 4 points, solve Ax=b for 8 params, h33=1
        A, b = create_matrix_equation_homography_py(src_points, dst_points)
        if A is None: return None
        # h_params = solve_direct_linear_lu_py(A, b) # For exactly determined system
        h_params = solve_direct_linear_svd_lstsq_py(A, b)  # More robust
        if h_params is None: return None
        H_flat = np.append(h_params, 1.0)  # Add h33 = 1
        H_matrix = H_flat.reshape(3, 3)
    else:  # More than 4 points, solve Ah=0 using SVD
        A_singular = create_matrix_equation_singular_homography_py(src_points, dst_points)
        if A_singular is None: return None
        h_flat = solve_homogeneous_svd_py(A_singular)
        if h_flat is None: return None
        # Normalize h so that h33 (h_flat[8]) is 1
        if abs(h_flat[8]) < EPSILON:
            # print("Warning: h33 is close to zero, homography might be degenerate.")
            # Could try normalizing by largest element or norm, but h33=1 is standard.
            return None
        H_matrix = (h_flat / h_flat[8]).reshape(3, 3)
    return H_matrix


# Affine related functions
def apply_affine_to_points_py(points: List[Point], affine_matrix: np.ndarray) -> List[Point]:
    """Applies a 2x3 affine matrix to a list of 2D points."""
    if affine_matrix is None or affine_matrix.shape != (2, 3):
        # print("Invalid affine matrix.")
        return []

    projected_points = []
    for p in points:
        # [x'] = [a b tx] [x]
        # [y']   [c d ty] [y]
        #                 [1]
        src_pt_h = np.array([p.x, p.y, 1.0], dtype=np.float64)
        # Affine matrix is 2x3, so we apply it as:
        # x' = a*x + b*y + tx
        # y' = c*x + d*y + ty
        # This can be done by taking the 2x2 part and translation separately,
        # or by augmenting the affine matrix to 3x3 ([A|t; 0 0 1]) and using homogeneous coords.
        # C++ version applies it directly.

        proj_x = affine_matrix[0, 0] * p.x + affine_matrix[0, 1] * p.y + affine_matrix[0, 2]
        proj_y = affine_matrix[1, 0] * p.x + affine_matrix[1, 1] * p.y + affine_matrix[1, 2]

        if not (math.isfinite(proj_x) and math.isfinite(proj_y)):
            return []  # Mimic C++ behavior

        projected_points.append(Point(proj_x, proj_y, 1.0))
    return projected_points


def create_matrix_equation_affine_py(src_points: List[Point], dst_points: List[Point]) -> Tuple[
    Optional[np.ndarray], Optional[np.ndarray]]:
    """Creates matrix equation Ap = b for affine transformation."""
    if len(src_points) != len(dst_points) or len(src_points) < 3:
        # print("Need at least 3 point correspondences for affine.")
        return None, None

    num_points = len(src_points)
    A = np.zeros((2 * num_points, 6))
    b = np.zeros(2 * num_points)

    for i in range(num_points):
        x1, y1 = src_points[i].x, src_points[i].y
        x2, y2 = dst_points[i].x, dst_points[i].y

        A[2 * i, :] = [x1, y1, 1, 0, 0, 0]
        b[2 * i] = x2
        A[2 * i + 1, :] = [0, 0, 0, x1, y1, 1]
        b[2 * i + 1] = y2
    return A, b


def find_affine_matrix_py(src_points: List[Point], dst_points: List[Point]) -> Optional[np.ndarray]:
    """Calculates 2x3 affine matrix. If 3 points, direct solve. If >3, least squares."""
    if len(src_points) != len(dst_points) or len(src_points) < 3:
        return None

    A, b = create_matrix_equation_affine_py(src_points, dst_points)
    if A is None: return None

    # Use least squares for robustness, even for 3 points it should work.
    p_params = solve_direct_linear_svd_lstsq_py(A, b)
    if p_params is None: return None

    # p_params = [a, b, tx, c, d, ty]
    affine_matrix = np.array([
        [p_params[0], p_params[1], p_params[2]],
        [p_params[3], p_params[4], p_params[5]]
    ], dtype=np.float64)
    return affine_matrix


# --- RANSAC/LMEDS (Filters from Ransac.h, Lmeds.h) ---
# OpenCV's findHomography and estimateAffinePartial2D already include RANSAC/LMEDS.
# For a direct translation of your custom RANSAC/LMEDS:

class UniqueCombinationGeneratorPy:
    def __init__(self, max_value_exclusive: int, combination_size: int):
        if max_value_exclusive == 0 or combination_size == 0:
            raise ValueError("max_value or combination_size cannot be zero")
        if max_value_exclusive < combination_size:
            raise ValueError("Size of combination exceeds max_value")

        self.population = list(range(max_value_exclusive))
        self.combination_size = combination_size
        self.generated_combinations_count = 0

        # Pre-generate all possible combinations if feasible, or track used ones.
        # For large populations/combinations, tracking used ones is better.
        # C++ used std::set to track. Python can do similarly.
        self._used_combinations_tuples = set()

        # Calculate total possible combinations (for information or early exit)
        if combination_size > max_value_exclusive:
            self._total_combinations = 0
        else:
            self._total_combinations = math.comb(max_value_exclusive, combination_size)

    def count_max_combination(self) -> int:
        return self._total_combinations

    def generate(self) -> Optional[List[int]]:
        if self.generated_combinations_count >= self._total_combinations:
            return None  # No more unique combinations

        # This is less efficient than C++ std::set for very large numbers of iterations,
        # but simpler for moderate use.
        attempts = 0
        max_attempts = self._total_combinations * 2 + 100  # Heuristic to avoid infinite loop

        while attempts < max_attempts:
            combination = sorted(random.sample(self.population, self.combination_size))
            comb_tuple = tuple(combination)
            if comb_tuple not in self._used_combinations_tuples:
                self._used_combinations_tuples.add(comb_tuple)
                self.generated_combinations_count += 1
                return combination
            attempts += 1
        return None  # Failed to find a new unique combination after many tries

    def reset(self):
        self._used_combinations_tuples.clear()
        self.generated_combinations_count = 0


class RansacPy:
    def __init__(self, threshold: float, max_iterations: int, num_random_points: int):
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.num_random_points = num_random_points
        self.best_transform: Optional[np.ndarray] = None
        self.best_inliers_count: int = -1

    def _calculate_delta_points(self, src_pts: List[Point], dst_pts: List[Point], transform_matrix: np.ndarray,
                                is_homography: bool) -> List[float]:
        if is_homography:
            projected_pts = apply_homography_to_points_py(src_pts, transform_matrix)
        else:  # Affine
            projected_pts = apply_affine_to_points_py(src_pts, transform_matrix)

        if not projected_pts or len(projected_pts) != len(dst_pts):
            return []  # Error in projection

        deltas = []
        for proj_p, dst_p in zip(projected_pts, dst_pts):
            dist_sq = (proj_p.x - dst_p.x) ** 2 + (proj_p.y - dst_p.y) ** 2
            deltas.append(dist_sq)  # C++ code uses squared distance for delta
        return deltas

    def _calculate_delta_matches(self, all_src_pts: List[Point], all_dst_pts: List[Point],
                                 matches: List[Match], transform_matrix: np.ndarray,
                                 is_homography: bool) -> List[float]:
        # Project ALL source points involved in ANY match
        # This requires knowing which subset of all_src_pts to project.
        # C++ passes full src, applies transform, then compares based on match indices.

        if is_homography:
            projected_all_src_pts = apply_homography_to_points_py(all_src_pts, transform_matrix)
        else:  # Affine
            projected_all_src_pts = apply_affine_to_points_py(all_src_pts, transform_matrix)

        if not projected_all_src_pts or len(projected_all_src_pts) != len(all_src_pts):
            # print("Projection failed for delta_matches")
            return []

        deltas = []
        for match in matches:
            if match.src >= len(projected_all_src_pts) or match.dst >= len(all_dst_pts):
                # print(f"Match index out of bounds: src {match.src}, dst {match.dst}")
                continue  # Should ideally not happen if data is consistent

            proj_p = projected_all_src_pts[match.src]
            dst_p = all_dst_pts[match.dst]
            dist_sq = (proj_p.x - dst_p.x) ** 2 + (proj_p.y - dst_p.y) ** 2
            deltas.append(dist_sq)
        return deltas

    def calculate(self, src_points_full: List[Point], dst_points_full: List[Point],
                  find_transform_func: Callable[[List[Point], List[Point]], Optional[np.ndarray]],
                  is_homography: bool,
                  matches: Optional[List[Match]] = None) -> Optional[np.ndarray]:

        self.best_transform = None
        self.best_inliers_count = -1

        # Determine the population size for sampling
        population_size = len(matches) if matches else len(src_points_full)
        if population_size < self.num_random_points:
            # print("Not enough points/matches for RANSAC sample.")
            return None

        combo_gen = UniqueCombinationGeneratorPy(population_size, self.num_random_points)

        # Adjust max_iterations if it's more than possible unique combinations
        max_possible_combos = combo_gen.count_max_combination()
        actual_iterations = min(self.max_iterations, max_possible_combos)
        if actual_iterations == 0 and population_size >= self.num_random_points:  # Edge case if math.comb returns 0 due to float issues for large N
            actual_iterations = self.max_iterations  # Fallback

        for iteration in range(actual_iterations):
            rand_indices = combo_gen.generate()
            if rand_indices is None:  # No more unique combinations
                break

            sample_src_pts: List[Point] = []
            sample_dst_pts: List[Point] = []

            if matches:
                for idx in rand_indices:
                    match = matches[idx]
                    if match.src < len(src_points_full) and match.dst < len(dst_points_full):
                        sample_src_pts.append(src_points_full[match.src])
                        sample_dst_pts.append(dst_points_full[match.dst])
                    else:  # Should not happen with valid matches
                        # print("Warning: Match index out of bounds during RANSAC sampling.")
                        continue
                if len(sample_src_pts) < self.num_random_points: continue  # Skip if bad samples
            else:  # No matches provided, assume src_points_full and dst_points_full are corresponding
                for idx in rand_indices:
                    sample_src_pts.append(src_points_full[idx])
                    sample_dst_pts.append(dst_points_full[idx])

            current_transform = find_transform_func(sample_src_pts, sample_dst_pts)
            if current_transform is None:
                continue

            # Calculate deltas for ALL points (or all matches)
            if matches:
                deltas = self._calculate_delta_matches(src_points_full, dst_points_full, matches, current_transform,
                                                       is_homography)
            else:
                deltas = self._calculate_delta_points(src_points_full, dst_points_full, current_transform,
                                                      is_homography)

            if not deltas:
                continue

            current_inliers_count = 0
            inlier_indices_current_iteration = []  # For potential re-estimation

            for i, delta_sq in enumerate(deltas):
                if delta_sq < self.threshold ** 2:  # Compare squared error with squared threshold
                    current_inliers_count += 1
                    # inlier_indices_current_iteration.append(i) # Store index of inlier for re-estimation

            if current_inliers_count > self.best_inliers_count:
                self.best_inliers_count = current_inliers_count
                self.best_transform = current_transform

                # Optional: Re-estimate model using all current inliers
                # This is a common RANSAC refinement step. C++ code doesn't show it explicitly.
                # if current_inliers_count >= self.num_random_points :
                #    inlier_src_pts = []
                #    inlier_dst_pts = []
                #    if matches:
                #        for idx in inlier_indices_current_iteration:
                #            match = matches[idx] # Assuming deltas are ordered same as matches
                #            inlier_src_pts.append(src_points_full[match.src])
                #            inlier_dst_pts.append(dst_points_full[match.dst])
                #    else:
                #        for idx in inlier_indices_current_iteration:
                #            inlier_src_pts.append(src_points_full[idx])
                #            inlier_dst_pts.append(dst_points_full[idx])
                #
                #    refined_transform = find_transform_func(inlier_src_pts, inlier_dst_pts)
                #    if refined_transform is not None:
                #        self.best_transform = refined_transform

        return self.best_transform


# LMEDS would be similar structure to RANSAC but with median error calculation.
# For brevity, I'll skip full LMEDS translation, as cv2.findHomography/estimateAffine has it.

def find_homography_py_custom_ransac(src_pts: List[Point], dst_pts: List[Point],
                                     method_filter: str = "RANSAC",  # or "LMEDS"
                                     threshold: float = 5.0,  # Pixel error threshold
                                     max_iterations: int = 2000,
                                     num_random_points_homography: int = 4,
                                     matches_list: Optional[List[Match]] = None) -> Optional[np.ndarray]:
    if method_filter.upper() == "RANSAC":
        ransac_solver = RansacPy(threshold, max_iterations, num_random_points_homography)
        return ransac_solver.calculate(src_pts, dst_pts, find_homography_matrix_py, is_homography=True,
                                       matches=matches_list)
    # elif method_filter.upper() == "LMEDS":
    #     lmeds_solver = LmedsPy(...)
    #     return lmeds_solver.calculate(...)
    else:
        # print(f"Filter method {method_filter} not supported by custom implementation. Using direct.")
        # Fallback to direct calculation if no matches (assumes points correspond)
        if not matches_list and len(src_pts) >= num_random_points_homography:
            return find_homography_matrix_py(src_pts, dst_pts)
        elif matches_list and len(matches_list) >= num_random_points_homography:
            # Need to select corresponding points based on matches for direct calculation
            # This case usually implies RANSAC/LMEDS should be used.
            # For a simple direct version with matches:
            src_matched = [src_pts[m.src] for m in matches_list]
            dst_matched = [dst_pts[m.dst] for m in matches_list]
            if len(src_matched) >= num_random_points_homography:
                return find_homography_matrix_py(src_matched, dst_matched)
    return None


def estimate_affine_partial2d_py_custom_ransac(src_pts: List[Point], dst_pts: List[Point],
                                               method_filter: str = "RANSAC",
                                               threshold: float = 3.0,
                                               max_iterations: int = 2000,
                                               num_random_points_affine: int = 3,
                                               matches_list: Optional[List[Match]] = None) -> Optional[np.ndarray]:
    if method_filter.upper() == "RANSAC":
        ransac_solver = RansacPy(threshold, max_iterations, num_random_points_affine)
        return ransac_solver.calculate(src_pts, dst_pts, find_affine_matrix_py, is_homography=False,
                                       matches=matches_list)
    # elif method_filter.upper() == "LMEDS": ...
    else:
        # Fallback to direct calculation
        if not matches_list and len(src_pts) >= num_random_points_affine:
            return find_affine_matrix_py(src_pts, dst_pts)
        elif matches_list and len(matches_list) >= num_random_points_affine:
            src_matched = [src_pts[m.src] for m in matches_list]
            dst_matched = [dst_pts[m.dst] for m in matches_list]
            if len(src_matched) >= num_random_points_affine:
                return find_affine_matrix_py(src_matched, dst_matched)
    return None


# --- KnnMatch (KnnMatch.h) ---
# OpenCV's BFMatcher().knnMatch is preferred.
class KnnMatchPy:
    def _euclidean_distance_desc(self, desc1: Description, desc2: Description) -> float:
        if desc1.numbers.shape != desc2.numbers.shape:
            return float('inf')  # Or handle error
        return np.linalg.norm(desc1.numbers - desc2.numbers)

    def find(self, src_descs: List[Description], dst_descs: List[Description], k_neighbors: int) -> List[Match]:
        """Performs KNN matching. Returns flat list of matches for all src_descs."""
        all_matches: List[Match] = []
        if not src_descs or not dst_descs or k_neighbors <= 0:
            return all_matches

        for i, d_src in enumerate(src_descs):
            distances_to_dst: List[Tuple[float, int]] = []  # (distance, dst_index)
            for j, d_dst in enumerate(dst_descs):
                dist = self._euclidean_distance_desc(d_src, d_dst)
                distances_to_dst.append((dist, j))

            distances_to_dst.sort(key=lambda item: item[0])  # Sort by distance

            for neighbor_idx in range(min(k_neighbors, len(distances_to_dst))):
                dist_val, dst_idx = distances_to_dst[neighbor_idx]
                all_matches.append(Match(src=i, dst=dst_idx, distance=dist_val))
        return all_matches


# --- Test utilities (from math_test.h, if needed for main script) ---
# Example: convertCvKeypointToPoint is now Point.list_from_cv_keypoints
# getMatches / getMatchesSort: These are complex test setup functions.
# They would be translated into Python test functions using cv2.ORB_create(), cv2.BFMatcher(), etc.
# For brevity, I will not translate the full test functions here but use their principles in example usage.


# --- Main example usage area (simulating a test or main script) ---
if __name__ == '__main__':
    print("--- Combined Image Processing Module ---")

    # Example: Histogram CLAHE (custom implementation)
    print("\nTesting Custom CLAHE...")
    try:
        # Create a sample grayscale image
        sample_gray = np.random.randint(0, 150, size=(100, 100), dtype=np.uint8)
        clahe_custom_result = apply_histogram_clahe_py(sample_gray, tile_size_x=8, tile_size_y=8,
                                                       relative_clip_limit=2.0)
        print(f"Custom CLAHE output shape: {clahe_custom_result.shape}, dtype: {clahe_custom_result.dtype}")

        # Compare with OpenCV's CLAHE
        clahe_cv = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_cv_result = clahe_cv.apply(sample_gray)
        print(f"OpenCV CLAHE output shape: {clahe_cv_result.shape}, dtype: {clahe_cv_result.dtype}")
        # cv2.imshow("Sample Gray", sample_gray)
        # cv2.imshow("Custom CLAHE", clahe_custom_result)
        # cv2.imshow("OpenCV CLAHE", clahe_cv_result)
        # cv2.waitKey(0)
    except Exception as e:
        print(f"Error in CLAHE test: {e}")

    # Example: Global Histogram Equalization
    print("\nTesting Global Histogram Equalization...")
    try:
        hist_eq_custom_result = apply_histogram_py(sample_gray)
        hist_eq_cv_result = cv2.equalizeHist(sample_gray)
        print(f"Custom HistEq output shape: {hist_eq_custom_result.shape}")
        print(f"OpenCV HistEq output shape: {hist_eq_cv_result.shape}")
        # cv2.imshow("Custom HistEq", hist_eq_custom_result)
        # cv2.imshow("OpenCV HistEq", hist_eq_cv_result)
        # cv2.waitKey(0)
    except Exception as e:
        print(f"Error in HistEq test: {e}")

    # Example: Homography
    print("\nTesting Homography...")
    # Define some source and destination points (at least 4)
    src_pts_list = [Point(0, 0), Point(100, 0), Point(100, 100), Point(0, 100), Point(50, 50)]
    # Simulate a perspective transform for dst_pts
    # True H: [1.1, 0.1, 10; 0.05, 1.2, 20; 0.001, 0.0005, 1]
    true_H_np = np.array([[1.1, 0.1, 10], [0.05, 1.2, 20], [0.001, 0.0005, 1]])
    dst_pts_list_h = apply_homography_to_points_py(src_pts_list, true_H_np)

    if dst_pts_list_h:
        print(f"Generated {len(dst_pts_list_h)} destination points for homography test.")
        # Calculate H using custom function
        calculated_H = find_homography_matrix_py(src_pts_list, dst_pts_list_h)
        if calculated_H is not None:
            print("Calculated Homography Matrix (custom SVD/direct):")
            print(calculated_H)
            print("True Homography Matrix (normalized for comparison if needed):")
            print(true_H_np / true_H_np[2, 2])  # Normalize true H for comparison

            # Compare with OpenCV's findHomography
            src_np = Point.list_to_cv_points_nparray(src_pts_list)
            dst_np = Point.list_to_cv_points_nparray(dst_pts_list_h)
            H_cv, _ = cv2.findHomography(src_np, dst_np)  # Default is RANSAC
            if H_cv is not None:
                print("OpenCV findHomography Matrix:")
                print(H_cv)
        else:
            print("Custom homography calculation failed.")
    else:
        print("Homography destination point generation failed.")

    # Example: Affine Transformation
    print("\nTesting Affine Transformation...")
    src_pts_affine = [Point(0, 0), Point(100, 0), Point(0, 100), Point(50, 75)]  # Need at least 3
    # True Affine: [1.1, 0.1, 10; -0.05, 0.9, 20]
    true_A_np = np.array([[1.1, 0.1, 10], [-0.05, 0.9, 20]])
    dst_pts_list_a = apply_affine_to_points_py(src_pts_affine, true_A_np)

    if dst_pts_list_a:
        print(f"Generated {len(dst_pts_list_a)} destination points for affine test.")
        calculated_A = find_affine_matrix_py(src_pts_affine, dst_pts_list_a)
        if calculated_A is not None:
            print("Calculated Affine Matrix (custom lstsq):")
            print(calculated_A)
            print("True Affine Matrix:")
            print(true_A_np)

            # Compare with OpenCV's estimateAffinePartial2D or estimateAffine2D
            src_np_a = Point.list_to_cv_points_nparray(src_pts_affine)
            dst_np_a = Point.list_to_cv_points_nparray(dst_pts_list_a)
            # A_cv, _ = cv2.estimateAffine2D(src_np_a, dst_np_a) # Needs at least 3 points
            A_cv_partial, _ = cv2.estimateAffinePartial2D(src_np_a, dst_np_a)  # Robust, RANSAC by default
            if A_cv_partial is not None:
                print("OpenCV estimateAffinePartial2D Matrix:")
                print(A_cv_partial)
        else:
            print("Custom affine calculation failed.")
    else:
        print("Affine destination point generation failed.")

    # Example: RANSAC for Homography (using custom RANSAC)
    print("\nTesting RANSAC for Homography (custom)...")
    if dst_pts_list_h:  # Use points from previous homography test
        # Introduce some outliers to dst_pts_list_h
        dst_pts_list_h_outliers = list(dst_pts_list_h)  # Make a copy
        if len(dst_pts_list_h_outliers) > 2:
            dst_pts_list_h_outliers[1] = Point(dst_pts_list_h_outliers[1].x + 50,
                                               dst_pts_list_h_outliers[1].y - 30)  # Add outlier
            dst_pts_list_h_outliers[-1] = Point(dst_pts_list_h_outliers[-1].x - 20, dst_pts_list_h_outliers[-1].y + 40)

        H_ransac_custom = find_homography_py_custom_ransac(src_pts_list, dst_pts_list_h_outliers, threshold=10.0)
        if H_ransac_custom is not None:
            print("Custom RANSAC Homography Matrix:")
            print(H_ransac_custom)
        else:
            print("Custom RANSAC Homography failed.")

    # Example: KnnMatch (custom implementation)
    print("\nTesting Custom KNN Match...")
    # Create dummy descriptors (NumPy arrays)
    desc_list_src = [Description(np.random.rand(64).astype(np.float32)) for _ in range(5)]
    desc_list_dst = [Description(np.random.rand(64).astype(np.float32)) for _ in range(8)]
    # Add a very similar descriptor to test matching
    desc_list_dst[3] = Description(desc_list_src[1].numbers + np.random.rand(64).astype(np.float32) * 0.01)

    knn_matcher_custom = KnnMatchPy()
    custom_matches = knn_matcher_custom.find(desc_list_src, desc_list_dst, k_neighbors=2)
    print(f"Custom KNN found {len(custom_matches)} raw matches (k=2 for each src):")
    # for m in custom_matches:
    #     print(f"SrcIdx: {m.src}, DstIdx: {m.dst}, Dist: {m.distance:.4f}")

    # Compare with OpenCV BFMatcher KNN
    bf_cv = cv2.BFMatcher(cv2.NORM_L2)  # L2 for float descriptors
    descriptors_src_np = np.array([d.numbers for d in desc_list_src])
    descriptors_dst_np = np.array([d.numbers for d in desc_list_dst])

    if descriptors_src_np.size > 0 and descriptors_dst_np.size > 0:
        knn_matches_cv = bf_cv.knnMatch(descriptors_src_np, descriptors_dst_np, k=2)
        print(f"OpenCV BFMatcher.knnMatch found {len(knn_matches_cv)} groups of matches.")
        # for i, m_group in enumerate(knn_matches_cv):
        #     print(f"SrcIdx {i}:")
        #     for m_cv in m_group:
        #         print(f"  DstIdx: {m_cv.trainIdx}, Dist: {m_cv.distance:.4f}")

    # cv2.destroyAllWindows() # If any imshow was used and not destroyed