import math

DM_PI = 3.14159265358979
DM_SM_A = 6378137.0  # WGS84 semi-major axis
DM_SM_B = 6356752.314 # WGS84 semi-minor axis
DM_SM_EccSquared = 6.69437999013e-03
UTMScaleFactor = 0.9996
R_EARTH = 6378137.0 # Also defined as DM_SM_A
DEGREE_TO_RAD = 0.01745329252

OVO_ACCURACY_INT = 0
OVO_ACCURACY_FLOAT = 1
OVO_ACCURACY_DOUBLE = 2

OVO_ANGLES_FROM_SOURCE = 0
OVO_ANGLES_FROM_VIDEO = 1 # Default for VideoProcessor source_flag

OVO_RANSAC = 8  # Corresponds to cv2.RANSAC for estimateAffinePartial2D
OVO_LMEDS = 4   # Corresponds to cv2.LMEDS