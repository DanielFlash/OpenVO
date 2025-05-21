import math
from dataclasses import dataclass, field

R_EARTH = 6378137.0  # from TypeOVO.h (dm_sm_a seems to be the WGS84 semi-major axis)
DEGREE_TO_RAD = 0.01745329252

@dataclass
class Pos_d3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

@dataclass
class Pos_f3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

@dataclass
class Pos_i3:
    x: int = 0
    y: int = 0
    z: int = 0

@dataclass
class Pos_d2:
    x: float = 0.0
    y: float = 0.0

    def set(self, x1: float, y1: float):
        self.x = x1
        self.y = y1

@dataclass
class Pos_f2:
    x: float = 0.0
    y: float = 0.0

    def set(self, x1: float, y1: float):
        self.x = x1
        self.y = y1

@dataclass
class Pos_i2:
    x: int = 0
    y: int = 0

    def set(self, x1: int, y1: int):
        self.x = x1
        self.y = y1

@dataclass
class CameraParams: # Renamed to follow Python conventions
    fov: float = 0.0
    fovx: float = 0.0
    fovy: float = 0.0
    type: int = 0
    resolution: Pos_i2 = field(default_factory=Pos_i2)

    def set(self, fov1: float = 0.0, fovx1: float = 0.0, fovy1: float = 0.0,
              type1: int = 0, res: Pos_i2 = None):
        self.fov = fov1
        self.fovx = fovx1
        self.fovy = fovy1
        self.type = type1
        self.resolution = res if res is not None else Pos_i2()

@dataclass
class PosAngle: # Renamed
    pitch: float = 0.0
    roll: float = 0.0
    yaw: float = 0.0

@dataclass
class PosWGS84: # Renamed
    latitude: float = 0.0
    longitude: float = 0.0
    force_zone_number: int = 0 # Assuming default for these if not specified
    force_zone_letter: int = 0 # Assuming it's an int, C++ char can be int

@dataclass
class AffineParams: # Renamed
    tx: int = 0
    ty: int = 0
    scale: float = 0.0
    angle: float = 0.0

# Constants (can be moved to ovo_constants.py)
OVO_ANGLES_FROM_SOURCE = 0
OVO_ANGLES_FROM_VIDEO = 1
OVO_RANSAC = 8  # cv2.RANSAC
OVO_LMEDS = 4 # cv2.LMEDS