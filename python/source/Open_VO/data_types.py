from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from .ovo_types import PosAngle, CameraParams, Pos_f2, Pos_d3, Pos_i2


@dataclass
class SurfaceImgData:
    imgName: str = ""
    imgW: int = 0
    imgH: int = 0
    imgTopLeftX: float = 0.0
    imgTopLeftY: float = 0.0
    imgBotRightX: float = 0.0
    imgBotRightY: float = 0.0


@dataclass
class SurfaceObjData:
    imgName: str = ""
    imgW: int = 0
    imgH: int = 0
    imgTopLeftX: float = 0.0
    imgTopLeftY: float = 0.0
    imgBotRightX: float = 0.0
    imgBotRightY: float = 0.0
    objLabel: int = 0
    bbX: int = 0
    bbY: int = 0
    bbW: int = 0
    bbH: int = 0


@dataclass
class SurfaceData:
    imgName: str = ""
    imgW: int = 0
    imgH: int = 0
    imgTopLeftX: float = 0.0
    imgTopLeftY: float = 0.0
    imgBotRightX: float = 0.0
    imgBotRightY: float = 0.0
    objId: int = 0
    objLabel: int = 0
    bbX: int = 0
    bbY: int = 0
    bbW: int = 0
    bbH: int = 0
    objCoordX: float = 0.0
    objCoordY: float = 0.0
    mappedTo: int = -1


@dataclass
class LocalData:
    overlapLevel: int = 0
    mappedTo: int = -1
    objId: int = -1
    objLabel: int = -1
    objCoordX: float = 0.0
    objCoordY: float = 0.0


@dataclass
class MapEdges:
    topLeftX: float = 0.0
    topLeftY: float = 0.0
    botRightX: float = 0.0
    botRightY: float = 0.0


@dataclass
class Detection:
    class_id: int = 0
    className: str = ""
    confidence: float = 0.0
    x: int = 0  # Bounding box top-left x
    y: int = 0  # Bounding box top-left y
    w: int = 0  # Bounding box width
    h: int = 0  # Bounding box height


@dataclass
class ObjectDist:
    # In Python, storing direct references like C++ pointers can be tricky
    # if the lists they come from are modified. It's often better to store
    # indices or ensure the source lists are stable.
    # For simplicity, we can store the objects themselves, assuming they are treated as immutable
    # for the duration of this struct's relevance, or use IDs/indices.
    localData: Optional[LocalData] = None  # Or localDataId: int
    surfaceData: Optional[SurfaceData] = None  # Or surfaceDataId: int
    dist: float = 0.0
    deltaX: float = 0.0
    deltaY: float = 0.0


@dataclass
class MemoryCell:
    state: List[int]
    action: List[int]
    reward: float
    next_state: List[int] = None  # Used by DQN, SARSA
    done: bool = None  # Used by DQN, SARSA
    next_action: List[int] = None  # Used by SARSA


@dataclass
class PPOMemoryCell:
    state: List[int]
    action: List[int]
    logits: List[float]
    log_probs: List[float]  # Should be a list containing a single float
    reward: float


@dataclass
class ActionResult:
    next_state: List[int]
    reward: float
    done: bool
