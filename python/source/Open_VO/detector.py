import cv2
import numpy as np
from typing import List, Dict, Optional
from .data_types import Detection
from .inferencers import BaseInferencer
from .inferencers import OnnxInferencerPy
from .inferencers import TorchInferencerPy


class DetectorPy:
    def __init__(self, labels_file: str, model_path: str,
                 cuda_enabled: bool = True, img_w: int = 640, img_h: int = 640,
                 score_thresh: float = 0.45, nms_thresh: float = 0.50, max_det: int = 100):
        self.labels_file = labels_file
        self.model_path = model_path
        self.labels: Dict[int, str] = self._load_labels()

        self.is_onnx_model = model_path.lower().endswith(".onnx")
        self.inferencer: Optional[BaseInferencer] = None

        if self.is_onnx_model:
            self.inferencer = OnnxInferencerPy(
                onnx_model_path=model_path, model_classes=self.labels,
                img_w=img_w, img_h=img_h, run_with_cuda=cuda_enabled,
                score_thresh=score_thresh, nms_thresh=nms_thresh, max_det=max_det
            )
        else:
            self.inferencer = TorchInferencerPy(
                torch_model_path=model_path, model_classes=self.labels,
                img_w=img_w, img_h=img_h, run_with_cuda=cuda_enabled,
                score_thresh=score_thresh, nms_thresh=nms_thresh, max_det=max_det
            )
            # TorchInferencerPy calls load_torch_network in its __init__

    def _load_labels(self) -> Dict[int, str]:
        labels_map: Dict[int, str] = {}
        try:
            with open(self.labels_file, 'r') as file:
                for line in file:
                    line = line.strip()
                    if not line: continue
                    parts = line.split(',', 1)  # Split only on the first comma
                    if len(parts) == 2:
                        try:
                            label_id = int(parts[0])
                            label_name = parts[1].strip()
                            labels_map[label_id] = label_name
                        except ValueError:
                            print(f"Warning: Could not parse label line: {line}")
                    else:
                        print(f"Warning: Malformed label line (expected id,name): {line}")
        except IOError as e:
            print(f"Error reading labels file {self.labels_file}: {e}")
        return labels_map

    def read_image(self, image_path: str) -> Optional[np.ndarray]:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Image File: {image_path} Is Not Found or empty.")
            return None
        return image

    def detect(self, image: np.ndarray) -> List[Detection]:
        if self.inferencer is None:
            print("Inferencer not initialized.")
            return []
        if image is None:
            print("Input image to detect is None.")
            return []
        return self.inferencer.run_inference(image)
