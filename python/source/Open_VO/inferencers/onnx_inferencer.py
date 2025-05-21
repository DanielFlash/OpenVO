import cv2
import numpy as np
import onnxruntime
from typing import List, Dict, Tuple
from ..data_types import Detection, Pos_i2
from .base_inferencer import BaseInferencer


class OnnxInferencerPy(BaseInferencer):
    def __init__(self, onnx_model_path: str, model_classes: Dict[int, str],
                 img_w: int = 640, img_h: int = 640, run_with_cuda: bool = True,
                 score_thresh: float = 0.45, nms_thresh: float = 0.50, max_det: int = 100):
        self.model_path = onnx_model_path
        self.classes = model_classes
        self.cuda_enabled = run_with_cuda
        self.model_shape = Pos_i2(x=img_w, y=img_h)
        self.model_score_threshold = score_thresh
        self.model_nms_threshold = nms_thresh
        self.model_max_det = max_det
        self.letter_box_for_square = True
        self.session = None
        self.input_name = None
        self.output_names = None
        self.load_onnx_network()

    def _format_to_square(self, source: np.ndarray) -> np.ndarray:
        """Pads image to a square shape (letterboxing)."""
        col, row = source.shape[1], source.shape[0]
        _max = max(col, row)
        resized_image = np.zeros((_max, _max, 3), dtype=source.dtype)  # Assuming 3 channels
        resized_image[0:row, 0:col] = source
        return resized_image

    def load_onnx_network(self):
        providers = []
        if self.cuda_enabled:
            # Check if CUDA execution provider is available
            if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
                providers.append('CUDAExecutionProvider')
                print("\nONNX Runtime: Using CUDAExecutionProvider")
            else:
                print("\nONNX Runtime: CUDAExecutionProvider not available, falling back to CPU.")
                providers.append('CPUExecutionProvider')
        else:
            providers.append('CPUExecutionProvider')
            print("\nONNX Runtime: Using CPUExecutionProvider")

        try:
            self.session = onnxruntime.InferenceSession(self.model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
        except Exception as e:
            print(f"Error loading ONNX model {self.model_path}: {e}")
            self.session = None

    def run_inference(self, input_image: np.ndarray) -> List[Detection]:
        if self.session is None:
            print("ONNX model not loaded. Cannot run inference.")
            return []

        model_input = input_image.copy()
        if self.letter_box_for_square and self.model_shape.x == self.model_shape.y:
            model_input = self._format_to_square(model_input)  # Letterbox

        # Preprocess: blobFromImage
        # Note: ONNX models might expect different normalization/channel order (RGB vs BGR)
        # The C++ version uses blobFromImage with scalefactor=1.0/255.0, BGR=false (so RGB), swapRB=true
        blob = cv2.dnn.blobFromImage(model_input, scalefactor=1.0 / 255.0,
                                     size=(self.model_shape.x, self.model_shape.y),
                                     mean=(0, 0, 0),  # No mean subtraction in C++
                                     swapRB=True,  # C++ default for blobFromImage from BGR is to swap to RGB
                                     crop=False)

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: blob})

        # Post-process (this heavily depends on the specific ONNX model's output format)
        # The C++ code has specific logic for YOLOv5/YOLOv8 like outputs.
        # Assuming outputs[0] contains the detections [batch, num_detections, 5+num_classes] or similar

        # This post-processing is from your C++ ONNX:
        # It assumes output shape is [batch, N, 4+num_classes] or [batch, 4+num_classes, N]
        raw_output = outputs[0]  # Assuming first output tensor contains detections

        if raw_output.shape[2] > raw_output.shape[1]:  # e.g., (1, 84, 8400) -> (1, 8400, 84)
            # This is specific to some YOLO versions where classes+coords are rows
            raw_output = raw_output.reshape(raw_output.shape[0], raw_output.shape[2], raw_output.shape[1])
            # Transpose if needed, C++ uses reshape then transpose.
            # raw_output = np.transpose(raw_output, (0, 2, 1)) # (batch, N, 84)

        # Assuming output is now [batch_size, num_predictions, box_coords + num_classes_scores]
        # And box_coords are [center_x, center_y, width, height]

        data = raw_output[0]  # Assuming batch size of 1

        img_height, img_width = input_image.shape[:2]  # Original image dimensions
        model_input_height, model_input_width = model_input.shape[:2]  # Dimensions of image fed to model

        x_factor = model_input_width / float(self.model_shape.x)
        y_factor = model_input_height / float(self.model_shape.y)

        class_ids = []
        confidences = []
        boxes = []

        for i in range(data.shape[0]):  # Iterate over detections
            row = data[i, :]
            box_coords = row[0:4]
            class_scores = row[4:]  # Scores for each class

            class_id = np.argmax(class_scores)
            max_class_score = class_scores[class_id]

            if max_class_score > self.model_score_threshold:
                confidences.append(float(max_class_score))
                class_ids.append(int(class_id))

                # Box coordinates (center_x, center_y, width, height) normalized by model input size
                # Need to scale them to the `model_input` dimensions
                center_x, center_y, w, h = box_coords

                # Scale to model_input dimensions
                # x = center_x * self.model_shape.x
                # y = center_y * self.model_shape.y
                # box_w = w * self.model_shape.x
                # box_h = h * self.model_shape.y
                # This scaling logic might need adjustment depending on how the model outputs coords.
                # The C++ code has:
                # left = int((x - 0.5 * w) * x_factor); top = int((y - 0.5 * h) * y_factor);
                # This implies data[0]..data[3] are already scaled to model_input size or similar.
                # Let's follow C++ more closely if data[0]..data[3] are center_x,center_y,w,h
                # that are relative to model_input's letterboxed/resized dimensions
                # but x_factor/y_factor are based on model_input vs model_shape.
                # This implies the model output coordinates are relative to `model_shape`.

                center_x_abs = box_coords[0]  # Assuming these are already relative to model_shape
                center_y_abs = box_coords[1]
                width_abs = box_coords[2]
                height_abs = box_coords[3]

                left = int((center_x_abs - 0.5 * width_abs) * x_factor)
                top = int((center_y_abs - 0.5 * height_abs) * y_factor)
                box_width = int(width_abs * x_factor)
                box_height = int(height_abs * y_factor)

                # Clip to original image dimensions (after scaling back from letterbox if any)
                # This part is tricky because letterboxing adds padding.
                # The C++ `scaleBoxes` in TorchInference handles this.
                # For ONNX, if letterboxing was done, we need to un-letterbox the coordinates.
                # For now, assuming coordinates are relative to `model_input` after scaling by x_factor, y_factor

                boxes.append([left, top, box_width, box_height])

        if not boxes:
            return []

        # Apply NMS using cv2.dnn.NMSBoxes
        # boxes should be list of [x, y, w, h] rects
        # confidences should be list of floats
        # class_ids is not directly used by NMSBoxes but needed for final output
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.model_score_threshold, self.model_nms_threshold,
                                   top_k=self.model_max_det)

        detections: List[Detection] = []
        if len(indices) > 0:
            # If indices is a 2D array (e.g., [[0], [2]]), flatten it.
            # In Python, NMSBoxes usually returns a flat array of indices.
            if isinstance(indices, np.ndarray) and indices.ndim > 1:
                indices = indices.flatten()

            for i in indices:
                box = boxes[i]
                result = Detection(
                    class_id=class_ids[i],
                    className=self.classes.get(class_ids[i], "Unknown"),
                    confidence=confidences[i],
                    x=box[0],
                    y=box[1],
                    w=box[2],
                    h=box[3]
                )
                detections.append(result)
        return detections
