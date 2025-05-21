import cv2
import numpy as np
import torch
import torchvision.ops.boxes as box_ops
from typing import List, Dict, Tuple, Optional
from ..data_types import Detection, Pos_i2
from .base_inferencer import BaseInferencer


class TorchInferencerPy(BaseInferencer):
    def __init__(self, torch_model_path: str, model_classes: Dict[int, str],
                 img_w: int = 640, img_h: int = 640, run_with_cuda: bool = True,
                 score_thresh: float = 0.45, nms_thresh: float = 0.50, max_det: int = 100):
        self.model_path = torch_model_path
        self.classes = model_classes
        self.device = None
        self.model_shape = Pos_i2(x=img_w, y=img_h)
        self.model_score_threshold = score_thresh
        self.model_nms_threshold = nms_thresh
        self.model_max_det = max_det
        self.letter_box_for_square = True
        self.net: Optional[torch.jit.ScriptModule] = None
        self.cuda_enabled_by_user = run_with_cuda
        self.actual_cuda_enabled = False
        self.load_torch_network()

    def _generate_scale(self, image_shape_hw: Tuple[int, int]) -> float:
        """Calculates resize scale factor to fit image into model_shape maintaining aspect ratio."""
        origin_h, origin_w = image_shape_hw
        target_h, target_w = self.model_shape.y, self.model_shape.x
        ratio_h = float(target_h) / float(origin_h)
        ratio_w = float(target_w) / float(origin_w)
        return min(ratio_h, ratio_w)

    def _format_to_square(self, source: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """
        Formats image to model_shape (square or not) using letterboxing.
        Returns formatted_image, scale_factor, (pad_w, pad_h)
        """
        if source.shape[1] == self.model_shape.x and source.shape[0] == self.model_shape.y:
            return source, 1.0, (0.0, 0.0)

        resize_scale = self._generate_scale((source.shape[0], source.shape[1]))
        new_shape_w = round(source.shape[1] * resize_scale)
        new_shape_h = round(source.shape[0] * resize_scale)

        pad_w = (self.model_shape.x - new_shape_w) / 2.0
        pad_h = (self.model_shape.y - new_shape_h) / 2.0

        top = round(pad_h - 0.1)  # Same rounding as C++
        bottom = round(pad_h + 0.1)
        left = round(pad_w - 0.1)
        right = round(pad_w + 0.1)

        resized_img = cv2.resize(source, (new_shape_w, new_shape_h), interpolation=cv2.INTER_AREA)

        # Add padding
        # Ensure padding values are non-negative
        top = max(0, top)
        bottom = max(0, bottom)
        left = max(0, left)
        right = max(0, right)

        # If the target size after padding is different due to rounding, adjust:
        # This can happen if (model_shape.width - new_shape_w) is odd.
        # The C++ copyMakeBorder handles this by padding to make total size correct.
        # Here, we ensure the total padding makes the image self.model_shape

        padded_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right,
                                        cv2.BORDER_CONSTANT, value=[114, 114, 114])  # BGR for pad value

        # It's possible that due to rounding, padded_img is not exactly model_shape.
        # If so, resize it one last time. More robust: calculate exact padding.
        current_h, current_w = padded_img.shape[:2]
        if current_h != self.model_shape.y or current_w != self.model_shape.x:
            # This can happen if the sum of pads doesn't exactly match.
            # A more robust way to pad to target size:
            delta_w = self.model_shape.x - new_shape_w
            delta_h = self.model_shape.y - new_shape_h
            pad_w_left = delta_w // 2
            pad_w_right = delta_w - pad_w_left
            pad_h_top = delta_h // 2
            pad_h_bottom = delta_h - pad_h_top

            padded_img = cv2.copyMakeBorder(resized_img, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right,
                                            cv2.BORDER_CONSTANT, value=[114, 114, 114])
            pad_w_actual = pad_w_left  # Use for unscaling
            pad_h_actual = pad_h_top  # Use for unscaling
        else:
            pad_w_actual = left
            pad_h_actual = top

        return padded_img, resize_scale, (pad_w_actual, pad_h_actual)

    def _xywh2xyxy(self, x_tensor: torch.Tensor) -> torch.Tensor:
        """Converts nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right."""
        y_tensor = torch.empty_like(x_tensor)
        dw = x_tensor[..., 2] / 2
        dh = x_tensor[..., 3] / 2
        y_tensor[..., 0] = x_tensor[..., 0] - dw
        y_tensor[..., 1] = x_tensor[..., 1] - dh
        y_tensor[..., 2] = x_tensor[..., 0] + dw
        y_tensor[..., 3] = x_tensor[..., 1] + dh
        return y_tensor

    def _scale_boxes(self, model_input_shape_hw: Tuple[int, int], boxes: torch.Tensor,
                     original_img_shape_hw: Tuple[int, int],
                     resize_scale: float, pad_wh: Tuple[float, float]) -> torch.Tensor:
        """
        Rescales boxes from model_input_shape (letterboxed) to original_img_shape.
        model_input_shape_hw: (height, width) of the image fed to the model (after letterboxing)
        boxes: [N, 4] tensor of (x1, y1, x2, y2) relative to model_input_shape
        original_img_shape_hw: (height, width) of the original image
        resize_scale: The scale factor used to resize original to fit model_input before padding
        pad_wh: (pad_width_applied_on_left, pad_height_applied_on_top)
        """
        pad_w, pad_h = pad_wh

        # Remove padding
        boxes[:, [0, 2]] -= pad_w  # x coordinates
        boxes[:, [1, 3]] -= pad_h  # y coordinates

        # Scale back to original image size
        boxes[:, :4] /= resize_scale

        # Clip to original image dimensions
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, original_img_shape_hw[1])  # x coordinates, clamp to width
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, original_img_shape_hw[0])  # y coordinates, clamp to height
        return boxes

    def _non_max_suppression_torch(self, prediction: torch.Tensor) -> List[torch.Tensor]:
        """
        Performs Non-Maximum Suppression (NMS) on inference results.
        prediction: [batch_size, num_classes + 5, num_anchors] or [batch_size, num_anchors, num_classes + 5]
        This needs to match the C++ output parsing for YOLO-like models.
        YOLOv5/v8 output: (bs, num_dets, xywh + conf + num_classes)
        or (bs, num_classes + 4 + (masks, if any), num_dets_per_level)
        The C++ code uses output shape: [bs, num_classes + 4 + num_masks, num_predictions]
        And it transposes to [bs, num_predictions, num_classes + 4 + num_masks]
        """
        # Assuming prediction is [bs, num_predictions, num_coords_conf_classes_masks]
        # bs = prediction.size(0)
        # nc = self.num_classes # Number of classes
        # nm = 0 # Number of masks, assume 0 for now like C++
        # mi = 4 + nc # Index of first mask coefficient

        # xc: True for predictions where max class_conf > score_thresh
        # C++: xc = prediction.index({ Slice(), Slice(4, mi) }).amax(1) > modelScoreThreshold;
        # This implies prediction[..., 4:mi] are class confidences.
        # Assumes prediction layout [bs, N, (x,y,w,h, obj_conf, cls1_conf, ..., clsN_conf, mask_data)] (YOLOv5 general format)
        # Or [bs, N, (x,y,w,h, cls1_conf, ..., clsN_conf)] if obj_conf is part of class scores or implicit.
        # The C++ code uses `prediction.index({ Slice(), Slice(4, mi) })` which seems to imply
        # class scores start at index 4. This is typical if objectness score is not separate or handled implicitly.

        conf_thres = self.model_score_threshold
        iou_thres = self.model_nms_threshold
        max_det = self.model_max_det

        # The exact NMS logic from C++ is quite specific.
        # For Python, torchvision.ops.nms or torchvision.ops.batched_nms is standard.
        # We need to get boxes [N, 4] (x1,y1,x2,y2) and scores [N] for each class.
        # Then apply NMS per class, or use multi-label NMS.

        output = [torch.zeros((0, 6), device=prediction.device)] * prediction.size(0)  # [x1, y1, x2, y2, conf, cls]

        for xi, x in enumerate(prediction):  # Iterate through batch
            # x is [N, C] where C = 4 (box) + 1 (obj_conf, if present) + num_classes
            # Or C = 4 (box) + num_classes (if obj_conf is multiplied into class_conf)

            # Filter by confidence
            # Assuming x[:, 4] is object confidence and x[:, 5:] are class scores for YOLOv5
            # Or, if it's like C++ `Slice(4, mi)`, then x[:, 4:4+num_classes] are class scores.
            # Let's assume output format [x,y,w,h, cls_score1, cls_score2, ...] for simplicity based on C++ NMS logic

            # Convert box from [center_x, center_y, width, height] to [x1, y1, x2, y2]
            box = self._xywh2xyxy(x[:4, :])

            # Detections matrix nx6 (xyxy, conf, cls)
            # Handle different YOLO output formats:
            if x.shape[0] == len(self.classes) + 4:  # [xywh, cls1, cls2, ...]
                # This case: obj_conf is implicit or not present. Confidence is max class score.
                conf, j = x[4:, :].max(0, keepdim=True)
                x = torch.cat((box, conf, j.float()), 0)[:, conf.view(-1) > conf_thres]
            else:  # Assume [xywh, obj_conf, cls1, cls2, ...]
                obj_conf = x[:, 4]
                class_conf, class_pred = x[:, 5:].max(1, keepdim=True)
                # conf = obj_conf * class_conf # Confidence = obj_conf * class_conf
                conf = class_conf  # C++ seems to use max class score as confidence for NMS selection

                # Filter by object confidence first (common practice)
                # x = x[obj_conf > some_obj_conf_threshold] # Optional pre-filter

                x = torch.cat((box, conf, class_pred.float()), 1)[conf.view(-1) > conf_thres]

            # If none remain after confidence thresholding
            if not x.size(0):
                continue

            # Batched NMS from torchvision
            # Sort by score
            # x = x[x[:, 4].argsort(descending=True)] # Sort by confidence

            # NMS
            # c = x[:, 5:6] * (4096 if multi_label else 1)  # classes
            # boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            # i = torchvision.ops.nms(boxes, scores, iou_thres) # NMS

            # Using torchvision.ops.batched_nms which handles classes internally for multi-label NMS
            # boxes (Tensor[N, 4]): boxes in (x1, y1, x2, y2) format
            # scores (Tensor[N]): scores for each one of the boxes
            # idxs (Tensor[N]): classes for each one of the boxes
            i = box_ops.batched_nms(x[:, :4], x[:, 4], x[:, 5], iou_thres)

            if i.size(0) > max_det:  # limit detections
                i = i[:max_det]

            output[xi] = x[i]
        return output

    def load_torch_network(self):
        try:
            self.net = torch.jit.load(self.model_path)
            self.net.eval()  # Set to evaluation mode

            if self.cuda_enabled_by_user and torch.cuda.is_available():
                self.device = torch.device('cuda')
                print("\nPyTorch: Attempting to run on CUDA")
                self.net.to(self.device)
                self.actual_cuda_enabled = True
            else:
                if self.cuda_enabled_by_user and not torch.cuda.is_available():
                    print("\nPyTorch: CUDA selected but not available. Running on CPU.")
                else:
                    print("\nPyTorch: Running on CPU")
                self.device = torch.device('cpu')
                self.net.to(self.device)
                self.actual_cuda_enabled = False
            # First inference can be slow, do a dummy forward pass
            # dummy_input = torch.rand(1, 3, self.model_shape.y, self.model_shape.x).to(self.device)
            # with torch.no_grad():
            #    self.net(dummy_input)
            # print("PyTorch model warm-up complete.")

        except Exception as e:
            print(f"Error loading PyTorch model {self.model_path}: {e}")
            self.net = None

    def run_inference(self, input_image: np.ndarray) -> List[Detection]:
        if self.net is None or self.device is None:
            print("PyTorch model not loaded. Cannot run inference.")
            return []

        original_shape_hw = (input_image.shape[0], input_image.shape[1])

        # Letterbox/Resize
        if self.letter_box_for_square and self.model_shape.x == self.model_shape.y:
            model_input_img, resize_scale, pad_wh = self._format_to_square(input_image)
        else:  # Just resize if not letterboxing to square
            model_input_img = cv2.resize(input_image, (self.model_shape.x, self.model_shape.y))
            resize_scale = 1.0  # Or calculate actual scale if aspect ratio changed
            pad_wh = (0.0, 0.0)

        # Convert to tensor: BGR to RGB, HWC to CHW, normalize
        img_tensor = torch.from_numpy(model_input_img).to(self.device)
        img_tensor = img_tensor.float() / 255.0  # Normalize to 0-1
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension if missing
        img_tensor = img_tensor.permute(0, 3, 1, 2)  # BHWC to BCHW (assuming input_image is HWC BGR)
        # If input_image is HWC RGB, no BGR->RGB needed.
        # OpenCV imread is BGR. If cvtColor to RGB before, then this permute is fine.
        # Assuming model_input_img is HWC and PyTorch model expects BCHW.
        # If model_input_img is BGR, need to swap channels.
        # img_tensor = img_tensor[:, [2,1,0], :, :] # BGR to RGB if needed here

        detections_final: List[Detection] = []
        with torch.no_grad():
            outputs = self.net(img_tensor)  # Forward pass

        # Post-process (this is highly dependent on the model's output structure)
        # The C++ code is for YOLO-like models.
        # `outputs` could be a tensor or a list/tuple of tensors.
        # Assuming `outputs` is a tensor or the first element is the main prediction tensor
        if isinstance(outputs, tuple) or isinstance(outputs, list):
            prediction_tensor = outputs[0]
        else:
            prediction_tensor = outputs

        # Apply NMS
        # The C++ nonMaxSuppression is complex. Using torchvision.ops.batched_nms is preferred in Python.
        # The prediction_tensor shape and content need to be adapted for batched_nms.
        # For now, using the _non_max_suppression_torch method (which itself needs robust implementation)

        processed_outputs = self._non_max_suppression_torch(prediction_tensor)  # List of tensors (one per batch image)

        for i in range(len(processed_outputs)):  # Iterate through batch
            output_per_image = processed_outputs[i]  # Tensor of [N, 6] (x1,y1,x2,y2,conf,cls)

            if output_per_image.numel() == 0:
                continue

            # Scale boxes back to original image coordinates
            boxes_scaled = self._scale_boxes(
                (model_input_img.shape[0], model_input_img.shape[1]),
                output_per_image[:4, :],  # xyxy boxes
                original_shape_hw,
                resize_scale,
                pad_wh
            )

            for det_idx in range(output_per_image.size(1)):
                x1, y1, x2, y2 = boxes_scaled[:, det_idx].cpu().numpy().astype(int)
                conf = output_per_image[4, det_idx].item()
                cls_id = int(output_per_image[5, det_idx].item())

                detections_final.append(Detection(
                    class_id=cls_id,
                    className=self.classes.get(cls_id, "Unknown"),
                    confidence=conf,
                    x=x1,
                    y=y1,
                    w=x2 - x1,
                    h=y2 - y1
                ))
        return detections_final
