from abc import ABC, abstractmethod
from typing import List
import numpy as np
from ..data_types import Detection


class BaseInferencer(ABC):
    @abstractmethod
    def run_inference(self, input_image: np.ndarray) -> List[Detection]:
        """
        Base method for detector implementation.
        Args:
            input_image: Input image as a NumPy array.
        Returns:
            A list of Detection objects.
        """
        pass
