from typing import List, Any
import cv2
import numpy as np

def detect_faces(detector: Any, img: np.ndarray) -> List[Any]:
    """
    Detect all faces in an image using YuNet (FaceDetectorYN).
    
    Args:
        detector: The pre-loaded FaceDetectorYN singleton.
        img: The image as a numpy BGR array.
        
    Returns:
        A list/array of face rows detected in the image.
    """
    height, width = img.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(img)
    return list(faces) if faces is not None else []
