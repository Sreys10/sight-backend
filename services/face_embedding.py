import numpy as np
from typing import Any

def extract_face_embedding(recognizer: Any, img: np.ndarray, face: Any) -> list:
    """
    Extract and normalize the 128-dimensional embedding from an OpenCV FaceRecognizerSF object.
    
    Args:
        recognizer: The pre-loaded FaceRecognizerSF singleton.
        img: The source BGR image.
        face: The face row returned by YuNet.
        
    Returns:
        A list of 128 floats representing the normalized face embedding.
    """
    # Align and crop the face to 112x112
    aligned = recognizer.alignCrop(img, face)
    
    # Extract the 128-dimensional feature
    feature = recognizer.feature(aligned)
    if feature is None or len(feature) == 0:
        raise ValueError("No embedding generated for the face.")
        
    embedding = feature[0]
    
    # Standardize to normal float list
    norm = np.linalg.norm(embedding)
    if norm > 0:
        normed = embedding / norm
    else:
        normed = embedding
        
    return normed.tolist()
