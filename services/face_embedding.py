import numpy as np
from typing import Any

def extract_face_embedding(face: Any) -> list:
    """
    Extract and normalize the 512-dimensional embedding from an InsightFace Face object.
    
    Args:
        face: The Face object returned by InsightFace app.get().
        
    Returns:
        A list of 512 floats representing the normalized face embedding.
    """
    embedding = face.embedding
    if embedding is None:
        raise ValueError("No embedding generated for the face.")
        
    # Standardize to normal float list
    norm = np.linalg.norm(embedding)
    if norm > 0:
        normed = embedding / norm
    else:
        normed = embedding
        
    return normed.tolist()
