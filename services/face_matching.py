import logging
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Dict, Any, Optional
from models import FaceEmbedding, Person

logger = logging.getLogger("face_recognition")

def search_face_in_db(
    db: Session, 
    query_embedding: List[float], 
    threshold: float = 0.60
) -> List[Dict[str, Any]]:
    """
    Search the database for the closest face matches using cosine similarity.
    
    Args:
        db: SQLAlchemy database session.
        query_embedding: 512-dimensional query vector.
        threshold: Minimum similarity threshold (default 0.60).
        
    Returns:
        List of dictionaries containing matched FaceEmbedding, Person, and similarity score.
    """
    try:
        # pgvector cosine_distance operator is <=>
        # cosine_similarity = 1.0 - cosine_distance
        cosine_distance_expr = FaceEmbedding.embedding.cosine_distance(query_embedding)
        similarity_expr = 1.0 - cosine_distance_expr
        
        # Query top 5 matches above threshold
        results = (
            db.query(FaceEmbedding, Person, similarity_expr.label("similarity"))
            .join(Person, FaceEmbedding.person_id == Person.id)
            .filter(similarity_expr >= threshold)
            .order_by(text("similarity DESC"))
            .limit(5)
            .all()
        )
        
        matches = []
        for face_emb, person, similarity in results:
            matches.append({
                "face_embedding": face_emb,
                "person": person,
                "similarity": float(similarity)
            })
            
        return matches
    except Exception as e:
        logger.error(f"Error during pgvector database search: {str(e)}")
        return []
