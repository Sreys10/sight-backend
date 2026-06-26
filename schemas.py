from pydantic import BaseModel, Field, ConfigDict
from uuid import UUID
from datetime import datetime
from typing import List, Optional

# Person Schema
class PersonBase(BaseModel):
    full_name: str = Field(..., description="Full name of the registered person")
    gender: Optional[str] = Field(None, description="Gender of the person")
    age: Optional[int] = Field(None, description="Age of the person")
    case_number: Optional[str] = Field(None, description="Associated case number/identifier")
    notes: Optional[str] = Field(None, description="Additional forensic/investigation notes")

class PersonCreate(PersonBase):
    pass

class PersonResponse(PersonBase):
    id: UUID
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

# FaceEmbedding Schema
class FaceEmbeddingResponse(BaseModel):
    id: UUID
    person_id: UUID
    image_path: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

# Search Response Schema
class MatchedPerson(BaseModel):
    id: UUID
    full_name: str = Field(..., description="Full name of the registered person")
    case_number: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[int] = None
    notes: Optional[str] = None
    created_at: datetime
    registered_images: List[str] = Field(default_factory=list, description="URLs/paths to registered photos")

    model_config = ConfigDict(
        from_attributes=True
    )

class FaceSearchMatch(BaseModel):
    face_index: int
    bounding_box: List[int] = Field(..., description="[x1, y1, x2, y2] coordinates")
    matched: bool
    confidence: Optional[float] = Field(None, description="Cosine similarity score as percentage (0-100)")
    person: Optional[MatchedPerson] = None
