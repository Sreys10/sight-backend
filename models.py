import uuid
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from database import Base

class Person(Base):
    __tablename__ = "person"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=text("gen_random_uuid()"))
    full_name = Column(String, nullable=False)
    gender = Column(String, nullable=True)
    age = Column(Integer, nullable=True)
    case_number = Column(String, nullable=True)
    notes = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=text("now()"))

    # Relationship to FaceEmbedding
    embeddings = relationship("FaceEmbedding", back_populates="person", cascade="all, delete-orphan")


class FaceEmbedding(Base):
    __tablename__ = "face_embedding"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=text("gen_random_uuid()"))
    person_id = Column(UUID(as_uuid=True), ForeignKey("person.id", ondelete="CASCADE"), nullable=False)
    image_path = Column(String, nullable=False)
    embedding = Column(Vector(512), nullable=False)  # 512-dimensional ArcFace vector
    created_at = Column(DateTime(timezone=True), server_default=text("now()"))

    # Relationship to Person
    person = relationship("Person", back_populates="embeddings")
