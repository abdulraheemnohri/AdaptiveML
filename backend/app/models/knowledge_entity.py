"""
Knowledge Entity and Relationship Models - For Knowledge Graph
"""

from sqlalchemy import Column, String, Text, Integer, Float, Boolean, JSON, DateTime, Enum, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base
from datetime import datetime
import enum


class EntityType(str, enum.Enum):
    CONCEPT = "concept"
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    EVENT = "event"
    OBJECT = "object"
    PROCESS = "process"
    THEORY = "theory"
    FACT = "fact"
    DEFINITION = "definition"
    QUESTION = "question"
    ANSWER = "answer"
    DOCUMENT = "document"
    CODE = "code"
    FORMULA = "formula"
    TERM = "term"


class KnowledgeEntity(Base):
    __tablename__ = "knowledge_entities"
    
    id = Column(String(36), primary_key=True, index=True, unique=True)
    name = Column(String(255), nullable=False)
    entity_type = Column(Enum(EntityType), nullable=False, index=True)
    description = Column(Text)
    content = Column(Text)
    metadata = Column(JSON, default={})
    source = Column(String(1024))
    source_type = Column(String(50))
    confidence = Column(Float, default=0.0)
    verified = Column(Boolean, default=False)
    verification_status = Column(String(50))
    verification_notes = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    outgoing_relationships = relationship("KnowledgeRelationship", foreign_keys="KnowledgeRelationship.source_id", backref="source_entity", cascade="all, delete-orphan")
    incoming_relationships = relationship("KnowledgeRelationship", foreign_keys="KnowledgeRelationship.target_id", backref="target_entity", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<KnowledgeEntity(id={self.id}, name={self.name}, type={self.entity_type.value})>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type.value,
            "description": self.description,
            "content": self.content,
            "metadata": self.metadata or {},
            "source": self.source,
            "source_type": self.source_type,
            "confidence": self.confidence,
            "verified": self.verified,
            "verification_status": self.verification_status,
            "verification_notes": self.verification_notes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class RelationshipType(str, enum.Enum):
    IS_A = "is_a"
    HAS_A = "has_a"
    PART_OF = "part_of"
    CAUSES = "causes"
    EFFECT_OF = "effect_of"
    RELATED_TO = "related_to"
    SYNONYM = "synonym"
    ANTONYM = "antonym"
    DEPENDS_ON = "depends_on"
    REQUIRES = "requires"
    USED_FOR = "used_for"
    EXAMPLE_OF = "example_of"
    INSTANCE_OF = "instance_of"
    SUBCLASS_OF = "subclass_of"
    SUPERCLASS_OF = "superclass_of"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    REFUTES = "refutes"


class KnowledgeRelationship(Base):
    __tablename__ = "knowledge_relationships"
    
    id = Column(String(36), primary_key=True, index=True, unique=True)
    source_id = Column(String(36), ForeignKey("knowledge_entities.id"), nullable=False)
    target_id = Column(String(36), ForeignKey("knowledge_entities.id"), nullable=False)
    relationship_type = Column(Enum(RelationshipType), nullable=False, index=True)
    metadata = Column(JSON, default={})
    weight = Column(Float, default=1.0)
    confidence = Column(Float, default=0.0)
    verified = Column(Boolean, default=False)
    source = Column(String(1024))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    source_entity = relationship("KnowledgeEntity", foreign_keys=[source_id])
    target_entity = relationship("KnowledgeEntity", foreign_keys=[target_id])
    
    def __repr__(self):
        return f"<KnowledgeRelationship(id={self.id}, source={self.source_id}, target={self.target_id}, type={self.relationship_type.value})>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship_type": self.relationship_type.value,
            "weight": self.weight,
            "metadata": self.metadata or {},
            "confidence": self.confidence,
            "verified": self.verified,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }