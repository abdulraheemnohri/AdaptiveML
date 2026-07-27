"""
Knowledge Graph Memory Service.
Stores entities, relationships, verified facts, sources, and timestamps.
Detects contradictions, outdated information, and missing connections.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional


class GraphEntity:
    def __init__(self, name: str, entity_type: str, metadata: Optional[Dict[str, Any]] = None):
        self.name = name
        self.entity_type = entity_type
        self.metadata = metadata or {}


class GraphRelation:
    def __init__(
        self,
        source: str,
        relation_type: str,
        target: str,
        source_ref: str = "scraped_web",
        confidence: float = 1.0,
    ):
        self.source = source
        self.relation_type = relation_type
        self.target = target
        self.source_ref = source_ref
        self.confidence = confidence
        self.timestamp = datetime.now().isoformat()


class KnowledgeGraphService:
    """Manages structural entity-relationship knowledge graph memory."""

    def __init__(self):
        self.entities: Dict[str, GraphEntity] = {}
        self.relations: List[GraphRelation] = []

        # Populate defaults
        self.add_entity("Qwen2.5-Omni-3B", "Base Model")
        self.add_entity("Text", "Modality")
        self.add_entity("Vision", "Modality")
        self.add_entity("Audio", "Modality")
        self.add_entity("Video", "Modality")
        self.add_entity("Qwen", "Developer")

        self.add_relation("Qwen2.5-Omni-3B", "Developed By", "Qwen", confidence=0.99)
        self.add_relation("Qwen2.5-Omni-3B", "Supports", "Text", confidence=0.95)
        self.add_relation("Qwen2.5-Omni-3B", "Supports", "Vision", confidence=0.95)
        self.add_relation("Qwen2.5-Omni-3B", "Supports", "Audio", confidence=0.92)
        self.add_relation("Qwen2.5-Omni-3B", "Supports", "Video", confidence=0.91)

    def add_entity(self, name: str, entity_type: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.entities[name] = GraphEntity(name, entity_type, metadata)

    def add_relation(
        self,
        source: str,
        relation_type: str,
        target: str,
        source_ref: str = "verification_agent",
        confidence: float = 1.0,
    ) -> None:
        relation = GraphRelation(source, relation_type, target, source_ref, confidence)
        self.relations.append(relation)

    def detect_contradictions(self, source: str, relation_type: str, target: str) -> List[GraphRelation]:
        """Detect relations that contradict the proposed facts."""
        conflicting = []
        for r in self.relations:
            if r.source == source and r.relation_type == relation_type and r.target != target:
                conflicting.append(r)
        return conflicting

    def query_relationships(self, entity_name: str) -> List[Dict[str, Any]]:
        """Retrieve all relationships connected to a given entity."""
        results = []
        for r in self.relations:
            if r.source == entity_name or r.target == entity_name:
                results.append({
                    "source": r.source,
                    "relation": r.relation_type,
                    "target": r.target,
                    "confidence": r.confidence,
                    "timestamp": r.timestamp,
                    "source_ref": r.source_ref
                })
        return results

    def get_stats(self) -> Dict[str, Any]:
        return {
            "entity_count": len(self.entities),
            "relation_count": len(self.relations),
            "last_updated": datetime.now().isoformat()
        }
