"""
Version Manager for Adaptive Qwen Omni.
Manages model and adapter versioning.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
import hashlib
import time

logger = logging.getLogger(__name__)


@dataclass
class VersionInfo:
    """Information about a version."""
    version: str
    base_model: str
    adapters: List[str] = field(default_factory=list)
    created_at: str = ""
    hash: str = ""
    size_bytes: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "base_model": self.base_model,
            "adapters": self.adapters,
            "created_at": self.created_at,
            "hash": self.hash,
            "size_bytes": self.size_bytes,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }


class VersionManager:
    """
    Manages versioning of models and adapters.
    
    Features:
    - Create version identifiers
    - Track version history
    - Rollback to previous versions
    - Compare versions
    """
    
    def __init__(
        self,
        version_format: str = "{base}-{timestamp}-{hash}",
        max_versions: int = 10,
        storage_path: str = "./versions",
    ):
        """
        Initialize version manager.
        
        Args:
            version_format: Format string for version identifiers
            max_versions: Maximum number of versions to keep
            storage_path: Path to store version information
        """
        self.version_format = version_format
        self.max_versions = max_versions
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.versions: Dict[str, VersionInfo] = {}
        self.version_history: List[str] = []
    
    def create_version(
        self,
        base_model: str,
        adapters: Optional[List[str]] = None,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new version identifier.
        
        Args:
            base_model: Base model name
            adapters: List of adapter names
            metrics: Performance metrics
            metadata: Additional metadata
            
        Returns:
            Version identifier
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Create hash
        version_data = f"{base_model}_{adapters}_{timestamp}"
        version_hash = hashlib.sha256(version_data.encode()).hexdigest()[:8]
        
        # Format version
        version = self.version_format.format(
            base=base_model.split('/')[-1] if '/' in base_model else base_model,
            timestamp=timestamp,
            hash=version_hash,
        )
        
        # Create version info
        info = VersionInfo(
            version=version,
            base_model=base_model,
            adapters=adapters or [],
            created_at=timestamp,
            hash=version_hash,
            metrics=metrics or {},
            metadata=metadata or {},
        )
        
        # Store version
        self.versions[version] = info
        self.version_history.append(version)
        
        # Rotate if needed
        if len(self.version_history) > self.max_versions:
            old_version = self.version_history.pop(0)
            del self.versions[old_version]
        
        # Save to storage
        self._save_version_info(info)
        
        logger.info(f"Created version: {version}")
        return version
    
    def _save_version_info(self, info: VersionInfo) -> None:
        """Save version info to storage."""
        import json
        
        info_path = self.storage_path / f"{info.version}.json"
        
        with open(info_path, "w") as f:
            json.dump(info.to_dict(), f, indent=2)
    
    def get_version(self, version: str) -> Optional[VersionInfo]:
        """Get version information."""
        return self.versions.get(version)
    
    def get_latest_version(self) -> Optional[VersionInfo]:
        """Get the latest version."""
        if not self.version_history:
            return None
        return self.versions.get(self.version_history[-1])
    
    def get_version_history(self) -> List[str]:
        """Get version history."""
        return self.version_history.copy()
    
    def rollback(self, version: str) -> bool:
        """
        Rollback to a specific version.
        
        Args:
            version: Version to rollback to
            
        Returns:
            True if rollback was successful
        """
        if version not in self.versions:
            logger.error(f"Version not found: {version}")
            return False
        
        # In actual implementation, this would restore the model/adapters
        # For now, just log the rollback
        logger.info(f"Rolling back to version: {version}")
        return True
    
    def rollback_to_previous(self) -> bool:
        """Rollback to the previous version."""
        if len(self.version_history) < 2:
            logger.error("No previous version to rollback to")
            return False
        
        previous_version = self.version_history[-2]
        return self.rollback(previous_version)
    
    def compare_versions(
        self,
        version1: str,
        version2: str,
    ) -> Dict[str, Any]:
        """
        Compare two versions.
        
        Args:
            version1: First version
            version2: Second version
            
        Returns:
            Comparison results
        """
        info1 = self.versions.get(version1)
        info2 = self.versions.get(version2)
        
        if not info1 or not info2:
            return {"error": "One or both versions not found"}
        
        comparison = {
            "version1": version1,
            "version2": version2,
            "metrics_diff": {},
        }
        
        # Compare metrics
        for metric in set(info1.metrics.keys()) | set(info2.metrics.keys()):
            val1 = info1.metrics.get(metric, 0)
            val2 = info2.metrics.get(metric, 0)
            comparison["metrics_diff"][metric] = val2 - val1
        
        return comparison
    
    def delete_version(self, version: str) -> bool:
        """Delete a version."""
        if version not in self.versions:
            return False
        
        del self.versions[version]
        
        if version in self.version_history:
            self.version_history.remove(version)
        
        # Delete from storage
        info_path = self.storage_path / f"{version}.json"
        if info_path.exists():
            info_path.unlink()
        
        logger.info(f"Deleted version: {version}")
        return True
    
    def clear_versions(self) -> None:
        """Clear all versions."""
        self.versions.clear()
        self.version_history.clear()
        
        # Clear storage
        for info_path in self.storage_path.glob("*.json"):
            info_path.unlink()
        
        logger.info("Cleared all versions")
    
    def list_versions(self) -> List[Dict[str, Any]]:
        """List all versions."""
        return [info.to_dict() for info in self.versions.values()]
