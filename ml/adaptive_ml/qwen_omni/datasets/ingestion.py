"""
Data Ingestion for Adaptive Qwen Omni.
Handles loading and preprocessing of multimodal datasets.
"""

import json
import csv
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union
import logging
import numpy as np
import torch

from adaptive_ml.qwen_omni.core import (
    ModalityType,
    TaskType,
    DomainType,
    MultimodalData,
    MultimodalEntry,
)

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset ingestion."""
    name: str
    path: Union[str, Path]
    format: str = "jsonl"  # jsonl, csv, json, parquet
    text_field: str = "text"
    image_field: str = "image"
    audio_field: str = "audio"
    video_field: str = "video"
    speech_field: str = "speech"
    instruction_field: str = "instruction"
    output_field: str = "output"
    domain_field: str = "domain"
    language_field: str = "language"
    
    # Chunking
    chunk_size: int = 1000
    max_length: Optional[int] = None
    
    # Filtering
    min_text_length: int = 10
    max_text_length: int = 4096
    allowed_modalities: List[ModalityType] = field(default_factory=list)
    
    # Sampling
    sample_size: Optional[int] = None
    sample_seed: int = 42


@dataclass
class DatasetStats:
    """Statistics for a dataset."""
    total_entries: int = 0
    entries_by_modality: Dict[ModalityType, int] = field(default_factory=dict)
    entries_by_domain: Dict[DomainType, int] = field(default_factory=dict)
    entries_by_task: Dict[TaskType, int] = field(default_factory=dict)
    avg_text_length: float = 0.0
    min_text_length: int = 0
    max_text_length: int = 0
    num_multimodal: int = 0
    num_text_only: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_entries": self.total_entries,
            "entries_by_modality": {k.value: v for k, v in self.entries_by_modality.items()},
            "entries_by_domain": {k.value: v for k, v in self.entries_by_domain.items()},
            "entries_by_task": {k.value: v for k, v in self.entries_by_task.items()},
            "avg_text_length": self.avg_text_length,
            "min_text_length": self.min_text_length,
            "max_text_length": self.max_text_length,
            "num_multimodal": self.num_multimodal,
            "num_text_only": self.num_text_only,
        }


class MultimodalDataset:
    """
    Dataset for multimodal data.
    Supports loading from various formats and provides iteration.
    """
    
    def __init__(
        self,
        config: DatasetConfig,
    ):
        """
        Initialize the multimodal dataset.
        
        Args:
            config: Dataset configuration
        """
        self.config = config
        self.path = Path(config.path)
        self.entries: List[MultimodalEntry] = []
        self.stats: DatasetStats = DatasetStats()
        self._loaded: bool = False
    
    def load(self) -> "MultimodalDataset":
        """Load the dataset."""
        if self._loaded:
            return self
        
        logger.info(f"Loading dataset: {self.config.name} from {self.path}")
        
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.path}")
        
        # Load based on format
        if self.config.format == "jsonl":
            self._load_jsonl()
        elif self.config.format == "json":
            self._load_json()
        elif self.config.format == "csv":
            self._load_csv()
        else:
            raise ValueError(f"Unsupported format: {self.config.format}")
        
        # Update stats
        self._update_stats()
        self._loaded = True
        
        logger.info(f"Loaded {len(self.entries)} entries from {self.path}")
        return self
    
    def _load_jsonl(self) -> None:
        """Load from JSON Lines format."""
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                entry = self._parse_entry(json.loads(line))
                if entry:
                    self.entries.append(entry)
    
    def _load_json(self) -> None:
        """Load from JSON format."""
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    entry = self._parse_entry(item)
                    if entry:
                        self.entries.append(entry)
            elif isinstance(data, dict):
                entry = self._parse_entry(data)
                if entry:
                    self.entries.append(entry)
    
    def _load_csv(self) -> None:
        """Load from CSV format."""
        import pandas as pd
        
        df = pd.read_csv(self.path)
        for _, row in df.iterrows():
            entry = self._parse_entry(row.to_dict())
            if entry:
                self.entries.append(entry)
    
    def _parse_entry(self, data: Dict[str, Any]) -> Optional[MultimodalEntry]:
        """Parse a single entry from raw data."""
        try:
            # Extract fields
            text = data.get(self.config.text_field)
            image = data.get(self.config.image_field)
            audio = data.get(self.config.audio_field)
            video = data.get(self.config.video_field)
            speech = data.get(self.config.speech_field)
            instruction = data.get(self.config.instruction_field)
            output = data.get(self.config.output_field)
            domain_str = data.get(self.config.domain_field, "general")
            language = data.get(self.config.language_field, "en")
            
            # Convert domain string to enum
            try:
                domain = DomainType(domain_str.lower())
            except ValueError:
                domain = DomainType.GENERAL
            
            # Create multimodal data
            multimodal_data = MultimodalData(
                text=text,
                image=image,
                audio=audio,
                video=video,
                speech=speech,
            )
            
            # Create entry
            entry = MultimodalEntry(
                id=str(len(self.entries)),
                data=multimodal_data,
                instruction=instruction,
                expected_output=output,
                domain=domain,
                language=language,
                source=str(self.path),
            )
            
            return entry
            
        except Exception as e:
            logger.warning(f"Failed to parse entry: {e}")
            return None
    
    def _update_stats(self) -> None:
        """Update dataset statistics."""
        self.stats.total_entries = len(self.entries)
        
        text_lengths = []
        for entry in self.entries:
            # Count modalities
            for modality in entry.data.modalities:
                self.stats.entries_by_modality[modality] = \
                    self.stats.entries_by_modality.get(modality, 0) + 1
            
            # Count domain
            self.stats.entries_by_domain[entry.domain] = \
                self.stats.entries_by_domain.get(entry.domain, 0) + 1
            
            # Count text length
            if entry.data.text:
                text_lengths.append(len(entry.data.text))
            
            # Count multimodal vs text-only
            if len(entry.data.modalities) > 1:
                self.stats.num_multimodal += 1
            else:
                self.stats.num_text_only += 1
        
        if text_lengths:
            self.stats.avg_text_length = sum(text_lengths) / len(text_lengths)
            self.stats.min_text_length = min(text_lengths)
            self.stats.max_text_length = max(text_lengths)
    
    def __iter__(self) -> Iterator[MultimodalEntry]:
        """Iterate over entries."""
        return iter(self.entries)
    
    def __len__(self) -> int:
        """Get number of entries."""
        return len(self.entries)
    
    def __getitem__(self, index: int) -> MultimodalEntry:
        """Get entry by index."""
        return self.entries[index]
    
    def sample(self, size: int, seed: int = 42) -> List[MultimodalEntry]:
        """Sample entries from the dataset."""
        import random
        random.seed(seed)
        return random.sample(self.entries, min(size, len(self.entries)))
    
    def filter_by_modality(self, modality: ModalityType) -> List[MultimodalEntry]:
        """Filter entries by modality."""
        return [
            entry for entry in self.entries 
            if modality in entry.data.modalities
        ]
    
    def filter_by_domain(self, domain: DomainType) -> List[MultimodalEntry]:
        """Filter entries by domain."""
        return [entry for entry in self.entries if entry.domain == domain]
    
    def filter_by_text_length(
        self,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> List[MultimodalEntry]:
        """Filter entries by text length."""
        results = []
        for entry in self.entries:
            if entry.data.text:
                length = len(entry.data.text)
                if (min_length is None or length >= min_length) and \
                   (max_length is None or length <= max_length):
                    results.append(entry)
        return results
    
    def chunk(self, chunk_size: int) -> List[List[MultimodalEntry]]:
        """Split dataset into chunks."""
        return [
            self.entries[i:i + chunk_size] 
            for i in range(0, len(self.entries), chunk_size)
        ]
    
    def get_batch(self, start: int, end: int) -> List[MultimodalEntry]:
        """Get a batch of entries."""
        return self.entries[start:end]
    
    def to_list(self) -> List[Dict[str, Any]]:
        """Convert to list of dictionaries."""
        return [entry.__dict__ for entry in self.entries]
    
    def save(self, path: Union[str, Path], format: str = "jsonl") -> None:
        """Save dataset to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "jsonl":
            with open(path, "w", encoding="utf-8") as f:
                for entry in self.entries:
                    f.write(json.dumps(self._entry_to_dict(entry)) + "\n")
        elif format == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump([self._entry_to_dict(e) for e in self.entries], f)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _entry_to_dict(self, entry: MultimodalEntry) -> Dict[str, Any]:
        """Convert entry to dictionary for serialization."""
        data_dict = {
            "text": entry.data.text,
            "image": entry.data.image,
            "audio": entry.data.audio,
            "video": entry.data.video,
            "speech": entry.data.speech,
        }
        return {
            "id": entry.id,
            "data": data_dict,
            "instruction": entry.instruction,
            "expected_output": entry.expected_output,
            "domain": entry.domain.value,
            "language": entry.language,
            "importance": entry.importance,
            "novelty": entry.novelty,
            "difficulty": entry.difficulty,
            "source": entry.source,
            "version": entry.version,
        }


class DataIngestion:
    """
    Handles ingestion of multiple datasets.
    Provides unified interface for loading and managing datasets.
    """
    
    def __init__(
        self,
        configs: Optional[List[DatasetConfig]] = None,
    ):
        """
        Initialize data ingestion.
        
        Args:
            configs: List of dataset configurations
        """
        self.configs = configs or []
        self.datasets: Dict[str, MultimodalDataset] = {}
        self._loaded: bool = False
    
    def add_config(self, config: DatasetConfig) -> None:
        """Add a dataset configuration."""
        self.configs.append(config)
    
    def load_all(self) -> "DataIngestion":
        """Load all configured datasets."""
        for config in self.configs:
            dataset = MultimodalDataset(config).load()
            self.datasets[config.name] = dataset
        self._loaded = True
        return self
    
    def load(self, name: str) -> Optional[MultimodalDataset]:
        """Load a specific dataset by name."""
        for config in self.configs:
            if config.name == name:
                dataset = MultimodalDataset(config).load()
                self.datasets[name] = dataset
                return dataset
        return None
    
    def get(self, name: str) -> Optional[MultimodalDataset]:
        """Get a loaded dataset by name."""
        return self.datasets.get(name)
    
    def get_all_entries(self) -> List[MultimodalEntry]:
        """Get all entries from all datasets."""
        all_entries = []
        for dataset in self.datasets.values():
            all_entries.extend(dataset.entries)
        return all_entries
    
    def get_stats(self) -> Dict[str, DatasetStats]:
        """Get statistics for all datasets."""
        return {name: dataset.stats for name, dataset in self.datasets.items()}
    
    def unload(self, name: str) -> bool:
        """Unload a dataset."""
        if name in self.datasets:
            del self.datasets[name]
            return True
        return False
    
    def unload_all(self) -> None:
        """Unload all datasets."""
        self.datasets.clear()
        self._loaded = False
    
    def merge_datasets(self, names: List[str]) -> MultimodalDataset:
        """Merge multiple datasets into one."""
        merged_entries = []
        for name in names:
            dataset = self.datasets.get(name)
            if dataset:
                merged_entries.extend(dataset.entries)
        
        # Create a new dataset with merged entries
        merged_config = DatasetConfig(
            name="merged",
            path="",
            format="jsonl",
        )
        merged_dataset = MultimodalDataset(merged_config)
        merged_dataset.entries = merged_entries
        merged_dataset._loaded = True
        merged_dataset._update_stats()
        
        return merged_dataset
