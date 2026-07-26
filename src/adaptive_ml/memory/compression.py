"""
Memory Compression for Adaptive ML Framework.
Implements FAISS-based compression for efficient similarity search in large replay buffers.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from adaptive_ml.core.config import AdaptiveMLConfig


@dataclass
class CompressionStats:
    """Statistics for memory compression."""

    compression_ratio: float = 1.0
    index_size: int = 0
    original_size: int = 0
    memory_savings: float = 0.0
    search_time_ms: float = 0.0
    build_time_ms: float = 0.0


class MemoryCompressor:
    """
    Compresses memory embeddings using FAISS for efficient similarity search.
    
    Supports multiple compression methods:
    - IVF (Inverted File): Partitions vectors into Voronoi cells
    - PQ (Product Quantization): Compresses vectors using product quantization
    - IVF_PQ: Combines IVF and PQ for better performance
    - HNSW: Hierarchical Navigable Small World graphs
    
    Compression reduces memory usage and speeds up similarity search,
    which is crucial for large replay buffers in continual learning.
    """

    def __init__(
        self,
        config: Optional[AdaptiveMLConfig] = None,
        dimension: int = 768,
        method: str = "ivf",
        nlist: int = 100,
        nprobe: int = 10,
        m: int = 8,
        nbits: int = 8,
        use_gpu: bool = False,
    ):
        """
        Initialize MemoryCompressor.
        
        Args:
            config: AdaptiveMLConfig instance
            dimension: Embedding dimension
            method: Compression method ("ivf", "pq", "ivf_pq", "hnsw")
            nlist: Number of IVF clusters
            nprobe: Number of probes for IVF
            m: PQ codebook size
            nbits: PQ quantization bits
            use_gpu: Use GPU for FAISS
        """
        self.config = config or AdaptiveMLConfig()
        self.dimension = dimension
        self.method = method
        self.nlist = nlist
        self.nprobe = nprobe
        self.m = m
        self.nbits = nbits
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # FAISS index
        self.index = None
        self._build_index()
        
        # Original vectors (for comparison)
        self.original_vectors: List[np.ndarray] = []
        
        # Statistics
        self.stats = CompressionStats()

    def _build_index(self) -> None:
        """Build FAISS index based on configuration."""
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "FAISS library not found. Please install with: "
                "pip install faiss-cpu or faiss-gpu"
            )
        
        if self.method == "ivf":
            # IVF index
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
            
        elif self.method == "pq":
            # PQ index
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFPQ(quantizer, self.dimension, self.nlist, self.m, self.nbits)
            
        elif self.method == "ivf_pq":
            # IVF + PQ
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFPQ(quantizer, self.dimension, self.nlist, self.m, self.nbits)
            
        elif self.method == "hnsw":
            # HNSW index
            self.index = faiss.IndexHNSWFlat(self.dimension, self.m)
            
        else:
            # Default: Flat L2
            self.index = faiss.IndexFlatL2(self.dimension)
        
        # Set number of probes for IVF
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = self.nprobe

    def add_vectors(self, vectors: List[np.ndarray]) -> None:
        """
        Add vectors to the compressed index.
        
        Args:
            vectors: List of vectors to add (each vector is a numpy array)
        """
        if not vectors:
            return
        
        # Convert to numpy array
        vectors_array = np.array(vectors, dtype=np.float32)
        
        # Store original vectors
        self.original_vectors.extend(vectors)
        
        # Add to index
        if not self.index.ntotal:
            # First vectors: train the index if needed
            if hasattr(self.index, "train") and hasattr(self.index, "is_trained"):
                # For IVF, we need at least nlist vectors to train
                if vectors_array.shape[0] >= getattr(self.index, "nlist", 1):
                    self.index.train(vectors_array)
                else:
                    # Not enough vectors to train, wait until we have more
                    # For now, use flat index - train with available vectors
                    if vectors_array.shape[0] > 0:
                        self.index.train(vectors_array)
        
        # Check if index is trained before adding
        if hasattr(self.index, "is_trained") and not self.index.is_trained:
            # Train with current vectors if we have enough
            if self.index.ntotal >= getattr(self.index, "nlist", 1):
                # Get all vectors so far
                all_vectors = np.array(self.original_vectors, dtype=np.float32)
                if all_vectors.shape[0] >= getattr(self.index, "nlist", 1):
                    self.index.train(all_vectors)
            else:
                # Not enough vectors yet, skip this batch
                self.original_vectors.extend(vectors)
                return
        
        self.index.add(vectors_array)
        
        # Update statistics
        self.stats.original_size = len(self.original_vectors)
        self.stats.index_size = self.index.ntotal
        self._update_compression_stats()

    def search(
        self,
        query_vectors: List[np.ndarray],
        k: int = 5,
    ) -> Tuple[List[List[int]], List[List[float]]]:
        """
        Search for nearest neighbors in the compressed index.
        
        Args:
            query_vectors: List of query vectors
            k: Number of nearest neighbors to return
            
        Returns:
            Tuple of (indices, distances) for each query
        """
        if not query_vectors or self.index.ntotal == 0:
            return [], []
        
        # Convert to numpy array
        query_array = np.array(query_vectors, dtype=np.float32)
        
        # Search
        import time
        start_time = time.time()
        
        distances, indices = self.index.search(query_array, k)
        
        search_time = (time.time() - start_time) * 1000  # ms
        self.stats.search_time_ms = search_time
        
        # Convert to lists
        indices_list = [list(idx) for idx in indices]
        distances_list = [list(dist) for dist in distances]
        
        return indices_list, distances_list

    def get_vectors(self, indices: List[int]) -> List[np.ndarray]:
        """
        Get original vectors by indices.
        
        Args:
            indices: List of indices
            
        Returns:
            List of vectors
        """
        return [self.original_vectors[i] for i in indices if i < len(self.original_vectors)]

    def _update_compression_stats(self) -> None:
        """Update compression statistics."""
        if self.index.ntotal > 0:
            # Estimate memory usage
            if self.method == "ivf":
                # IVF: nlist * dimension * 4 bytes for centroids
                # + ntotal * dimension * 4 bytes for vectors
                centroid_memory = self.nlist * self.dimension * 4
                vector_memory = self.index.ntotal * self.dimension * 4
                total_memory = centroid_memory + vector_memory
                original_memory = len(self.original_vectors) * self.dimension * 4
                
            elif self.method == "pq":
                # PQ: nlist * m * dimension/m * 4 bytes for codebooks
                # + ntotal * m * nbits/8 bytes for codes
                codebook_memory = self.nlist * self.m * (self.dimension // self.m) * 4
                code_memory = self.index.ntotal * self.m * (self.nbits // 8)
                total_memory = codebook_memory + code_memory
                original_memory = len(self.original_vectors) * self.dimension * 4
                
            elif self.method == "ivf_pq":
                # IVF_PQ: similar to PQ but with IVF structure
                codebook_memory = self.nlist * self.m * (self.dimension // self.m) * 4
                code_memory = self.index.ntotal * self.m * (self.nbits // 8)
                centroid_memory = self.nlist * self.dimension * 4
                total_memory = codebook_memory + code_memory + centroid_memory
                original_memory = len(self.original_vectors) * self.dimension * 4
                
            elif self.method == "hnsw":
                # HNSW: approximately ntotal * m * dimension * 4 bytes
                total_memory = self.index.ntotal * self.m * self.dimension * 4
                original_memory = len(self.original_vectors) * self.dimension * 4
                
            else:  # flat
                total_memory = self.index.ntotal * self.dimension * 4
                original_memory = len(self.original_vectors) * self.dimension * 4
            
            self.stats.compression_ratio = original_memory / total_memory if total_memory > 0 else 1.0
            self.stats.memory_savings = (1.0 - 1.0 / self.stats.compression_ratio) * 100

    def clear(self) -> None:
        """Clear all vectors from the index."""
        self.index.reset()
        self.original_vectors = []
        self.stats = CompressionStats()

    def get_stats(self) -> CompressionStats:
        """Get compression statistics."""
        self._update_compression_stats()
        return self.stats

    def __len__(self) -> int:
        """Get number of vectors in the index."""
        return self.index.ntotal

    def __repr__(self) -> str:
        return (
            f"MemoryCompressor(method={self.method}, "
            f"dimension={self.dimension}, "
            f"size={len(self)}, "
            f"compression_ratio={self.stats.compression_ratio:.2f}x)"
        )


class CompressedReplayBuffer:
    """
    Replay buffer with built-in compression for efficient similarity search.
    
    This extends the basic replay buffer with FAISS-based compression,
    enabling efficient nearest neighbor search even with large buffers.
    """

    def __init__(
        self,
        config: Optional[AdaptiveMLConfig] = None,
        embedding_dim: int = 768,
        method: str = "ivf",
        nlist: int = 100,
        nprobe: int = 10,
        m: int = 8,
        nbits: int = 8,
    ):
        """
        Initialize CompressedReplayBuffer.
        
        Args:
            config: AdaptiveMLConfig instance (optional)
            embedding_dim: Dimension of embeddings
            method: Compression method
            nlist: Number of IVF clusters
            nprobe: Number of probes
            m: PQ codebook size
            nbits: PQ quantization bits
        """
        self.config = config
        self.embedding_dim = embedding_dim
        
        # Create compressor
        self.compressor = MemoryCompressor(
            config=config,
            dimension=embedding_dim,
            method=method,
            nlist=nlist,
            nprobe=nprobe,
            m=m,
            nbits=nbits,
        )
        
        # Storage for embeddings and metadata
        self.embeddings: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []

    def add_embedding(
        self,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Add an embedding to the compressed buffer.
        
        Args:
            embedding: Embedding vector
            metadata: Optional metadata
            
        Returns:
            Index of the added embedding
        """
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embedding.shape[0]}")
        
        # Store embedding and metadata
        idx = len(self.embeddings)
        self.embeddings.append(embedding)
        self.metadata.append(metadata or {})
        
        # Add to compressor
        self.compressor.add_vectors([embedding])
        
        return idx

    def add_embeddings(
        self,
        embeddings: List[np.ndarray],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> List[int]:
        """
        Add multiple embeddings to the compressed buffer.
        
        Args:
            embeddings: List of embedding vectors
            metadata_list: Optional list of metadata
            
        Returns:
            List of indices
        """
        indices = []
        for i, embedding in enumerate(embeddings):
            metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else None
            idx = self.add_embedding(embedding, metadata)
            indices.append(idx)
        return indices

    def search_similar(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[int], List[float], List[Dict[str, Any]]]:
        """
        Search for similar embeddings in the buffer.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of nearest neighbors
            filter_metadata: Optional metadata filter
            
        Returns:
            Tuple of (indices, distances, metadata)
        """
        # Search in compressed index
        indices, distances = self.compressor.search([query_embedding], k)
        
        if not indices or not indices[0]:
            return [], [], []
        
        # Get metadata for results
        result_indices = indices[0]
        result_distances = distances[0]
        result_metadata = [self.metadata[i] for i in result_indices if i < len(self.metadata)]
        
        # Apply metadata filter if provided
        if filter_metadata:
            filtered_indices = []
            filtered_distances = []
            filtered_metadata = []
            
            for idx, dist, meta in zip(result_indices, result_distances, result_metadata):
                # Check if metadata matches filter
                match = True
                for key, value in filter_metadata.items():
                    if key not in meta or meta[key] != value:
                        match = False
                        break
                
                if match:
                    filtered_indices.append(idx)
                    filtered_distances.append(dist)
                    filtered_metadata.append(meta)
            
            return filtered_indices, filtered_distances, filtered_metadata
        
        return result_indices, result_distances, result_metadata

    def get_embedding(self, idx: int) -> Optional[np.ndarray]:
        """Get embedding by index."""
        if 0 <= idx < len(self.embeddings):
            return self.embeddings[idx]
        return None

    def get_metadata(self, idx: int) -> Optional[Dict[str, Any]]:
        """Get metadata by index."""
        if 0 <= idx < len(self.metadata):
            return self.metadata[idx]
        return None

    def clear(self) -> None:
        """Clear all embeddings from the buffer."""
        self.embeddings = []
        self.metadata = []
        self.compressor.clear()

    def get_stats(self) -> CompressionStats:
        """Get compression statistics."""
        return self.compressor.get_stats()

    def __len__(self) -> int:
        """Get number of embeddings in the buffer."""
        return len(self.embeddings)

    def __repr__(self) -> str:
        return (
            f"CompressedReplayBuffer(size={len(self)}, "
            f"dimension={self.embedding_dim}, "
            f"compression={self.compressor.method})"
        )
