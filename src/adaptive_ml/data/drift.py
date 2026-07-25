"""
Drift Detection for Adaptive ML Framework.
Detects statistical, semantic, and concept drift in data streams.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from adaptive_ml.core.config import AdaptiveMLConfig
from adaptive_ml.core.types import DriftResult, DriftType


@dataclass
class DriftWindow:
    """A sliding window of data for drift detection."""

    data: List[Any]  # List of data points (embeddings, features, etc.)
    labels: Optional[List[Any]] = None  # Optional labels
    timestamps: Optional[List[float]] = None  # Optional timestamps
    max_size: int = 1000
    
    def add(self, data_point: Any, label: Optional[Any] = None, timestamp: Optional[float] = None) -> None:
        """Add a new data point to the window."""
        self.data.append(data_point)
        if self.labels is not None:
            self.labels.append(label)
        if self.timestamps is not None:
            self.timestamps.append(timestamp)
        
        # Trim if exceeding max size
        if len(self.data) > self.max_size:
            self.data = self.data[-self.max_size:]
            if self.labels is not None:
                self.labels = self.labels[-self.max_size:]
            if self.timestamps is not None:
                self.timestamps = self.timestamps[-self.max_size:]
    
    def get_data(self) -> np.ndarray:
        """Get data as numpy array."""
        return np.array(self.data)
    
    def clear(self) -> None:
        """Clear the window."""
        self.data = []
        if self.labels is not None:
            self.labels = []
        if self.timestamps is not None:
            self.timestamps = []


class DriftDetector:
    """
    Detects drift in data streams using multiple methods.
    
    Supported drift types:
    - Statistical drift: KS-test, PSI, Wasserstein distance
    - Semantic drift: PCA reconstruction error, embedding similarity
    - Concept drift: Combination of statistical and semantic drift
    
    Usage:
        detector = DriftDetector(config)
        
        # Add reference data (initial distribution)
        for x in reference_data:
            detector.add_reference(x)
        
        # Monitor new data
        for x in new_data:
            result = detector.check_drift(x)
            if result.is_drift:
                print(f"Drift detected: {result.drift_type} with score {result.score}")
    """

    def __init__(
        self,
        config: Optional[AdaptiveMLConfig] = None,
        embedding_fn: Optional[Callable] = None,
    ):
        """
        Initialize DriftDetector.
        
        Args:
            config: AdaptiveMLConfig instance
            embedding_fn: Optional function to compute embeddings from raw data
        """
        self.config = config or AdaptiveMLConfig()
        self.embedding_fn = embedding_fn
        
        # Initialize windows
        self.reference_window = DriftWindow(
            data=[],
            max_size=self.config.drift.reference_size,
        )
        self.current_window = DriftWindow(
            data=[],
            max_size=self.config.drift.window_size,
        )
        
        # PCA for semantic drift
        self.pca = None
        self.pca_components = min(10, self.config.drift.reference_size)
        
        # Track statistics for PSI
        self.reference_stats: Dict[str, Dict] = {}

    def add_reference(self, data: Any, label: Optional[Any] = None) -> None:
        """Add data to the reference window."""
        if self.embedding_fn is not None:
            data = self.embedding_fn(data)
        
        self.reference_window.add(data, label)
        
        # Update PCA if we have enough data
        if len(self.reference_window.data) >= self.pca_components:
            self._fit_pca()

    def add_current(self, data: Any, label: Optional[Any] = None) -> None:
        """Add data to the current window."""
        if self.embedding_fn is not None:
            data = self.embedding_fn(data)
        
        self.current_window.add(data, label)

    def check_drift(self, data: Any = None, label: Optional[Any] = None) -> DriftResult:
        """
        Check for drift in the current window compared to reference.
        
        Args:
            data: Optional new data point to add and check
            label: Optional label for the new data point
        
        Returns:
            DriftResult with drift type, score, and whether drift was detected
        """
        if data is not None:
            self.add_current(data, label)
        
        # Check if we have enough data
        if len(self.reference_window.data) < 10 or len(self.current_window.data) < 10:
            return DriftResult(
                drift_type=DriftType.NONE,
                score=0.0,
                threshold=self.config.drift.concept_threshold,
                is_drift=False,
                details={"message": "Insufficient data for drift detection"},
            )
        
        # Run all drift tests
        statistical_result = self._check_statistical_drift()
        semantic_result = self._check_semantic_drift()
        
        # Concept drift is combination of both
        concept_score = (statistical_result.score + semantic_result.score) / 2
        concept_threshold = self.config.drift.concept_threshold
        
        # Determine overall result
        if statistical_result.is_drift and semantic_result.is_drift:
            drift_type = DriftType.CONCEPT
            score = concept_score
            threshold = concept_threshold
        elif statistical_result.is_drift:
            drift_type = DriftType.STATISTICAL
            score = statistical_result.score
            threshold = self.config.drift.statistical_threshold
        elif semantic_result.is_drift:
            drift_type = DriftType.SEMANTIC
            score = semantic_result.score
            threshold = self.config.drift.semantic_threshold
        else:
            drift_type = DriftType.NONE
            score = concept_score
            threshold = concept_threshold
        
        is_drift = score > threshold
        
        return DriftResult(
            drift_type=drift_type,
            score=score,
            threshold=threshold,
            is_drift=is_drift,
            details={
                "statistical": statistical_result.details,
                "semantic": semantic_result.details,
            },
        )

    def _check_statistical_drift(self) -> DriftResult:
        """Check for statistical drift using the configured test."""
        ref_data = self.reference_window.get_data()
        curr_data = self.current_window.get_data()
        
        test = self.config.drift.statistical_test.lower()
        threshold = self.config.drift.statistical_threshold
        
        if test == "ks":
            score, p_value = self._ks_test(ref_data, curr_data)
        elif test == "psi":
            score = self._psi_test(ref_data, curr_data)
            p_value = 0.0  # PSI doesn't have p-value
        elif test == "wasserstein":
            score = self._wasserstein_distance(ref_data, curr_data)
            p_value = 0.0  # Wasserstein doesn't have p-value
        else:
            raise ValueError(f"Unknown statistical test: {test}")
        
        # For KS test, use p-value; for others, use score
        if test == "ks":
            is_drift = p_value < threshold
        else:
            is_drift = score > threshold
        
        return DriftResult(
            drift_type=DriftType.STATISTICAL,
            score=score if test != "ks" else 1 - p_value,
            threshold=threshold,
            is_drift=is_drift,
            details={
                "test": test,
                "p_value": p_value if test == "ks" else None,
                "score": score,
            },
        )

    def _check_semantic_drift(self) -> DriftResult:
        """Check for semantic drift using PCA reconstruction error."""
        ref_data = self.reference_window.get_data()
        curr_data = self.current_window.get_data()
        
        threshold = self.config.drift.semantic_threshold
        
        # Fit PCA on reference data
        if self.pca is None:
            self._fit_pca()
        
        if self.pca is None:
            return DriftResult(
                drift_type=DriftType.SEMANTIC,
                score=0.0,
                threshold=threshold,
                is_drift=False,
                details={"message": "PCA not fitted"},
            )
        
        # Compute reconstruction error for both windows
        ref_error = self._reconstruction_error(ref_data)
        curr_error = self._reconstruction_error(curr_data)
        
        # Score is the relative increase in error
        if ref_error > 0:
            score = (curr_error - ref_error) / ref_error
        else:
            score = curr_error
        
        is_drift = score > threshold
        
        return DriftResult(
            drift_type=DriftType.SEMANTIC,
            score=score,
            threshold=threshold,
            is_drift=is_drift,
            details={
                "ref_error": float(ref_error),
                "curr_error": float(curr_error),
                "relative_increase": float(score),
            },
        )

    def _fit_pca(self) -> None:
        """Fit PCA on reference data."""
        ref_data = self.reference_window.get_data()
        
        if len(ref_data) < self.pca_components:
            return
        
        try:
            self.pca = PCA(n_components=self.pca_components)
            self.pca.fit(ref_data)
        except Exception as e:
            print(f"Warning: Failed to fit PCA: {e}")
            self.pca = None

    def _reconstruction_error(self, data: np.ndarray) -> float:
        """Compute PCA reconstruction error."""
        if self.pca is None:
            return 0.0
        
        try:
            reconstructed = self.pca.inverse_transform(self.pca.transform(data))
            error = np.mean(np.linalg.norm(data - reconstructed, axis=1))
            return float(error)
        except Exception:
            return 0.0

    def _ks_test(self, ref_data: np.ndarray, curr_data: np.ndarray) -> Tuple[float, float]:
        """
        Kolmogorov-Smirnov test for statistical drift.
        
        Returns:
            Tuple of (KS statistic, p-value)
        """
        # Flatten data for KS test
        ref_flat = ref_data.flatten()
        curr_flat = curr_data.flatten()
        
        try:
            stat, p_value = stats.ks_2samp(ref_flat, curr_flat)
            return float(stat), float(p_value)
        except Exception:
            return 0.0, 1.0

    def _psi_test(self, ref_data: np.ndarray, curr_data: np.ndarray) -> float:
        """
        Population Stability Index (PSI) for statistical drift.
        
        Returns:
            PSI value (higher = more drift)
        """
        # Flatten data
        ref_flat = ref_data.flatten()
        curr_flat = curr_data.flatten()
        
        try:
            # Bin data into 10 bins
            ref_hist, bin_edges = np.histogram(ref_flat, bins=10)
            curr_hist, _ = np.histogram(curr_flat, bins=bin_edges)
            
            # Add small constant to avoid division by zero
            ref_hist = ref_hist + 1e-10
            curr_hist = curr_hist + 1e-10
            
            # Compute PSI
            total_ref = np.sum(ref_hist)
            total_curr = np.sum(curr_hist)
            
            psi = 0.0
            for i in range(len(ref_hist)):
                p_ref = ref_hist[i] / total_ref
                p_curr = curr_hist[i] / total_curr
                psi += (p_curr - p_ref) * np.log(p_curr / p_ref)
            
            return float(psi)
        except Exception:
            return 0.0

    def _wasserstein_distance(self, ref_data: np.ndarray, curr_data: np.ndarray) -> float:
        """
        Wasserstein distance (Earth Mover's Distance) for statistical drift.
        
        Returns:
            Wasserstein distance
        """
        # Flatten data
        ref_flat = ref_data.flatten()
        curr_flat = curr_data.flatten()
        
        try:
            return float(stats.wasserstein_distance(ref_flat, curr_flat))
        except Exception:
            return 0.0

    def reset(self) -> None:
        """Reset all windows and statistics."""
        self.reference_window.clear()
        self.current_window.clear()
        self.pca = None
        self.reference_stats = {}

    def set_embedding_fn(self, embedding_fn: Callable) -> None:
        """Set the embedding function for semantic drift detection."""
        self.embedding_fn = embedding_fn

    def get_window_stats(self) -> Dict[str, Any]:
        """Get statistics about the current windows."""
        return {
            "reference_size": len(self.reference_window.data),
            "current_size": len(self.current_window.data),
            "pca_fitted": self.pca is not None,
        }
