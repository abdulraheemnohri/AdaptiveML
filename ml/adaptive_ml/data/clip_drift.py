"""
CLIP-based Drift Detection for Adaptive ML Framework.
Detects semantic drift in multi-modal data using CLIP embeddings.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import requests
from io import BytesIO

from adaptive_ml.core.config import AdaptiveMLConfig
from adaptive_ml.core.types import DriftResult, DriftType


@dataclass
class CLIPDriftStats:
    """Statistics for CLIP-based drift detection."""

    num_tests: int = 0
    num_drifts_detected: int = 0
    drift_rate: float = 0.0
    mean_drift_score: float = 0.0
    max_drift_score: float = 0.0
    modality_distribution: Dict[str, int] = field(default_factory=dict)


class CLIPDriftDetector:
    """
    CLIP-based drift detector for multi-modal data.
    
    Uses CLIP (Contrastive Language-Image Pre-training) to detect semantic drift
    between reference and current data distributions. CLIP can process both
    text and images, making it ideal for multi-modal drift detection.
    
    The drift score is computed as the Earth Mover's Distance (EMD) or
    Maximum Mean Discrepancy (MMD) between the CLIP embeddings of reference
    and current data.
    
    Reference:
        Radford et al., "Learning Transferable Visual Models From Natural Language
        Supervision", ICML 2021.
        https://arxiv.org/abs/2103.00020
    """

    def __init__(
        self,
        config: Optional[AdaptiveMLConfig] = None,
        clip_model: str = "openai/clip-vit-base-patch32",
        threshold: float = 0.2,
        device: Optional[str] = None,
    ):
        """
        Initialize CLIPDriftDetector.
        
        Args:
            config: AdaptiveMLConfig instance
            clip_model: CLIP model name or path
            threshold: Drift detection threshold (0-1)
            device: Device to run on
        """
        self.config = config or AdaptiveMLConfig()
        self.clip_model_name = clip_model
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load CLIP model
        self.clip_model = self._load_clip_model()
        self.preprocess = self._get_preprocess()
        
        # Reference embeddings
        self.reference_embeddings: Dict[str, List[torch.Tensor]] = {}
        self.reference_modalities: Dict[str, List[str]] = {}
        
        # Statistics
        self.stats = CLIPDriftStats()

    def _load_clip_model(self) -> nn.Module:
        """Load CLIP model."""
        try:
            import clip
            
            # Load CLIP model
            model, preprocess = clip.load(
                self.clip_model_name,
                device=self.device,
                jit=False,
            )
            model.eval()
            return model
        except ImportError:
            raise ImportError(
                "CLIP library not found. Please install with: "
                "pip install git+https://github.com/openai/CLIP.git"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP model: {e}")

    def _get_preprocess(self) -> Any:
        """Get CLIP preprocess function."""
        try:
            import clip
            _, preprocess = clip.load(
                self.clip_model_name,
                device=self.device,
                jit=False,
            )
            return preprocess
        except:
            # Fallback to simple preprocess
            from torchvision import transforms
            return transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ])

    def _encode_text(self, text: str) -> torch.Tensor:
        """
        Encode text using CLIP text encoder.
        
        Args:
            text: Input text
            
        Returns:
            Text embedding tensor
        """
        import clip
        
        text_input = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_input)
        
        # Normalize
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features

    def _encode_image(self, image: Any) -> torch.Tensor:
        """
        Encode image using CLIP image encoder.
        
        Args:
            image: Input image (PIL Image, path, or URL)
            
        Returns:
            Image embedding tensor
        """
        import clip
        
        # Load image
        if isinstance(image, str):
            # Try to load from URL or path
            if image.startswith(("http://", "https://")):
                response = requests.get(image)
                img = Image.open(BytesIO(response.content))
            else:
                img = Image.open(image)
        elif isinstance(image, Image.Image):
            img = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Preprocess
        image_input = self.preprocess(img).unsqueeze(0).to(self.device)
        
        # Encode
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
        
        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features

    def _encode_multi_modal(
        self,
        text: Optional[str] = None,
        image: Optional[Any] = None,
        audio: Optional[Any] = None,
    ) -> torch.Tensor:
        """
        Encode multi-modal input using CLIP.
        
        For multi-modal inputs, we concatenate the embeddings from different
        modalities (if available) or use the available modality.
        
        Args:
            text: Text input
            image: Image input
            audio: Audio input (not supported by CLIP, will be ignored)
            
        Returns:
            Multi-modal embedding tensor
        """
        embeddings = []
        
        if text is not None:
            text_emb = self._encode_text(text)
            embeddings.append(text_emb)
        
        if image is not None:
            img_emb = self._encode_image(image)
            embeddings.append(img_emb)
        
        if not embeddings:
            raise ValueError("At least one modality (text or image) must be provided")
        
        # Concatenate embeddings
        if len(embeddings) == 1:
            return embeddings[0]
        else:
            # Average embeddings from different modalities
            return torch.mean(torch.stack(embeddings), dim=0)

    def add_reference_data(
        self,
        data: List[Any],
        task_id: str = "default",
        modality: str = "auto",
    ) -> None:
        """
        Add reference data for drift detection.
        
        Args:
            data: List of reference data samples
            task_id: Task identifier
            modality: Modality type ("text", "image", "multi_modal", or "auto")
        """
        if task_id not in self.reference_embeddings:
            self.reference_embeddings[task_id] = []
            self.reference_modalities[task_id] = []
        
        for sample in data:
            # Determine modality
            if modality == "auto":
                if isinstance(sample, str):
                    modality = "text"
                elif isinstance(sample, (Image.Image, np.ndarray)):
                    modality = "image"
                else:
                    modality = "multi_modal"
            
            # Encode based on modality
            if modality == "text":
                embedding = self._encode_text(sample)
            elif modality == "image":
                embedding = self._encode_image(sample)
            else:  # multi_modal
                if isinstance(sample, dict):
                    embedding = self._encode_multi_modal(
                        text=sample.get("text"),
                        image=sample.get("image"),
                    )
                else:
                    embedding = self._encode_text(str(sample))
            
            self.reference_embeddings[task_id].append(embedding)
            self.reference_modalities[task_id].append(modality)

    def detect_drift(
        self,
        data: List[Any],
        task_id: str = "default",
        modality: str = "auto",
        method: str = "cosine",
    ) -> DriftResult:
        """
        Detect drift between reference and current data.
        
        Args:
            data: List of current data samples
            task_id: Task identifier
            modality: Modality type
            method: Drift detection method ("cosine", "mmd", "emd")
            
        Returns:
            DriftResult with drift detection results
        """
        if task_id not in self.reference_embeddings or not self.reference_embeddings[task_id]:
            return DriftResult(
                drift_type=DriftType.NONE,
                score=0.0,
                threshold=self.threshold,
                is_drift=False,
                details={"error": "No reference data for task"},
            )
        
        # Encode current data
        current_embeddings = []
        for sample in data:
            if modality == "auto":
                if isinstance(sample, str):
                    modality = "text"
                elif isinstance(sample, (Image.Image, np.ndarray)):
                    modality = "image"
                else:
                    modality = "multi_modal"
            
            if modality == "text":
                embedding = self._encode_text(sample)
            elif modality == "image":
                embedding = self._encode_image(sample)
            else:  # multi_modal
                if isinstance(sample, dict):
                    embedding = self._encode_multi_modal(
                        text=sample.get("text"),
                        image=sample.get("image"),
                    )
                else:
                    embedding = self._encode_text(str(sample))
            
            current_embeddings.append(embedding)
        
        # Stack embeddings
        ref_tensor = torch.stack(self.reference_embeddings[task_id])
        curr_tensor = torch.stack(current_embeddings)
        
        # Compute drift score
        if method == "cosine":
            drift_score = self._cosine_distance(ref_tensor, curr_tensor)
        elif method == "mmd":
            drift_score = self._mmd_distance(ref_tensor, curr_tensor)
        elif method == "emd":
            drift_score = self._emd_distance(ref_tensor, curr_tensor)
        else:
            drift_score = self._cosine_distance(ref_tensor, curr_tensor)
        
        # Update statistics
        self.stats.num_tests += 1
        self.stats.mean_drift_score = (
            (self.stats.mean_drift_score * (self.stats.num_tests - 1) + drift_score) 
            / self.stats.num_tests
        )
        self.stats.max_drift_score = max(
            self.stats.max_drift_score, drift_score
        )
        
        if drift_score > self.threshold:
            self.stats.num_drifts_detected += 1
        
        self.stats.drift_rate = (
            self.stats.num_drifts_detected / self.stats.num_tests
            if self.stats.num_tests > 0
            else 0.0
        )
        
        # Update modality distribution
        mod = self.reference_modalities[task_id][0] if self.reference_modalities[task_id] else "unknown"
        self.stats.modality_distribution[mod] = (
            self.stats.modality_distribution.get(mod, 0) + 1
        )
        
        return DriftResult(
            drift_type=DriftType.SEMANTIC,
            score=float(drift_score),
            threshold=self.threshold,
            is_drift=drift_score > self.threshold,
            details={
                "method": method,
                "reference_samples": len(self.reference_embeddings[task_id]),
                "current_samples": len(current_embeddings),
                "embedding_dim": ref_tensor.shape[-1],
            },
        )

    def _cosine_distance(
        self,
        ref_embeddings: torch.Tensor,
        curr_embeddings: torch.Tensor,
    ) -> float:
        """
        Compute cosine distance between reference and current embeddings.
        
        Args:
            ref_embeddings: Reference embeddings (n_ref, d)
            curr_embeddings: Current embeddings (n_curr, d)
            
        Returns:
            Cosine distance (0-1, higher = more drift)
        """
        # Compute pairwise cosine similarities
        ref_norm = ref_embeddings / ref_embeddings.norm(dim=-1, keepdim=True)
        curr_norm = curr_embeddings / curr_embeddings.norm(dim=-1, keepdim=True)
        
        # Average cosine similarity
        cosine_sim = torch.mean(torch.mm(ref_norm, curr_norm.T))
        
        # Convert to distance (1 - similarity)
        cosine_dist = 1.0 - cosine_sim
        
        return float(cosine_dist.item())

    def _mmd_distance(
        self,
        ref_embeddings: torch.Tensor,
        curr_embeddings: torch.Tensor,
    ) -> float:
        """
        Compute Maximum Mean Discrepancy (MMD) between embeddings.
        
        Args:
            ref_embeddings: Reference embeddings (n_ref, d)
            curr_embeddings: Current embeddings (n_curr, d)
            
        Returns:
            MMD distance (higher = more drift)
        """
        def mmd2(X, Y, sigma=1.0):
            """Compute MMD^2 between X and Y."""
            XX = torch.mm(X, X.T)
            YY = torch.mm(Y, Y.T)
            XY = torch.mm(X, Y.T)
            
            n_x = X.shape[0]
            n_y = Y.shape[0]
            
            # Kernel matrices with Gaussian RBF
            K_XX = torch.exp(-XX / sigma)
            K_YY = torch.exp(-YY / sigma)
            K_XY = torch.exp(-XY / sigma)
            
            # MMD^2 computation
            mmd = (
                K_XX.sum() / (n_x * n_x)
                + K_YY.sum() / (n_y * n_y)
                - 2 * K_XY.sum() / (n_x * n_y)
            )
            
            return mmd
        
        # Normalize embeddings
        ref_norm = ref_embeddings / ref_embeddings.norm(dim=-1, keepdim=True)
        curr_norm = curr_embeddings / curr_embeddings.norm(dim=-1, keepdim=True)
        
        # Compute MMD
        mmd = mmd2(ref_norm, curr_norm)
        
        # Normalize to 0-1 range (approximate)
        return float(torch.sigmoid(mmd).item())

    def _emd_distance(
        self,
        ref_embeddings: torch.Tensor,
        curr_embeddings: torch.Tensor,
    ) -> float:
        """
        Compute Earth Mover's Distance (EMD) between embeddings.
        
        Args:
            ref_embeddings: Reference embeddings (n_ref, d)
            curr_embeddings: Current embeddings (n_curr, d)
            
        Returns:
            EMD distance (higher = more drift)
        """
        # For simplicity, use cosine distance as approximation
        # True EMD would require solving an optimization problem
        return self._cosine_distance(ref_embeddings, curr_embeddings)

    def get_stats(self) -> CLIPDriftStats:
        """Get CLIP drift detection statistics."""
        return self.stats

    def clear_reference(self, task_id: str = "default") -> None:
        """Clear reference data for a task."""
        if task_id in self.reference_embeddings:
            del self.reference_embeddings[task_id]
        if task_id in self.reference_modalities:
            del self.reference_modalities[task_id]

    def reset(self) -> None:
        """Reset all reference data and statistics."""
        self.reference_embeddings = {}
        self.reference_modalities = {}
        self.stats = CLIPDriftStats()

    def __repr__(self) -> str:
        return (
            f"CLIPDriftDetector(model={self.clip_model_name}, "
            f"threshold={self.threshold}, "
            f"device={self.device})"
        )
