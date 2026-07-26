"""
Advanced Evaluation Metrics for Adaptive ML Framework.
Implements perplexity, BLEU, ROUGE, and other advanced metrics for comprehensive evaluation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from adaptive_ml.core.config import AdaptiveMLConfig


@dataclass
class AdvancedMetricsResult:
    """Results from advanced metrics computation."""

    perplexity: Optional[float] = None
    bleu_1: Optional[float] = None
    bleu_2: Optional[float] = None
    bleu_3: Optional[float] = None
    bleu_4: Optional[float] = None
    rouge_1: Optional[float] = None
    rouge_2: Optional[float] = None
    rouge_l: Optional[float] = None
    f1_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    accuracy: Optional[float] = None
    
    # Multi-modal metrics
    vision_accuracy: Optional[float] = None
    audio_accuracy: Optional[float] = None
    
    # Additional details
    details: Dict[str, Any] = field(default_factory=dict)


class PerplexityCalculator:
    """
    Computes perplexity for language models.
    
    Perplexity is a standard metric for evaluating language models:
        PP = exp(L) where L is the average negative log-likelihood
    
    Lower perplexity indicates better model performance.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[AdaptiveMLConfig] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize PerplexityCalculator.
        
        Args:
            model: Language model
            config: AdaptiveMLConfig instance
            device: Device to run on
        """
        self.model = model
        self.config = config or AdaptiveMLConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def compute(
        self,
        dataloader: DataLoader,
        max_length: Optional[int] = None,
    ) -> float:
        """
        Compute perplexity on a dataset.
        
        Args:
            dataloader: DataLoader with input data
            max_length: Maximum sequence length for computation
            
        Returns:
            Perplexity score
        """
        self.model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                    targets = batch[1] if len(batch) > 1 else None
                else:
                    inputs = batch.get("input_ids", batch.get("inputs"))
                    targets = batch.get("labels", batch.get("targets"))
                
                inputs = inputs.to(self.device)
                if targets is not None:
                    targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Get logits
                if isinstance(outputs, torch.Tensor):
                    logits = outputs
                else:
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                
                # Compute loss
                if targets is not None:
                    # Shift targets for causal LM
                    if logits.shape[1] == targets.shape[1]:
                        targets = targets[:, 1:]
                        logits = logits[:, :-1, :]
                    
                    # Compute cross-entropy loss
                    loss = F.cross_entropy(
                        logits.view(-1, logits.shape[-1]),
                        targets.view(-1),
                        reduction="sum",
                    )
                    
                    # Count non-padding tokens
                    num_tokens = (targets != -100).sum().item() if targets.dim() > 1 else targets.shape[0]
                else:
                    # For unsupervised, use all tokens
                    loss = F.cross_entropy(
                        logits.view(-1, logits.shape[-1]),
                        torch.argmax(logits, dim=-1).view(-1),
                        reduction="sum",
                    )
                    num_tokens = logits.shape[0] * logits.shape[1]
                
                total_loss += loss.item()
                total_tokens += num_tokens
        
        # Compute average loss per token
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
            perplexity = float(np.exp(avg_loss))
        else:
            perplexity = float('inf')
        
        return perplexity


class BLEUCalculator:
    """
    Computes BLEU score for text generation evaluation.
    
    BLEU (Bilingual Evaluation Understudy) measures the quality of
    machine-generated text by comparing n-grams with reference text.
    
    Supports BLEU-1 through BLEU-4.
    """

    def __init__(
        self,
        n: int = 4,
        weights: Optional[List[float]] = None,
    ):
        """
        Initialize BLEUCalculator.
        
        Args:
            n: Maximum n-gram order (1-4)
            weights: Custom weights for each n-gram order
        """
        self.n = min(max(n, 1), 4)
        self.weights = weights or [1.0 / self.n] * self.n

    def compute(
        self,
        hypotheses: List[str],
        references: List[List[str]],
    ) -> Dict[str, float]:
        """
        Compute BLEU scores for hypotheses against references.
        
        Args:
            hypotheses: List of generated text strings
            references: List of lists of reference text strings
            
        Returns:
            Dictionary with BLEU-1 through BLEU-n scores
        """
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        except ImportError:
            raise ImportError(
                "NLTK library not found. Please install with: pip install nltk"
            )
        
        # Tokenize
        def tokenize(text):
            return text.lower().split()
        
        results = {}
        
        for i in range(1, self.n + 1):
            weights = [0.0] * self.n
            weights[i - 1] = 1.0
            
            scores = []
            for hyp, refs in zip(hypotheses, references):
                hyp_tokens = tokenize(hyp)
                ref_tokens_list = [tokenize(ref) for ref in refs]
                
                # Use smoothing for short sentences
                smoothing = SmoothingFunction().method1
                
                score = sentence_bleu(
                    ref_tokens_list,
                    hyp_tokens,
                    weights=weights,
                    smoothing_function=smoothing,
                )
                scores.append(score)
            
            results[f"bleu_{i}"] = float(np.mean(scores)) if scores else 0.0
        
        return results


class ROUGECalculator:
    """
    Computes ROUGE score for text generation evaluation.
    
    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures
    the quality of generated text by comparing it to reference text
    using n-gram overlap.
    
    Supports ROUGE-1, ROUGE-2, and ROUGE-L (longest common subsequence).
    """

    def __init__(self):
        """Initialize ROUGECalculator."""
        pass

    def compute(
        self,
        hypotheses: List[str],
        references: List[List[str]],
    ) -> Dict[str, float]:
        """
        Compute ROUGE scores for hypotheses against references.
        
        Args:
            hypotheses: List of generated text strings
            references: List of lists of reference text strings
            
        Returns:
            Dictionary with ROUGE-1, ROUGE-2, and ROUGE-L scores
        """
        try:
            from rouge import Rouge
        except ImportError:
            raise ImportError(
                "rouge library not found. Please install with: pip install rouge"
            )
        
        # Initialize ROUGE
        rouge = Rouge()
        
        # Compute scores
        rouge_1_scores = []
        rouge_2_scores = []
        rouge_l_scores = []
        
        for hyp, refs in zip(hypotheses, references):
            # Use first reference
            ref = refs[0] if refs else ""
            
            scores = rouge.get_scores(hyp, ref)
            rouge_1_scores.append(scores[0]["f"])
            rouge_2_scores.append(scores[0]["f"])
            rouge_l_scores.append(scores[0]["f"])
        
        return {
            "rouge_1": float(np.mean(rouge_1_scores)) if rouge_1_scores else 0.0,
            "rouge_2": float(np.mean(rouge_2_scores)) if rouge_2_scores else 0.0,
            "rouge_l": float(np.mean(rouge_l_scores)) if rouge_l_scores else 0.0,
        }


class F1Calculator:
    """
    Computes F1 score, precision, and recall for classification tasks.
    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        average: str = "macro",
    ):
        """
        Initialize F1Calculator.
        
        Args:
            num_classes: Number of classes (for multi-class)
            average: Averaging method ("micro", "macro", "weighted", None)
        """
        self.num_classes = num_classes
        self.average = average

    def compute(
        self,
        predictions: List[int],
        targets: List[int],
    ) -> Dict[str, float]:
        """
        Compute F1, precision, and recall.
        
        Args:
            predictions: List of predicted class indices
            targets: List of true class indices
            
        Returns:
            Dictionary with F1, precision, recall scores
        """
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        # Convert to numpy arrays
        y_true = np.array(targets)
        y_pred = np.array(predictions)
        
        # Compute scores
        f1 = f1_score(y_true, y_pred, average=self.average, zero_division=0)
        precision = precision_score(y_true, y_pred, average=self.average, zero_division=0)
        recall = recall_score(y_true, y_pred, average=self.average, zero_division=0)
        
        return {
            "f1_score": float(f1),
            "precision": float(precision),
            "recall": float(recall),
        }


class AdvancedEvaluator:
    """
    Comprehensive evaluator with advanced metrics.
    
    Combines multiple evaluation metrics for thorough assessment of
    model performance, especially for language generation tasks.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[AdaptiveMLConfig] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize AdvancedEvaluator.
        
        Args:
            model: Model to evaluate
            config: AdaptiveMLConfig instance
            device: Device to run on
        """
        self.model = model
        self.config = config or AdaptiveMLConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize metric calculators
        self.perplexity_calc = PerplexityCalculator(model, config, device)
        self.bleu_calc = BLEUCalculator(n=4)
        self.rouge_calc = ROUGECalculator()
        self.f1_calc = F1Calculator()

    def evaluate(
        self,
        dataloader: DataLoader,
        compute_perplexity: bool = True,
        compute_bleu: bool = True,
        compute_rouge: bool = True,
        compute_f1: bool = True,
    ) -> AdvancedMetricsResult:
        """
        Evaluate model using advanced metrics.
        
        Args:
            dataloader: DataLoader with evaluation data
            compute_perplexity: Whether to compute perplexity
            compute_bleu: Whether to compute BLEU scores
            compute_rouge: Whether to compute ROUGE scores
            compute_f1: Whether to compute F1 scores
            
        Returns:
            AdvancedMetricsResult with all computed metrics
        """
        result = AdvancedMetricsResult()
        
        # Collect predictions and targets
        all_predictions = []
        all_targets = []
        all_hypotheses = []
        all_references = []
        
        self.model.eval()
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                    targets = batch[1] if len(batch) > 1 else None
                else:
                    inputs = batch.get("input_ids", batch.get("inputs"))
                    targets = batch.get("labels", batch.get("targets"))
                
                inputs = inputs.to(self.device)
                if targets is not None:
                    targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Get predictions
                if isinstance(outputs, torch.Tensor):
                    logits = outputs
                else:
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                
                # Store for metric computation
                if targets is not None:
                    preds = torch.argmax(logits, dim=-1)
                    all_predictions.extend(preds.cpu().numpy().flatten())
                    all_targets.extend(targets.cpu().numpy().flatten())
                
                # For text generation, store hypotheses and references
                # (This would require a generation step, which we skip for now)

        # Compute metrics
        if compute_f1 and all_predictions and all_targets:
            f1_results = self.f1_calc.compute(all_predictions, all_targets)
            result.f1_score = f1_results.get("f1_score")
            result.precision = f1_results.get("precision")
            result.recall = f1_results.get("recall")
            result.accuracy = float(np.mean(np.array(all_predictions) == np.array(all_targets)))
        
        # Compute perplexity
        if compute_perplexity:
            try:
                result.perplexity = self.perplexity_calc.compute(dataloader)
            except Exception as e:
                result.details["perplexity_error"] = str(e)
        
        # Note: BLEU and ROUGE require text generation, which is more complex
        # These would be computed separately with generation functions
        
        return result

    def evaluate_generation(
        self,
        hypotheses: List[str],
        references: List[List[str]],
    ) -> AdvancedMetricsResult:
        """
        Evaluate text generation quality using BLEU and ROUGE.
        
        Args:
            hypotheses: List of generated text strings
            references: List of lists of reference text strings
            
        Returns:
            AdvancedMetricsResult with BLEU and ROUGE scores
        """
        result = AdvancedMetricsResult()
        
        # Compute BLEU
        try:
            bleu_results = self.bleu_calc.compute(hypotheses, references)
            result.bleu_1 = bleu_results.get("bleu_1")
            result.bleu_2 = bleu_results.get("bleu_2")
            result.bleu_3 = bleu_results.get("bleu_3")
            result.bleu_4 = bleu_results.get("bleu_4")
        except Exception as e:
            result.details["bleu_error"] = str(e)
        
        # Compute ROUGE
        try:
            rouge_results = self.rouge_calc.compute(hypotheses, references)
            result.rouge_1 = rouge_results.get("rouge_1")
            result.rouge_2 = rouge_results.get("rouge_2")
            result.rouge_l = rouge_results.get("rouge_l")
        except Exception as e:
            result.details["rouge_error"] = str(e)
        
        return result

    def __repr__(self) -> str:
        return (
            f"AdvancedEvaluator(model={type(self.model).__name__}, "
            f"device={self.device})"
        )
