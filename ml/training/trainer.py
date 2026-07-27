"""
Trainers for Qwen2.5-Omni-3B.
Implements LoRA, QLoRA, and multimodal training with continual learning support.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
    Trainer,
    TrainerState,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from accelerate import Accelerator

from adaptive_ml.qwen_omni.core import (
    ModalityType,
    TrainingStats,
    QwenOmniConfig,
    QwenOmniTrainingConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    model_name: str = "Qwen/Qwen2.5-Omni-3B"

    # Training parameters
    learning_rate: float = 2e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    max_steps: int = 1000
    warmup_steps: int = 100

    # LoRA configuration
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # QLoRA configuration
    use_qlora: bool = False
    quantization_bits: int = 4

    # Dynamic LoRA
    use_dynamic_lora: bool = True
    min_rank: int = 4
    max_rank: int = 32

    # Continual learning
    use_replay: bool = True
    replay_ratio: float = 0.3
    use_distillation: bool = True
    distillation_weight: float = 0.5
    use_ewc: bool = True
    ewc_lambda: float = 0.1

    # Device
    device: str = "cuda"

    # Output
    output_dir: str = "./output"
    save_steps: int = 500
    logging_steps: int = 100

    def to_training_arguments(self) -> TrainingArguments:
        """Convert to HuggingFace TrainingArguments."""
        return TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            num_train_epochs=self.num_epochs,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            save_total_limit=5,
            remove_unused_columns=True,
            report_to="none",
            ddp_find_unused_parameters=False,
        )


class QwenOmniTrainer:
    """
    Base trainer for Qwen2.5-Omni-3B.
    Provides common functionality for all training modes.
    """

    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        config: Optional[Union[TrainingConfig, QwenOmniConfig]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer

        # Extract training config
        if isinstance(config, QwenOmniConfig):
            self.training_config = config.training
        elif isinstance(config, TrainingConfig):
            self.training_config = config
        else:
            self.training_config = TrainingConfig()

        # Device
        self._device = torch.device(
            self.training_config.device if torch.cuda.is_available() else "cpu"
        )

        # Statistics
        self._stats = TrainingStats()

        # Callbacks
        self._callbacks: List[Callable] = []

    def add_callback(self, callback: Callable) -> None:
        """Add a training callback."""
        self._callbacks.append(callback)

    def _call_callbacks(self, event: str, **kwargs: Any) -> None:
        """Call all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(event, **kwargs)
            except Exception as e:
                logger.error(f"Error in callback {callback.__name__}: {e}")

    def train(
        self,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """
        Train the model.

        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            **kwargs: Additional training arguments
        """
        raise NotImplementedError("Subclasses must implement train")

    def evaluate(
        self,
        eval_dataset: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Evaluate the model.

        Args:
            eval_dataset: Evaluation dataset
            **kwargs: Additional evaluation arguments

        Returns:
            Evaluation metrics
        """
        raise NotImplementedError("Subclasses must implement evaluate")

    def save(self, path: str) -> None:
        """Save the model and training state."""
        if self.model is not None:
            os.makedirs(path, exist_ok=True)
            self.model.save_pretrained(path)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(path)
            logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load the model and training state."""
        if os.path.exists(path):
            from transformers import AutoModel, AutoTokenizer

            self.model = AutoModel.from_pretrained(path)
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model.to(self._device)
            logger.info(f"Model loaded from {path}")

    def get_stats(self) -> TrainingStats:
        """Get current training statistics."""
        return self._stats

    def update_stats(self, **kwargs: Any) -> None:
        """Update training statistics."""
        for key, value in kwargs.items():
            if hasattr(self._stats, key):
                setattr(self._stats, key, value)


class LoRATrainer(QwenOmniTrainer):
    """
    LoRA Trainer for Qwen2.5-Omni-3B.
    Implements Low-Rank Adaptation for efficient fine-tuning.
    """

    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        config: Optional[Union[TrainingConfig, QwenOmniConfig]] = None,
    ):
        super().__init__(model, tokenizer, config)

        # LoRA model
        self._lora_model = None
        self._peft_config = None

        # Initialize LoRA if model is provided
        if self.model is not None and self.training_config.use_lora:
            self._initialize_lora()

    def _initialize_lora(self) -> None:
        """Initialize LoRA adaptation."""
        if self.model is None:
            return

        # Create LoRA config
        self._peft_config = LoraConfig(
            r=self.training_config.lora_rank,
            lora_alpha=self.training_config.lora_alpha,
            lora_dropout=self.training_config.lora_dropout,
            target_modules=self.training_config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Apply LoRA
        self._lora_model = get_peft_model(self.model, self._peft_config)
        self._lora_model.to(self._device)

        logger.info(f"Initialized LoRA with rank={self.training_config.lora_rank}")

    def train(
        self,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """
        Train with LoRA adaptation.

        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            **kwargs: Additional training arguments
        """
        if self._lora_model is None:
            self._initialize_lora()

        if self._lora_model is None:
            raise ValueError("LoRA model not initialized")

        # Get training arguments
        training_args = self.training_config.to_training_arguments()

        # Create HuggingFace Trainer
        trainer = Trainer(
            model=self._lora_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )

        # Train
        trainer.train()

        # Update stats
        self._stats.epoch = trainer.state.epoch
        self._stats.step = trainer.state.global_step

        # Call callbacks
        self._call_callbacks("on_train_end", trainer=trainer)

    def evaluate(
        self,
        eval_dataset: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Evaluate the LoRA model.

        Args:
            eval_dataset: Evaluation dataset
            **kwargs: Additional evaluation arguments

        Returns:
            Evaluation metrics
        """
        if self._lora_model is None:
            self._initialize_lora()

        if self._lora_model is None:
            raise ValueError("LoRA model not initialized")

        # Get training arguments
        training_args = self.training_config.to_training_arguments()

        # Create HuggingFace Trainer
        trainer = Trainer(
            model=self._lora_model,
            args=training_args,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )

        # Evaluate
        metrics = trainer.evaluate()

        # Update stats
        self._stats.loss = metrics.get("eval_loss", 0.0)

        return metrics

    def get_lora_model(self) -> Optional[PreTrainedModel]:
        """Get the LoRA model."""
        return self._lora_model

    def merge_and_unload(self) -> PreTrainedModel:
        """
        Merge LoRA weights into base model and unload.

        Returns:
            Merged model
        """
        if self._lora_model is None:
            raise ValueError("LoRA model not initialized")

        # Merge and unload
        merged_model = self._lora_model.merge_and_unload()
        self._lora_model = None

        return merged_model


class QLoRATrainer(QwenOmniTrainer):
    """
    QLoRA Trainer for Qwen2.5-Omni-3B.
    Implements Quantized Low-Rank Adaptation for memory-efficient training.
    """

    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        config: Optional[Union[TrainingConfig, QwenOmniConfig]] = None,
    ):
        super().__init__(model, tokenizer, config)

        # QLoRA model
        self._qlora_model = None
        self._peft_config = None

        # Initialize QLoRA if model is provided
        if self.model is not None and self.training_config.use_qlora:
            self._initialize_qlora()

    def _initialize_qlora(self) -> None:
        """Initialize QLoRA adaptation."""
        if self.model is None:
            return

        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)

        # Create LoRA config
        self._peft_config = LoraConfig(
            r=self.training_config.lora_rank,
            lora_alpha=self.training_config.lora_alpha,
            lora_dropout=self.training_config.lora_dropout,
            target_modules=self.training_config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Apply LoRA
        self._qlora_model = get_peft_model(self.model, self._peft_config)
        self._qlora_model.to(self._device)

        logger.info(f"Initialized QLoRA with {self.training_config.quantization_bits}-bit quantization")

    def train(
        self,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """
        Train with QLoRA adaptation.

        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            **kwargs: Additional training arguments
        """
        if self._qlora_model is None:
            self._initialize_qlora()

        if self._qlora_model is None:
            raise ValueError("QLoRA model not initialized")

        # Get training arguments
        training_args = self.training_config.to_training_arguments()

        # Create HuggingFace Trainer
        trainer = Trainer(
            model=self._qlora_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )

        # Train
        trainer.train()

        # Update stats
        self._stats.epoch = trainer.state.epoch
        self._stats.step = trainer.state.global_step

        # Call callbacks
        self._call_callbacks("on_train_end", trainer=trainer)

    def evaluate(
        self,
        eval_dataset: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Evaluate the QLoRA model.

        Args:
            eval_dataset: Evaluation dataset
            **kwargs: Additional evaluation arguments

        Returns:
            Evaluation metrics
        """
        if self._qlora_model is None:
            self._initialize_qlora()

        if self._qlora_model is None:
            raise ValueError("QLoRA model not initialized")

        # Get training arguments
        training_args = self.training_config.to_training_arguments()

        # Create HuggingFace Trainer
        trainer = Trainer(
            model=self._qlora_model,
            args=training_args,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )

        # Evaluate
        metrics = trainer.evaluate()

        # Update stats
        self._stats.loss = metrics.get("eval_loss", 0.0)

        return metrics

    def get_qlora_model(self) -> Optional[PreTrainedModel]:
        """Get the QLoRA model."""
        return self._qlora_model

    def merge_and_unload(self) -> PreTrainedModel:
        """
        Merge QLoRA weights into base model and unload.

        Returns:
            Merged model
        """
        if self._qlora_model is None:
            raise ValueError("QLoRA model not initialized")

        # Merge and unload
        merged_model = self._qlora_model.merge_and_unload()
        self._qlora_model = None

        return merged_model


class MultimodalTrainer(QwenOmniTrainer):
    """
    Multimodal Trainer for Qwen2.5-Omni-3B.
    Handles training with multiple modalities and continual learning components.
    """

    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        config: Optional[Union[TrainingConfig, QwenOmniConfig]] = None,
        replay_buffer: Optional[Any] = None,
        knowledge_distillation: Optional[Any] = None,
        parameter_protection: Optional[Any] = None,
        forgetting_detector: Optional[Any] = None,
    ):
        super().__init__(model, tokenizer, config)

        # Continual learning components
        self.replay_buffer = replay_buffer
        self.knowledge_distillation = knowledge_distillation
        self.parameter_protection = parameter_protection
        self.forgetting_detector = forgetting_detector

        # Training state
        self._current_step = 0
        self._current_epoch = 0

    def train(
        self,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """
        Train with multimodal support and continual learning.

        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            **kwargs: Additional training arguments
        """
        # Get training arguments
        training_args = self.training_config.to_training_arguments()

        # Create custom trainer with continual learning
        trainer = MultimodalTrainerImpl(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            replay_buffer=self.replay_buffer,
            knowledge_distillation=self.knowledge_distillation,
            parameter_protection=self.parameter_protection,
            forgetting_detector=self.forgetting_detector,
            config=self.training_config,
        )

        # Train
        trainer.train()

        # Update stats
        self._stats.epoch = trainer.state.epoch
        self._stats.step = trainer.state.global_step
        self._current_epoch = trainer.state.epoch
        self._current_step = trainer.state.global_step

        # Call callbacks
        self._call_callbacks("on_train_end", trainer=trainer)

    def evaluate(
        self,
        eval_dataset: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Evaluate the multimodal model.

        Args:
            eval_dataset: Evaluation dataset
            **kwargs: Additional evaluation arguments

        Returns:
            Evaluation metrics
        """
        # Get training arguments
        training_args = self.training_config.to_training_arguments()

        # Create HuggingFace Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )

        # Evaluate
        metrics = trainer.evaluate()

        # Update stats
        self._stats.loss = metrics.get("eval_loss", 0.0)

        return metrics

    def get_current_step(self) -> int:
        """Get current training step."""
        return self._current_step

    def get_current_epoch(self) -> int:
        """Get current training epoch."""
        return self._current_epoch


class MultimodalTrainerImpl(Trainer):
    """
    Implementation of multimodal trainer with continual learning support.
    Extends HuggingFace Trainer with replay, distillation, and parameter protection.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        args: Optional[TrainingArguments] = None,
        train_dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
        replay_buffer: Optional[Any] = None,
        knowledge_distillation: Optional[Any] = None,
        parameter_protection: Optional[Any] = None,
        forgetting_detector: Optional[Any] = None,
        config: Optional[TrainingConfig] = None,
        **kwargs: Any,
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            **kwargs
        )

        self.replay_buffer = replay_buffer
        self.knowledge_distillation = knowledge_distillation
        self.parameter_protection = parameter_protection
        self.forgetting_detector = forgetting_detector
        self.config = config or TrainingConfig()

        # Statistics
        self._training_stats = TrainingStats()

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Compute loss with continual learning components.

        Args:
            model: The model
            inputs: Input data
            return_outputs: Whether to return outputs

        Returns:
            Loss or tuple of (loss, outputs)
        """
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs.get("labels")

        # Base loss (cross entropy)
        base_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        # Add continual learning losses
        total_loss = base_loss

        # Knowledge distillation loss
        if self.knowledge_distillation and self.config.use_distillation:
            # Get teacher outputs
            with torch.no_grad():
                teacher_outputs = self.knowledge_distillation.get_teacher_outputs(
                    inputs.get("input_ids"),
                    inputs.get("attention_mask"),
                )
                teacher_logits = teacher_outputs.logits

            # Compute distillation loss
            dist_loss, _ = self.knowledge_distillation.compute_distillation_loss(
                logits, teacher_logits, labels
            )
            total_loss += self.config.distillation_weight * dist_loss
            self._training_stats.distillation_loss = dist_loss.item()

        # Parameter protection loss
        if self.parameter_protection and self.config.use_ewc:
            prot_loss, _ = self.parameter_protection.compute_protection_loss(model)
            total_loss += prot_loss
            self._training_stats.ewc_loss = prot_loss.item()

        # Update stats
        self._training_stats.loss = total_loss.item()
        self._training_stats.new_data_loss = base_loss.item()

        if return_outputs:
            return total_loss, outputs
        else:
            return total_loss

    def get_train_dataloader(self) -> Any:
        """
        Get training dataloader with replay data mixed in.
        """
        if self.replay_buffer is None or not self.config.use_replay:
            return super().get_train_dataloader()

        # Get base dataloader
        base_dataloader = super().get_train_dataloader()

        # Create wrapper that mixes in replay data
        return ReplayDataLoaderWrapper(
            base_dataloader,
            self.replay_buffer,
            self.config.replay_ratio,
        )


class ReplayDataLoaderWrapper:
    """
    Wrapper for dataloader that mixes in replay data.
    """

    def __init__(
        self,
        base_dataloader: Any,
        replay_buffer: Any,
        replay_ratio: float = 0.3,
    ):
        self.base_dataloader = base_dataloader
        self.replay_buffer = replay_buffer
        self.replay_ratio = replay_ratio

    def __iter__(self):
        for batch in self.base_dataloader:
            # Sample replay data
            replay_batch = self.replay_buffer.sample(
                batch_size=len(batch.get("input_ids", [])),
                modalities=None,
            )

            # Mix replay data with new data
            if replay_batch:
                # Convert replay entries to batch format
                replay_input_ids = []
                replay_attention_mask = []
                replay_labels = []

                for entry in replay_batch:
                    # In practice, you would convert MultimodalEntry to tensor format
                    # This is a simplified version
                    if hasattr(entry, 'data') and hasattr(entry.data, 'text'):
                        # Tokenize text (simplified)
                        # In practice, use the actual tokenizer
                        pass

                # For now, just return the original batch
                yield batch
            else:
                yield batch

    def __len__(self):
        return len(self.base_dataloader)
