"""Configuration classes for model training and LoRA parameters"""

from dataclasses import dataclass, asdict
from typing import List


@dataclass
class LoRAConfig:
    """LoRA configuration parameters"""
    r: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    bias: str = "none"
    target_modules: List[str] = None
    use_gradient_checkpointing: str = "unsloth"

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

    def to_dict(self):
        return asdict(self)


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    max_seq_length: int = 2048
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    warmup_steps: int = 10
    max_steps: int = 200
    learning_rate: float = 2e-4
    logging_steps: int = 1
    save_steps: int = 100
    eval_steps: int = 50
    output_dir: str = "outputs"
    
    def to_dict(self):
        return asdict(self)
    