"""Strategy Pattern implementation for different fine-tuning approaches"""

from abc import ABC, abstractmethod
from .config import LoRAConfig


class FineTuningStrategy(ABC):
    """Abstract base class for fine-tuning strategies"""
    
    @abstractmethod
    def get_peft_model(self, model, lora_config: LoRAConfig):
        """Apply fine-tuning strategy to model"""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return strategy name for logging"""
        pass
    
    @abstractmethod
    def prepare_model_for_training(self, model):
        """Additional model preparation if needed"""
        pass


class UnslothStrategy(FineTuningStrategy):
    """Unsloth-based fine-tuning strategy - optimized for speed and memory"""
    
    def get_peft_model(self, model, lora_config: LoRAConfig):
        from unsloth import FastLanguageModel
        
        print("Applying Unsloth LoRA adaptation...")
        return FastLanguageModel.get_peft_model(
            model,
            r=lora_config.r,
            target_modules=lora_config.target_modules,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout,
            bias=lora_config.bias,
            use_gradient_checkpointing=lora_config.use_gradient_checkpointing,
            random_state=3407,
        )
    
    def get_strategy_name(self) -> str:
        return "Unsloth"
    
    def prepare_model_for_training(self, model):
        # Unsloth handles optimization internally
        return model


class StandardLoRAStrategy(FineTuningStrategy):
    """Standard LoRA fine-tuning strategy using PEFT library"""
    
    def get_peft_model(self, model, lora_config: LoRAConfig):
        from peft import get_peft_model, LoraConfig as PeftLoraConfig, prepare_model_for_kbit_training
        
        print("Applying Standard LoRA adaptation...")
        peft_config = PeftLoraConfig(
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout,
            bias=lora_config.bias,
            target_modules=lora_config.target_modules,
            task_type="CAUSAL_LM"
        )
        
        model = prepare_model_for_kbit_training(model)
        return get_peft_model(model, peft_config)
    
    def get_strategy_name(self) -> str:
        return "StandardLoRA"
    
    def prepare_model_for_training(self, model):
        # Enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()
        return model


class QLoRAStrategy(FineTuningStrategy):
    """QLoRA strategy - LoRA with quantization"""
    
    def get_peft_model(self, model, lora_config: LoRAConfig):
        from peft import get_peft_model, LoraConfig as PeftLoraConfig, prepare_model_for_kbit_training
        
        print("Applying QLoRA adaptation (4-bit quantized)...")
        model = prepare_model_for_kbit_training(model)
        
        peft_config = PeftLoraConfig(
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout,
            bias=lora_config.bias,
            target_modules=lora_config.target_modules,
            task_type="CAUSAL_LM"
        )
        
        return get_peft_model(model, peft_config)
    
    def get_strategy_name(self) -> str:
        return "QLoRA"
    
    def prepare_model_for_training(self, model):
        model.gradient_checkpointing_enable()
        return model
    