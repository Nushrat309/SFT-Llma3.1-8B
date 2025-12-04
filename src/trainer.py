"""Main fine-tuning orchestrator using Strategy Pattern"""

import torch
from pathlib import Path
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer
from accelerate import Accelerator

from .config import LoRAConfig, TrainingConfig
from .strategies import FineTuningStrategy, UnslothStrategy
from .data_processor import DatasetProcessor
from .evaluator import Evaluator


class LLAMAFineTuner:
    """Main fine-tuning orchestrator using Strategy Pattern"""
    
    def __init__(self, strategy: FineTuningStrategy = None):
        self.lora_config = LoRAConfig()
        self.train_config = TrainingConfig()
        self.model = None
        self.tokenizer = None
        self.strategy = strategy or UnslothStrategy()  # Default to Unsloth
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        
        print(f"Initialized LLAMAFineTuner with {self.strategy.get_strategy_name()} strategy")
        print(f"Using {torch.cuda.device_count()} GPU(s) - Device: {self.device}")
    
    def set_strategy(self, strategy: FineTuningStrategy):
        """Allow runtime strategy switching"""
        self.strategy = strategy
        print(f"Strategy switched to: {strategy.get_strategy_name()}")
    
    def load_model(self):
        """Load base model and apply selected fine-tuning strategy"""
        print(f"\n{'='*60}")
        print(f"Loading model with {self.strategy.get_strategy_name()} strategy")
        print(f"{'='*60}")
        
        # Load base model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
            max_seq_length=self.train_config.max_seq_length,
            dtype=None,
            load_in_4bit=True,
            device_map="auto",
        )
        
        # Apply selected strategy
        model = self.strategy.get_peft_model(model, self.lora_config)
        model = self.strategy.prepare_model_for_training(model)
        
        self.model = model
        self.tokenizer = tokenizer
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model loaded successfully with {self.strategy.get_strategy_name()}")
        print(f"Trainable parameters: {trainable_params:,}")

    def prepare_dataset(self, csv_path: str = None):
        """Load and prepare dataset"""
        processor = DatasetProcessor(self.tokenizer, self.train_config.max_seq_length)
        train, val, test = processor.load_bengali_empathetic_dataset(csv_path)
        return train, val, test
"""Compare different fine-tuning strategies"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    LLAMAFineTuner,
    UnslothStrategy,
    StandardLoRAStrategy,
    QLoRAStrategy,
    ExperimentDatabase
)


def compare_strategies(csv_path: str = None):
    """Compare different fine-tuning strategies"""
    
    strategies = [
        UnslothStrategy(),
        StandardLoRAStrategy(),
        QLoRAStrategy()
    ]
    
    results = {}
    db = ExperimentDatabase()
    
    try:
        for strategy in strategies:
            print(f"\n{'='*60}")
            print(f"Training with {strategy.get_strategy_name()}")
            print(f"{'='*60}")
            
            # Initialize fine-tuner with strategy
            finetuner = LLAMAFineTuner(strategy=strategy)
            
            # Load model and data
            finetuner.load_model()
            train_ds, val_ds, test_ds = finetuner.prepare_dataset(csv_path)
            
            # Train
            stats = finetuner.train(train_ds, val_ds)
            
            # Evaluate
            metrics, preds, refs = finetuner.evaluate(test_ds)
            
            # Log to database
            exp_id = db.log_experiment(
                model_name="Llama-3.1-8B-Bengali-Empathetic",
                strategy_name=strategy.get_strategy_name(),
                lora_config=finetuner.lora_config.to_dict(),
                train_loss=stats.metrics.get("train_loss", 0),
                val_loss=stats.metrics.get("eval_loss", 0),
                metrics=metrics
            )
            
            # Store results
            results[strategy.get_strategy_name()] = {
                "exp_id": exp_id,
                "metrics": metrics,
                "train_loss": stats.metrics.get("train_loss", 0),
                "val_loss": stats.metrics.get("eval_loss", 0)
            }
            
            # Save model
            model_path = f"outputs/models/bengali_empathetic_llama_{strategy.get_strategy_name()}"
            finetuner.save_model(model_path)
        
        # Print comparison table
        print("\n" + "="*60)
        print("STRATEGY COMPARISON")
        print("="*60)
        print(f"\n{'Strategy':<20} {'Train Loss':<12} {'Val Loss':<12} {'Perplexity':<12} {'BLEU':<10}")
        print("-" * 66)
        
        for strategy_name, result in results.items():
            print(f"{strategy_name:<20} "
                  f"{result['train_loss']:<12.4f} "
                  f"{result['val_loss']:<12.4f} "
                  f"{result['metrics']['perplexity']:<12.2f} "
                  f"{result['metrics']['bleu']:<10.4f}")
        
        # Save comparison to file
        comparison_file = Path("results/strategy_comparison.txt")
        comparison_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(comparison_file, "w", encoding="utf-8") as f:
            f.write("Strategy Comparison Results\n")
            f.write("="*60 + "\n\n")
            f.write(f"{'Strategy':<20} {'Train Loss':<12} {'Val Loss':<12} {'Perplexity':<12} {'BLEU':<10}\n")
            f.write("-" * 66 + "\n")
            
            for strategy_name, result in results.items():
                f.write(f"{strategy_name:<20} "
                       f"{result['train_loss']:<12.4f} "
                       f"{result['val_loss']:<12.4f} "
                       f"{result['metrics']['perplexity']:<12.2f} "
                       f"{result['metrics']['bleu']:<10.4f}\n")
        
        print(f"\nComparison saved to {comparison_file}")
        
        return results
        
    finally:
        db.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare fine-tuning strategies")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to CSV dataset file"
    )
    
    args = parser.parse_args()
    compare_strategies(csv_path=args.data)
    def train(self, train_ds, val_ds):
        """Execute training loop"""
        print(f"\n{'='*60}")
        print(f"Starting Training with {self.strategy.get_strategy_name()}")
        print(f"{'='*60}")
        
        args = SFTConfig(
            per_device_train_batch_size=self.train_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.train_config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.train_config.gradient_accumulation_steps,
            warmup_steps=self.train_config.warmup_steps,
            max_steps=self.train_config.max_steps,
            learning_rate=self.train_config.learning_rate,
            logging_steps=self.train_config.logging_steps,
            save_steps=self.train_config.save_steps,
            eval_steps=self.train_config.eval_steps,
            output_dir=self.train_config.output_dir,
            optim="adamw_8bit",
            seed=3407,
            report_to="none",
            fp16=True,          # T4 compatible
            bf16=False,         # Disabled for T4
            packing=False,      # Memory efficient
            max_seq_length=self.train_config.max_seq_length,
            ddp_find_unused_parameters=False,
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            dataset_text_field="text",
            args=args,
        )
        
        trainer = self.accelerator.prepare(trainer)
        print("Training started...")
        stats = trainer.train()
        print("Training completed!")
        
        return stats

    def evaluate(self, test_ds):
        """Run evaluation"""
        evaluator = Evaluator(device=self.device)
        return evaluator.comprehensive_evaluation(self.model, self.tokenizer, test_ds)

    def save_model(self, path="outputs/models/bengali_empathetic_llama"):
        """Save fine-tuned model"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained_merged(path, self.tokenizer, save_method="merged_16bit")
        print(f"Model saved to {path}")
