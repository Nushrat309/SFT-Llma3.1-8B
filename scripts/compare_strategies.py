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
