"""Main training script"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    LLAMAFineTuner,
    UnslothStrategy,
    StandardLoRAStrategy,
    QLoRAStrategy,
    ExperimentDatabase
)


def run_training(strategy_name="unsloth", csv_path=None):
    """Execute complete fine-tuning pipeline"""
    
    # Initialize database
    db = ExperimentDatabase()
    
    try:
        # Select strategy
        strategies = {
            "unsloth": UnslothStrategy(),
            "lora": StandardLoRAStrategy(),
            "qlora": QLoRAStrategy()
        }
        strategy = strategies.get(strategy_name.lower(), UnslothStrategy())
        
        # Initialize fine-tuner
        finetuner = LLAMAFineTuner(strategy=strategy)
        
        # Load model and data
        finetuner.load_model()
        train_ds, val_ds, test_ds = finetuner.prepare_dataset(csv_path)

        # Baseline evaluation
        print("\n" + "="*60)
        print("BASELINE EVALUATION (before training)")
        print("="*60)
        baseline_metrics, _, _ = finetuner.evaluate(test_ds)

        # Fine-tuning
        print("\n" + "="*60)
        print("STARTING FINE-TUNING")
        print("="*60)
        stats = finetuner.train(train_ds, val_ds)

        # Final evaluation
        print("\n" + "="*60)
        print("FINAL EVALUATION (after training)")
        print("="*60)
        final_metrics, preds, refs = finetuner.evaluate(test_ds)

        # Log experiment
        exp_id = db.log_experiment(
            model_name="Llama-3.1-8B-Bengali-Empathetic",
            strategy_name=finetuner.strategy.get_strategy_name(),
            lora_config=finetuner.lora_config.to_dict(),
            train_loss=stats.metrics.get("train_loss", 0),
            val_loss=stats.metrics.get("eval_loss", 0),
            metrics=final_metrics
        )
        
        print(f"\nExperiment logged with ID: {exp_id}")

        # Log sample responses
        for i, (pred, ref) in enumerate(zip(preds[:5], refs[:5])):
            db.log_response(exp_id, ref[:100], pred[:100])

        # Save model
        model_path = f"outputs/models/bengali_empathetic_llama_{finetuner.strategy.get_strategy_name()}"
        finetuner.save_model(model_path)

        # Print comparison
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Strategy: {finetuner.strategy.get_strategy_name()}")
        print(f"\nBaseline Metrics:")
        for k, v in baseline_metrics.items():
            print(f"  {k}: {v:.4f}")
        print(f"\nFinal Metrics:")
        for k, v in final_metrics.items():
            print(f"  {k}: {v:.4f}")
        print(f"\nImprovement:")
        for k in final_metrics.keys():
            if k == "perplexity":
                improvement = baseline_metrics[k] - final_metrics[k]
                print(f"  {k}: {improvement:.4f} (lower is better)")
            else:
                improvement = final_metrics[k] - baseline_metrics[k]
                print(f"  {k}: {improvement:.4f} (higher is better)")

        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        
        return exp_id

    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA on Bengali Empathetic Conversations")
    parser.add_argument(
        "--strategy",
        type=str,
        default="unsloth",
        choices=["unsloth", "lora", "qlora"],
        help="Fine-tuning strategy to use"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to CSV dataset file"
    )
    
    args = parser.parse_args()
    run_training(strategy_name=args.strategy, csv_path=args.data)
    