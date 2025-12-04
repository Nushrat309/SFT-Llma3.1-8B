"""Standalone evaluation script for trained models"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import LLAMAFineTuner, ExperimentDatabase
from src.evaluator import Evaluator


def evaluate_model(model_path: str, csv_path: str = None, exp_id: int = None):
    """Evaluate a trained model"""
    
    print(f"\n{'='*60}")
    print(f"EVALUATING MODEL: {model_path}")
    print(f"{'='*60}")
    
    # Load model
    finetuner = LLAMAFineTuner()
    finetuner.load_model()
    
    # Load dataset
    _, _, test_ds = finetuner.prepare_dataset(csv_path)
    
    # Run evaluation
    evaluator = Evaluator(device=finetuner.device)
    metrics, preds, refs = evaluator.comprehensive_evaluation(
        finetuner.model, 
        finetuner.tokenizer, 
        test_ds
    )
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    # Show sample responses
    print("\n" + "="*60)
    print("SAMPLE RESPONSES")
    print("="*60)
    for i, (pred, ref) in enumerate(zip(preds[:3], refs[:3]), 1):
        print(f"\n--- Sample {i} ---")
        print(f"Reference: {ref[:200]}...")
        print(f"Generated: {pred[:200]}...")
    
    # Save to database if exp_id provided
    if exp_id:
        db = ExperimentDatabase()
        for pred, ref in zip(preds, refs):
            db.log_response(exp_id, ref, pred)
        db.close()
        print(f"\nResponses logged to experiment {exp_id}")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to CSV dataset file"
    )
    parser.add_argument(
        "--exp-id",
        type=int,
        default=None,
        help="Experiment ID for logging"
    )
    
    args = parser.parse_args()
    evaluate_model(args.model, args.data, args.exp_id)
    