"""Generate sample responses from trained model"""

import sys
import argparse
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import LLAMAFineTuner
from unsloth import FastLanguageModel
from transformers import TextStreamer


def generate_responses(model_path: str = None, interactive: bool = False):
    """Generate responses from trained model"""
    # Initialize and load model
    finetuner = LLAMAFineTuner()

    if model_path:
        print(f"Loading model from {model_path}...")
        # Load saved model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        finetuner.model = model
        finetuner.tokenizer = tokenizer
    else:
        print("Loading base model...")
        finetuner.load_model()

    FastLanguageModel.for_inference(finetuner.model)

    # Test prompts
    test_prompts = [
        {
            "situation": "পরীক্ষায় ব্যর্থতা",
            "title": "বন্ধু পরীক্ষায় ফেল করেছে",
            "message": "আমি খুব দুঃখ পেয়েছি, কী করব বুঝতে পারছি না।"
        },
        {
            "situation": "কাজের চাপ",
            "title": "অফিসে অতিরিক্ত কাজের চাপ",
            "message": "আমি খুব ক্লান্ত, সব সামলাতে পারছি না।"
        },
        {
            "situation": "পারিবারিক সমস্যা",
            "title": "পরিবারের সাথে মতবিরোধ",
            "message": "আমি খুব বিচলিত, কেউ আমার কথা বুঝছে না।"
        }
    ]

    print("\n" + "="*60)
    print("INFERENCE DEMO")
    print("="*60)

    if interactive:
        print("\nInteractive mode. Type 'quit' to exit.\n")
        while True:
            situation = input("বিষয়: ")
            if situation.lower() == 'quit':
                break
            title = input("পরিস্থিতি: ")
            message = input("অনুভূতি: ")
            
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
আপনি একজন সহানুভূতিশীল বাংলা সহকারী।<|eot_id|><|start_header_id|>user<|end_header_id|>
বিষয়: {situation}
পরিস্থিতি: {title}
অনুভূতি: {message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
            print("\nGenerated Response:")
            inputs = finetuner.tokenizer([prompt], return_tensors="pt").to(finetuner.device)
            streamer = TextStreamer(finetuner.tokenizer, skip_prompt=True, skip_special_tokens=True)
            _ = finetuner.model.generate(
                **inputs, 
                streamer=streamer, 
                max_new_tokens=200, 
                temperature=0.7, 
                do_sample=True
            )
            print("\n" + "-"*60)
    else:
        for i, test in enumerate(test_prompts, 1):
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
আপনি একজন সহানুভূতিশীল বাংলা সহকারী।<|eot_id|><|start_header_id|>user<|end_header_id|>
বিষয়: {test['situation']}
পরিস্থিতি: {test['title']}
অনুভূতি: {test['message']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
            print(f"\n--- Test Prompt {i} ---")
            print(f"Situation: {test['situation']}")
            print(f"Message: {test['message']}")
            print("\nGenerated Response:")
            
            inputs = finetuner.tokenizer([prompt], return_tensors="pt").to(finetuner.device)
            streamer = TextStreamer(finetuner.tokenizer, skip_prompt=True, skip_special_tokens=True)
            _ = finetuner.model.generate(
                **inputs, 
                streamer=streamer, 
                max_new_tokens=200, 
                temperature=0.7, 
                do_sample=True
            )
            print("\n" + "-"*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate responses from trained model")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model (optional, uses base model if not provided)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    args = parser.parse_args()
    generate_responses(args.model, args.interactive)
