"""Dataset loading and preprocessing utilities"""

import pandas as pd
from datasets import Dataset
from typing import Tuple


class DatasetProcessor:
    """Handles dataset loading and preprocessing"""
    
    def __init__(self, tokenizer, max_seq_length=2048):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def format_prompt(self, situation, title, message, response=""):
        """Format conversation in LLaMA 3.1 chat template"""
        system = "আপনি একজন সহানুভূতিশীল বাংলা সহকারী।"
        user = f"বিষয়: {situation}\nপরিস্থিতি: {title}\nঅনুভূতি: {message}"
        
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{system}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n"
            f"{user}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
            f"{response}<|eot_id|>"
        )

    def load_bengali_empathetic_dataset(self, csv_path: str = None) -> Tuple[Dataset, Dataset, Dataset]:
        """Load and preprocess Bengali Empathetic Conversations dataset"""
        try:
            if csv_path is None:
                csv_path = "data/BengaliEmpatheticConversationsCorpus.csv"
            
            df = pd.read_csv(csv_path)
            df = df.rename(columns={
                "Topics": "situation",
                "Question-Title": "title",
                "Questions": "message",
                "Answers": "response"
            })
            print(f"Loaded {len(df)} samples from dataset")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Using dummy data for demonstration...")
            data = {
                "situation": ["দুঃখ"] * 100,
                "title": ["ব্যর্থতা"] * 100,
                "message": ["আমি খুব হতাশ"] * 100,
                "response": ["আমি বুঝতে পারছি তুমি কেমন অনুভব করছো..."] * 100
            }
            df = pd.DataFrame(data)

        # Format all examples
        df["text"] = df.apply(
            lambda row: self.format_prompt(
                row.situation, row.title, row.message, row.response
            ), 
            axis=1
        )
        
        # Create dataset splits
        dataset = Dataset.from_pandas(df)
        split = dataset.train_test_split(test_size=0.2, seed=42)
        val_test = split["test"].train_test_split(test_size=0.5, seed=42)
        
        print(f"Train: {len(split['train'])}, Val: {len(val_test['train'])}, Test: {len(val_test['test'])}")
        return split["train"], val_test["train"], val_test["test"]
    