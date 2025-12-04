"""Model evaluation with multiple metrics"""

import torch
import evaluate
from typing import Dict, List, Tuple
from unsloth import FastLanguageModel


class Evaluator:
    """Handles model evaluation with multiple metrics"""
    
    def __init__(self, device="cuda"):
        self.rouge = evaluate.load('rouge')
        self.bleu = evaluate.load('sacrebleu')
        self.device = device

    def calculate_perplexity(self, model, tokenizer, dataset, max_samples=100):
        """Calculate perplexity on dataset"""
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        samples = dataset.select(range(min(max_samples, len(dataset))))
        
        with torch.no_grad():
            for ex in samples:
                inputs = tokenizer(
                    ex["text"], 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=2048
                ).to(self.device)
                
                labels = inputs["input_ids"].clone()
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss.item()
                tokens = inputs["attention_mask"].sum().item()
                
                total_loss += loss * tokens
                total_tokens += tokens
        
        perplexity = torch.exp(torch.tensor(total_loss / total_tokens)).item()
        return perplexity

    def calculate_bleu_rouge(self, preds: List[str], refs: List[str]) -> Dict[str, float]:
        """Calculate BLEU and ROUGE scores"""
        rouge_scores = self.rouge.compute(predictions=preds, references=refs)
        bleu_score = self.bleu.compute(
            predictions=preds, 
            references=[[r] for r in refs]
        )["score"]
        
        return {
            "rouge1": rouge_scores["rouge1"],
            "rouge2": rouge_scores["rouge2"],
            "rougeL": rouge_scores["rougeL"],
            "bleu": bleu_score
        }

    def generate_evaluation_responses(self, model, tokenizer, dataset, num_samples=20) -> Tuple[List[str], List[str]]:
        """Generate responses for evaluation"""
        FastLanguageModel.for_inference(model)
        preds, refs = [], []
        
        samples = dataset.select(range(min(num_samples, len(dataset))))
        
        for ex in samples:
            # Extract prompt (everything before assistant response)
            prompt = ex["text"].split("<|start_header_id|>assistant<|end_header_id|>")[0]
            prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
            
            # Extract reference response
            ref = ex["text"].split("<|start_header_id|>assistant<|end_header_id|>")[1]
            ref = ref.split("<|eot_id|>")[0].strip()
            
            # Generate prediction
            inputs = tokenizer([prompt], return_tensors="pt").to(self.device)
            output = model.generate(
                **inputs, 
                max_new_tokens=128, 
                temperature=0.7, 
                do_sample=True
            )
            
            pred = tokenizer.decode(output[0], skip_special_tokens=True)
            pred = pred.split("assistant")[-1].strip()
            
            preds.append(pred)
            refs.append(ref)
        
        return preds, refs

    def comprehensive_evaluation(self, model, tokenizer, test_dataset) -> Tuple[Dict[str, float], List[str], List[str]]:
        """Run complete evaluation pipeline"""
        print("\n=== Starting Comprehensive Evaluation ===")
        
        # Calculate perplexity
        ppl = self.calculate_perplexity(model, tokenizer, test_dataset)
        print(f"Perplexity: {ppl:.2f}")
        
        # Generate responses and calculate BLEU/ROUGE
        preds, refs = self.generate_evaluation_responses(model, tokenizer, test_dataset)
        scores = self.calculate_bleu_rouge(preds, refs)
        scores["perplexity"] = ppl
        
        # Print all metrics
        print("\nEvaluation Metrics:")
        for k, v in scores.items():
            print(f"  {k}: {v:.4f}")
        
        return scores, preds, refs
    