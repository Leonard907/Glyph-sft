from evaluate import load
from datasets import load_dataset
from tqdm import tqdm
import json
import torch

try:
    from alignscore import AlignScore
except ImportError:
    print("alignscore not installed")

def parse_non_ascii(text):
    """Remove or handle non-ASCII characters"""
    return text.encode('ascii', 'ignore').decode('ascii')

def eval_bertscore(input_output_pairs):
    """Evaluate BERTScore for prediction-reference pairs"""
    bertscore = load('bertscore')
    
    predictions = []
    references = []
    
    print("Preparing data for BERTScore...")
    for pred, ref in input_output_pairs:
        predictions.append(parse_non_ascii(pred))
        references.append(parse_non_ascii(ref))
    
    print("Computing BERTScore...")
    scores = bertscore.compute(
        predictions=predictions, 
        references=references, 
        model_type="microsoft/deberta-xlarge-mnli", 
        batch_size=4
    )
    
    avg_precision = sum(scores["precision"]) / len(scores["precision"])
    avg_recall = sum(scores["recall"]) / len(scores["recall"])
    avg_f1 = sum(scores["f1"]) / len(scores["f1"])
    
    return {
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1
    }

def eval_rouge(input_output_pairs):
    """Evaluate ROUGE scores for prediction-reference pairs"""
    rouge = load('rouge')
    
    predictions = []
    references = []
    
    for pred, ref in input_output_pairs:
        predictions.append(parse_non_ascii(pred))
        references.append(parse_non_ascii(ref))
    
    print("Computing ROUGE...")
    scores = rouge.compute(
        predictions=predictions, 
        references=references, 
        use_stemmer=True
    )
    
    # Compute geometric mean of ROUGE-1, ROUGE-2, and ROUGE-L F1 scores
    rouge1 = scores['rouge1']
    rouge2 = scores['rouge2']
    rougeL = scores['rougeL']
    geometric_mean = (rouge1 * rouge2 * rougeL) ** (1/3)
    
    return {
        "geometric_mean": geometric_mean,
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeL": rougeL
    }

def eval_alignscore(input_output_pairs, reference_texts):
    """
    Evaluate AlignScore for prediction-reference pairs
    reference_texts: list of input documents for factuality checking
    """
    scorer = AlignScore(
        model='roberta-large', 
        batch_size=32, 
        device='cuda:0', 
        ckpt_path='/mnt/lituou/AlignScore-large.ckpt', 
        evaluation_mode='nli_sp', 
        verbose=False
    )
    
    total_score = 0
    print("Computing AlignScore...")
    
    for i, (pred, ref) in enumerate(tqdm(input_output_pairs)):
        context = parse_non_ascii(reference_texts[i])
        claim = parse_non_ascii(pred)
        score = scorer.score(contexts=[context], claims=[claim])[0]
        total_score += score
    
    avg_score = total_score / len(input_output_pairs)
    return avg_score

# Main evaluation code
if __name__ == "__main__":
    # Load dataset and predictions
    dt = load_dataset("shipWr3ck/supersummary", split="test")
    predictions = json.load(open("/mnt/lituou/Glyph-sft/scripts/results/gemma3-27b_supersummary_dpi72.json"))
    
    # Prepare input-output pairs
    input_output_pairs = []
    reference_texts = []  # For AlignScore (input documents)
    
    for item in dt:
        gold_summary = item["output"]
        title = item["title"]
        input_text = item["input"]  # Assuming this exists in the dataset
        
        if title in predictions:
            predicted_summary = predictions[title]
            input_output_pairs.append((predicted_summary, gold_summary))
            reference_texts.append(input_text)
    
    print(f"Evaluating {len(input_output_pairs)} predictions...\n")
    
    # Run evaluations with torch.no_grad() for efficiency
    with torch.no_grad():
        # 1. ROUGE
        print("=" * 50)
        print("ROUGE Evaluation")
        print("=" * 50)
        rouge_scores = eval_rouge(input_output_pairs)
        print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
        print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
        print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
        print(f"Geometric Mean: {rouge_scores['geometric_mean']:.4f}\n")
        
        # 2. BERTScore
        print("=" * 50)
        print("BERTScore Evaluation")
        print("=" * 50)
        bertscore_results = eval_bertscore(input_output_pairs)
        print(f"Precision: {bertscore_results['precision']:.4f}")
        print(f"Recall: {bertscore_results['recall']:.4f}")
        print(f"F1: {bertscore_results['f1']:.4f}\n")
        
        # 3. AlignScore (if input texts are available)
        if reference_texts:
            print("=" * 50)
            print("AlignScore Evaluation")
            print("=" * 50)
            alignscore_result = eval_alignscore(input_output_pairs, reference_texts)
            print(f"AlignScore: {alignscore_result:.4f}\n")
        
        # Save all results
        results = {
            "rouge": rouge_scores,
            "bertscore": bertscore_results,
        }
        
        if reference_texts:
            results["alignscore"] = alignscore_result
        
        # Save to file
        output_file = "/mnt/lituou/Glyph-sft/scripts/results/gemma3-27b_supersummary_dpi72_eval.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_file}")