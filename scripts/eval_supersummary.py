from evaluate import load
from datasets import load_dataset
from tqdm import tqdm
import json
import torch
import os
from pathlib import Path
from bert_score import BERTScorer

try:
    from alignscore import AlignScore
except ImportError:
    print("alignscore not installed")

def parse_non_ascii(text):
    """Remove or handle non-ASCII characters"""
    return text.encode('ascii', 'ignore').decode('ascii')

def prepare_all_predictions(json_files, dataset):
    """
    Prepare all predictions from all JSON files
    Returns: dict mapping filename to (input_output_pairs, reference_texts)
    """
    all_data = {}
    
    for json_file in json_files:
        filename = json_file.name
        
        try:
            with open(json_file, 'r') as f:
                predictions = json.load(f)
            
            input_output_pairs = []
            reference_texts = []
            
            for item in dataset:
                gold_summary = item["output"]
                title = item["title"]
                input_text = item["input"]
                
                if title in predictions:
                    predicted_summary = predictions[title].split("</think>")[-1].strip()
                    input_output_pairs.append((predicted_summary, gold_summary))
                    reference_texts.append(input_text)
            
            if len(input_output_pairs) > 0:
                all_data[filename] = {
                    'pairs': input_output_pairs,
                    'references': reference_texts
                }
            else:
                print(f"  WARNING: No matching predictions found in {filename}")
                
        except Exception as e:
            print(f"  ERROR loading {filename}: {str(e)}")
            continue
    
    return all_data

def eval_rouge_all(all_data):
    """Evaluate ROUGE scores for all files at once"""
    print("=" * 70)
    print("Computing ROUGE for all files...")
    print("=" * 70)
    
    rouge = load('rouge')
    results = {}
    
    for filename, data in tqdm(all_data.items(), desc="ROUGE"):
        predictions = []
        references = []
        
        for pred, ref in data['pairs']:
            predictions.append(parse_non_ascii(pred))
            references.append(parse_non_ascii(ref))
        
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
        
        results[filename] = {
            "rouge1": rouge1,
            "rouge2": rouge2,
            "rougeL": rougeL,
            "geometric_mean": geometric_mean
        }
        
        print(f"  {filename}: ROUGE-1={rouge1:.4f}, ROUGE-2={rouge2:.4f}, ROUGE-L={rougeL:.4f}, GM={geometric_mean:.4f}")
    
    return results

def eval_bertscore_all(all_data):
    """Evaluate BERTScore for all files at once"""
    print("\n" + "=" * 70)
    print("Computing BERTScore for all files...")
    print("=" * 70)
    
    scorer = BERTScorer(model_type="microsoft/deberta-xlarge-mnli", batch_size=1, device='cuda:0')
    results = {}
    
    for filename, data in tqdm(all_data.items(), desc="BERTScore"):
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        
        for pred, ref in tqdm(data['pairs'], desc=f"  {filename}", leave=False):
            pred_clean = parse_non_ascii(pred)
            ref_clean = parse_non_ascii(ref)
            
            precision, recall, f1 = scorer.score([pred_clean], [ref_clean])
            
            total_precision += precision.item()
            total_recall += recall.item()
            total_f1 += f1.item()
        
        avg_precision = total_precision / len(data['pairs'])
        avg_recall = total_recall / len(data['pairs'])
        avg_f1 = total_f1 / len(data['pairs'])
        
        results[filename] = {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1
        }
        
        print(f"  {filename}: Precision={avg_precision:.4f}, Recall={avg_recall:.4f}, F1={avg_f1:.4f}")
    
    return results

def eval_alignscore_all(all_data):
    """Evaluate AlignScore for all files at once"""
    print("\n" + "=" * 70)
    print("Computing AlignScore for all files...")
    print("=" * 70)
    
    scorer = AlignScore(
        model='roberta-large', 
        batch_size=32, 
        device='cuda:0', 
        ckpt_path='/mnt/lituou/AlignScore-large.ckpt', 
        evaluation_mode='nli_sp', 
        verbose=False
    )
    
    results = {}
    
    for filename, data in all_data.items():
        print(f"  Processing {filename}...")
        total_score = 0
        
        for i, (pred, ref) in enumerate(tqdm(data['pairs'], desc=f"  {filename}", leave=False)):
            context = parse_non_ascii(data['references'][i])
            claim = parse_non_ascii(pred)
            score = scorer.score(contexts=[context], claims=[claim])[0]
            total_score += score
        
        avg_score = total_score / len(data['pairs'])
        results[filename] = avg_score
        print(f"    AlignScore: {avg_score:.4f}")
    
    return results

# Main evaluation code
if __name__ == "__main__":
    results_folder = "/mnt/lituou/Glyph-sft/scripts/results"
    output_file = os.path.join(results_folder, "full_eval.json")
    
    # Load dataset once
    print("Loading dataset...")
    dt = load_dataset("shipWr3ck/supersummary", split="test")
    
    # Find all JSON files in the folder
    json_files = list(Path(results_folder).glob("*.json"))
    # Exclude the output file itself if it exists
    json_files = [f for f in json_files if f.name != "full_eval.json"]
    
    print(f"Found {len(json_files)} JSON files to evaluate\n")
    
    # Prepare all predictions
    print("Preparing all predictions...")
    all_data = prepare_all_predictions(json_files, dt)
    print(f"Successfully loaded {len(all_data)} files\n")
    
    if len(all_data) == 0:
        print("No valid prediction files found. Exiting.")
        exit(1)
    
    # Run all evaluations with torch.no_grad() for efficiency
    with torch.no_grad():
        # 1. ROUGE - all files
        # rouge_results = eval_rouge_all(all_data)
        
        # # 2. BERTScore - all files
        # bertscore_results = eval_bertscore_all(all_data)
        
        # 3. AlignScore - all files
        alignscore_results = eval_alignscore_all(all_data)
        # alignscore_results = {filename: 0 for filename in all_data.keys()}
    
    # Combine all results
    all_results = {}
    for filename in all_data.keys():
        all_results[filename] = {
            "rouge": rouge_results[filename],
            "bertscore": bertscore_results[filename],
            "alignscore": alignscore_results[filename],
            "num_predictions": len(all_data[filename]['pairs'])
        }
    
    # Save all results to file
    print("\n" + "=" * 70)
    print("Saving results...")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"All results saved to {output_file}")
    print(f"Total files evaluated: {len(all_results)}")
    
    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Filename':<50} {'ROUGE-GM':<12} {'BERTScore':<12} {'Predictions':<12}")
    print("-" * 70)
    for filename, results in sorted(all_results.items()):
        rouge_gm = results['rouge']['geometric_mean']
        bert_f1 = results['bertscore']['f1']
        num_preds = results['num_predictions']
        print(f"{filename:<50} {rouge_gm:<12.4f} {bert_f1:<12.4f} {num_preds:<12}")
    print("=" * 70)