import os
import json
import argparse
import traceback
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Process QA dataset with pure text question answering")
    parser.add_argument(
        '--mode', 
        choices=['sequential', 'concurrent'], 
        default='concurrent',
        help='Processing mode for inference: sequential or concurrent (default: concurrent)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=30,
        help='Number of concurrent workers for inference (only used in concurrent mode, default: 30)'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Model to use for processing'
    )
    parser.add_argument(
        '--port',
        type=str,
        help='Port number for the service'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=8192,
        help='Maximum number of tokens in response (default: 8192)'
    )
    parser.add_argument(
        '--max-input-tokens',
        type=int,
        default=None,
        help='Maximum number of input tokens. If set, docs will be truncated to fit (default: None)'
    )
    return parser.parse_args()

# Parse command line arguments
args = parse_arguments()
port = args.port
model = args.model
mode = args.mode
num_workers = args.workers
max_tokens = args.max_tokens
max_input_tokens = args.max_input_tokens
task = "loong_pure_text" # Hard code as QA task

# Hard-coded input jsonl file path
INPUT_JSONL_FILE = './loong_process_100k.jsonl'  # Hard-coded input file

MODEL_ID_MAPPING = {
    "gemma3-12b": "google/gemma3-12b-it",
    "qwen3-8b": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen3-30b": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "qwen3-30b-thinking": "Qwen/Qwen3-VL-30B-A3B-Thinking",
    "kimi-a3b": "moonshotai/Kimi-VL-A3B-Thinking",
    "glyph": "zai-org/Glyph"
}

# Configuration
OUTPUT_JSON_FILE = f'./results/{model}_{task}.json'
MAX_WORKERS = num_workers

# Load tokenizer if max_input_tokens is specified
tokenizer = None
if max_input_tokens is not None:
    if model not in MODEL_ID_MAPPING:
        raise ValueError(f"Model '{model}' not found in MODEL_ID_MAPPING. Available models: {list(MODEL_ID_MAPPING.keys())}")
    model_id = MODEL_ID_MAPPING[model]
    print(f"Loading tokenizer for model: {model_id}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        print(f"Tokenizer loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer for {model_id}: {e}")

# Clear proxy settings
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)

def load_jsonl(file_path):
    """Load data from jsonl file"""
    items = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                # Validate required fields
                if 'final_question' not in item:
                    print(f"Warning: Item at line {line_num} missing 'final_question', skipping")
                    continue
                if 'docs' not in item:
                    print(f"Warning: Item at line {line_num} missing 'docs', skipping")
                    continue
                # Add index as unique identifier if not present
                if 'id' not in item:
                    item['id'] = f"item_{line_num}"
                items.append(item)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue
    
    return items

def truncate_docs(docs, question, max_input_tokens, tokenizer):
    """
    Truncate docs to fit within max_input_tokens while keeping question intact.
    
    Args:
        docs: The document text to truncate
        question: The question text (must remain intact)
        max_input_tokens: Maximum total input tokens
        tokenizer: The tokenizer to use
    
    Returns:
        Truncated docs text
    """
    if tokenizer is None or max_input_tokens is None:
        return docs
    
    # Tokenize the question and separator
    separator = "\n\n"
    question_tokens = tokenizer.encode(question, add_special_tokens=False)
    separator_tokens = tokenizer.encode(separator, add_special_tokens=False)
    
    # Calculate available tokens for docs
    used_tokens = len(question_tokens) + len(separator_tokens)
    available_tokens = max_input_tokens - used_tokens
    
    if available_tokens <= 0:
        raise ValueError(f"Question and separator already exceed max_input_tokens ({max_input_tokens})")
    
    # Tokenize docs
    docs_tokens = tokenizer.encode(docs, add_special_tokens=False)
    
    # If docs fit within available tokens, return as is
    if len(docs_tokens) <= available_tokens:
        return docs
    
    # Truncate docs tokens to fit
    truncated_docs_tokens = docs_tokens[:available_tokens]
    
    # Decode back to text
    truncated_docs = tokenizer.decode(truncated_docs_tokens, skip_special_tokens=True)
    
    return truncated_docs

def run_inference(item):
    """Run text-only inference on document and question"""
    item_id = item.get('id', 'unknown')
    question = item.get('final_question', '')
    docs = item.get('docs', '')
    
    # Truncate docs if max_input_tokens is specified
    if max_input_tokens is not None and tokenizer is not None:
        try:
            docs = truncate_docs(docs, question, max_input_tokens, tokenizer)
        except Exception as e:
            error_msg = f"Truncation error: {e}"
            print(f"Warning for item '{item_id}': {error_msg}")
            # Continue with original docs if truncation fails
    
    # Concatenate docs and question with "\n\n" separator
    prompt = f"{docs}\n\n{question}"
    
    try:
        # Build request payload for text-only inference
        messages = [{'role': 'user', 'content': prompt}]
        
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "top_p": 1.0,
            "top_k": 1,
            "repetition_penalty": 1.1,
        }
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        api_url = f"http://localhost:{port}/v1/chat/completions"
        
        # Make API request
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=6000)
        
        # Check HTTP status
        if response.status_code != 200:
            error_msg = f"HTTP error: {response.status_code}, Response: {response.text}"
            print(f"Error for item '{item_id}': {error_msg}")
            return {
                'id': item_id,
                'question': question,
                'answer': f"API Error: {error_msg}",
                'success': False,
                'error': error_msg
            }
        
        # Parse JSON response
        try:
            result = response.json()
        except json.JSONDecodeError as e:
            error_msg = f"JSON decode error: {e}, Raw response: {response.text}"
            print(f"Error for item '{item_id}': {error_msg}")
            return {
                'id': item_id,
                'question': question,
                'answer': f"JSON Error: {error_msg}",
                'success': False,
                'error': error_msg
            }
        
        # Validate response structure
        if 'choices' not in result or not result['choices']:
            error_msg = f"Invalid response structure: {result}"
            print(f"Error for item '{item_id}': {error_msg}")
            return {
                'id': item_id,
                'question': question,
                'answer': f"Invalid Response: {error_msg}",
                'success': False,
                'error': error_msg
            }
        
        if 'message' not in result['choices'][0] or 'content' not in result['choices'][0]['message']:
            error_msg = f"Missing content in response: {result}"
            print(f"Error for item '{item_id}': {error_msg}")
            return {
                'id': item_id,
                'question': question,
                'answer': f"Missing Content: {error_msg}",
                'success': False,
                'error': error_msg
            }
        
        answer = result['choices'][0]['message']['content']
        
        return {
            'id': item_id,
            'question': question,
            'answer': answer,
            'success': True,
            'error': None
        }
        
    except requests.exceptions.Timeout:
        error_msg = "Request timeout"
        print(f"Error for item '{item_id}': {error_msg}")
        return {
            'id': item_id,
            'question': question,
            'answer': f"Timeout Error: {error_msg}",
            'success': False,
            'error': error_msg
        }
    except requests.exceptions.ConnectionError as e:
        error_msg = f"Connection error: {e}"
        print(f"Error for item '{item_id}': {error_msg}")
        return {
            'id': item_id,
            'question': question,
            'answer': f"Connection Error: {error_msg}",
            'success': False,
            'error': error_msg
        }
    except Exception as e:
        error_trace = traceback.format_exc()
        error_msg = f"Unexpected error: {type(e).__name__}: {e}"
        print(f"Error for item '{item_id}': {error_msg}")
        return {
            'id': item_id,
            'question': question,
            'answer': f"Unexpected Error: {error_trace}",
            'success': False,
            'error': error_msg
        }

# Create base output directory if it doesn't exist
os.makedirs(os.path.dirname(OUTPUT_JSON_FILE), exist_ok=True)

# Load input data from jsonl file
print(f"Loading data from {INPUT_JSONL_FILE}...")
dt = load_jsonl(INPUT_JSONL_FILE)
print(f"Loaded {len(dt)} items from {INPUT_JSONL_FILE}")

# Load existing answers if output file exists
qa_results = {}

if os.path.exists(OUTPUT_JSON_FILE):
    try:
        with open(OUTPUT_JSON_FILE, 'r', encoding='utf-8') as f:
            qa_results = json.load(f)
        print(f"Loaded {len(qa_results)} existing answers from {OUTPUT_JSON_FILE}")
    except (json.JSONDecodeError, FileNotFoundError):
        print(f"Could not load existing answers from {OUTPUT_JSON_FILE}, starting fresh")
        qa_results = {}
else:
    print(f"No existing result file found, creating new {OUTPUT_JSON_FILE}")
    # Initialize empty JSON file
    with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump({}, f)

def save_answer_to_json(item_id, question, answer):
    """Thread-safe function to save a single answer to JSON file"""
    # Load existing results
    try:
        with open(OUTPUT_JSON_FILE, 'r', encoding='utf-8') as f:
            existing_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_results = {}
    
    # Add new answer
    existing_results[item_id] = {
        'question': question,
        'answer': answer
    }
    
    # Save back to file
    with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(existing_results, f, indent=2, ensure_ascii=False)

# Run inference (sequential or concurrent based on mode)
print("=" * 80)
print(f"RUNNING INFERENCE ({mode.upper()} mode)")
print("=" * 80)

items_needing_inference = []

# Filter out items that already have answers
for idx, item in enumerate(dt):
    item_id = item.get('id', f"item_{idx}")
    if item_id not in qa_results:
        items_needing_inference.append(item)

print(f"Total items: {len(dt)}")
print(f"Already have answers (skipped): {len(qa_results)}")
print(f"Items needing inference: {len(items_needing_inference)}")

if items_needing_inference:
    successful_inference = 0
    failed_inference = 0
    
    if mode == 'sequential':
        # Sequential inference
        for item in tqdm(items_needing_inference, desc="Running inference", unit="item"):
            result = run_inference(item)
            
            if result['success'] and result['answer']:
                save_answer_to_json(result['id'], result['question'], result['answer'])
                qa_results[result['id']] = {
                    'question': result['question'],
                    'answer': result['answer']
                }
                successful_inference += 1
            else:
                # Save error message to JSON
                save_answer_to_json(result['id'], result['question'], result['answer'])
                qa_results[result['id']] = {
                    'question': result['question'],
                    'answer': result['answer']
                }
                failed_inference += 1
                error_msg = result['error'] if result['error'] else "Failed to generate answer"
                print(f"\nFailed inference for item '{result['id']}': {error_msg}")
    
    else:  # concurrent mode
        # Concurrent inference
        print(f"Using {MAX_WORKERS} concurrent workers for inference...")
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all inference tasks
            future_to_item = {executor.submit(run_inference, item): item for item in items_needing_inference}
            
            # Process completed tasks with progress tracking
            with tqdm(total=len(items_needing_inference), desc="Running inference", unit="item") as pbar:
                for future in as_completed(future_to_item):
                    result = future.result()
                    
                    if result['success'] and result['answer']:
                        save_answer_to_json(result['id'], result['question'], result['answer'])
                        qa_results[result['id']] = {
                            'question': result['question'],
                            'answer': result['answer']
                        }
                        successful_inference += 1
                        pbar.set_postfix({
                            'Success': successful_inference, 
                            'Failed': failed_inference,
                            'Current': result['id'][:30] + '...' if len(result['id']) > 30 else result['id']
                        })
                    else:
                        # Save error message to JSON
                        save_answer_to_json(result['id'], result['question'], result['answer'])
                        qa_results[result['id']] = {
                            'question': result['question'],
                            'answer': result['answer']
                        }
                        failed_inference += 1
                        error_msg = result['error'] if result['error'] else "Failed to generate answer"
                        print(f"\nFailed inference for item '{result['id']}': {error_msg}")
                        pbar.set_postfix({
                            'Success': successful_inference, 
                            'Failed': failed_inference,
                            'Current': f"FAILED: {result['id'][:20]}..."
                        })
                    
                    pbar.update(1)
    
    print(f"\nInference phase complete!")
    print(f"Successful inference: {successful_inference}")
    print(f"Failed inference: {failed_inference}")
else:
    print("\nNo items need inference (all already processed)")

# Final summary
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(f"Total items in dataset: {len(dt)}")
print(f"Total answers in output file: {len(qa_results)}")
print(f"All answers saved to: {OUTPUT_JSON_FILE}")

