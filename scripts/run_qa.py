import os
import glob
import json
import argparse
import traceback
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from word2png_function import text_to_images
from vlm_inference import vlm_inference

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Process QA dataset with text-to-image rendering and VLM question answering")
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
        '--dpi',
        type=int,
        default=72
    )
    parser.add_argument(
        '--render-only',
        action='store_true',
        help='Only render images without running inference'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=256000,
        help='Maximum number of input tokens when truncating images (default: 256000)'
    )
    return parser.parse_args()

# Parse command line arguments
args = parse_arguments()
port = args.port
model = args.model
mode = args.mode
num_workers = args.workers
dpi = args.dpi
render_only = args.render_only
max_input_tokens = args.max_tokens
task = "loong" # Hard code as QA task

IMAGE_TOKENS = 259

if "qwen" in model.lower():
    if dpi == 72:
        IMAGE_TOKENS = 470
    elif dpi == 96:
        IMAGE_TOKENS = 842
elif "glyph" in model.lower():
    if dpi == 72:
        IMAGE_TOKENS = 632
    elif dpi == 96:
        IMAGE_TOKENS = 1122

MAX_INPUT_IMAGES = max_input_tokens // IMAGE_TOKENS

# Hard-coded input jsonl file path
INPUT_JSONL_FILE = './loong_process_100k.jsonl'  # Hard-coded input file

# Configuration
CONFIG_EN_PATH = f'../config/config_en_dpi{dpi}.json'
OUTPUT_BASE_DIR = f'./{task}_images_dpi{dpi}'
OUTPUT_JSON_FILE = f'./results_{task}/{model}_{task}_dpi{dpi}.json'
MAX_WORKERS = num_workers

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

def render_doc_images(item, item_index=None):
    """Render a single document to images"""
    doc_text = item['docs']
    item_id = item.get('id', f"item_{item_index if item_index is not None else 'unknown'}")
    question = item.get('final_question', '')
    
    try:
        # Clean item ID for use as directory name
        safe_id = "".join(c for c in str(item_id) if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_id = safe_id.replace(' ', '_')
        
        doc_image_dir = os.path.join(OUTPUT_BASE_DIR, safe_id)
        
        # Check if images are already rendered
        if os.path.exists(doc_image_dir) and os.listdir(doc_image_dir):
            image_paths = glob.glob(os.path.join(doc_image_dir, '*.png'))
            return {
                'id': item_id,
                'question': question,
                'safe_id': safe_id,
                'image_paths': sorted(image_paths),
                'success': True,
                'error': None,
                'skipped': True
            }
        else:
            # Render document text to images
            image_paths = text_to_images(
                text=doc_text,
                output_dir=OUTPUT_BASE_DIR,
                config_path=CONFIG_EN_PATH,
                unique_id=safe_id
            )
            
            return {
                'id': item_id,
                'question': question,
                'safe_id': safe_id,
                'image_paths': sorted(image_paths),
                'success': True,
                'error': None,
                'skipped': False
            }
        
    except Exception as e:
        return {
            'id': item_id,
            'question': question,
            'safe_id': None,
            'image_paths': None,
            'success': False,
            'error': str(e),
            'skipped': False
        }

def run_inference(item_info):
    """Run VLM inference on rendered document images"""
    try:
        answer = vlm_inference(
            question=item_info['question'],
            image_paths=item_info['image_paths'],
            api_url=f"http://localhost:{port}/v1/chat/completions",
            model_name=model,
            max_input_tokens=max_input_tokens,
            max_tokens=16384 if "thinking" in model.lower() else 8192,
            max_images=MAX_INPUT_IMAGES
        )
        
        return {
            'id': item_info['id'],
            'question': item_info['question'],
            'answer': answer,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        error_trace = traceback.format_exc()
        return {
            'id': item_info['id'],
            'question': item_info['question'],
            'answer': f"Image Inference Error: {error_trace}",
            'success': False,
            'error': error_trace  # Use full error trace for output
        }

# Create base output directory if it doesn't exist
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
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

# PHASE 1: Render all images sequentially
print("=" * 80)
print("PHASE 1: RENDERING IMAGES (Sequential)")
print("=" * 80)

items_needing_inference = []
rendering_failed = 0
rendering_skipped = 0
rendering_success = 0

for idx, item in enumerate(tqdm(dt, desc="Rendering document images", unit="item")):
    # Skip items that already have answers
    item_id = item.get('id', f"item_{idx}")
    if item_id in qa_results:
        rendering_skipped += 1
        continue
    
    result = render_doc_images(item, item_index=idx)
    
    if result['success']:
        if result['skipped']:
            print(f"\nImages for item '{result['id']}' already exist, using existing images.")
        else:
            print(f"\nRendered {len(result['image_paths'])} images for item '{result['id']}'")
        
        items_needing_inference.append(result)
        rendering_success += 1
    else:
        rendering_failed += 1
        print(f"\nFailed to render item '{result['id']}': {result['error']}")

print(f"\nRendering phase complete!")
print(f"Total items: {len(dt)}")
print(f"Already have answers (skipped): {len(qa_results)}")
print(f"Successfully rendered/loaded: {rendering_success}")
print(f"Failed to render: {rendering_failed}")

# PHASE 2: Run inference (sequential or concurrent based on mode)
if render_only:
    print("\n" + "=" * 80)
    print("RENDER-ONLY MODE: Skipping inference phase")
    print("=" * 80)
    print(f"All images have been rendered to: {OUTPUT_BASE_DIR}")
elif items_needing_inference:
    print("\n" + "=" * 80)
    print(f"PHASE 2: RUNNING INFERENCE ({mode.upper()} mode)")
    print("=" * 80)
    
    successful_inference = 0
    failed_inference = 0
    
    if mode == 'sequential':
        # Sequential inference
        for item_info in tqdm(items_needing_inference, desc="Running inference", unit="item"):
            result = run_inference(item_info)
            
            if result['success'] and result['answer']:
                save_answer_to_json(result['id'], result['question'], result['answer'])
                qa_results[result['id']] = {
                    'question': result['question'],
                    'answer': result['answer']
                }
                successful_inference += 1
            else:
                # Save full error trace to JSON
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
            future_to_item = {executor.submit(run_inference, item_info): item_info for item_info in items_needing_inference}
            
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
                        # Save full error trace to JSON
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

