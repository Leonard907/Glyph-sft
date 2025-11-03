from datasets import load_dataset
import os
import glob
import json
import argparse
import sys
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from word2png_function import text_to_images
from vlm_inference import vlm_inference

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Process book dataset with text-to-image rendering and VLM summarization")
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
    return parser.parse_args()

# Parse command line arguments
args = parse_arguments()
port = args.port
model = args.model
mode = args.mode
num_workers = args.workers
dpi = args.dpi
task = "supersummary" # Hard code as no other tasks are supported

dt = load_dataset("shipWr3ck/supersummary", split="test")

# Configuration
CONFIG_EN_PATH = f'../config/config_en_dpi{dpi}.json'
OUTPUT_BASE_DIR = f'./{task}_images_dpi{dpi}'
OUTPUT_JSON_FILE = f'./results/{model}_{task}_dpi{dpi}.json'
MAX_WORKERS = num_workers

SUMMARIZATION_PROMPT = """The input images collectively represent all the pages of a book, where each image contains a portion of the text from the book. Your task is to produce a plot summary of this book from the image contents, describing the main storylines of the book and the key characters involved. Follow these instructions on how to write the plot summary:

1. Focus on describing concrete details from the book, avoid vague descriptions of themes or motifs.
2. Ensure the summary is coherent and flows well. Write in complete paragraphs and ensure logical connections between paragraphs.
3. Identify and cover the main storylines and characters, avoid minor details or subplots that do not contribute to the main narrative.
4. Construct the summary based on solely the content present in the images, it is STRICTLY PROHIBITED to introduce any external information or assumptions about the book.
5. Your summary should be comprehensive and ideally be around 800 words in length. Always starting with "Plot Summary:" on the first line, followed by the plot summary on the subsequent lines.

Now please generate the plot summary based on these instructions."""

def render_book_images(item):
    """Render a single book to images"""
    book_text = item['input']
    book_title = item['title']
    
    try:
        # Clean book title for use as directory name
        safe_title = "".join(c for c in book_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_title = safe_title.replace(' ', '_')
        
        book_image_dir = os.path.join(OUTPUT_BASE_DIR, safe_title)
        
        # Check if images are already rendered
        if os.path.exists(book_image_dir) and os.listdir(book_image_dir):
            image_paths = glob.glob(os.path.join(book_image_dir, '*.png'))
            return {
                'title': book_title,
                'safe_title': safe_title,
                'image_paths': sorted(image_paths),
                'success': True,
                'error': None,
                'skipped': True
            }
        else:
            # Render book text to images
            image_paths = text_to_images(
                text=book_text,
                output_dir=OUTPUT_BASE_DIR,
                config_path=CONFIG_EN_PATH,
                unique_id=safe_title
            )
            
            return {
                'title': book_title,
                'safe_title': safe_title,
                'image_paths': sorted(image_paths),
                'success': True,
                'error': None,
                'skipped': False
            }
        
    except Exception as e:
        return {
            'title': book_title,
            'safe_title': None,
            'image_paths': None,
            'success': False,
            'error': str(e),
            'skipped': False
        }

def run_inference(book_info):
    """Run VLM inference on rendered book images"""
    try:
        summary = vlm_inference(
            question=SUMMARIZATION_PROMPT,
            image_paths=book_info['image_paths'],
            api_url=f"http://localhost:{port}/v1/chat/completions",
            model_name=model
        )
        
        return {
            'title': book_info['title'],
            'summary': summary,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'title': book_info['title'],
            'summary': None,
            'success': False,
            'error': str(e)
        }

# Create base output directory if it doesn't exist
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_JSON_FILE), exist_ok=True)

# Load existing summaries if output file exists
book_summaries = {}

if os.path.exists(OUTPUT_JSON_FILE):
    try:
        with open(OUTPUT_JSON_FILE, 'r', encoding='utf-8') as f:
            book_summaries = json.load(f)
        print(f"Loaded {len(book_summaries)} existing summaries from {OUTPUT_JSON_FILE}")
    except (json.JSONDecodeError, FileNotFoundError):
        print(f"Could not load existing summaries from {OUTPUT_JSON_FILE}, starting fresh")
        book_summaries = {}
else:
    print(f"No existing summary file found, creating new {OUTPUT_JSON_FILE}")
    # Initialize empty JSON file
    with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump({}, f)

def save_summary_to_json(title, summary):
    """Thread-safe function to save a single summary to JSON file"""
    # Load existing summaries
    try:
        with open(OUTPUT_JSON_FILE, 'r', encoding='utf-8') as f:
            existing_summaries = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_summaries = {}
    
    # Add new summary
    existing_summaries[title] = summary
    
    # Save back to file
    with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(existing_summaries, f, indent=2, ensure_ascii=False)

# PHASE 1: Render all images sequentially
print("=" * 80)
print("PHASE 1: RENDERING IMAGES (Sequential)")
print("=" * 80)

books_needing_inference = []
rendering_failed = 0
rendering_skipped = 0
rendering_success = 0

for item in tqdm(dt, desc="Rendering book images", unit="book"):
    # Skip books that already have summaries
    if item['title'] in book_summaries:
        rendering_skipped += 1
        continue
    
    result = render_book_images(item)
    
    if result['success']:
        if result['skipped']:
            print(f"\nImages for '{result['title']}' already exist, using existing images.")
        else:
            print(f"\nRendered {len(result['image_paths'])} images for '{result['title']}'")
        
        books_needing_inference.append(result)
        rendering_success += 1
    else:
        rendering_failed += 1
        print(f"\nFailed to render '{result['title']}': {result['error']}")

print(f"\nRendering phase complete!")
print(f"Total books: {len(dt)}")
print(f"Already have summaries (skipped): {len(book_summaries)}")
print(f"Successfully rendered/loaded: {rendering_success}")
print(f"Failed to render: {rendering_failed}")

# PHASE 2: Run inference (sequential or concurrent based on mode)
if books_needing_inference:
    print("\n" + "=" * 80)
    print(f"PHASE 2: RUNNING INFERENCE ({mode.upper()} mode)")
    print("=" * 80)
    
    successful_inference = 0
    failed_inference = 0
    
    if mode == 'sequential':
        # Sequential inference
        for book_info in tqdm(books_needing_inference, desc="Running inference", unit="book"):
            result = run_inference(book_info)
            
            if result['success'] and result['summary']:
                save_summary_to_json(result['title'], result['summary'])
                book_summaries[result['title']] = result['summary']
                successful_inference += 1
                print(f"\nGenerated summary for '{result['title']}'")
            else:
                failed_inference += 1
                error_msg = result['error'] if result['error'] else "Failed to generate summary"
                print(f"\nFailed inference for '{result['title']}': {error_msg}")
    
    else:  # concurrent mode
        # Concurrent inference
        print(f"Using {MAX_WORKERS} concurrent workers for inference...")
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all inference tasks
            future_to_book = {executor.submit(run_inference, book_info): book_info for book_info in books_needing_inference}
            
            # Process completed tasks with progress tracking
            with tqdm(total=len(books_needing_inference), desc="Running inference", unit="book") as pbar:
                for future in as_completed(future_to_book):
                    result = future.result()
                    
                    if result['success'] and result['summary']:
                        save_summary_to_json(result['title'], result['summary'])
                        book_summaries[result['title']] = result['summary']
                        successful_inference += 1
                        pbar.set_postfix({
                            'Success': successful_inference, 
                            'Failed': failed_inference,
                            'Current': result['title'][:30] + '...' if len(result['title']) > 30 else result['title']
                        })
                    else:
                        failed_inference += 1
                        error_msg = result['error'] if result['error'] else "Failed to generate summary"
                        print(f"\nFailed inference for '{result['title']}': {error_msg}")
                        pbar.set_postfix({
                            'Success': successful_inference, 
                            'Failed': failed_inference,
                            'Current': f"FAILED: {result['title'][:20]}..."
                        })
                    
                    pbar.update(1)
    
    print(f"\nInference phase complete!")
    print(f"Successful inference: {successful_inference}")
    print(f"Failed inference: {failed_inference}")
else:
    print("\nNo books need inference (all already processed)")

# Final summary
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(f"Total books in dataset: {len(dt)}")
print(f"Total summaries in output file: {len(book_summaries)}")
print(f"All summaries saved to: {OUTPUT_JSON_FILE}")