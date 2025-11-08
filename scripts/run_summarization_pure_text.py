from datasets import load_dataset
import os
import json
import argparse
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests
from transformers import AutoTokenizer


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description=(
            "Process the SuperSummary dataset by sending the book text as a "
            "pure-text prompt to the model"
        )
    )
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
        '--tokenizer',
        type=str,
        default='Qwen/Qwen2.5-7B-Instruct',
        help='HuggingFace tokenizer to use for token counting (default: Qwen/Qwen2.5-7B-Instruct)'
    )
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=120000,
        help='Maximum number of tokens for book text (default: 255000)'
    )
    return parser.parse_args()


# Parse command line arguments
args = parse_arguments()
port = args.port
model = args.model or "glyph"
mode = args.mode
num_workers = args.workers
task = "supersummary"  # Hard code as no other tasks are supported
max_book_tokens = args.max_tokens

# Load tokenizer
print(f"Loading tokenizer: {args.tokenizer}")
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
print(f"Tokenizer loaded successfully")

dt = load_dataset("shipWr3ck/supersummary", split="test")

# Configuration
OUTPUT_JSON_FILE = f'./results/{model}_{task}_pure_text.json'
MAX_WORKERS = num_workers

SUMMARIZATION_PROMPT = """You will be given full text from a book. Your task is to produce a plot summary of this book, describing the main storylines of the book and the key characters involved. Follow these instructions on how to write the plot summary:

1. Focus on describing concrete details from the book, avoid vague descriptions of themes or motifs.
2. Ensure the summary is coherent and flows well. Write in complete paragraphs and ensure logical connections between paragraphs.
3. Identify and cover the main storylines and characters, avoid minor details or subplots that do not contribute to the main narrative.
4. Construct the summary based on solely the content provided, it is STRICTLY PROHIBITED to introduce any external information or assumptions about the book.
5. Your summary should be comprehensive and ideally be around 800 words in length. Always starting with "Plot Summary:" on the first line, followed by the plot summary on the subsequent lines.

Full Book Text:
<book>
{book_text}
</book>

Now please generate the plot summary based on the full book text and instructions."""


def truncate_book_text(book_text, max_tokens):
    """Truncate book text to stay below max_tokens using the loaded tokenizer."""
    # Tokenize the book text
    tokens = tokenizer.encode(book_text, add_special_tokens=False)
    
    # If already below limit, return as is
    if len(tokens) <= max_tokens:
        return book_text, len(tokens)
    
    # Truncate tokens to max_tokens
    truncated_tokens = tokens[:max_tokens]
    
    # Decode back to text
    truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    
    return truncated_text, len(truncated_tokens)


def build_prompt(book_title, book_text):
    """Construct the prompt for a single book, truncating if necessary."""
    # Truncate book text to stay below token limit
    truncated_text, token_count = truncate_book_text(book_text, max_book_tokens)
    
    if not truncated_text or not truncated_text.strip():
        raise ValueError("Book text is empty after truncation")
    
    # Format the prompt with the truncated text
    prompt = SUMMARIZATION_PROMPT.format(book_text=truncated_text)
    
    return prompt, token_count


def prepare_book_prompt(item):
    """Prepare the prompt for a single book entry."""
    book_text = item['input']
    book_title = item['title']

    try:
        prompt, token_count = build_prompt(book_title, book_text)
        return {
            'title': book_title,
            'prompt': prompt,
            'token_count': token_count,
            'success': True,
            'error': None
        }
    except Exception as e:
        return {
            'title': book_title,
            'prompt': None,
            'token_count': 0,
            'success': False,
            'error': str(e)
        }


def run_text_inference(book_info):
    """Run inference for a prepared book prompt via direct text-only request."""
    api_url = f"http://localhost:{port}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": book_info['prompt']
            }
        ],
        "max_tokens": 8192,
        "temperature": 0.0001,
        "top_p": 1.0,
        "top_k": 1,
        "repetition_penalty": 1.1,
        "skip_special_tokens": False,
        "stop_token_ids": [151329, 151348, 151336],
        "include_stop_str_in_output": True
    }

    try:
        response = requests.post(
            api_url,
            headers={'Content-Type': 'application/json'},
            data=json.dumps(payload),
            timeout=1200
        )
        response.raise_for_status()
        result = response.json()

        if not result.get('choices'):
            raise ValueError(f"Invalid response structure: {result}")

        summary = result['choices'][0]['message']['content']

        return {
            'title': book_info['title'],
            'summary': summary,
            'success': True,
            'error': None
        }

    except Exception as e:
        error_trace = traceback.format_exc()
        return {
            'title': book_info['title'],
            'summary': f"Text Inference Error: {error_trace}",
            'success': False,
            'error': str(e)
        }


# Create output directory if it doesn't exist
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
    with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump({}, f)


def save_summary_to_json(title, summary):
    """Thread-safe function to save a single summary to JSON file."""
    try:
        with open(OUTPUT_JSON_FILE, 'r', encoding='utf-8') as f:
            existing_summaries = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_summaries = {}

    existing_summaries[title] = summary

    with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(existing_summaries, f, indent=2, ensure_ascii=False)


# PHASE 1: Prepare prompts
print("=" * 80)
print("PHASE 1: PREPARING TEXT PROMPTS")
print(f"Max tokens per book: {max_book_tokens}")
print("=" * 80)

books_needing_inference = []
preparation_success = 0
preparation_failed = 0
preparation_skipped = 0

for item in tqdm(dt, desc="Preparing book prompts", unit="book"):
    if item['title'] in book_summaries:
        preparation_skipped += 1
        continue

    result = prepare_book_prompt(item)

    if result['success']:
        books_needing_inference.append(result)
        preparation_success += 1
        print(
            f"\nPrepared prompt with {result['token_count']} tokens for "
            f"'{result['title']}'"
        )
    else:
        preparation_failed += 1
        print(f"\nFailed to prepare '{result['title']}': {result['error']}")

print(f"\nPrompt preparation complete!")
print(f"Total books: {len(dt)}")
print(f"Already have summaries (skipped): {len(book_summaries)}")
print(f"Successfully prepared: {preparation_success}")
print(f"Failed to prepare: {preparation_failed}")


# PHASE 2: Run inference
if books_needing_inference:
    print("\n" + "=" * 80)
    print(f"PHASE 2: RUNNING INFERENCE ({mode.upper()} mode)")
    print("=" * 80)

    successful_inference = 0
    failed_inference = 0

    if mode == 'sequential':
        for book_info in tqdm(books_needing_inference, desc="Running inference", unit="book"):
            result = run_text_inference(book_info)

            if result['success'] and result['summary']:
                save_summary_to_json(result['title'], result['summary'])
                book_summaries[result['title']] = result['summary']
                successful_inference += 1
                print(f"\nGenerated summary for '{result['title']}'")
            else:
                save_summary_to_json(result['title'], result['summary'])
                book_summaries[result['title']] = result['summary']
                failed_inference += 1
                error_msg = result['error'] if result['error'] else "Failed to generate summary"
                print(f"\nFailed inference for '{result['title']}': {error_msg}")

    else:
        print(f"Using {MAX_WORKERS} concurrent workers for inference...")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_book = {
                executor.submit(run_text_inference, book_info): book_info for book_info in books_needing_inference
            }

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
                        save_summary_to_json(result['title'], result['summary'])
                        book_summaries[result['title']] = result['summary']
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