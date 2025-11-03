#!/usr/bin/env python3
"""
Simple Text to Images Inference

Usage:
    python inference_code.py
"""
from word2png_function import text_to_images

CONFIG_EN_PATH = '../config/config_en.json'
OUTPUT_DIR = './output_images'
INPUT_FILE = '/mnt/lituou/Book_Builder/epub/All_Fours_full.txt'

# Read text from file
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    text = f.read()
# import time 
# a = time.time()
images = text_to_images(
    text= text,
    output_dir=OUTPUT_DIR,
    config_path=CONFIG_EN_PATH,
    unique_id='book_test2'
)
# b = time.time()
# print(f"Time taken: {b - a} seconds")

print(f"\nGenerated {len(images)} image(s):")
for img_path in images:
    print(f"  {img_path}")