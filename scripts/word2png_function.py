#!/usr/bin/env python3
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import io
import os
import json
import numpy as np
import gc
from pdf2image import pdfinfo_from_bytes, convert_from_bytes
import re
from multiprocessing import Pool
from tqdm import tqdm
from xml.sax.saxutils import escape
import shutil

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib import colors

# Alignment mapping
ALIGN_MAP = {
    "LEFT": TA_LEFT,
    "CENTER": TA_CENTER,
    "RIGHT": TA_RIGHT,
    "JUSTIFY": TA_JUSTIFY,
}

# Global variables for multiprocessing
GLOBAL_CONFIG = None
OUTPUT_DIR = None
recover = False


def load_config(config_path):
    """Load configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Convert colors
    if 'page-bg-color' in config and isinstance(config['page-bg-color'], str):
        config['page-bg-color'] = colors.HexColor(config['page-bg-color'])
    if 'font-color' in config and isinstance(config['font-color'], str):
        config['font-color'] = colors.HexColor(config['font-color'])
    if 'para-bg-color' in config and isinstance(config['para-bg-color'], str):
        config['para-bg-color'] = colors.HexColor(config['para-bg-color'])
    if 'para-border-color' in config and isinstance(config['para-border-color'], str):
        config['para-border-color'] = colors.HexColor(config['para-border-color'])
    if 'highlight-color' in config and isinstance(config['highlight-color'], str):
        config['highlight-color'] = colors.HexColor(config['highlight-color'])
    
    # Convert alignment
    if 'alignment' in config and isinstance(config['alignment'], str):
        config['alignment'] = ALIGN_MAP.get(config['alignment'], TA_JUSTIFY)
    
    # Convert page size
    if 'page-size' in config and isinstance(config['page-size'], str):
        config['page-size'] = tuple(map(float, config['page-size'].split(',')))
    
    return config


def apply_highlight_ranges(text, highlight_ranges, highlight_color='#FF0000'):
    """
    Apply color highlighting to specific character ranges in text
    
    Args:
        text: Original text string
        highlight_ranges: List of tuples [(start, end), ...] where start and end are character indices
        highlight_color: Color to apply (default red)
        
    Returns:
        Text with HTML markup for highlighting
    """
    if not highlight_ranges:
        return text
    
    # Sort ranges by start position
    sorted_ranges = sorted(highlight_ranges, key=lambda x: x[0])
    
    # Validate and merge overlapping ranges
    merged_ranges = []
    for start, end in sorted_ranges:
        if start < 0 or end > len(text) or start >= end:
            continue  # Skip invalid ranges
        
        if merged_ranges and start <= merged_ranges[-1][1]:
            # Merge overlapping ranges
            merged_ranges[-1] = (merged_ranges[-1][0], max(merged_ranges[-1][1], end))
        else:
            merged_ranges.append((start, end))
    
    # Build result string with markup
    result_parts = []
    last_pos = 0
    
    for start, end in merged_ranges:
        # Add text before highlight
        if start > last_pos:
            result_parts.append(escape(text[last_pos:start]))
        
        # Add highlighted text
        highlighted_text = escape(text[start:end])
        result_parts.append(f'<font color="{highlight_color}">{highlighted_text}</font>')
        
        last_pos = end
    
    # Add remaining text
    if last_pos < len(text):
        result_parts.append(escape(text[last_pos:]))
    
    return ''.join(result_parts)


def text_to_images(text, output_dir, config_path=None, config_dict=None, unique_id=None, highlight_ranges=None):
    """
    Convert text to images - Inference interface
    
    Args:
        text: Input text content
        output_dir: Image output directory
        config_path: Configuration file path (optional)
        config_dict: Configuration dictionary (optional, higher priority than config_path)
        unique_id: Unique identifier (optional, auto-generated if not provided)
        highlight_ranges: List of character index ranges to highlight [(start, end), ...]
        
    Returns:
        list: List of generated image paths
        
    Example:
        >>> images = text_to_images(
        ...     text="Hello World",
        ...     output_dir="./output",
        ...     config_path="config.json",
        ...     highlight_ranges=[(0, 5)]  # Highlight "Hello"
        ... )
        >>> print(images)  # ['./output/xxx/page_001.png', ...]
    """
    # Load configuration
    if config_dict is None:
        if config_path is None:
            raise ValueError("Must provide either config_path or config_dict")
        config = load_config(config_path)
    else:
        config = config_dict.copy()
        # Convert special fields in config
        if 'page-bg-color' in config and isinstance(config['page-bg-color'], str):
            config['page-bg-color'] = colors.HexColor(config['page-bg-color'])
        if 'font-color' in config and isinstance(config['font-color'], str):
            config['font-color'] = colors.HexColor(config['font-color'])
        if 'para-bg-color' in config and isinstance(config['para-bg-color'], str):
            config['para-bg-color'] = colors.HexColor(config['para-bg-color'])
        if 'para-border-color' in config and isinstance(config['para-border-color'], str):
            config['para-border-color'] = colors.HexColor(config['para-border-color'])
        if 'highlight-color' in config and isinstance(config['highlight-color'], str):
            config['highlight-color'] = colors.HexColor(config['highlight-color'])
        if 'alignment' in config and isinstance(config['alignment'], str):
            config['alignment'] = ALIGN_MAP.get(config['alignment'], TA_JUSTIFY)
        if 'page-size' in config and isinstance(config['page-size'], str):
            config['page-size'] = tuple(map(float, config['page-size'].split(',')))
    
    # Generate unique ID
    if unique_id is None:
        import hashlib
        unique_id = hashlib.md5(text.encode()).hexdigest()[:16]
    
    # Extract configuration parameters
    page_size = config.get('page-size', A4)
    margin_x = config.get('margin-x', 20)
    margin_y = config.get('margin-y', 20)
    font_path = config.get('font-path')
    assert font_path, "Must provide font-path"
    
    font_name = os.path.basename(font_path).split('.')[0]
    font_size = config.get('font-size', 9)
    line_height = config.get('line-height') or (font_size + 1)
    
    page_bg_color = config.get('page-bg-color', colors.HexColor('#FFFFFF'))
    font_color = config.get('font-color', colors.HexColor('#000000'))
    para_bg_color = config.get('para-bg-color', colors.HexColor('#FFFFFF'))
    para_border_color = config.get('para-border-color', colors.HexColor('#FFFFFF'))
    highlight_color = config.get('highlight-color', '#FF0000')
    
    first_line_indent = config.get('first-line-indent', 0)
    left_indent = config.get('left-indent', 0)
    right_indent = config.get('right-indent', 0)
    alignment = config.get('alignment', TA_JUSTIFY)
    space_before = config.get('space-before', 0)
    space_after = config.get('space-after', 0)
    border_width = config.get('border-width', 0)
    border_padding = config.get('border-padding', 0)
    
    horizontal_scale = config.get('horizontal-scale', 1.0)
    dpi = config.get('dpi', 72)
    auto_crop_last_page = config.get('auto-crop-last-page', False)
    auto_crop_width = config.get('auto-crop-width', False)
    newline_markup = config.get('newline-markup', '<br/>')
    
    # Register font
    try:
        pdfmetrics.registerFont(TTFont(font_name, font_path))
    except:
        pass  # Font already registered
    
    # Create PDF
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=page_size,
        leftMargin=margin_x,
        rightMargin=margin_x,
        topMargin=margin_y,
        bottomMargin=margin_y,
    )
    
    # Create paragraph style
    styles = getSampleStyleSheet()
    RE_CJK = re.compile(r'[\u4E00-\u9FFF]')
    
    custom = ParagraphStyle(
        name="Custom",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=font_size,
        leading=line_height,
        textColor=font_color,
        backColor=para_bg_color,
        borderColor=para_border_color,
        borderWidth=border_width,
        borderPadding=border_padding,
        firstLineIndent=first_line_indent,
        wordWrap="CJK" if RE_CJK.search(text) else None,
        leftIndent=left_indent,
        rightIndent=right_indent,
        alignment=alignment,
        spaceBefore=space_before,
        spaceAfter=space_after,
    )
    
    # Process text with highlighting
    text = text.replace('\xad', '').replace('\u200b', '')
    
    # Apply highlighting if ranges provided
    if highlight_ranges:
        processed_text = apply_highlight_ranges(text, highlight_ranges, highlight_color)
    else:
        processed_text = escape(text)
    
    # Replace multiple spaces
    def replace_spaces(s):
        # Don't replace spaces inside font tags
        parts = re.split(r'(<font[^>]*>.*?</font>)', s)
        result = []
        for part in parts:
            if part.startswith('<font'):
                result.append(part)
            else:
                result.append(re.sub(r' {2,}', lambda m: '&nbsp;'*len(m.group()), part))
        return ''.join(result)
    
    processed_text = replace_spaces(processed_text)
    parts = processed_text.split('\n')
    
    # Create paragraphs in batches
    story = []
    turns = 30
    def balance_font_tags(text):
        """
        Properly balance font tags by tracking the stack of open tags.
        Handles cases where tags are improperly nested or closed.
        """
        stack = []
        result = []
        i = 0
        
        while i < len(text):
            # Check for opening font tag
            font_open = re.match(r'<font[^>]*>', text[i:])
            if font_open:
                tag = font_open.group(0)
                stack.append(tag)
                result.append(tag)
                i += len(tag)
                continue
            
            # Check for closing font tag
            font_close = re.match(r'</font>', text[i:])
            if font_close:
                if stack:
                    stack.pop()
                    result.append('</font>')
                else:
                    # Closing tag without opening - extract color and add opening tag
                    result.insert(0, '<font color="#FF0000">')
                    result.append('</font>')
                i += len(font_close.group(0))
                continue
            
            # Regular character
            result.append(text[i])
            i += 1
        
        # Close any remaining unclosed tags
        while stack:
            result.append('</font>')
            stack.pop()
        
        return ''.join(result)


    # Use in your loop:
    for i in range(0, len(parts), turns):
        tmp_text = newline_markup.join(parts[i:i+turns])
        
        # Balance font tags properly
        tmp_text = balance_font_tags(tmp_text)
        
        try:
            story.append(Paragraph(tmp_text, custom))
        except Exception as e:
            print(f"Error creating paragraph: {e}")
            print(f"Text snippet: {tmp_text[:200]}...")
            # Optionally use pdb for debugging
            # import pdb; pdb.set_trace()
    
    # Build PDF
    doc.build(
        story,
        onFirstPage=lambda c, d: (c.saveState(), c.setFillColor(page_bg_color), c.rect(0, 0, page_size[0], page_size[1], stroke=0, fill=1), c.restoreState()),
        onLaterPages=lambda c, d: (c.saveState(), c.setFillColor(page_bg_color), c.rect(0, 0, page_size[0], page_size[1], stroke=0, fill=1), c.restoreState())
    )
    
    pdf_bytes = buf.getvalue()
    buf.close()
    
    # Create output directory
    out_root = os.path.join(output_dir, unique_id)
    os.makedirs(out_root, exist_ok=True)
    
    # Convert PDF to images
    info = pdfinfo_from_bytes(pdf_bytes)
    num_pages = total = info["Pages"]
    batch = 20
    image_paths = []
    
    for start in range(1, total + 1, batch):
        end = min(start + batch - 1, total)
        images = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=start, last_page=end)
        
        for offset, img in enumerate(images, start=start):
            w, h = img.size
            
            # Horizontal scaling
            if horizontal_scale != 1.0:
                img = img.resize((int(w * horizontal_scale), h))
            
            # Adaptive cropping
            if auto_crop_width or (auto_crop_last_page and offset == num_pages):
                gray = np.array(img.convert("L"))
                bg_gray = np.median(gray[:2, :2])
                tolerance = 5
                mask = np.abs(gray - bg_gray) > tolerance
                
                if auto_crop_width:
                    cols = np.where(mask.any(axis=0))[0]
                    if cols.size:
                        rightmost_col = cols[-1] + 1
                        right = min(img.width, rightmost_col + margin_x)
                        img = img.crop((0, 0, right, img.height))
                
                if auto_crop_last_page and offset == num_pages:
                    rows = np.where(mask.any(axis=1))[0]
                    if rows.size:
                        last_row = rows[-1]
                        lower = min(img.height, last_row + margin_y)
                        img = img.crop((0, 0, img.width, lower))
            
            out_path = os.path.join(out_root, f"page_{offset:03d}.png")
            img.save(out_path, 'PNG')
            image_paths.append(os.path.abspath(out_path))
            img.close()
        
        images.clear()
        del images
    
    del pdf_bytes
    gc.collect()
    
    return image_paths


def process_one(item):
    """Process single item - for batch processing"""
    global GLOBAL_CONFIG, OUTPUT_DIR, recover
    
    _id = item.get('unique_id')
    assert _id
    
    # Check recovery mode
    if recover and os.path.exists(os.path.join(OUTPUT_DIR, _id)):
        item['image_paths'] = []
        return item
    
    # Parse configuration
    item_config = item.get('config', {}) or {}
    config = {**GLOBAL_CONFIG, **item_config}
    
    # Process special fields in item config
    if 'page-size' in item_config and isinstance(item_config['page-size'], str):
        config['page-size'] = tuple(map(float, item_config['page-size'].split(',')))
    if 'page-bg-color' in item_config and isinstance(item_config['page-bg-color'], str):
        config['page-bg-color'] = colors.HexColor(item_config['page-bg-color'])
    if 'font-color' in item_config and isinstance(item_config['font-color'], str):
        config['font-color'] = colors.HexColor(item_config['font-color'])
    if 'para-bg-color' in item_config and isinstance(item_config['para-bg-color'], str):
        config['para-bg-color'] = colors.HexColor(item_config['para-bg-color'])
    if 'para-border-color' in item_config and isinstance(item_config['para-border-color'], str):
        config['para-border-color'] = colors.HexColor(item_config['para-border-color'])
    if 'highlight-color' in item_config and isinstance(item_config['highlight-color'], str):
        config['highlight-color'] = colors.HexColor(item_config['highlight-color'])
    if 'alignment' in item_config and isinstance(item_config['alignment'], str):
        config['alignment'] = ALIGN_MAP.get(item_config['alignment'], TA_JUSTIFY)
    
    # Get text and highlight ranges
    text = item.get('context', '')
    assert text
    
    highlight_ranges = item.get('highlight_ranges', None)
    
    # Call inference function
    image_paths = text_to_images(
        text=text,
        output_dir=OUTPUT_DIR,
        config_dict=config,
        unique_id=_id,
        highlight_ranges=highlight_ranges
    )
    
    item['image_paths'] = image_paths
    return item


def batch_process_to_images(json_path, output_dir, output_jsonl_path, 
                            config_path, processes=16, is_recover=False, batch_size=100):
    """Batch process JSON data to generate images"""
    global GLOBAL_CONFIG, OUTPUT_DIR, recover
    
    # Set global variables
    GLOBAL_CONFIG = load_config(config_path)
    OUTPUT_DIR = output_dir
    recover = is_recover
    
    print(f"Loaded config from: {config_path}")
    
    # Prepare output directory
    if not recover:
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        if os.path.exists(output_jsonl_path):
            os.remove(output_jsonl_path)
    
    # Read data
    with open(json_path, 'r', encoding='utf-8') as f:
        data_to_process = json.load(f)
    
    # Get already processed IDs
    processed_ids = set()
    if recover and os.path.exists(output_jsonl_path):
        with open(output_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    processed_ids.add(item.get('unique_id'))
                except:
                    continue
        print(f"Found {len(processed_ids)} already processed items")
    
    # Filter processed items
    data_to_process = [item for item in data_to_process 
                      if item.get('unique_id') not in processed_ids]
    print(f"Remaining items to process: {len(data_to_process)}")
    
    if not data_to_process:
        print("All items processed")
        return
    
    # Parallel processing
    batch_buffer = []
    
    with Pool(processes=processes) as pool:
        for result_item in tqdm(pool.imap_unordered(process_one, data_to_process, chunksize=1), 
                               total=len(data_to_process)):
            if result_item:
                batch_buffer.append(result_item)
                _id = result_item.get('unique_id', 'UNKNOWN')
                count = len(result_item.get('image_paths', []))
                tqdm.write(f"{_id}: generated {count} pages")
                
                # Batch write
                if len(batch_buffer) >= batch_size:
                    with open(output_jsonl_path, 'a', encoding='utf-8') as f:
                        for item in batch_buffer:
                            f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    batch_buffer = []
    
    # Write remaining items
    if batch_buffer:
        with open(output_jsonl_path, 'a', encoding='utf-8') as f:
            for item in batch_buffer:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print("Processing complete")


if __name__ == '__main__':
    # Example 1: Single text inference with highlighting
    CONFIG_PATH = '/mnt/lituou/Glyph-sft/config/config_en_dpi72.json'
    text = open('input.txt', 'r').read()
    OUTPUT_DIR = './output_images'
    
    # Highlight "test" (characters 10-14) and "Second" (characters 20-26)
    highlight_ranges = []
    
    images = text_to_images(
        text=text,
        output_dir=OUTPUT_DIR,
        config_path=CONFIG_PATH,
        unique_id='test_001',
        highlight_ranges=highlight_ranges
    )
    print(f"Generated {len(images)} images:")
    for img in images:
        print(f"  {img}")
    
    # Example 2: Batch processing with highlighting
    # Each item in JSON can have a "highlight_ranges" field: [[start1, end1], [start2, end2], ...]
    # CONFIG_PATH = '../config/config.json'
    # JSON_PATH = '../evaluation/mrcr/data/processed_2needle_0-128k.json'
    # OUTPUT_JSONL_PATH = '../evaluation/mrcr/data/processed_2needle_0-128k.jsonl'
    # OUTPUT_DIR = '../evaluation/mrcr/rendered_images'
    
    # batch_process_to_images(
    #     json_path=JSON_PATH,
    #     output_dir=OUTPUT_DIR,
    #     output_jsonl_path=OUTPUT_JSONL_PATH,
    #     config_path=CONFIG_PATH,
    #     processes=16,
    #     is_recover=True,
    #     batch_size=100
    # )