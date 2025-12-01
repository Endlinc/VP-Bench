import base64
import io
import json
import os
import pandas as pd
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

# ----------------------
# Configuration (Easy to adjust)
# ----------------------
INPUT_JSON = "/Volumes/Ming-Data/Dataset/VP_Bench/vp_bench_stage_1/update_stage_1/stage1_data_w_context_unique.json"
OUTPUT_PARQUET = "vp_bench_stage_1.parquet"  # 改为Parquet格式

# Image directories (primary first, then fallback)
PRIMARY_IMAGE_DIR = "/Volumes/Ming-Data/Dataset/VP_Bench/vp_bench_stage_1/update_stage_1/stage1_visual_sampled_images"
FALLBACK_IMAGE_DIR = "/Volumes/Ming-Data/Dataset/VP_Bench/vp_bench_stage_1/update_stage_1/rough_location_images"

JSON_IMAGE_FILENAME_FIELD = "vp_image"  # Only use vp_image for filename
UNIQUE_INDEX_START = 1  # Start number for unique index
CHUNK_SIZE = 100  # Process 100 entries per chunk (adjust based on your memory)


# ----------------------
# Reusable Helper Functions (Unchanged)
# ----------------------
def resize_image_by_factor(img, factor=1):
    w, h = img.size
    new_w, new_h = int(w * factor), int(h * factor)
    return img.resize((new_w, new_h))


def encode_image_to_base64(img, target_size=-1, fmt='JPEG'):
    if img.mode in ('RGBA', 'P', 'LA'):
        img = img.convert('RGB')
    if target_size > 0:
        img.thumbnail((target_size, target_size))

    img_buffer = io.BytesIO()
    img.save(img_buffer, format=fmt)
    image_data = img_buffer.getvalue()
    ret = base64.b64encode(image_data).decode('utf-8')

    # Handle min edge and max size constraints
    max_size = int(os.environ.get('VLMEVAL_MAX_IMAGE_SIZE', 1e9))
    min_edge = int(os.environ.get('VLMEVAL_MIN_IMAGE_EDGE', 1e2))

    if min(img.size) < min_edge:
        factor = min_edge / min(img.size)
        image_new = resize_image_by_factor(img, factor)
        img_buffer = io.BytesIO()
        image_new.save(img_buffer, format=fmt)
        ret = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

    factor = 1
    while len(ret) > max_size:
        factor *= 0.7
        image_new = resize_image_by_factor(img, factor)
        img_buffer = io.BytesIO()
        image_new.save(img_buffer, format=fmt)
        ret = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

    if factor < 1:
        print(f'Warning: Image resized to {factor:.2f}x (exceeds max size {max_size})')

    return ret


def encode_image_file_to_base64(image_path, target_size=-1, fmt='JPEG'):
    with Image.open(image_path) as img:
        return encode_image_to_base64(img, target_size=target_size, fmt=fmt)


def extract_question_components(full_question: str, context: str) -> tuple[str, dict[str, str]]:
    lines = [line.strip() for line in full_question.split('\n') if line.strip()]

    # Skip duplicate context lines
    context_lines = context.split('\n') if context else []
    skip_lines = 0
    for ctx_line, q_line in zip(context_lines, lines):
        if ctx_line.strip() == q_line.strip():
            skip_lines += 1
        else:
            break

    # Extract question and options
    user_question = ""
    option_dict = {"A": "", "B": "", "C": "", "D": ""}
    for line in lines[skip_lines:]:
        if line.startswith(('A.', 'B.', 'C.', 'D.')):
            letter = line[0]
            option_dict[letter] = line[2:].strip()
        elif not user_question and not line.startswith("Answer with"):
            user_question = line

    return user_question, option_dict


def get_image_base64(entry: dict) -> str | None:
    # Get filename ONLY from vp_image
    image_filename = entry.get(JSON_IMAGE_FILENAME_FIELD)
    if not image_filename:
        warnings.warn(f"vp_image empty (Original ID: {entry.get('id', 'N/A')})")
        return None

    # Check primary → fallback directory
    primary_path = os.path.join(PRIMARY_IMAGE_DIR, image_filename)
    fallback_path = os.path.join(FALLBACK_IMAGE_DIR, image_filename)

    if os.path.exists(primary_path):
        valid_path = primary_path
    elif os.path.exists(fallback_path):
        valid_path = fallback_path
    else:
        warnings.warn(f"Image not found: {image_filename} (ID: {entry.get('id', 'N/A')})")
        return None

    # Encode image
    try:
        with Image.open(valid_path) as img:
            img.verify()  # Validate integrity
        return encode_image_file_to_base64(valid_path)
    except Exception as e:
        warnings.warn(f"Encode failed: {valid_path} (Error: {str(e)[:50]})")
        return None


# ----------------------
# Chunked Processing Core Logic
# ----------------------
def process_chunk(chunk_data: list[dict], start_index: int) -> pd.DataFrame:
    """Process a single chunk of data and return DataFrame"""
    processed_rows = []
    current_index = start_index

    for item in chunk_data:
        # Extract core info (added vp_shape → category and question_type → l2-category)
        context = item.get("context", "").strip()
        full_question = item.get("question", "").strip()
        correct_answer = item.get("answer", "").strip()
        vp_amount = item.get("vp_amount", "").strip()
        vp_shape = item.get("vp_shape", "").strip()
        vp_properties = item.get("vp_properties", "").strip()

        # Extract question and options
        user_question, option_dict = extract_question_components(full_question, context)

        # Encode image
        image_base64 = get_image_base64(item)

        # Build row (added category and l2-category columns)
        processed_rows.append({
            "index": current_index,
            "context": context,
            "question": user_question,
            "A": option_dict["A"],
            "B": option_dict["B"],
            "C": option_dict["C"],
            "D": option_dict["D"],
            "answer": correct_answer,
            "image": image_base64,
            "vp_amount": vp_amount,
            "vp_shape": vp_shape,
            "vp_properties": vp_properties
        })

        current_index += 1

    return pd.DataFrame(processed_rows), current_index


def run_chunked_processing():
    """Main function: Load → Process chunks → Merge → Save as Parquet"""
    # Step 1: Load full JSON (split into chunks later)
    print(f"Loading input JSON: {INPUT_JSON}")
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    total_entries = len(raw_data)
    print(f"Total entries to process: {total_entries}")

    # Step 2: Initialize variables
    current_start_index = UNIQUE_INDEX_START
    total_chunks = (total_entries + CHUNK_SIZE - 1) // CHUNK_SIZE  # Ceiling division
    processed_dfs = []  # 收集所有分块的DataFrame

    # Step 3: Process each chunk
    for chunk_idx in range(total_chunks):
        # Calculate chunk boundaries
        start = chunk_idx * CHUNK_SIZE
        end = min((chunk_idx + 1) * CHUNK_SIZE, total_entries)
        chunk_data = raw_data[start:end]

        print(f"\n--- Processing Chunk {chunk_idx + 1}/{total_chunks} (Entries {start + 1}-{end}) ---")

        # Process chunk
        chunk_df, next_start_index = process_chunk(chunk_data, current_start_index)
        processed_dfs.append(chunk_df)  # 收集分块结果

        # Update index for next chunk
        current_start_index = next_start_index

    # Step 4: Merge all chunks and save as Parquet
    print(f"\nMerging all chunks and saving to Parquet...")
    final_df = pd.concat(processed_dfs, ignore_index=True)
    final_df.to_parquet(
        OUTPUT_PARQUET,
        engine='pyarrow',
        index=False,
        compression='zstd'
    )

    # Final summary
    print(f"\n✅ Processing completed!")
    print(f"- Total chunks processed: {total_chunks}")
    print(f"- Total entries processed: {len(final_df)}")
    print(f"- Output Parquet: {OUTPUT_PARQUET}")
    print(f"- Columns: {list(final_df.columns)}")
    print(f"- File size: {os.path.getsize(OUTPUT_PARQUET) / 1024 / 1024:.2f} MB")


# ----------------------
# Run the Script
# ----------------------
if __name__ == "__main__":
    # Install dependencies first: pip install pandas pillow pyarrow
    try:
        run_chunked_processing()
    except Exception as e:
        print(f"\n❌ Processing failed: {str(e)}")