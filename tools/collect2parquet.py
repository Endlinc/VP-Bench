import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd


def read_image_bytes(image_path: Path) -> Optional[bytes]:
    """Read an image as raw bytes. Returns None if missing or unreadable."""
    try:
        with image_path.open("rb") as f:
            return f.read()
    except (FileNotFoundError, OSError):
        return None


def process_annotation_file(
    json_path: Path,
    images_dir: Path,
    dataset_name: str,
) -> List[Dict[str, Any]]:
    """
    Parse one annotation JSON file and return a list of flattened rows:
    one row per region per image.
    """
    rows: List[Dict[str, Any]] = []

    with json_path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[WARN] Failed to parse JSON: {json_path} ({e})")
            return rows

    if not isinstance(data, list):
        print(f"[WARN] Expected list at top level: {json_path}")
        return rows

    for item_idx, item in enumerate(data):
        meta_source = item.get("meta_source", None)
        data_file = item.get("data_file", None)
        regions = item.get("region", [])

        if not data_file:
            print(f"[WARN] Missing 'data_file' in {json_path}, item {item_idx}")
            continue

        img_path = images_dir / data_file
        img_bytes = read_image_bytes(img_path)

        if img_bytes is None:
            print(f"[WARN] Could not read image bytes: {img_path}")

        if not isinstance(regions, list):
            print(f"[WARN] 'region' should be a list in {json_path}, item {item_idx}")
            continue

        for region_idx, region in enumerate(regions):
            row: Dict[str, Any] = {
                "dataset_name": dataset_name,
                "annotation_file": str(json_path),
                "image_path": str(img_path),
                "image_bytes": img_bytes,  # raw image data
                "meta_source": meta_source,
                "data_file": data_file,
                "region_index": region_idx,
                "meta_bbox": region.get("meta_bbox"),
                "meta_polygon": region.get("meta_polygon"),
                "meta_annotation": region.get("meta_annotation"),
                "question": region.get("question"),
                "answer": region.get("answer"),
            }
            rows.append(row)

    return rows


def collect_dataset(
    root_dir: Path,
    annotations_subdir: str,
    images_subdir: str,
) -> pd.DataFrame:
    """
    Walk all subdirectories under root_dir, find annotation and image directories,
    and collect all region annotations into a single DataFrame.
    """
    all_rows: List[Dict[str, Any]] = []

    for sub in sorted(p for p in root_dir.iterdir() if p.is_dir()):
        dataset_name = sub.name
        ann_dir = sub / annotations_subdir
        img_dir = sub / images_subdir

        if not ann_dir.is_dir():
            print(f"[INFO] Skip {sub}: missing annotations dir '{annotations_subdir}'")
            continue
        if not img_dir.is_dir():
            print(f"[INFO] Skip {sub}: missing images dir '{images_subdir}'")
            continue

        print(f"[INFO] Processing dataset: {dataset_name}")
        json_files = sorted(ann_dir.glob("*.json"))

        if not json_files:
            print(f"[INFO] No JSON files found in {ann_dir}")
            continue

        for json_path in json_files:
            rows = process_annotation_file(json_path, img_dir, dataset_name)
            all_rows.extend(rows)

    if not all_rows:
        print("[WARN] No rows collected. DataFrame will be empty.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Collect image + annotation regions into a Parquet file."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory containing sub-datasets.",
    )
    parser.add_argument(
        "--annotations_subdir",
        type=str,
        default="annotations",
        help="Name of the annotation subdirectory inside each dataset (default: annotations).",
    )
    parser.add_argument(
        "--images_subdir",
        type=str,
        default="data",
        help="Name of the image subdirectory inside each dataset (default: images).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output Parquet file.",
    )

    args = parser.parse_args()

    root_dir = Path(args.root_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not root_dir.is_dir():
        raise SystemExit(f"Root directory does not exist or is not a directory: {root_dir}")

    df = collect_dataset(
        root_dir=root_dir,
        annotations_subdir=args.annotations_subdir,
        images_subdir=args.images_subdir,
    )

    print(f"[INFO] Collected {len(df)} rows.")
    if len(df) == 0:
        print("[WARN] Nothing to save; exiting without writing Parquet.")
        return

    # Use pyarrow backend for better support of nested/list types
    df.to_parquet(output_path, index=False)
    print(f"[INFO] Saved Parquet to: {output_path}")


if __name__ == "__main__":
    main()
