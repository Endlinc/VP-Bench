import argparse
import json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from PIL import Image

import pyarrow as pa
import pyarrow.parquet as pq


# ---------- Image helpers ----------

def load_image_bytes(image_path: Path) -> Optional[bytes]:
    """Load image and return PNG-encoded RGB bytes. Return None on failure."""
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            buf = BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
    except Exception as e:
        print(f"[WARN] Failed to load image {image_path}: {e}")
        return None


def ensure_list_data_files(data_file_field: Union[str, List[str], None]) -> List[str]:
    """Normalize data_file field to a list of strings."""
    if isinstance(data_file_field, str):
        return [data_file_field]
    elif isinstance(data_file_field, list):
        return [str(x) for x in data_file_field]
    else:
        return []


def build_image_dict(
    images_dir: Path,
    data_files: List[str],
    cache: Dict[str, Optional[bytes]],
) -> Dict[str, Optional[bytes]]:
    """
    Load all data_files into a dict: view_id -> image bytes.
    Uses a shared cache to avoid re-reading the same file.
    """
    image_dict: Dict[str, Optional[bytes]] = {}
    for rel_path in data_files:
        if rel_path not in cache:
            img_path = images_dir / rel_path
            cache[rel_path] = load_image_bytes(img_path)
        image_dict[rel_path] = cache[rel_path]
        if cache[rel_path] is None:
            print(f"[WARN] Missing image bytes for {rel_path}")
    return image_dict


def guess_images_dir(json_path: Path) -> Path:
    """
    Heuristic to find images directory for a given JSON:

    1) <dataset_root>/images if exists (e.g., EMOTIC/annotations/foo.json → EMOTIC/images)
    2) <json_dir>/images if exists
    3) Fallback: json_dir (assume data_file is relative to the JSON itself)
    """
    json_dir = json_path.parent
    dataset_root = json_dir.parent

    cand1 = dataset_root / "data"
    cand2 = json_dir / "data"

    if cand1.is_dir():
        return cand1
    if cand2.is_dir():
        return cand2
    return json_dir


# ---------- Region normalization ----------

def build_meta_bbox(region: Dict[str, Any]) -> Optional[str]:
    """
    Merge all bbox-related fields into a single JSON string.

    Supported keys:
      - meta_bbox
      - meta_obj_bbox, meta_sbj_bbox, meta_relation_bbox
    """
    merged: Dict[str, Any] = {}

    # Generic bbox
    if "meta_bbox" in region:
        merged["meta_bbox"] = region["meta_bbox"]

    # OpenPSG-style
    for key in ("meta_obj_bbox", "meta_sbj_bbox", "meta_relation_bbox"):
        if key in region:
            merged[key] = region[key]

    if not merged:
        return None

    # If only meta_bbox exists, you might want it directly; still safe as JSON
    if list(merged.keys()) == ["meta_bbox"]:
        return json.dumps(merged["meta_bbox"], ensure_ascii=False)
    return json.dumps(merged, ensure_ascii=False)


def build_meta_polygon(region: Dict[str, Any]) -> Optional[str]:
    """
    Merge all polygon-related fields into a single JSON string.

    Supported keys:
      - meta_polygon
      - meta_obj_polygon, meta_sbj_polygon, meta_relation_polygon
    """
    merged: Dict[str, Any] = {}

    if "meta_polygon" in region:
        merged["meta_polygon"] = region["meta_polygon"]

    for key in ("meta_obj_polygon", "meta_sbj_polygon", "meta_relation_polygon"):
        if key in region:
            merged[key] = region[key]

    if not merged:
        return None

    if list(merged.keys()) == ["meta_polygon"]:
        return json.dumps(merged["meta_polygon"], ensure_ascii=False)
    return json.dumps(merged, ensure_ascii=False)


def build_meta_annotation(region: Dict[str, Any]) -> Optional[str]:
    """
    Convert meta_annotation to a string:

      - dict -> JSON
      - string / other -> str(...)
    """
    if "meta_annotation" not in region:
        return None
    raw = region["meta_annotation"]
    if isinstance(raw, dict):
        return json.dumps(raw, ensure_ascii=False)
    if raw is None:
        return None
    return str(raw)


def normalize_region_generic(
    data_files: List[str],
    region: Dict[str, Any],
) -> (Dict[str, Dict[str, List[int]]],
      Dict[str, Dict[str, List[List[int]]]],
      Dict[str, str]):
    """
    Handle generic 'meta_bbox' / 'meta_polygon' regions (emotic, RoomSpace).
    """
    meta_bbox_out: Dict[str, Dict[str, List[int]]] = {}
    meta_polygon_out: Dict[str, Dict[str, List[List[int]]]] = {}

    # meta_bbox can be list (single view) or dict[view -> list]
    meta_bbox = region.get("meta_bbox")
    if meta_bbox is not None:
        if isinstance(meta_bbox, list):
            # assume single view
            if data_files:
                view = data_files[0]
                meta_bbox_out.setdefault(view, {})["default"] = meta_bbox
        elif isinstance(meta_bbox, dict):
            for view, bbox in meta_bbox.items():
                if isinstance(bbox, list):
                    meta_bbox_out.setdefault(view, {})["default"] = bbox

    # meta_polygon similarly
    meta_polygon = region.get("meta_polygon")
    if meta_polygon is not None:
        if isinstance(meta_polygon, list):
            if data_files:
                view = data_files[0]
                meta_polygon_out.setdefault(view, {})["default"] = meta_polygon
        elif isinstance(meta_polygon, dict):
            for view, poly in meta_polygon.items():
                if isinstance(poly, list):
                    meta_polygon_out.setdefault(view, {})["default"] = poly

    # meta_annotation → dict role -> string
    raw_ann = region.get("meta_annotation")
    if isinstance(raw_ann, dict):
        meta_annotation_out = {str(k): str(v) for k, v in raw_ann.items()}
    elif raw_ann is None:
        meta_annotation_out = {}
    else:
        meta_annotation_out = {"default": str(raw_ann)}

    return meta_bbox_out, meta_polygon_out, meta_annotation_out


def normalize_region_openpsg(
    data_files: List[str],
    region: Dict[str, Any],
) -> (Dict[str, Dict[str, List[int]]],
      Dict[str, Dict[str, List[List[int]]]],
      Dict[str, str]):
    """
    Handle OpenPSG-style regions with meta_obj/sbj/relation_*.
    """
    meta_bbox_out: Dict[str, Dict[str, List[int]]] = {}
    meta_polygon_out: Dict[str, Dict[str, List[List[int]]]] = {}
    meta_annotation_out: Dict[str, str] = {}

    if not data_files:
        return meta_bbox_out, meta_polygon_out, meta_annotation_out

    view = data_files[0]

    # Bounding boxes by role
    for role, bbox_key in [
        ("object", "meta_obj_bbox"),
        ("subject", "meta_sbj_bbox"),
        ("relation", "meta_relation_bbox"),
    ]:
        bbox = region.get(bbox_key)
        if isinstance(bbox, list):
            meta_bbox_out.setdefault(view, {})[role] = bbox

    # Polygons by role
    for role, poly_key in [
        ("object", "meta_obj_polygon"),
        ("subject", "meta_sbj_polygon"),
        ("relation", "meta_relation_polygon"),
    ]:
        poly = region.get(poly_key)
        if isinstance(poly, list):
            meta_polygon_out.setdefault(view, {})[role] = poly

    # meta_annotation may already be a dict with object/subject/relation
    raw_ann = region.get("meta_annotation")
    if isinstance(raw_ann, dict):
        meta_annotation_out = {str(k): str(v) for k, v in raw_ann.items()}
    elif raw_ann is None:
        meta_annotation_out = {}
    else:
        meta_annotation_out = {"default": str(raw_ann)}

    return meta_bbox_out, meta_polygon_out, meta_annotation_out


def merge_nested_bbox_poly(
    a_bbox: Dict[str, Dict[str, List[int]]],
    a_poly: Dict[str, Dict[str, List[List[int]]]],
    b_bbox: Dict[str, Dict[str, List[int]]],
    b_poly: Dict[str, Dict[str, List[List[int]]]],
) -> (Dict[str, Dict[str, List[int]]],
      Dict[str, Dict[str, List[List[int]]]]):
    """
    Merge two nested view->role->value dicts.
    """
    out_bbox = {k: v.copy() for k, v in a_bbox.items()}
    out_poly = {k: v.copy() for k, v in a_poly.items()}

    for view, roles in b_bbox.items():
        out_bbox.setdefault(view, {}).update(roles)
    for view, roles in b_poly.items():
        out_poly.setdefault(view, {}).update(roles)

    return out_bbox, out_poly

# ---------- JSON processing ----------

def process_annotation_file(
    json_path: Path,
    image_cache: Dict[str, Optional[bytes]],
) -> List[Dict[str, Any]]:
    """
    Read one JSON annotation file and return a list of rows
    in the unified schema (but NOT as a DataFrame yet).
    """
    rows: List[Dict[str, Any]] = []

    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to parse {json_path}: {e}")
        return rows

    if not isinstance(data, list):
        print(f"[WARN] Top level of {json_path} is not a list.")
        return rows

    images_dir = guess_images_dir(json_path)

    for item in data:
        meta_source = item.get("meta_source", "")

        data_files = ensure_list_data_files(item.get("data_file"))
        if not data_files:
            print(f"[WARN] No valid data_file in {json_path}")
            continue

        # Load all images for this instance
        image_dict = build_image_dict(images_dir, data_files, image_cache)

        # Two patterns:
        #  - 'region': emotic / RoomSpace-style
        #  - 'regions': OpenPSG-style
        if "region" in item:
            region_list = item.get("region", [])
        elif "regions" in item:
            region_list = item.get("regions", [])
        else:
            region_list = []

        if not isinstance(region_list, list):
            print(f"[WARN] 'region(s)' not a list in {json_path}")
            continue

        for region in region_list:
            # Decide if this is OpenPSG-style (has meta_obj_bbox etc.)
            is_openpsg = any(
                key in region
                for key in ("meta_obj_bbox", "meta_sbj_bbox", "meta_relation_bbox")
            )

            if is_openpsg:
                bbox1, poly1, ann1 = normalize_region_openpsg(data_files, region)
                bbox2, poly2, ann2 = normalize_region_generic(data_files, region)
                meta_bbox, meta_polygon = merge_nested_bbox_poly(
                    bbox1, poly1, bbox2, poly2
                )
                meta_annotation = {**ann2, **ann1}  # give OpenPSG dict precedence
            else:
                meta_bbox, meta_polygon, meta_annotation = normalize_region_generic(
                    data_files, region
                )

            row = {
                "meta_source": meta_source,
                "data_file": data_files,
                "image": image_dict,
                "meta_bbox": meta_bbox,
                "meta_polygon": meta_polygon,
                "meta_annotation": meta_annotation,
                "question": region.get("question"),
                "answer": region.get("answer"),
            }
            rows.append(row)

    return rows


# ---------- Chunked collection & Parquet writing ----------

def collect_and_write_parquet(
    root_dir: Path,
    output: Path,
    chunk_size: int = 5000,
):
    """
    Recursively walk root_dir, process JSON files, and write chunks
    to separate Parquet files using pandas.to_parquet.

    If output = /path/to/vp_bench_stage_2_meta.parquet,
    chunks will be:
      /path/to/vp_bench_stage_2_meta.part0000.parquet
      /path/to/vp_bench_stage_2_meta.part0001.parquet
      ...
    """
    json_files: List[Path] = []
    for p in root_dir.rglob("*.json"):
        if ".git" in p.parts:
            continue
        json_files.append(p)

    print(f"[INFO] Found {len(json_files)} JSON files under {root_dir}")

    image_cache: Dict[str, Optional[bytes]] = {}
    buffer: List[Dict[str, Any]] = []
    chunk_idx: int = 0

    cols = [
        "meta_source",
        "data_file",
        "image",
        "meta_bbox",
        "meta_polygon",
        "meta_annotation",
        "question",
        "answer",
    ]

    def flush_buffer():
        nonlocal buffer, chunk_idx
        if not buffer:
            return

        df = pd.DataFrame(buffer)
        df = df[cols]  # enforce column order

        # Replace NaN with None so bytes/strings survive cleanly
        # df = df.where(pd.notnull(df), None)

        # Construct chunk filename
        chunk_path = output.with_name(
            f"{output.stem}.part{chunk_idx:04d}{output.suffix}"
        )

        print(f"[INFO] Writing chunk {chunk_idx} with {len(df)} rows to {chunk_path}")
        df.to_parquet(chunk_path, index=False)  # uses pyarrow / fastparquet under the hood

        buffer = []
        chunk_idx += 1

    for jp in sorted(json_files):
        print(f"[INFO] Processing JSON: {jp}")
        rows = process_annotation_file(jp, image_cache)
        buffer.extend(rows)
        if len(buffer) >= chunk_size:
            flush_buffer()

    # final flush
    if buffer:
        flush_buffer()

    if chunk_idx == 0:
        print("[WARN] No rows collected. No Parquet written.")
    else:
        print(f"[INFO] Finished writing {chunk_idx} chunk file(s).")

# ---------- CLI ----------


def main():
    parser = argparse.ArgumentParser(
        description="Collect VP metadata into multiple chunked Parquet files."
    )
    parser.add_argument(
        "--root_dir",
        required=True,
        type=str,
        help="Root directory containing metadata JSONs (e.g., /home/ming/Datasets/VP-METADATA).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help=(
            "Base output Parquet path. Chunks will be saved as "
            "<stem>.partXXXX<suffix>, e.g. vp_bench_stage_2_meta.part0000.parquet"
        ),
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=5000,
        help="Number of rows per Parquet chunk (default: 5000).",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()

    if not root_dir.is_dir():
        raise SystemExit(f"Root dir does not exist: {root_dir}")

    collect_and_write_parquet(root_dir, output, chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()
