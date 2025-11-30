import argparse
import base64
import io
import json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from PIL import Image


# ---------- Image helpers ----------

def encode_image_file_to_base64(image_path: Path) -> Optional[str]:
    """
    Open image and return base64-encoded PNG.
    Return None and report an error if the image cannot be loaded.
    """
    try:
        image = Image.open(image_path)
        if image.mode in ("RGBA", "P", "LA"):
            image = image.convert("RGB")

        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")

        image_data = img_buffer.getvalue()
        ret = base64.b64encode(image_data).decode("utf-8")
        return ret
    except Exception as e:
        # Report error but do not hard-crash so you can see all bad images
        print(f"[ERROR] Failed to load image {image_path}: {e}")
        return None


def ensure_list_data_files(data_file_field: Union[str, List[str], None]) -> List[str]:
    """Normalize data_file field to a list of strings."""
    if isinstance(data_file_field, str):
        return [data_file_field]
    elif isinstance(data_file_field, list):
        return [str(x) for x in data_file_field]
    else:
        return []


def build_image_list(
    images_dir: Path,
    data_files: List[str],
    cache: Dict[str, Optional[str]],
) -> List[Optional[str]]:
    """
    Load all data_files into a dict: rel_path -> base64 string (or None on error).

    Uses a shared cache to avoid re-reading the same file.
    """
    image_list: List[Optional[str]] = []
    for rel_path in data_files:
        if rel_path not in cache:
            img_path = images_dir / Path("data") / rel_path
            temp = encode_image_file_to_base64(img_path)
            cache[rel_path] = temp
        image_list.append(cache[rel_path])
        if cache[rel_path] is None:
            print(f"[WARN] Missing/failed image bytes for {rel_path}")
    return image_list


def guess_images_dir_for_anno(json_path: Path, image_root_dir: Path) -> Path:
    """
    Decide which image sub-folder to use based on the annotation file name/path.

    Rules (case-insensitive, based on 'anno_file is containing ...'):
      - if contains 'emotic'           -> image_root_dir / 'EMOTIC-Scaled'
      - if contains 'mapillary_vistas' -> image_root_dir / 'MapillaryVistas-Scaled'
      - if contains 'mia'             -> image_root_dir / 'MIA-Scaled'
      - if contains 'sd-100'          -> image_root_dir / 'SD-100-Scaled'
      - if contains 'see_click'       -> image_root_dir / 'SeeClick-Scaled'
      - if contains 'sgg'             -> image_root_dir / 'SceneGraphRcongnition-Scaled'
    """
    full_str = str(json_path).lower()

    subdir = None
    if "emotic" in full_str:
        subdir = "EMOTIC-Scaled"
    elif "mapillary_vistas" in full_str:
        subdir = "MapillaryVistas-Scaled"
    elif "mia" in full_str:
        subdir = "MIA-Scaled"
    elif "sd-100" in full_str:
        subdir = "SD-100-Scaled"
    elif "see_click" in full_str:
        subdir = "SeeClick-Scaled"
    elif "sgg" in full_str:
        subdir = "SceneGraphRcongnition-Scaled"

    if subdir is not None:
        images_dir = image_root_dir / subdir
        if not images_dir.is_dir():
            raise SystemExit(
                f"[ERROR] Expected image directory for {json_path} not found: {images_dir}"
            )
        return images_dir

    # Fallback if nothing matched
    print(
        f"[WARN] Could not infer dataset type from annotation path {json_path}. "
        f"Using image_root_dir directly: {image_root_dir}"
    )
    if not image_root_dir.is_dir():
        raise SystemExit(f"[ERROR] image_root_dir does not exist: {image_root_dir}")
    return image_root_dir


def derive_meta_source_from_anno(json_path: Path) -> str:
    """
    Derive meta_source from the annotation file path.

    Rules (case-insensitive):
      - if contains 'emotic'           -> 'emotic'
      - if contains 'mapillary_vistas' -> 'mapillary_vistas'
      - if contains 'mia'             -> 'MIA'
      - if contains 'sd-100'          -> 'SD-100'
      - if contains 'see_click'       -> 'see_click'
      - if contains 'sgg'             -> 'SGG'
    """
    full_str = str(json_path).lower()

    if "emotic" in full_str:
        return "emotic"
    if "mapillary_vistas" in full_str:
        return "mapillary_vistas"
    if "mia" in full_str:
        return "MIA"
    if "sd-100" in full_str:
        return "SD-100"
    if "see_click" in full_str:
        return "see_click"
    if "sgg" in full_str:
        return "SGG"

    # fallback if nothing matches
    return ""


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
):
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
):
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
):
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
    image_root_dir: Path,
    image_cache: Dict[str, Optional[str]],
) -> List[Dict[str, Any]]:
    """
    Read one JSON annotation file and return a list of rows
    in the unified schema (but NOT as a DataFrame yet).

    NOTE: In raw annotation, `question` is a single string:
        "question\\nA\\nB\\nC\\nD"
    We parse and store them separately as `question`, `A`, `B`, `C`, `D`.
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

    images_dir = guess_images_dir_for_anno(json_path, image_root_dir)
    meta_source = derive_meta_source_from_anno(json_path)

    for item in data:
        data_files = ensure_list_data_files(item.get("original_image"))
        if not data_files:
            print(f"[WARN] No valid data_file in {json_path}")
            continue

        # Load all images for this instance
        image_list = build_image_list(images_dir, data_files, image_cache)

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
                meta_annotation = {**ann2, **ann1}  # OpenPSG dict precedence
            else:
                meta_bbox, meta_polygon, meta_annotation = normalize_region_generic(
                    data_files, region
                )

            # --- Parse concatenated question string into question + A/B/C/D ---
            q_concat = region.get("question")
            q_text = None
            A = B = C = D = None

            if isinstance(q_concat, str):
                lines = [ln.strip() for ln in q_concat.split("\n")]
                if len(lines) > 0:
                    q_text = lines[0] or None
                if len(lines) > 1:
                    A = lines[1] or None
                if len(lines) > 2:
                    B = lines[2] or None
                if len(lines) > 3:
                    C = lines[3] or None
                if len(lines) > 4:
                    D = lines[4] or None
            else:
                # If not a string, keep as-is in question, others stay None
                q_text = q_concat

            row = {
                "meta_source": meta_source,
                "data_file": data_files,
                "image": image_list,
                "meta_bbox": meta_bbox,
                "meta_polygon": meta_polygon,
                "question": q_text,
                "A": A,
                "B": B,
                "C": C,
                "D": D,
                "answer": region.get("answer"),
            }
            rows.append(row)

    return rows


# ---------- Chunked collection & Parquet writing ----------

def collect_and_write_parquet(
    anno_dir: Path,
    image_root_dir: Path,
    output: Path,
    chunk_size: int = 5000,
):
    """
    Recursively walk anno_dir, process JSON files, and write chunks
    to separate Parquet files using pandas.to_parquet.

    If output = /path/to/vp_bench_stage_2_meta.parquet,
    chunks will be:
      /path/to/vp_bench_stage_2_meta.part0000.parquet
      /path/to/vp_bench_stage_2_meta.part0001.parquet
      ...
    """
    json_files: List[Path] = []
    for p in anno_dir.rglob("*.json"):
        if ".git" in p.parts:
            continue
        json_files.append(p)

    print(f"[INFO] Found {len(json_files)} JSON files under {anno_dir}")

    image_cache: Dict[str, Optional[str]] = {}
    buffer: List[Dict[str, Any]] = []
    chunk_idx: int = 0

    cols = [
        "meta_source",
        "data_file",
        "image",
        "meta_bbox",
        "meta_polygon",
        "question",
        "A",
        "B",
        "C",
        "D",
        "answer",
    ]

    def flush_buffer():
        nonlocal buffer, chunk_idx
        if not buffer:
            return

        df = pd.DataFrame(buffer)
        df = df[cols]  # enforce column order

        # Construct chunk filename
        chunk_path = output.with_name(
            f"{output.stem}.part{chunk_idx:04d}{output.suffix}"
        )

        print(f"[INFO] Writing chunk {chunk_idx} with {len(df)} rows to {chunk_path}")
        df.to_parquet(
            chunk_path,
            index=False,
            compression="zstd",
        )

        buffer = []
        chunk_idx += 1

    for jp in sorted(json_files):
        print(f"[INFO] Processing JSON: {jp}")
        rows = process_annotation_file(jp, image_root_dir, image_cache)
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
        "--anno_dir",
        required=True,
        type=str,
        help="Directory containing annotation JSONs.",
    )
    parser.add_argument(
        "--image_root_dir",
        required=True,
        type=str,
        help=(
            "Root directory where images are stored in subfolders like "
            "EMOTIC-Scaled, MapillaryVistas-Scaled, MIA-Scaled, SD-100-Scaled, "
            "SeeClick-Scaled, SceneGraphRcongnition-Scaled."
        ),
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

    anno_dir = Path(args.anno_dir).expanduser().resolve()
    image_root_dir = Path(args.image_root_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()

    if not anno_dir.is_dir():
        raise SystemExit(f"Annotation dir does not exist: {anno_dir}")
    if not image_root_dir.is_dir():
        raise SystemExit(f"image_root_dir does not exist: {image_root_dir}")

    collect_and_write_parquet(
        anno_dir=anno_dir,
        image_root_dir=image_root_dir,
        output=output,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
