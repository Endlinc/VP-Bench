import argparse
import base64
import io
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable

import pandas as pd
from PIL import Image, ImageDraw


# ========= Text normalization & metrics =========

_punc_regex = re.compile(r"[^\w\s]", flags=re.UNICODE)

def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = _punc_regex.sub(" ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def exact_match(pred: str, gold: str) -> bool:
    return normalize_text(pred) == normalize_text(gold)


# ========= Parquet helpers =========

def iter_parquet_files(path: Path) -> Iterable[Path]:
    """
    path can be:
      - single .parquet file
      - directory of parquet files
      - glob with '*'
    """
    if "*" in str(path):
        files = sorted(path.parent.glob(path.name))
    elif path.is_dir():
        files = sorted(path.glob("*.parquet"))
    else:
        files = [path]

    for f in files:
        if f.is_file():
            yield f


# ========= VP extraction / visualization =========

def ensure_list_data_files(data_file_field: Any) -> List[str]:
    """Normalize data_file to a list of strings."""
    if isinstance(data_file_field, list):
        return [str(x) for x in data_file_field]
    if isinstance(data_file_field, str):
        # try to parse JSON if it looks like '["a","b"]'
        try:
            maybe_list = json.loads(data_file_field)
            if isinstance(maybe_list, list):
                return [str(x) for x in maybe_list]
        except Exception:
            pass
        return [data_file_field]
    return []


def parse_json_field(field_value: Any) -> Any:
    """Parse a JSON string field if needed; otherwise return as-is."""
    if isinstance(field_value, str):
        try:
            return json.loads(field_value)
        except Exception:
            return None
    return field_value


def choose_view_and_bbox(
    meta_bbox_parsed: Any,
    data_files: List[str],
) -> Tuple[Optional[str], Optional[List[int]]]:
    """
    Decide which view and which bbox to use for visualization.
    Handles:
      - list → single bbox, first data_file is view
      - dict[view -> dict[role -> bbox]] → pick data_file[0] if possible
    """
    if meta_bbox_parsed is None:
        return None, None

    # Case 1: plain list [x1, y1, x2, y2]
    if isinstance(meta_bbox_parsed, list) and len(meta_bbox_parsed) == 4:
        view = data_files[0] if data_files else "default_view"
        return view, meta_bbox_parsed

    # Case 2: nested dict: {view: {role: bbox}}
    if isinstance(meta_bbox_parsed, dict):
        # pick view
        view_candidates = list(meta_bbox_parsed.keys())
        if not view_candidates:
            return None, None

        if data_files:
            if data_files[0] in meta_bbox_parsed:
                view = data_files[0]
            else:
                view = view_candidates[0]
        else:
            view = view_candidates[0]

        role_dict = meta_bbox_parsed.get(view, {})
        if not isinstance(role_dict, dict) or not role_dict:
            return view, None

        # choose bbox role priority
        for role in ("relation", "object", "subject", "default"):
            if role in role_dict:
                bbox = role_dict[role]
                if isinstance(bbox, list) and len(bbox) == 4:
                    return view, bbox

        # fallback: first bbox
        first_bbox = next(iter(role_dict.values()))
        if isinstance(first_bbox, list) and len(first_bbox) == 4:
            return view, first_bbox

    # Unknown / unsupported
    return None, None


def overlay_bbox_on_image(image_b64: str, bbox: List[int]) -> bytes:
    """
    Decode base64 → image, draw red rectangle, return PNG bytes.
    """
    raw = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")

    x1, y1, x2, y2 = bbox
    draw = ImageDraw.Draw(img)
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ========= Prompt construction =========

def format_question_with_vp(
    question: str,
    vp_text: str,
    explanation_prefix: Optional[str] = None,
    options: Optional[List[str]] = None,
) -> str:
    """
    1) replace <VP>
    2) add explanation in front
    3) add options if given
    """
    q = question or ""
    q = q.replace("<VP>", vp_text)

    if explanation_prefix:
        q = explanation_prefix.strip() + " " + q

    if options:
        lines = [q, "", "Options:"]
        for i, opt in enumerate(options):
            label = chr(ord("A") + i)
            lines.append(f"{label}. {opt}")
        q = "\n".join(lines)

    return q


def extract_options_from_row(row: Dict[str, Any]) -> Optional[List[str]]:
    """
    If your Parquet has an 'options' column (stringified JSON or list),
    parse it and return list of options. Else return None.
    """
    if "options" not in row:
        return None
    val = row["options"]

    if isinstance(val, list):
        return [str(x) for x in val]

    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            # treat as single option, mostly useless, so ignore
            return None

    return None


# ========= Model wrapper (YOU MUST IMPLEMENT) =========

def call_model_with_image(prompt: str, image_bytes: bytes) -> str:
    """
    Stub: call your MLLM here.

    You can implement with OpenAI, vLLM, LMDeploy, etc.
    For example (pseudo-code):

    from openai import OpenAI
    client = OpenAI()

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64," + base64.b64encode(image_bytes).decode("utf-8")
                        },
                    },
                ]
            }
        ]
    )
    return resp.choices[0].message.content

    For now this function raises NotImplementedError so you don't accidentally
    run it without filling in your backend.
    """
    raise NotImplementedError("Implement call_model_with_image() for your backend.")


# ========= Main evaluation loop =========

def evaluate_parquet(
    parquet_path: Path,
    vp_text: str,
    explanation_prefix: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> None:
    total = 0
    correct = 0

    for f in iter_parquet_files(parquet_path):
        print(f"[INFO] Reading {f}")
        df = pd.read_parquet(f)

        for _, row in df.iterrows():
            if max_samples is not None and total >= max_samples:
                break

            row_dict = row.to_dict()

            question = row_dict.get("question", "")
            gold_answer = row_dict.get("answer", "")
            if pd.isna(question) or pd.isna(gold_answer):
                continue

            # ---- Step 2–4: build prompt ----
            options = extract_options_from_row(row_dict)
            prompt = format_question_with_vp(
                question=question,
                vp_text=vp_text,
                explanation_prefix=explanation_prefix,
                options=options,
            )

            # ---- Step 5: visualize VP & get image bytes ----
            data_files = ensure_list_data_files(row_dict.get("data_file"))
            meta_bbox_parsed = parse_json_field(row_dict.get("meta_bbox"))
            image_dict_parsed = parse_json_field(row_dict.get("image"))

            if not data_files or not isinstance(image_dict_parsed, dict):
                print("[WARN] Missing data_file or image dict; skipping row.")
                continue

            view, bbox = choose_view_and_bbox(meta_bbox_parsed, data_files)
            if view is None or bbox is None:
                print("[WARN] Could not determine view/bbox; skipping row.")
                continue

            img_b64 = image_dict_parsed.get(view)
            if not isinstance(img_b64, str):
                print(f"[WARN] No base64 image for view {view}; skipping row.")
                continue

            try:
                image_bytes = overlay_bbox_on_image(img_b64, bbox)
            except Exception as e:
                print(f"[WARN] Failed to overlay bbox on image for view {view}: {e}")
                continue

            # ---- Step 6: call model ----
            try:
                pred_answer = call_model_with_image(prompt, image_bytes)
            except NotImplementedError:
                print("[ERROR] call_model_with_image() not implemented.")
                print("        Please implement it before running evaluation.")
                return
            except Exception as e:
                print(f"[WARN] Model call failed: {e}")
                continue

            # ---- Step 7: accumulate ----
            total += 1
            if exact_match(str(pred_answer), str(gold_answer)):
                correct += 1

            if total % 50 == 0:
                acc = correct / total if total > 0 else 0.0
                print(f"[INFO] Processed {total} samples, running acc={acc*100:.2f}%")

        if max_samples is not None and total >= max_samples:
            break

    # ---- Step 8: final accuracy ----
    if total == 0:
        print("[WARN] No valid samples evaluated.")
        return

    final_acc = correct / total
    print("\n===== FINAL RESULTS =====")
    print(f"Total evaluated samples: {total}")
    print(f"Exact-match accuracy:    {final_acc*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MLLM on VP-Bench parquet outcome with VP visualization."
    )
    parser.add_argument(
        "--parquet",
        required=True,
        type=str,
        help=(
            "Path to a parquet file, directory, or glob pattern. "
            "E.g. vp_bench_stage_2_meta.part*.parquet"
        ),
    )
    parser.add_argument(
        "--vp_text",
        required=True,
        type=str,
        help="Text that replaces <VP> in the question, e.g. 'the highlighted region'.",
    )
    parser.add_argument(
        "--explanation_prefix",
        type=str,
        default="In this question, <VP> refers to the highlighted region in the image.",
        help="Explanation text prefixed before the question.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on number of samples to evaluate (for quick testing).",
    )
    args = parser.parse_args()

    parquet_path = Path(args.parquet).expanduser().resolve()

    evaluate_parquet(
        parquet_path=parquet_path,
        vp_text=args.vp_text,
        explanation_prefix=args.explanation_prefix,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
