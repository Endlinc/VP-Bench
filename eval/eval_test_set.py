import base64
import io
import json
import os

import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

from visual_prompts.utils import PromptManager


def return_parquets(path):
    par_pths = os.listdir(path)
    du_pths = list()
    for par_pth in par_pths:
        if par_pth.endswith(".parquet"):
            du_pths.append(par_pth)
    return du_pths

def decode_img(base64_str):
    img_bytes = base64.b64decode(base64_str)

    # Step 2: Wrap bytes in a file-like stream
    buffer = io.BytesIO(img_bytes)

    # Step 3: Load stream into PIL.Image
    decoded_img = Image.open(buffer)
    return decoded_img

PAR_DIR = "/Volumes/Ming-Data/Dataset/VP_Bench/vp_bench_stage_2/VP-Bench-Stage-2"
anno_pths = return_parquets(PAR_DIR)
acc_meter = {
    "total": 0,
    "correct": 0
}
v_prompts = [
            'bounding_box',
            'mask',
            'fill_contour',
            'circle',
            'arrow',
            'number_label',  # Add NumberLabelPrompt
            'alphabet_label',  # Add AlphabetLabelPrompt
            'single_point',
            'scribble'
        ]

vp_manager = PromptManager()
vp_prompt = vp_manager.get_prompt(v_prompts[0])
color = ["red"]

# Load processor (supports both Chinese/English)
processor = AutoProcessor.from_pretrained("Qwen/Qwen-VL-Chat")

# Load model (use bf16 for better GPU performance if supported)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-VL-Chat",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

for anno_pth in anno_pths:
    # Load a single Parquet file
    df = pd.read_parquet(os.path.join(PAR_DIR, anno_pth))  # Uses pyarrow by default (if installed)

    # View the loaded data
    print(df.head())

    for row in df.itertuples(index=False):
        source = row.meta_source
        img_nms = row.data_file
        imgs = row.image
        imgs = json.loads(imgs)
        meta_bbox = row.meta_bbox
        meta_bbox = json.loads(meta_bbox)
        meta_poly = row.meta_polygon
        meta_poly = json.loads(meta_poly)
        question = row.question
        answer = row.answer
        if "PSG" in source:
            img_nm = img_nms[0]
            img = decode_img(imgs[img_nm])

            s_bbox = meta_bbox[img_nm]["subject"]
            o_bbox = meta_bbox[img_nm]["object"]
            s_poly = meta_poly[img_nm]["subject"]
            s_poly = [sum(s_poly, [])]
            o_poly = meta_poly[img_nm]["object"]
            o_poly = [sum(o_poly, [])]

            anno = [
                {
                    "bbox": s_bbox,
                    "segmentation": s_poly,
                },
                {
                    "bbox": o_bbox,
                    "segmentation": o_poly,
                }
            ]

            img = vp_prompt.apply(img, anno, {"width": img.width, "height": img.height}, color=color)

            imgs = [img]
            conv = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]
                }
            ]
            inputs = processor(
                images=imgs,
                text=conv,
                return_tensors="pt",
            )
            output = model.generate(**inputs)
        elif "RoomSpace" in source:
            f_img_nm = img_nms[0]
            t_img_nm = img_nms[1]
            f_img = decode_img(imgs[f_img_nm])
            t_img = decode_img(imgs[t_img_nm])

            f_bbox = meta_bbox[f_img_nm]["default"]
            t_bbox = meta_bbox[t_img_nm]["default"]
            f_poly = meta_poly[f_img_nm]["default"]
            f_poly = [sum(f_poly, [])]
            t_poly = meta_poly[t_img_nm]["default"]
            t_poly = [sum(t_poly, [])]

            f_anno = [
                {
                    "bbox": f_bbox,
                    "segmentation": f_poly,
                }
            ]
            t_anno = [
                {
                    "bbox": t_bbox,
                    "segmentation": t_img,
                }
            ]

            f_img = vp_prompt.apply(f_img, f_anno, {"width": f_img.width, "height": f_img.height}, color=color)
            t_img = vp_prompt.apply(t_img, t_anno, {"width": t_img.width, "height": t_img.height}, color=color)

            imgs = [f_img, t_img]
            conv = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]
                }
            ]
            inputs = processor(
                images=imgs,
                text=conv,
                return_tensors="pt",
            )
            output = model.generate(**inputs)
        else:
            img_nm = img_nms[0]
            img = decode_img(imgs[img_nm])

            bbox = meta_bbox[img_nm]["default"]
            poly = meta_poly[img_nm]["default"]
            poly = [sum(poly, [])]

            anno = [
                {
                    "bbox": bbox,
                    "segmentation": poly,
                }
            ]

            img = vp_prompt.apply(img, anno, {"width": img.width, "height": img.height}, color=color)

            imgs = [img]
            conv = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]
                }
            ]
            inputs = processor(
                images=imgs,
                text=conv,
                return_tensors="pt",
            )
            output = model.generate(**inputs)
