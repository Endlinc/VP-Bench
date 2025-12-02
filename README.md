# VP-Bench: A Comprehensive Benchmark for Visual Prompting in Multimodal Large Language Models

<p align="center">
        &nbsp&nbsp🤗 <a href="https://huggingface.co/datasets/jinpeng0528/VP-Bench">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/VP-Bench">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://arxiv.org/pdf/2511.11438">Paper</a>&nbsp&nbsp
</p>

---

![VP-Bench Dataset Overview](./assets/dataset-overview.png)

---

VP-Bench is a two-stage benchmark for testing how multimodal large language models handle **visual prompts**—boxes, masks, arrows, scribbles, etc. Functionally, Stage 1 measures core VP perception across 34k+ images and nearly 39k questions, probing existence, counting, localization and reference under diverse VP shapes and styles. Stage 2 evaluates how adding VPs changes performance on six downstream tasks, including medical imaging, GUI grounding, 3D/scene understanding and emotion recognition. The research focuses on quantifying models’ true reliance on VPs versus text priors, analyzing sensitivity to prompt design, and understanding why current models still fall short of robust, human-like VP-guided reasoning. 

---

## Release Checklist

- [x] **Stage 1 evaluation dataset on VLMEvalKit**  
  Integrated into **[VLMEvalKit](https://github.com/open-compass/VLMEvalKit)** with a VLMEvalKit-compatible test file.

- [x] **Public dataset release (Stage 1 & Stage 2)**  
  - Stage 1: <https://modelscope.cn/datasets/VP-Bench/VP-Bench-Stage-1>  
  - Stage 2: <https://modelscope.cn/datasets/VP-Bench/VP-Bench-Stage-2>

- [ ] **Full evaluation code release (this repo)**  
  Stage 1 & Stage 2 evaluation scripts and configs (coming soon).

---

## Data File Format

VP-Bench uses Parquet files to store the evaluation data. Below we describe the schema of the two main files.

---

### Stage 1: `vp_bench_stage_1.parquet`

Each row corresponds to a single multiple-choice VP-perception question on one image.

* `index`
  Integer. Unique sample index within this file.

* `context`
  String. Optional textual context or background information shown before the question.

* `question`
  String. The natural language question presented to the model/user.

* `A`, `B`, `C`, `D`
  String. The four answer options for the multiple-choice question.

* `answer`
  String. The ground-truth answer label (e.g., `"A"`, `"B"`, `"C"`, `"D"`).

* `image`
  String (base64). The image content encoded as a base64 PNG string.

* `vp_amount`
  Integer. Number of visual prompts (VPs) associated with this sample (e.g., how many boxes/masks are drawn).

* `vp_shape`
  String or list of strings. The shape type(s) of the visual prompt(s), such as bounding boxes, polygons, etc.

*`vp_properties`  
  String. A textual description of the attributes of the visualized VP(s), detailing how the visual prompt is rendered or characterized (e.g., style, semantics, or other VP-specific properties).

---

### Stage 2: `vp_bench_stage_2_meta.parquet`

Each row describes a Stage-2 sample and its linkage to underlying source data and visual prompt metadata.

* `meta_source`
  String. Identifier of the original dataset or source benchmark from which this sample is derived.

* `data_file`
  String or list of strings. Path(s) to the underlying data file(s) containing the full question–answer records used to construct this Stage-2 sample.

* `image`
  List. A list of image identifiers (e.g., encoded images) associated with this sample.

* `meta_bbox`
  List. Visual prompt metadata in bounding-box form (e.g., a list of `[x_min, y_min, x_max, y_max]` or similar structures for each VP).

* `meta_polygon`
  List. Visual prompt metadata in polygon form (e.g., a list of point lists defining each polygonal VP).

* `question`
  String. The natural language question for this Stage-2 sample.

* `A`, `B`, `C`, `D`
  String. The multiple-choice answer options.

* `answer`
  String or `null`. Ground-truth answer label if available (e.g., `"A"`–`"D"`); may be missing for some samples that are used only for model prediction or human evaluation.

---

### Stage 2: Using Metadata to Build Visual-Prompted QA

The Stage 2 meta file (`vp_bench_stage_2_meta.parquet`) acts as an **intermediate data resource** for constructing diverse visual-prompted QA samples. Each row links source questions, images, and visual prompt metadata, but **does not** yet contain the rendered visual prompts on the image or the `<color> <vp>` tags in the question text.

To build fully visualized QA examples, you should:

1. **Load the Stage 2 meta record**

   * Read one row from `vp_bench_stage_2_meta.parquet`.
   * Retrieve fields such as `image`, `question`, `A`–`D`, `meta_bbox`, and `meta_polygon`.

2. **Visualize the VP on the image**

   * Choose a specific image entry `img_nm` and its corresponding metadata:

     * `meta_bbox[img_nm]["default"]` for the bounding box
     * `meta_polygon[img_nm]["default"]` for the polygon/mask
   * Use the provided visualization framework (via `PromptManager`) to draw the VP on the image.

3. **Inject `<color> <vp>` tags into the text**

   * In the original `question` from the meta file, replace the text span that refers to the visual prompt with a pair of tags:

     * `<color>`: indicates the VP color used in visualization (e.g., `red`, `blue`).
     * `<vp>`: indicates that this span corresponds to a visual prompt region.
   * For example, a phrase like “the ball on the left” might become `the <red> <vp>` in the question after you render a red VP over that region.

This two-step process (visualizing the VP on the image + tagging the corresponding text) lets you build **VP-aware QA** suitable for evaluating how models use explicit visual prompts.

---

## Visualization Framework Usage

We provide a lightweight visualization framework to help you render VPs on images. To use it, first import the `PromptManager` class from the Python source code:

```python
from your_project.prompt_manager import PromptManager  # adjust import path as needed
```

Then follow the typical workflow below:

```python
# 1. Initialize the manager and select a VP style
vp_manager = PromptManager()
vp_prompt = vp_manager.get_prompt(v_prompts[0])  # v_prompts is a list of VP styles/configs

# 2. Retrieve bounding box and polygon metadata for a given image
bbox = meta_bbox[img_nm]["default"]          # e.g., [x_min, y_min, x_max, y_max]
poly = meta_polygon[img_nm]["default"]       # e.g., list of [x, y] points

# 3. Construct the annotation list expected by the VP renderer
anno = [
    {
        "bbox": bbox,
        "segmentation": poly,
    }
]

# 4. Apply the visual prompt to the image
img = vp_prompt.apply(
    img,
    anno,
    {"width": img.width, "height": img.height},
    color=color,  # e.g., "red", "blue" — this should match the <color> tag in your question
)
```

After you render the VP (`img`) and decide on a `color`, update the question string from the meta file by replacing the corresponding phrase with the `<color> <vp>` token pair. This ensures the **image**, **visual prompt**, and **textual reference** are consistently aligned for Stage 2 QA experiments.

---

## Citation

If you use VP-Bench in your research, please cite:

> Xu, M., Chen, J., Zhao, Y., Li, J. C. L., Qiu, Y., Du, Z., Wu, M., Zhang, P., Li, K., Yang, H., Ma, W., Wei, J., Li, Q., Liu, K., & Lei, W.
> *VP-Bench: A Comprehensive Benchmark for Visual Prompting in Multimodal Large Language Models*. AAAI 2026 / arXiv:2511.11438. ([arXiv][1])

### BibTeX

```bibtex
@article{xu2025vpbench,
  title   = {VP-Bench: A Comprehensive Benchmark for Visual Prompting in Multimodal Large Language Models},
  author  = {Xu, Mingjie and Chen, Jinpeng and Zhao, Yuzhi and Li, Jason Chun Lok and Qiu, Yue and Du, Zekang and Wu, Mengyang and Zhang, Pingping and Li, Kun and Yang, Hongzheng and Ma, Wenao and Wei, Jiaheng and Li, Qinbin and Liu, Kangcheng and Lei, Wenqiang},
  journal = {AAAI 2026},
  year    = {2025}
}
```
