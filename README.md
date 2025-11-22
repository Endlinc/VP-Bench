# VP-Bench: A Comprehensive Benchmark for Visual Prompting in Multimodal Large Language Models

**AAAI 2026 (to appear)**  

This repository provides the **official Stage 1 evaluation code** for

> **VP-Bench: A Comprehensive Benchmark for Visual Prompting in Multimodal Large Language Models**

VP-Bench evaluates whether multimodal LLMs can  
1. **Perceive visual prompts (VPs)** such as bounding boxes, masks, scribbles, arrows, etc.  
2. **Use these prompts** to solve real downstream tasks.

At the moment this repo focuses on:

- **Stage 1 – VP Perception**  
  🔓 **Released**: evaluation logic integrated into **[VLMEvalKit](https://github.com/open-compass/VLMEvalKit)** and a VLMEvalKit-compatible [Stage 1 test set](https://modelscope.cn/datasets/VP-Bench/VP-Bench_stage_1_VLMEvalKit/resolve/master/vp_bench_stage_1.tsv).
- **Stage 2 – VP-Enhanced Downstream Tasks**  
  ⏳ **Coming soon**: new tasks and evaluation code (will be added to this repo later).

---

## Repository Structure

Current layout:

```text
VP-Bench/
├─ vlmeval/
│  ├─ __init__.py
│  └─ dataset/
│     ├─ __init__.py
│     └─ image_vqa.py   # Modified VLMEvalKit dataset / evaluator for VP-Bench Stage 1
├─ .gitignore
├─ LICENSE
└─ README.md
