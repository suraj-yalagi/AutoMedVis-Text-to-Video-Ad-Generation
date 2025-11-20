# AutoMedVis — Text-to-Video Ad Generation for Medical Visuals

AutoMedVis-Text-to-Video-Ad-Generation is a research / demo codebase for producing short, high-quality medical advertisement-style videos from plain-text prompts and lightweight templates. The project demonstrates an end-to-end pipeline tying together text-conditioning, motion-aware generation, multi-scene composition, simple audio/voiceover integration, and final editing for short (5–30s) ads and explainer clips suitable for medical or healthcare contexts.

Important note: This repository is intended for research, prototyping, and demonstration only. Produced media should never be used as medical advice. Always consult licensed medical professionals and follow regulations for healthcare content in your jurisdiction.

Table of contents
- About
- Features
- Quick demo
- Installation
- Usage
  - Prepare prompts
  - Run inference (single prompt)
  - Batch generation
  - Fine-tuning / training (high-level)
  - Postprocessing and assembly
- Project structure
- Dependencies & requirements
- Dataset & model notes
- Examples
- Best practices & ethical considerations
- Contributing
- License
- Contact

About
AutoMedVis creates short video ads or explainer clips by converting structured text prompts into animated scenes. The pipeline focuses on:
- Faithful, controlled visual generation for medical contexts (e.g., clinic scenes, device closeups, simplified anatomy diagrams).
- Easy-to-author text templates so non-technical users can create video variations.
- Composable scenes, captions, and audio voiceovers.
- Modular components so you can replace individual models (text encoder, video generator, TTS) with alternatives.

Features
- Text-to-video generation pipeline (prompt → frames → assembled video).
- Scene templating: sequence multiple short scenes with transition options.
- Simple storyboard-driven layout: set duration, shot type, and caption for each scene.
- Optional TTS voiceover / audio overlay.
- Batch mode for generating multiple ad variations.
- Export to MP4 with optional subtitle burn-in.

Quick demo
1. Clone the repo:
   git clone https://github.com/suraj-yalagi/AutoMedVis-Text-to-Video-Ad-Generation.git
2. Install dependencies (see Installation).
3. Run a single prompt generation (example):
   python scripts/generate.py --prompt "Friendly doctor explaining diabetes prevention, bright clinic, 10s" --out demo.mp4

Installation
1. Clone:
   git clone https://github.com/suraj-yalagi/AutoMedVis-Text-to-Video-Ad-Generation.git
   cd AutoMedVis-Text-to-Video-Ad-Generation

2. Python environment (recommended):
   - Python 3.8+ (3.10 recommended)
   - Create virtualenv:
     python -m venv venv
     source venv/bin/activate   # macOS / Linux
     venv\Scripts\activate      # Windows

3. Install Python dependencies:
   pip install -r requirements.txt

4. Optional GPU/CUDA:
   - For reasonable generation speed you will want an NVIDIA GPU and CUDA-compatible PyTorch.
   - Install a matching torch + torchvision with CUDA support; adjust install line on https://pytorch.org

Usage

Prepare prompts
- The pipeline expects a JSON/YAML storyboard or a single-line prompt.
- Storyboard example (storyboards/example_story.json):
 
