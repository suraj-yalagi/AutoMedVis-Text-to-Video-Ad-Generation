# AutoMedVis: Text-to-Video Ad Generation

AutoMedVis is an AI-powered project designed to generate engaging medical advertisement videos from textual descriptions. By leveraging cutting-edge deep learning models for both text-to-image and image-to-video synthesis, this repository enables seamless creation of medical promotional content—automatically, efficiently, and with minimal human intervention.

## Features

- **Text-to-Image Synthesis:** Converts medical product descriptions into high-quality medical images.
- **Image-to-Video Generation:** Transforms synthesized images into dynamic video advertisements.
- **Customizable Workflows:** Easily adapt input prompts and settings for different medical domains or ad requirements.
- **Automated Video Assembly:** Stitch generated frames into smooth, coherent ad videos.
- **Scalable & Extensible:** Modular design for integration with other AI models or pipelines.

## Example Workflow

1. **Input:** Provide a text description of a medical product or service.
2. **Image Generation:** The system generates relevant images based on the description.
3. **Video Synthesis:** Images are animated and merged into a short advertisement video.
4. **Output:** Download or deploy the final medical ad video.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Chinmay6824/AutoMedVis-Text-to-Video-Ad-Generation.git
   cd AutoMedVis-Text-to-Video-Ad-Generation
   ```

2. **Install Dependencies:**
   - Python 3.8+
   - [PyTorch](https://pytorch.org/)
   - [OpenCV](https://opencv.org/)
   - [Transformers](https://huggingface.co/transformers/)
   - Additional requirements:
     ```bash
     pip install -r requirements.txt
     ```

3. **(Optional) Set Up GPU Support:**
   - Ensure CUDA is installed for accelerated model inference.

## Usage

```bash
python main.py --input "Describe your medical product here."
```

- Generated videos will be saved in the `output/` directory.

### Arguments (Example)

- `--input`: Text prompt describing the medical product/service.
- `--output`: Output directory for videos (default: `output/`).
- Additional arguments may be available for model selection or video length.

## Example

```bash
python main.py --input "A compact digital thermometer with quick temperature readings, designed for children and adults."
```

## Repository Structure

```
AutoMedVis-Text-to-Video-Ad-Generation/
├── main.py
├── models/
│   ├── text_to_image.py
│   ├── image_to_video.py
├── utils/
│   ├── video_utils.py
│   └── ...
├── requirements.txt
├── output/
└── README.md
```

## Contributing

Contributions are welcome! Please open issues or pull requests for bug fixes, improvements, or new features.

## License

This project is licensed under the MIT License.

## Contact

For questions or collaborations, open an issue or contact [Chinmay6824](https://github.com/Chinmay6824).

---

*Empowering medical advertising with AI-driven creativity!*
