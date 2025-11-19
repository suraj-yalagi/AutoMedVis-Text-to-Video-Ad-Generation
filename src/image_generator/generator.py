from typing import List, Optional
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import os

class MedicalImageGenerator:
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: Optional[str] = None
    ):
        """Initialize the image generator with Stable Diffusion."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.device = device
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            use_safetensors=True,
            safety_checker=None
        )
        
        # Use DPM-Solver++ for faster inference
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipeline.scheduler.config
        )
        
        # Enable memory optimization
        self.pipeline.enable_attention_slicing()
        if device == "cuda":
            self.pipeline.enable_model_cpu_offload()
        
        self.pipeline = self.pipeline.to(device)
        
        # Medical-specific negative prompts
        self.negative_prompt = """
        blurry, distorted, unrealistic, low quality, low resolution,
        incorrect anatomy, incorrect medical equipment, unsafe medical practices,
        inappropriate medical content, disturbing imagery, graphic content
        """

    def generate_images(
        self,
        prompts: List[str],
        num_images_per_prompt: int = 1,
        output_dir: str = "generated_images",
        **kwargs
    ) -> List[str]:
        """Generate medical images based on prompts."""
        os.makedirs(output_dir, exist_ok=True)
        generated_paths = []
        
        for i, prompt in enumerate(prompts):
            # Enhance prompt with medical context
            enhanced_prompt = f"""
            Professional medical illustration, {prompt},
            high quality, detailed, accurate medical representation,
            professional lighting, clear focus, medical setting
            """
            
            # Generate images
            images = self.pipeline(
                prompt=enhanced_prompt,
                negative_prompt=self.negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                **kwargs
            ).images
            
            # Save images
            for j, image in enumerate(images):
                filename = f"medical_image_{i}_{j}.png"
                path = os.path.join(output_dir, filename)
                image.save(path)
                generated_paths.append(path)
                
        return generated_paths

    def generate_medicine_visualization(
        self,
        medicine_type: str,
        usage_context: str,
        num_images: int = 1,
        output_dir: str = "generated_images"
    ) -> List[str]:
        """Generate specific medicine-related images."""
        prompts = [
            f"Professional medical illustration of {medicine_type}",
            f"Person using {medicine_type} in {usage_context}",
            f"Medical professional explaining {medicine_type} usage"
        ]
        
        return self.generate_images(
            prompts=prompts,
            num_images_per_prompt=num_images,
            output_dir=output_dir
        )

    def generate_symptom_visualization(
        self,
        symptoms: List[str],
        num_images: int = 1,
        output_dir: str = "generated_images"
    ) -> List[str]:
        """Generate images representing medical symptoms."""
        prompts = [
            f"Professional medical illustration showing {symptom}"
            for symptom in symptoms
        ]
        
        return self.generate_images(
            prompts=prompts,
            num_images_per_prompt=num_images,
            output_dir=output_dir
        ) 