from typing import Optional, List
import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import os

class MedicalMusicGenerator:
    def __init__(
        self,
        model_name: str = "facebook/musicgen-medium",
        device: Optional[str] = None
    ):
        """Initialize the music generator with AudioCraft."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.device = device
        self.model = MusicGen.get_pretrained(model_name)
        self.model.set_generation_params(
            duration=30,  # 30 seconds for typical ad length
            temperature=0.8,
            top_k=250,
            top_p=0.0
        )

    def generate_music(
        self,
        prompt: str,
        output_dir: str = "generated_music",
        num_samples: int = 1,
        **kwargs
    ) -> List[str]:
        """Generate background music based on the prompt."""
        os.makedirs(output_dir, exist_ok=True)
        generated_paths = []
        
        # Enhance prompt with medical context
        enhanced_prompt = f"""
        Background music for medical advertisement, {prompt},
        professional, calm, reassuring, medical setting,
        suitable for healthcare content
        """
        
        # Generate music
        wav = self.model.generate(
            descriptions=[enhanced_prompt],
            progress=True,
            return_tokens=False
        )
        
        # Save the generated music
        for i in range(num_samples):
            filename = f"medical_music_{i}.wav"
            path = os.path.join(output_dir, filename)
            audio_write(
                path,
                wav[0].cpu(),
                self.model.sample_rate,
                strategy="loudness",
                loudness_compressor=True
            )
            generated_paths.append(path)
            
        return generated_paths

    def generate_medicine_music(
        self,
        medicine_type: str,
        target_audience: str,
        mood: str = "calm",
        output_dir: str = "generated_music"
    ) -> List[str]:
        """Generate specific music for medicine advertisement."""
        prompt = f"""
        Background music for {medicine_type} advertisement,
        targeting {target_audience},
        {mood} and reassuring mood,
        professional medical setting
        """
        
        return self.generate_music(
            prompt=prompt,
            output_dir=output_dir
        )

    def generate_symptom_music(
        self,
        symptoms: List[str],
        mood: str = "calm",
        output_dir: str = "generated_music"
    ) -> List[str]:
        """Generate music suitable for symptom-related content."""
        prompt = f"""
        Background music for medical content about {', '.join(symptoms)},
        {mood} and professional mood,
        suitable for healthcare education
        """
        
        return self.generate_music(
            prompt=prompt,
            output_dir=output_dir
        ) 