from typing import Dict, List, Optional
import os
from ..text_processor.processor import MedicineTextProcessor
from ..image_generator.generator import MedicalImageGenerator
from ..music_generator.generator import MedicalMusicGenerator
from ..video_composer.composer import MedicalVideoComposer

class MedicalAdGenerator:
    def __init__(
        self,
        output_dir: str = "generated_ads",
        device: Optional[str] = None
    ):
        """Initialize the medical ad generator pipeline."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.text_processor = MedicineTextProcessor()
        self.image_generator = MedicalImageGenerator(device=device)
        self.music_generator = MedicalMusicGenerator(device=device)
        self.video_composer = MedicalVideoComposer(
            output_dir=os.path.join(output_dir, "videos")
        )

    def generate_ad(
        self,
        medicine_description: str,
        medicine_name: Optional[str] = None,
        num_images: int = 3,
        output_filename: Optional[str] = None
    ) -> Dict[str, str]:
        """Generate a complete medical advertisement."""
        # Step 1: Process text and generate script
        keywords = self.text_processor.extract_keywords(medicine_description)
        script, image_prompts = self.text_processor.generate_script(keywords)
        
        # Step 2: Generate images
        image_paths = self.image_generator.generate_images(
            prompts=image_prompts[:num_images],
            output_dir=os.path.join(self.output_dir, "images")
        )
        
        # Step 3: Generate music
        music_paths = self.music_generator.generate_medicine_music(
            medicine_type=keywords.get('medicine_type', [''])[0],
            target_audience=keywords.get('target_users', ['general'])[0],
            output_dir=os.path.join(self.output_dir, "music")
        )
        
        # Step 4: Compose video
        if output_filename is None:
            output_filename = f"{medicine_name or 'medical_ad'}.mp4"
            
        video_path = self.video_composer.create_medicine_ad(
            medicine_name=medicine_name or "Medicine",
            image_paths=image_paths,
            audio_path=music_paths[0],
            script=script,
            output_filename=output_filename
        )
        
        return {
            "script": script,
            "image_paths": image_paths,
            "music_paths": music_paths,
            "video_path": video_path
        }

    def generate_symptom_ad(
        self,
        symptoms: List[str],
        description: str,
        num_images: int = 3,
        output_filename: Optional[str] = None
    ) -> Dict[str, str]:
        """Generate a symptom-focused medical advertisement."""
        # Step 1: Process text and generate script
        keywords = self.text_processor.extract_keywords(description)
        script, image_prompts = self.text_processor.generate_script(keywords)
        
        # Step 2: Generate images
        image_paths = self.image_generator.generate_symptom_visualization(
            symptoms=symptoms,
            num_images=num_images,
            output_dir=os.path.join(self.output_dir, "images")
        )
        
        # Step 3: Generate music
        music_paths = self.music_generator.generate_symptom_music(
            symptoms=symptoms,
            output_dir=os.path.join(self.output_dir, "music")
        )
        
        # Step 4: Compose video
        if output_filename is None:
            output_filename = f"{'_'.join(symptoms)}_ad.mp4"
            
        video_path = self.video_composer.create_symptom_ad(
            symptoms=symptoms,
            image_paths=image_paths,
            audio_path=music_paths[0],
            script=script,
            output_filename=output_filename
        )
        
        return {
            "script": script,
            "image_paths": image_paths,
            "music_paths": music_paths,
            "video_path": video_path
        } 