from typing import List, Optional
from moviepy.editor import (
    VideoFileClip,
    ImageClip,
    AudioFileClip,
    CompositeVideoClip,
    TextClip,
    concatenate_videoclips
)
import os
from PIL import Image
import numpy as np

class MedicalVideoComposer:
    def __init__(
        self,
        output_dir: str = "generated_videos",
        resolution: tuple = (1920, 1080),
        fps: int = 30
    ):
        """Initialize the video composer."""
        self.output_dir = output_dir
        self.resolution = resolution
        self.fps = fps
        os.makedirs(output_dir, exist_ok=True)

    def create_video(
        self,
        image_paths: List[str],
        audio_path: str,
        script: str,
        output_filename: str = "medical_ad.mp4",
        duration_per_image: float = 5.0
    ) -> str:
        """Create a video advertisement from images and audio."""
        # Load and prepare images
        image_clips = []
        for image_path in image_paths:
            # Load and resize image
            img = Image.open(image_path)
            img = img.resize(self.resolution, Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            img_array = np.array(img)
            
            # Create image clip
            clip = ImageClip(img_array)
            clip = clip.set_duration(duration_per_image)
            image_clips.append(clip)
            
        # Concatenate image clips
        video = concatenate_videoclips(image_clips, method="compose")
        
        # Load and add audio
        audio = AudioFileClip(audio_path)
        video = video.set_audio(audio)
        
        # Add text overlay
        text_clips = self._create_text_clips(script, video.duration)
        if text_clips:
            video = CompositeVideoClip([video] + text_clips)
            
        # Write final video
        output_path = os.path.join(self.output_dir, output_filename)
        video.write_videofile(
            output_path,
            fps=self.fps,
            codec="libx264",
            audio_codec="aac"
        )
        
        return output_path

    def _create_text_clips(
        self,
        script: str,
        total_duration: float
    ) -> List[TextClip]:
        """Create text clips for the script."""
        text_clips = []
        script_lines = script.split('\n')
        duration_per_line = total_duration / len(script_lines)
        
        for i, line in enumerate(script_lines):
            if not line.strip():
                continue
                
            # Create text clip
            text_clip = TextClip(
                line,
                fontsize=40,
                color='white',
                font='Arial-Bold',
                stroke_color='black',
                stroke_width=2
            )
            
            # Position and time the text
            text_clip = text_clip.set_position(('center', 'bottom'))
            text_clip = text_clip.set_start(i * duration_per_line)
            text_clip = text_clip.set_duration(duration_per_line)
            
            text_clips.append(text_clip)
            
        return text_clips

    def create_medicine_ad(
        self,
        medicine_name: str,
        image_paths: List[str],
        audio_path: str,
        script: str,
        output_filename: Optional[str] = None
    ) -> str:
        """Create a medicine-specific advertisement."""
        if output_filename is None:
            output_filename = f"{medicine_name}_ad.mp4"
            
        return self.create_video(
            image_paths=image_paths,
            audio_path=audio_path,
            script=script,
            output_filename=output_filename
        )

    def create_symptom_ad(
        self,
        symptoms: List[str],
        image_paths: List[str],
        audio_path: str,
        script: str,
        output_filename: Optional[str] = None
    ) -> str:
        """Create a symptom-specific advertisement."""
        if output_filename is None:
            output_filename = f"{'_'.join(symptoms)}_ad.mp4"
            
        return self.create_video(
            image_paths=image_paths,
            audio_path=audio_path,
            script=script,
            output_filename=output_filename
        ) 