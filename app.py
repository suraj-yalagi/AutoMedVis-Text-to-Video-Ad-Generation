import streamlit as st
import os
from PIL import Image
import numpy as np
import torch

# Set page config
st.set_page_config(
    page_title="Medical Ad Generator",
    page_icon="💊",
    layout="wide"
)

# Create necessary directories
os.makedirs("generated_images", exist_ok=True)
os.makedirs("generated_music", exist_ok=True)
os.makedirs("generated_videos", exist_ok=True)

# Main title
st.title("💊 Medical Ad Generator")
st.markdown("""
Generate professional medical advertisements using AI. This tool creates:
- 🖼️ Medical images
- 🎶 Background music
- 🎬 Complete video ads
""")

# Sidebar
st.sidebar.title("Settings")
ad_type = st.sidebar.radio(
    "Select Ad Type",
    ["Medicine Ad", "Symptom Ad"]
)

def load_text_processor():
    with st.spinner("Loading text processor..."):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            model_name = "gpt2"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            st.success("Text processor loaded successfully!")
            return tokenizer, model
        except Exception as e:
            st.error(f"Error loading text processor: {str(e)}")
            return None, None

def load_image_generator():
    with st.spinner("Loading image generator..."):
        try:
            from diffusers import StableDiffusionPipeline
            model_id = "runwayml/stable-diffusion-v1-5"
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                use_safetensors=True,
                safety_checker=None
            )
            pipeline.enable_attention_slicing()
            if torch.cuda.is_available():
                pipeline.enable_model_cpu_offload()
            st.success("Image generator loaded successfully!")
            return pipeline
        except Exception as e:
            st.error(f"Error loading image generator: {str(e)}")
            return None

# Main content
if ad_type == "Medicine Ad":
    st.header("Medicine Advertisement")
    
    col1, col2 = st.columns(2)
    
    with col1:
        medicine_name = st.text_input("Medicine Name")
        medicine_description = st.text_area(
            "Medicine Description",
            placeholder="Enter detailed description of the medicine, including its type, usage, target symptoms, and benefits..."
        )
        
    with col2:
        num_images = st.slider("Number of Images", 1, 5, 3)
        output_filename = st.text_input(
            "Output Filename",
            value=f"{medicine_name or 'medical_ad'}.mp4"
        )
        
    if st.button("Generate Ad"):
        if not medicine_description:
            st.error("Please enter a medicine description")
        else:
            try:
                # Step 1: Load and initialize text processor
                tokenizer, model = load_text_processor()
                if tokenizer and model:
                    st.info("Generating script...")
                    # Add your text processing logic here
                    script = "Sample script for " + medicine_name if medicine_name else "Sample script"
                    st.text(script)
                
                # Step 2: Load and initialize image generator
                pipeline = load_image_generator()
                if pipeline:
                    st.info("Generating images...")
                    # Generate images based on medicine description
                    prompts = [
                        f"Professional medical illustration of {medicine_name}, high quality, detailed",
                        f"Person using {medicine_name} medicine, professional medical setting",
                        f"Medical professional explaining {medicine_name} usage, clear and professional"
                    ]
                    
                    # Create output directory
                    output_dir = "generated_images"
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Generate and display images
                    cols = st.columns(min(3, num_images))
                    for i, prompt in enumerate(prompts[:num_images]):
                        with cols[i]:
                            with st.spinner(f"Generating image {i+1}..."):
                                # Generate image
                                image = pipeline(
                                    prompt=prompt,
                                    negative_prompt="blurry, distorted, unrealistic, low quality, low resolution",
                                    num_inference_steps=30,  # Reduced for faster generation
                                    guidance_scale=7.5
                                ).images[0]
                                
                                # Save image
                                image_path = os.path.join(output_dir, f"medical_image_{i}.png")
                                image.save(image_path)
                                
                                # Display image
                                st.image(image, caption=f"Image {i+1}")
                    
                    st.success(f"Generated {num_images} images successfully!")
                
                st.success("Advertisement components generated successfully!")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                
else:  # Symptom Ad
    st.header("Symptom Advertisement")
    
    col1, col2 = st.columns(2)
    
    with col1:
        symptoms = st.text_input(
            "Symptoms",
            placeholder="Enter symptoms separated by commas"
        ).split(",")
        symptoms = [s.strip() for s in symptoms if s.strip()]
        
        description = st.text_area(
            "Description",
            placeholder="Enter description of the symptoms and their impact..."
        )
        
    with col2:
        num_images = st.slider("Number of Images", 1, 5, 3)
        output_filename = st.text_input(
            "Output Filename",
            value=f"{'_'.join(symptoms)}_ad.mp4"
        )
        
    if st.button("Generate Ad"):
        if not symptoms or not description:
            st.error("Please enter symptoms and description")
        else:
            try:
                # Step 1: Load and initialize text processor
                tokenizer, model = load_text_processor()
                if tokenizer and model:
                    st.info("Generating script...")
                    # Add your text processing logic here
                    script = "Sample script for symptoms: " + ", ".join(symptoms)
                    st.text(script)
                
                # Step 2: Load and initialize image generator
                pipeline = load_image_generator()
                if pipeline:
                    st.info("Generating images...")
                    # Add your image generation logic here
                    st.success("Images would be generated here")
                
                st.success("Basic components loaded successfully! Full functionality coming soon.")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
### About
This tool uses AI to generate medical advertisements. It combines:
- Text understanding with GPT-2
- Image generation with Stable Diffusion
- Video composition with MoviePy

Status: Currently in testing phase. Components are loaded on demand to ensure stability.
""") 