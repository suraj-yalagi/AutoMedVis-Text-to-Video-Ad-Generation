from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqGeneration
import torch

class MedicineTextProcessor:
    def __init__(self, model_name: str = "google/flan-t5-base"):
        """Initialize the text processor with a language model."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqGeneration.from_pretrained(model_name)
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

    def extract_keywords(self, description: str) -> Dict[str, List[str]]:
        """Extract key information from medicine description."""
        prompt = f"""
        Extract the following information from this medicine description:
        - Medicine type
        - Usage instructions
        - Target symptoms
        - Target users
        - Key benefits
        
        Description: {description}
        """
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
        outputs = self.model.generate(**inputs, max_length=200)
        extracted_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse the extracted text into structured format
        return self._parse_extracted_text(extracted_text)

    def generate_script(self, keywords: Dict[str, List[str]]) -> Tuple[str, List[str]]:
        """Generate a video script and image prompts based on extracted keywords."""
        prompt = f"""
        Create a 30-second advertisement script and image prompts for a medicine with:
        Type: {keywords.get('medicine_type', [''])[0]}
        Usage: {keywords.get('usage_instructions', [''])[0]}
        Symptoms: {', '.join(keywords.get('target_symptoms', []))}
        Users: {', '.join(keywords.get('target_users', []))}
        Benefits: {', '.join(keywords.get('key_benefits', []))}
        
        Format the output as:
        SCRIPT: [the script]
        IMAGES: [list of image prompts]
        """
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
        outputs = self.model.generate(**inputs, max_length=500)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse the generated text into script and image prompts
        script, image_prompts = self._parse_generated_text(generated_text)
        return script, image_prompts

    def _parse_extracted_text(self, text: str) -> Dict[str, List[str]]:
        """Parse the extracted text into structured format."""
        # This is a simplified parser - you might want to enhance it
        keywords = {
            'medicine_type': [],
            'usage_instructions': [],
            'target_symptoms': [],
            'target_users': [],
            'key_benefits': []
        }
        
        # Basic parsing logic - you can enhance this based on your needs
        lines = text.split('\n')
        current_key = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.lower().replace(' ', '_')
                if key in keywords:
                    current_key = key
                    keywords[key].append(value.strip())
            elif current_key:
                keywords[current_key].append(line.strip())
                
        return keywords

    def _parse_generated_text(self, text: str) -> Tuple[str, List[str]]:
        """Parse the generated text into script and image prompts."""
        script = ""
        image_prompts = []
        
        current_section = None
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('SCRIPT:'):
                current_section = 'script'
                script = line.replace('SCRIPT:', '').strip()
            elif line.startswith('IMAGES:'):
                current_section = 'images'
                image_prompts = [p.strip() for p in line.replace('IMAGES:', '').split(',')]
            elif current_section == 'script':
                script += '\n' + line
            elif current_section == 'images':
                image_prompts.append(line)
                
        return script, image_prompts 