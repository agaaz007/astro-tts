import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import os

def download_model():
    print("Downloading model and tokenizers...")
    
    # Create model directory if it doesn't exist
    os.makedirs("model", exist_ok=True)
    
    # Download model
    model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts")
    model.save_pretrained("model")
    
    # Download tokenizers
    tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
    tokenizer.save_pretrained("model/tokenizer")
    
    description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    description_tokenizer.save_pretrained("model/desc_tokenizer")
    
    print("Model and tokenizers downloaded successfully!")

if __name__ == "__main__":
    download_model()
