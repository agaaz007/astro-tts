import os
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration

def download_model():
    model_name = "ai4bharat/indic-parler-tts"
    model_dir = "model"
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Downloading model and tokenizer from {model_name}...")
    
    # Download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=model_dir
    )
    tokenizer.save_pretrained(model_dir)
    
    # Download model
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="float16",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        cache_dir=model_dir
    )
    model.save_pretrained(model_dir)
    
    print(f"Model and tokenizer downloaded and saved to {model_dir}")

if __name__ == "__main__":
    download_model()
