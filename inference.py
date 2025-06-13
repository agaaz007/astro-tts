import io
import torch
import soundfile as sf
import base64
import sys # Make sure sys is imported
from transformers import AutoTokenizer
from transformers.utils import is_flash_attn_2_available

from threading import Thread
from flask import Flask, request, Response, stream_with_context, jsonify
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer
import numpy as np
import os, pathlib

app = Flask(__name__)

# ───────────────────────────────────────────────────────────────
# 1) LOAD MODEL & TOKENIZER
# ───────────────────────────────────────────────────────────────

# 1a) pick torch.device and dtype
torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Use float16 as per your download_model.py for consistency and GPU efficiency
torch_dtype  = torch.float16
print("using torch device: with dtype ", torch_dtype)

# 1b) decide which model repo / folder to load
env_model = os.getenv("MODEL_ID", "").strip()

default_dir = pathlib.Path("/opt/ml/model")
if env_model:
    # 1) explicit override
    model_name = env_model
    print(f"Using model specified via MODEL_ID env var: '{model_name}'", file=sys.stderr)
elif (default_dir / "config.json").is_file():
    # 2) pre-packed artefact (SageMaker)
    model_name = str(default_dir)
    print("Using model found inside /opt/ml/model .", file=sys.stderr)
else:
    # 3) fallback to public HF repo
    model_name = "ai4bharat/indic-parler-tts"
    print(f"/opt/ml/model empty – downloading '{model_name}' from the Hub.", file=sys.stderr)

# 1c) single tokenizer for both "prompt" and "description"
tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 1c) load Parler-TTS with Flash Attention 2 preference and fallback logic
# Decide best attention backend
if is_flash_attn_2_available() and torch.cuda.is_available():
    attention_implementation = "flash_attention_2"
    print("Using Flash Attention 2 backend.")
elif torch.__version__ >= "2.0" and torch.cuda.is_available():
    attention_implementation = "sdpa"
    print("Flash Attention 2 unavailable – falling back to PyTorch SDPA.")
else:
    attention_implementation = "eager"
    print("Using default (eager) attention implementation.")

try:
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation=attention_implementation
    ).to(torch_device)
    model.eval()
except Exception as e:
    print(f"Failed to load model with {attention_implementation} attention: {e}", file=sys.stderr)
    print("Attempting to load model with default 'eager' attention as a final fallback.", file=sys.stderr)
    try:
        model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="eager"
        ).to(torch_device)
        model.eval()
    except Exception as e_eager:
        print(f"Failed to load model even with 'eager' attention: {e_eager}", file=sys.stderr)
        raise # Re-raise if even eager fails, as model loading is critical

# ───────────────────────────────────────────────────────────────
# 2) FLASK ROUTE FOR STREAMING RAW PCM AUDIO
# ───────────────────────────────────────────────────────────────

@app.route("/invocations", methods=["POST"])
def invocations():
    """
    Expects JSON:
    { "prompt": "...", "description": "..." }
    Streams back raw 16-bit PCM audio chunks directly.
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON payload"}), 400

    text = data.get("prompt", "").strip()
    desc_text = data.get("description", "").strip()

    if not text:
        return jsonify({"error": "No prompt provided"}), 400
    if not desc_text:
        return jsonify({"error": "No description provided"}), 400

    # Tokenize the description (style/voice)
    _d = tok(desc_text, return_tensors="pt").to(torch_device)
    desc_ids  = _d.input_ids
    desc_mask = _d.attention_mask

    # Tokenize the prompt (speech transcript)
    _p = tok(text, return_tensors="pt").to(torch_device)
    prompt_ids  = _p.input_ids
    prompt_mask = _p.attention_mask

    # Build a ParlerTTSStreamer that yields every chunk until the very end
    frame_rate = model.audio_encoder.config.frame_rate
    play_steps_in_s = 0.5
    play_steps = int(frame_rate * play_steps_in_s)
    streamer = ParlerTTSStreamer(model, device=torch_device, play_steps=play_steps)

    gen_kwargs = dict(
        input_ids              = desc_ids,
        attention_mask         = desc_mask,
        prompt_input_ids       = prompt_ids,
        prompt_attention_mask  = prompt_mask,
        streamer               = streamer,
        do_sample              = True,
        temperature            = 0.6,
        min_new_tokens         = 5,
    )

    def generate_pcm_chunks():
        """
        Generator that yields raw 16-bit PCM audio chunks (bytes).
        """
        with torch.no_grad():
            # Kick off model.generate in a background thread
            thread = Thread(target=model.generate, kwargs=gen_kwargs)
            thread.start()

            for new_audio in streamer:
                # streamer may return either a Tensor (GPU/CPU) or an np.ndarray
                if isinstance(new_audio, torch.Tensor):
                    pcm_tensor = new_audio.cpu().numpy()
                else:
                    pcm_tensor = new_audio  # already a NumPy array

                # Convert float16 / float32 PCM → int16
                audio_int16 = (pcm_tensor * 32767).astype(np.int16)
                yield audio_int16.tobytes()

            thread.join()

    # Return a chunked HTTP response, streaming raw PCM bytes
    return Response(
        stream_with_context(generate_pcm_chunks()),
        mimetype="audio/pcm",
        headers={"Transfer-Encoding": "chunked"}
    )

# ───────────────────────────────────────────────────────────────
# 3) FLASK ROUTE FOR HEALTHCHECK (ping)
# ───────────────────────────────────────────────────────────────

@app.route("/ping", methods=["GET"])
def ping():
    return "pong", 200

# ───────────────────────────────────────────────────────────────
# 4) RUN THE FLASK APP
# ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, threaded=False)