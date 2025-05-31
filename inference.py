# inference.py
from flask import Flask, request, jsonify
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import io
import base64
import sys
import torch._dynamo

app = Flask(__name__)

# Check and log Flash Attention availability
try:
    import flash_attn
    print("[INFO] Flash Attention is available and imported.", file=sys.stdout)
except ImportError:
    print("[INFO] Flash Attention is NOT available.", file=sys.stdout)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Enable TensorFloat-32 for speed (works on most modern GPUs)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("[INFO] TensorFloat-32 (TF32) enabled for faster matmul operations", file=sys.stdout)

model = ParlerTTSForConditionalGeneration.from_pretrained(
    "ai4bharat/indic-parler-tts",
    attn_implementation="eager"
).to(device)
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

# Quantization functions for performance optimization
def try_onnx_quantization(model, tokenizer, description_tokenizer):
    """Try to quantize model using ONNX/Optimum for better performance"""
    try:
        print("[INFO] Attempting ONNX quantization with Optimum...", file=sys.stdout)
        from optimum.onnxruntime import ORTModelForSpeechSeq2Seq, ORTQuantizer, ORTConfig
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
        import tempfile
        import os
        
        # Create temporary directory for ONNX model
        with tempfile.TemporaryDirectory() as temp_dir:
            # Export to ONNX format
            print("[INFO] Exporting model to ONNX format...", file=sys.stdout)
            
            # Try to use Optimum's automatic export
            try:
                onnx_model = ORTModelForSpeechSeq2Seq.from_pretrained(
                    "ai4bharat/indic-parler-tts",
                    export=True,
                    cache_dir=temp_dir
                )
                print("[INFO] ✅ ONNX export successful", file=sys.stdout)
                
                # Apply quantization
                quantization_config = AutoQuantizationConfig.avx512_vnni(is_static=False)
                quantizer = ORTQuantizer.from_pretrained(onnx_model)
                quantized_model = quantizer.quantize(
                    save_dir=temp_dir,
                    quantization_config=quantization_config
                )
                print("[INFO] ✅ ONNX quantization successful", file=sys.stdout)
                return quantized_model
                
            except Exception as e:
                print(f"[INFO] Optimum ONNX export failed: {str(e)[:100]}...", file=sys.stdout)
                return None
                
    except ImportError as e:
        print(f"[INFO] ONNX quantization libraries not available: {str(e)[:100]}...", file=sys.stdout)
        return None
    except Exception as e:
        print(f"[INFO] ONNX quantization failed: {str(e)[:100]}...", file=sys.stdout)
        return None

def try_tensorrt_quantization(model):
    """Try to optimize model using TensorRT (if available)"""
    try:
        print("[INFO] Attempting TensorRT optimization...", file=sys.stdout)
        import torch_tensorrt
        
        # Create sample inputs for TensorRT compilation
        sample_input_ids = torch.randint(0, 1000, (1, 10)).to(device)
        sample_attention_mask = torch.ones(1, 10).to(device)
        
        # Compile with TensorRT
        trt_model = torch_tensorrt.compile(
            model,
            inputs=[
                torch_tensorrt.Input(
                    min_shape=[1, 1],
                    opt_shape=[1, 50], 
                    max_shape=[1, 100],
                    dtype=torch.float32
                )
            ],
            enabled_precisions={torch.float32, torch.half},
            workspace_size=1 << 30  # 1GB
        )
        print("[INFO] ✅ TensorRT optimization successful", file=sys.stdout)
        return trt_model
        
    except ImportError:
        print("[INFO] TensorRT not available", file=sys.stdout)
        return None
    except Exception as e:
        print(f"[INFO] TensorRT optimization failed: {str(e)[:100]}...", file=sys.stdout)
        return None

def try_optimum_quantization(model):
    """Try Hugging Face Optimum dynamic quantization"""
    try:
        print("[INFO] Attempting Optimum dynamic quantization...", file=sys.stdout)
        from optimum.intel import INCQuantizer
        
        quantizer = INCQuantizer.from_pretrained(model)
        quantized_model = quantizer.quantize()
        print("[INFO] ✅ Optimum quantization successful", file=sys.stdout)
        return quantized_model
        
    except ImportError:
        print("[INFO] Optimum Intel quantization not available", file=sys.stdout)
        return None
    except Exception as e:
        print(f"[INFO] Optimum quantization failed: {str(e)[:100]}...", file=sys.stdout)
        return None

# Try quantization methods in order of preference
print("[INFO] Attempting model quantization for performance...", file=sys.stdout)
quantized_model = None

# Try ONNX quantization first (most compatible)
if quantized_model is None:
    quantized_model = try_onnx_quantization(model, tokenizer, description_tokenizer)

# Try Optimum quantization as fallback
if quantized_model is None:
    quantized_model = try_optimum_quantization(model)

# Try TensorRT optimization (most aggressive)
if quantized_model is None:
    quantized_model = try_tensorrt_quantization(model)

# Use quantized model if available
if quantized_model is not None:
    print("[INFO] ✅ Using quantized model for inference", file=sys.stdout)
    model = quantized_model
else:
    print("[INFO] No quantization method succeeded, using original model", file=sys.stdout)

torch._dynamo.config.suppress_errors = True

# Try compiling with detailed logging
print("[INFO] Attempting to compile model.forward with torch.compile...", file=sys.stdout)
try:
    model.forward = torch.compile(model.forward, mode="default")
    print("[INFO] ✅ Model compilation SUCCEEDED with 'default' mode", file=sys.stdout)
except Exception as e:
    print(f"[INFO] ❌ Model compilation FAILED: {str(e)[:100]}...", file=sys.stdout)
    print("[INFO] Falling back to eager mode", file=sys.stdout)

# Warm up
dummy = tokenizer("Warmup", return_tensors="pt").to(device)
try:
    print("[INFO] Starting model warmup...", file=sys.stdout)
    # Handle different model types for warmup
    if is_quantized and 'ONNX' in str(model.__class__):
        print("[INFO] Skipping warmup for ONNX model (not needed)", file=sys.stdout)
    else:
        model.generate(
            input_ids=dummy.input_ids,
            attention_mask=dummy.attention_mask,
            prompt_input_ids=dummy.input_ids,
            prompt_attention_mask=dummy.attention_mask,
            temperature=0.7,
            max_new_tokens=5  # Short warmup for speed
        )
        print("[INFO] Model warmup completed successfully", file=sys.stdout)
except Exception as e:
    print(f"[INFO] Warmup failed (not critical): {str(e)[:100]}...", file=sys.stdout)

# Log compilation stats after warmup
def log_compilation_stats():
    try:
        if hasattr(torch._dynamo.utils, 'compilation_metrics'):
            print("[INFO] Compilation metrics available", file=sys.stdout)
        if hasattr(model.forward, '_dynamo_compiled_fn'):
            print("[INFO] Model forward pass is compiled", file=sys.stdout)
        else:
            print("[INFO] Model forward pass is in eager mode", file=sys.stdout)
    except:
        pass

# Check compilation status
log_compilation_stats()

# Log final model type for debugging
print(f"[INFO] Final model type: {type(model).__name__}", file=sys.stdout)
is_quantized = hasattr(model, '__class__') and ('ONNX' in str(model.__class__) or 'Quantized' in str(model.__class__))
print(f"[INFO] Model is quantized: {is_quantized}", file=sys.stdout)

# Configure model for optimal FP32 performance
print("[INFO] Configuring model for optimal FP32 performance...", file=sys.stdout)
try:
    # Enable static cache for faster key-value caching
    if hasattr(model, 'generation_config'):
        model.generation_config.cache_implementation = "static"
        model.generation_config.do_sample = True
        model.generation_config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        print("[INFO] Static cache and generation config optimized", file=sys.stdout)
    
    # Enable memory efficient attention if available
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        print("[INFO] SDPA available for memory efficient attention", file=sys.stdout)
    
    # Optimize for single sequence generation
    model.eval()
    torch.cuda.empty_cache()
    print("[INFO] Model set to eval mode and CUDA cache cleared", file=sys.stdout)
    
except Exception as e:
    print(f"[INFO] Some optimizations failed: {str(e)[:100]}", file=sys.stdout)

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"})

@app.route("/invocations", methods=["POST"])
def invoke():
    data = request.get_json()
    prompt = data.get("prompt", "")
    description = data.get("description", "")

    # Input validation
    if not prompt or not description or not isinstance(prompt, str) or not isinstance(description, str):
        return jsonify({"error": "Both 'prompt' and 'description' must be non-empty strings."}), 400

    prompt_ids = tokenizer(prompt, return_tensors="pt").to(device)
    desc_ids = description_tokenizer(description, return_tensors="pt").to(device)

    # Try BF16 first, fallback to FP16, then alternative mixed precision approaches, then FP32
    print(f"[INFO] GPU: {torch.cuda.get_device_name()}, BF16 supported: {torch.cuda.is_bf16_supported()}", file=sys.stdout)
    print(f"[INFO] Model dtype: {next(model.parameters()).dtype}", file=sys.stdout)
    
    try:
        print("[INFO] Attempting BF16 with autocast...", file=sys.stdout)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            generation = model.generate(
                input_ids=desc_ids.input_ids,
                attention_mask=desc_ids.attention_mask,
                prompt_input_ids=prompt_ids.input_ids,
                prompt_attention_mask=prompt_ids.attention_mask,
                temperature=0.7
            )
        if not torch.isfinite(generation).all():
            raise ValueError("NaN/Inf detected in BF16 output")
        print("[INFO] Inference mode: BF16", file=sys.stdout)
    except Exception as e_bf16:
        print(f"[INFO] BF16 autocast failed: {type(e_bf16).__name__}: {str(e_bf16)[:150]}", file=sys.stdout)
        # Try FP16 as fallback
        try:
            print("[INFO] Attempting FP16 with autocast...", file=sys.stdout)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                generation = model.generate(
                    input_ids=desc_ids.input_ids,
                    attention_mask=desc_ids.attention_mask,
                    prompt_input_ids=prompt_ids.input_ids,
                    prompt_attention_mask=prompt_ids.attention_mask,
                    temperature=0.7
                )
            if not torch.isfinite(generation).all():
                raise ValueError("NaN/Inf detected in FP16 output")
            print("[INFO] Inference mode: FP16", file=sys.stdout)
        except Exception as e_fp16:
            print(f"[INFO] FP16 autocast failed: {type(e_fp16).__name__}: {str(e_fp16)[:150]}", file=sys.stdout)
            # Try model.half() approach (direct FP16 without autocast)
            try:
                print("[INFO] Attempting direct FP16 (model.half())...", file=sys.stdout)
                # Convert model to FP16 directly
                model_fp16 = model.half()
                prompt_ids_fp16 = tokenizer(prompt, return_tensors="pt").to(device).half()
                desc_ids_fp16 = description_tokenizer(description, return_tensors="pt").to(device).half()
                
                generation = model_fp16.generate(
                    input_ids=desc_ids_fp16.input_ids,
                    attention_mask=desc_ids_fp16.attention_mask,
                    prompt_input_ids=prompt_ids_fp16.input_ids,
                    prompt_attention_mask=prompt_ids_fp16.attention_mask,
                    temperature=0.7
                )
                if not torch.isfinite(generation).all():
                    raise ValueError("NaN/Inf detected in direct FP16 output")
                print("[INFO] Inference mode: Direct FP16", file=sys.stdout)
                # Convert back to FP32 for future requests
                model.float()
            except Exception as e_direct_fp16:
                print(f"[INFO] Direct FP16 failed: {type(e_direct_fp16).__name__}: {str(e_direct_fp16)[:150]}", file=sys.stdout)
                # Convert back to FP32 if half() failed
                try:
                    model.float()
                except:
                    pass
                # Try INT8 quantized inference if available
                try:
                    # Check if we already have a quantized model
                    if hasattr(model, '__class__') and 'ONNX' in str(model.__class__):
                        print("[INFO] Already using ONNX quantized model, skipping additional quantization", file=sys.stdout)
                        raise ImportError("Already quantized")
                    
                    import torch
                    if hasattr(torch, 'quantization'):
                        print("[INFO] Attempting PyTorch dynamic INT8 quantization...", file=sys.stdout)
                        model_int8 = torch.quantization.quantize_dynamic(
                            model, {torch.nn.Linear}, dtype=torch.qint8
                        )
                        generation = model_int8.generate(
                            input_ids=desc_ids.input_ids,
                            attention_mask=desc_ids.attention_mask,
                            prompt_input_ids=prompt_ids.input_ids,
                            prompt_attention_mask=prompt_ids.attention_mask,
                            temperature=0.7
                        )
                        if not torch.isfinite(generation).all():
                            raise ValueError("NaN/Inf detected in INT8 output")
                        print("[INFO] Inference mode: PyTorch INT8", file=sys.stdout)
                    else:
                        raise ImportError("Torch quantization not available")
                except Exception as e_int8:
                    print(f"[INFO] PyTorch INT8 failed: {type(e_int8).__name__}: {str(e_int8)[:150]}", file=sys.stdout)
                    # Final fallback to FP32
                    try:
                        print("[INFO] Falling back to FP32 (no autocast)...", file=sys.stdout)
                        generation = model.generate(
                            input_ids=desc_ids.input_ids,
                            attention_mask=desc_ids.attention_mask,
                            prompt_input_ids=prompt_ids.input_ids,
                            prompt_attention_mask=prompt_ids.attention_mask,
                            temperature=0.7
                        )
                        if not torch.isfinite(generation).all():
                            return jsonify({"error": "Model output contains NaN or Inf in all modes. Try different input."}), 500
                        print("[INFO] Inference mode: FP32", file=sys.stdout)
                    except Exception as e_fp32:
                        return jsonify({"error": f"Model inference failed in all precision modes: {str(e_fp32)}"}), 500

    audio = generation.cpu().numpy().squeeze()
    buffer = io.BytesIO()
    sf.write(buffer, audio, model.config.sampling_rate, format='WAV')
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")

    return jsonify({"audio_base64": encoded})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
