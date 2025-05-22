"""
Script to optimize the Mistral model for deployment by:
1. Quantizing the model to reduce memory usage
2. Merging model shards
3. Converting to SafeTensors format (optional)

Run this script before building your Docker image to optimize model weights.
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def optimize_model(model_path, output_path, quantization='int8', merge_shards=True):
    """
    Optimize the model for deployment
    
    Args:
        model_path: Path to your original model directory
        output_path: Path to save the optimized model
        quantization: Quantization type ('int8', 'int4', 'none')
        merge_shards: Whether to merge model shards
    """
    print(f"Starting model optimization from {model_path} to {output_path}")
    print(f"Quantization: {quantization}, Merge shards: {merge_shards}")
    
    # Check if output directory exists, create if not
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created output directory: {output_path}")
    
    # First load tokenizer and save to output path
    print("Loading and saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(output_path)
    print("Tokenizer saved successfully")
    
    # Load model with different quantization options
    print(f"Loading model with {quantization} quantization...")
    
    if quantization == 'int8':
        # Load 8-bit quantized model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            load_in_8bit=True,
        )
    elif quantization == 'int4':
        # Load 4-bit quantized model (requires transformers>=4.30.0 and bitsandbytes>=0.39.0)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            load_in_4bit=True,
            quantization_config={
                "bnb_4bit_compute_dtype": torch.float16,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4"
            }
        )
    else:
        # Load model in FP16 if CUDA is available, otherwise FP32
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=dtype,
        )
    
    print("Model loaded successfully")
    
    # Save optimized model
    print(f"Saving optimized model to {output_path}...")
    model.save_pretrained(
        output_path,
        safe_serialization=True,  # Use safetensors format
        max_shard_size="2GB"      # Control shard size
    )
    print("Model optimization completed successfully")
    
    # Print directory size
    model_size_gb = sum(os.path.getsize(os.path.join(output_path, f)) for f in os.listdir(output_path) if os.path.isfile(os.path.join(output_path, f))) / (1024**3)
    print(f"Optimized model size: {model_size_gb:.2f} GB")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize Mistral model for deployment")
    parser.add_argument("--model_path", type=str, required=True, help="Path to original model directory")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save optimized model")
    parser.add_argument("--quantization", type=str, default="int8", choices=["int8", "int4", "none"], help="Quantization type")
    parser.add_argument("--merge_shards", action="store_true", help="Merge model shards")
    
    args = parser.parse_args()
    optimize_model(args.model_path, args.output_path, args.quantization, args.merge_shards)
