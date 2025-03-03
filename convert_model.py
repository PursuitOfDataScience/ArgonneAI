#!/usr/bin/env python3
"""
Utility to convert a pipeline-parallel model to a standard single-device model
and save it to a new directory for easy inference.
"""
import os
import torch
import re
import shutil
import argparse
from pathlib import Path

try:
    from safetensors.torch import save_file as safe_save
    from safetensors.torch import load_file as safe_load
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("safetensors not available. Install with: pip install safetensors")

# Import directly from the current directory
from mp_pretrain import load_bpe_tokenizer, ArgonneConfig, ArgonneModelParallel

def fix_pipeline_state_dict(state_dict):
    """
    Converts pipeline-parallel state_dict keys to match single-GPU model structure.
    """
    new_state_dict = {}
    
    # Pattern for pipeline stage keys
    pipeline_pattern = r'pipeline_stages\.(\d+)\.(\d+)\.(.*)'
    
    # First pass to detect structure
    processed_blocks = {}
    for key in state_dict.keys():
        if not key.startswith('pipeline_stages.'):
            continue
            
        match = re.match(pipeline_pattern, key)
        if match:
            gpu_idx, block_in_gpu_idx = int(match.group(1)), int(match.group(2))
            processed_blocks.setdefault(gpu_idx, set()).add(block_in_gpu_idx)
    
    # Determine blocks_per_gpu (maximum block index + 1)
    blocks_per_gpu = 1
    if processed_blocks:
        blocks_per_gpu = max(max(indices) for indices in processed_blocks.values()) + 1
        print(f"Detected {blocks_per_gpu} blocks per GPU in the model")
    
    # Second pass to convert keys
    for key, value in state_dict.items():
        if key.startswith('pipeline_stages.'):
            match = re.match(pipeline_pattern, key)
            if match:
                gpu_idx, block_in_gpu_idx, rest = int(match.group(1)), int(match.group(2)), match.group(3)
                # Calculate global block index
                global_block_idx = gpu_idx * blocks_per_gpu + block_in_gpu_idx
                new_key = f'blocks.{global_block_idx}.{rest}'
                new_state_dict[new_key] = value
        else:
            # Copy other weights unchanged
            new_state_dict[key] = value
    
    return new_state_dict

def convert_and_save_model(source_dir="Argonne_LLM", target_dir="Argonne_LLM_Inference"):
    """
    Convert a pipeline-parallel model to a standard single-device model
    and save it to a new directory for easy inference.
    """
    print(f"Converting model from {source_dir} to {target_dir}...")
    
    # Create target directory
    os.makedirs(target_dir, exist_ok=True)
    
    # Copy configuration files
    config_files = ["config.json", "generation_config.json", "tokenizer_config.json", 
                  "special_tokens_map.json", "tokenizer.json"]
    
    for file in config_files:
        source_file = os.path.join(source_dir, file)
        target_file = os.path.join(target_dir, file)
        if os.path.exists(source_file):
            print(f"Copying {file}...")
            shutil.copy2(source_file, target_file)
    
    # Load config
    config_path = os.path.join(source_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    import json
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Create config and model
    config = ArgonneConfig.from_dict(config_dict)
    model = ArgonneModelParallel(config)
    
    # Load model weights
    model_safetensors_path = os.path.join(source_dir, "model.safetensors")
    model_bin_path = os.path.join(source_dir, "pytorch_model.bin")
    
    if os.path.exists(model_safetensors_path) and SAFETENSORS_AVAILABLE:
        print(f"Loading weights from {model_safetensors_path}")
        state_dict = safe_load(model_safetensors_path)
    elif os.path.exists(model_bin_path):
        print(f"Loading weights from {model_bin_path}")
        state_dict = torch.load(model_bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No model weights found in {source_dir}")
    
    # Check if weights use pipeline structure
    has_pipeline = any(k.startswith('pipeline_stages.') for k in state_dict.keys())
    if has_pipeline:
        print("Converting pipeline structure to blocks structure...")
        state_dict = fix_pipeline_state_dict(state_dict)
    else:
        print("Model already uses blocks structure, no conversion needed.")
    
    # Load state dict into model
    model.load_state_dict(state_dict, strict=False)
    
    # Save model in both formats
    # 1. PyTorch format
    torch_path = os.path.join(target_dir, "pytorch_model.bin")
    print(f"Saving in PyTorch format: {torch_path}")
    torch.save(model.state_dict(), torch_path)
    
    # 2. SafeTensors format (if available)
    if SAFETENSORS_AVAILABLE:
        safetensors_path = os.path.join(target_dir, "model.safetensors")
        print(f"Saving in SafeTensors format: {safetensors_path}")
        safe_save(model.state_dict(), safetensors_path)
    
    # Also save the custom tokenizer
    print("Saving tokenizer...")
    tokenizer = load_bpe_tokenizer()
    tokenizer.save_pretrained(target_dir)
    
    print(f"Model successfully converted and saved to {target_dir}")
    print("Now you can use the simple inference script with this directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a pipeline-parallel model to standard format")
    parser.add_argument("--source", type=str, default="Argonne_LLM",
                       help="Source directory containing the pipeline-parallel model")
    parser.add_argument("--target", type=str, default="Argonne_LLM_Inference",
                       help="Target directory for the converted model")
    
    args = parser.parse_args()
    convert_and_save_model(args.source, args.target)
