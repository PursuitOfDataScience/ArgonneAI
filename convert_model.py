'''
    This script converts a model to be compatible with Hugging Face's transformers library.
    It handles the following steps:
    1. Copies or creates the `config.json` file.
    2. Copies the model weights from the original directory.
    3. Creates an `index.json` file with metadata.
    4. Copies tokenizer files from the original directory.
    5. Prints a message indicating the model is ready for use.
'''

import os
import torch
import json
import shutil
from mp_pretrain import ArgonneConfig, ArgonneModelParallel
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

# Register the model with Hugging Face's Auto classes
AutoConfig.register("argonne", ArgonneConfig)
AutoModel.register(ArgonneConfig, ArgonneModelParallel)
AutoModelForCausalLM.register(ArgonneConfig, ArgonneModelParallel)

# Path to your existing model directory
input_dir = "Argonne_LLM"  
output_dir = "Argonne_LLM_Fixed"  

# Create output directory
os.makedirs(output_dir, exist_ok=True)

print(f"Converting model in {input_dir} and saving to {output_dir}")

# Step 1: Copy config.json if it exists, or create it
config_path = os.path.join(input_dir, "config.json")
if os.path.exists(config_path):
    print(f"Using existing config from {config_path}")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    config = ArgonneConfig(**config_dict)
else:
    print("Creating default config")
    # Create a default config - adjust parameters to match your model
    config = ArgonneConfig(
        vocab_size=12000,
        block_size=2048,
        n_layer=12,  # Adjust these to match your model
        n_head=12,   # Adjust these to match your model
        n_embd=1296  # Adjust these to match your model
    )

# Save config to new location
config.save_pretrained(output_dir)
print(f"Config saved to {output_dir}/config.json")

# Step 2: Copy model weights
model_bin_path = os.path.join(input_dir, "pytorch_model.bin")
if os.path.exists(model_bin_path):
    print(f"Copying model weights from {model_bin_path}")
    shutil.copy(model_bin_path, os.path.join(output_dir, "pytorch_model.bin"))
else:
    print(f"⚠️ Model weights not found at {model_bin_path}")
    # Try loading from checkpoint if available
    checkpoint_dir = "pretrained"
    if os.path.exists(checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint") and f.endswith(".pth")]
        if checkpoint_files:
            latest_checkpoint = sorted(checkpoint_files)[-1]
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
            print(f"Loading from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model = ArgonneModelParallel(config)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
            print(f"Model weights saved to {output_dir}/pytorch_model.bin")
        else:
            print(f"⚠️ No checkpoint files found in {checkpoint_dir}")
    else:
        print(f"⚠️ Checkpoint directory {checkpoint_dir} not found")

# Step 3: Create the critical index.json file with metadata
model_bin = os.path.join(output_dir, "pytorch_model.bin")
if os.path.exists(model_bin):
    print(f"Creating index file for {model_bin}")
    # Load state dict to get keys
    state_dict = torch.load(model_bin, map_location="cpu")
    
    # Create the index file required by transformers
    index_dict = {
        "metadata": {
            "format": "pt",  # This is what's missing in your case
            "total_size": sum(t.numel() * t.element_size() for t in state_dict.values())
        },
        "weight_map": {param_name: "pytorch_model.bin" for param_name in state_dict.keys()}
    }
    
    # Save the index file
    index_path = os.path.join(output_dir, "pytorch_model.bin.index.json")
    with open(index_path, 'w') as f:
        json.dump(index_dict, f, indent=2)
    
    print(f"Created index file at {index_path}")
else:
    print(f"⚠️ Cannot create index file: {model_bin} doesn't exist")

# Step 4: Copy tokenizer files
tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"]
tokenizer_dirs = [input_dir, "bpe_tokenizer"]

found_tokenizer_files = False
for tokenizer_dir in tokenizer_dirs:
    if not os.path.exists(tokenizer_dir):
        print(f"Tokenizer directory {tokenizer_dir} not found, trying next...")
        continue
    
    for file in tokenizer_files:
        src_path = os.path.join(tokenizer_dir, file)
        if os.path.exists(src_path):
            dst_path = os.path.join(output_dir, file)
            print(f"Copying tokenizer file: {src_path} -> {dst_path}")
            shutil.copy(src_path, dst_path)
            found_tokenizer_files = True
    
    if found_tokenizer_files:
        break

if not found_tokenizer_files:
    print("No tokenizer files found. The model may work but you'll need to specify a tokenizer separately.")

print("\nModel conversion complete!")
print(f"You can now load your model with: AutoModelForCausalLM.from_pretrained('{output_dir}')")