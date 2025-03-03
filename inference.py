"""
Clean and simple inference script for the Argonne LLM model.
Uses direct model loading to avoid metadata errors.
"""
import os
import torch
import json
from mp_pretrain import load_bpe_tokenizer, ArgonneConfig, ArgonneModelParallel

def load_model(model_dir):
    """
    Load a model directly from files without using from_pretrained
    to avoid metadata issues.
    """
    # Load config
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Create config and model
    config = ArgonneConfig(**config_dict)
    model = ArgonneModelParallel(config)
    
    # Try to load weights from PyTorch format
    model_bin_path = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.exists(model_bin_path):
        print(f"Loading weights from {model_bin_path}")
        state_dict = torch.load(model_bin_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    else:
        # Try SafeTensors format
        try:
            from safetensors.torch import load_file
            model_safetensors_path = os.path.join(model_dir, "model.safetensors")
            if os.path.exists(model_safetensors_path):
                print(f"Loading weights from {model_safetensors_path}")
                state_dict = load_file(model_safetensors_path)
                model.load_state_dict(state_dict, strict=False)
            else:
                raise FileNotFoundError(f"No model weights found in {model_dir}")
        except ImportError:
            raise ImportError("Neither PyTorch weights nor safetensors package available")
    
    print("Model weights loaded successfully")
    return model, config

def main():
    # The directory containing the converted model
    model_dir = "Argonne_LLM_Inference"  # Use this after running convert_model.py
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = load_bpe_tokenizer()
    
    # Load model using direct method instead of from_pretrained
    print("Loading model...")
    model, config = load_model(model_dir)

    # Print model parameters in total
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters in model: {total_params}")
    
    # Set up for inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.eval()
    model.to(device)
    
    # Add devices attribute needed by generate method
    if not hasattr(model, 'devices'):
        model.devices = [device]
    
    # Example prompts to test
    prompts = [
        "Once upon a time, ",
        "The meaning of life is ",
        "In the future, artificial intelligence will "
    ]
    
    # Generate text for each prompt
    for prompt in prompts:
        print(f"\n{'=' * 40}")
        print(f"PROMPT: '{prompt}'")
        print(f"{'=' * 40}")
        
        # Tokenize the prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # Generate text
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=150,
                temperature=0.7,
                top_k=50
            )
        
        # Decode the generated text
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print("\nGENERATED TEXT:")
        print(generated_text)

if __name__ == "__main__":
    main()
