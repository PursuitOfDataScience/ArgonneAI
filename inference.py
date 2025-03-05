import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Register the model architecture with AutoModel
from mp_pretrain import ArgonneConfig, ArgonneModelParallel
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

# Register the model with Hugging Face's Auto classes
AutoConfig.register("argonne", ArgonneConfig)
AutoModel.register(ArgonneConfig, ArgonneModelParallel)
AutoModelForCausalLM.register(ArgonneConfig, ArgonneModelParallel)

def main():
    # Load model and tokenizer using the Auto classes
    model_dir = "Argonne_LLM_Fixed"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    
    # Setup for inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Add the 'devices' attribute that model.generate() expects
    if not hasattr(model, 'devices'):
        model.devices = [device]
    
    # Set up pipeline stages to None if model was loaded without distribution
    if not hasattr(model, 'pipeline_stages') or model.pipeline_stages is None:
        model.pipeline_stages = None
    
    # Generate text from a prompt
    prompt = "The future of AI research is "
    
    # Extract just the input_ids from tokenizer output
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    # Generate text
    outputs = model.generate(
        input_ids,
        max_new_tokens=100,
        temperature=0.7,
        top_k=50
    )
    
    # Print the result
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated text:\n{generated_text}")

if __name__ == "__main__":
    main()