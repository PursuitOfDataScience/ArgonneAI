from transformers import PreTrainedTokenizerFast, PreTrainedModel, PretrainedConfig
from mp_pretrain import load_bpe_tokenizer, ArgonneConfig, ArgonneModelParallel
import torch

def main():
    # Load the BPE tokenizer
    loaded_tokenizer = load_bpe_tokenizer()

    # Load the model configuration and model (everything on CPU by default)
    loaded_config = ArgonneConfig.from_pretrained("Argonne_LLM")
    loaded_model = ArgonneModelParallel.from_pretrained("Argonne_LLM", config=loaded_config)
    loaded_model.eval()

    if torch.cuda.is_available():
        loaded_model.to("cuda")

    # Define the prompt and create input IDs on CPU first
    prompt = "Once upon a time, "
    input_ids = loaded_tokenizer.encode(prompt, return_tensors="pt")

    # --- NEW: If model is on GPU, move input_ids to GPU as well ---
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")

    # Generate text
    with torch.no_grad():
        output_ids = loaded_model.generate(
            input_ids,
            max_new_tokens=50,
            temperature=0.8,
            top_k=50
        )

    # Decode and print the generated text
    print(loaded_tokenizer.decode(output_ids[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
