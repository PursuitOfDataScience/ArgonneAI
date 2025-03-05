import os
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer
from transformers import (
    PreTrainedTokenizerFast,
    PretrainedConfig,
    PreTrainedModel
)
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
import glob

os.environ["HF_DATASETS_CACHE"] = "./.cache"

#####################################
# BPE Tokenizer Utilities
#####################################

def create_text_file_from_arrow(arrow_files, output_file="all_text_for_tokenizer.txt"):
    """
    Given a list of Arrow files, extract the 'text' column and write
    it to a single text file (one text example per line).
    """
    print(f"Creating a combined text file '{output_file}' from Arrow files...")
    with open(output_file, "w", encoding="utf-8") as wf:
        for arrow_path in tqdm(arrow_files):
            # Load the Arrow file in *streaming* mode to avoid large memory usage
            ds = load_dataset("arrow", data_files=[arrow_path], streaming=True)
            # If "train" split exists, use ds["train"], else ds is the dataset
            if "train" in ds:
                ds = ds["train"]
            for example in ds:
                text = example.get("text", "")
                # Write one line of text
                wf.write(text.replace("\n", " ") + "\n")

def train_bpe_tokenizer(text_file, vocab_size=12000):
    """
    Train a ByteLevel BPE tokenizer on a *plain-text file* and save it.
    """
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[text_file],
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[
            "<|start_of_text|>",
            "<pad>",
            "<|end_of_text|>",
            "<unk>",
            "<mask>"
        ]
    )

    os.makedirs("bpe_tokenizer", exist_ok=True)
    tokenizer.save_model("bpe_tokenizer")

    # Save the full tokenizer JSON representation
    with open(os.path.join("bpe_tokenizer", "tokenizer.json"), "w", encoding="utf-8") as f:
        f.write(tokenizer._tokenizer.to_str())

    # Create a tokenizer configuration
    tokenizer_config = {
        "model_max_length": 2048,
        "bos_token": "<|start_of_text|>",
        "eos_token": "<|end_of_text|>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "mask_token": "<mask>"
    }
    with open(os.path.join("bpe_tokenizer", "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f)

    # Create a Hugging Face PreTrainedTokenizerFast instance
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join("bpe_tokenizer", "tokenizer.json"),
        bos_token="<|start_of_text|>",
        eos_token="<|end_of_text|>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>"
    )
    hf_tokenizer.save_pretrained("bpe_tokenizer")
    return hf_tokenizer


def load_bpe_tokenizer():
    """Load a previously trained BPE tokenizer in Hugging Face format."""
    hf_tokenizer = PreTrainedTokenizerFast.from_pretrained("bpe_tokenizer", use_fast=True)
    return hf_tokenizer

#####################################
# STREAMING MODE
#####################################

def streaming_token_generator(data_files, hf_tokenizer):
    """
    Yields tokenized examples from a streaming dataset (no shuffle).
    data_files should be a list of Arrow files.
    """
    dataset = load_dataset("arrow", data_files=data_files, streaming=True)
    if "train" in dataset:
        dataset = dataset["train"]

    for example in dataset:
        text = example["text"] if "text" in example else ""
        token_ids = hf_tokenizer.encode(text)
        if len(token_ids) > 0:
            yield token_ids

#####################################
# NON-STREAMING: Full Pass
#####################################

def load_nonstream_data(data_files, hf_tokenizer, block_size, num_proc=8):
    """
    Loads the entire dataset in memory either from a cached processed directory
    or processes it in parallel if not yet cached.
    Returns a list of token ID sequences.
    """

    processed_dir = "processed_data/tokenized_data"
    if os.path.exists(processed_dir):
        print(f"Loading cached dataset from '{processed_dir}'...")
        ds = load_from_disk(processed_dir)
        tokenized_data = ds["token_ids"]
        return tokenized_data

    print("No cached dataset found. Processing in parallel...")

    ds_dict = load_dataset("arrow", data_files=data_files, streaming=False)
    if "train" in ds_dict:
        ds = ds_dict["train"]
    else:
        ds = ds_dict

    def tokenize_and_truncate(example):
        text = example["text"] if "text" in example else ""
        token_ids = hf_tokenizer.encode(text)
        if len(token_ids) < block_size + 1:
            return {"token_ids": None}
        token_ids = token_ids[:block_size+1]
        return {"token_ids": token_ids}

    ds = ds.map(
        tokenize_and_truncate,
        batched=False,
        num_proc=num_proc
    )
    ds = ds.filter(lambda ex: ex["token_ids"] is not None,
                   num_proc=num_proc)

    if "text" in ds.column_names:
        ds = ds.remove_columns(["text"])

    os.makedirs(os.path.dirname(processed_dir), exist_ok=True)
    ds.save_to_disk(processed_dir)
    print(f"Processed dataset saved to '{processed_dir}'.")

    tokenized_data = ds["token_ids"]
    return tokenized_data

def collate_batch(token_list_batch, block_size):
    """
    Convert a list of token-ID lists into x,y Tensors for causal LM.
    We'll truncate if longer than block_size+1, skip if shorter.
    """
    x_list, y_list = [], []
    for tokens in token_list_batch:
        if len(tokens) < block_size + 1:
            continue
        tokens = tokens[:block_size+1]
        x_list.append(tokens[:-1])
        y_list.append(tokens[1:])

    if not x_list:
        return None, None

    x_tensor = torch.tensor(x_list, dtype=torch.long)
    y_tensor = torch.tensor(y_list, dtype=torch.long)
    return x_tensor, y_tensor

#####################################
# Model Definition 
#####################################

class ArgonneConfig(PretrainedConfig):
    model_type = "argonne"
    def __init__(self, vocab_size=12000, block_size=2048, n_layer=24, n_head=24, n_embd=1296, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding dim must be divisible by n_head"
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
                 .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        b, t, c = x.size()
        q = self.query(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = self.key(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :t, :t] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = self.resid_drop(self.proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ArgonneModelParallel(PreTrainedModel):
    config_class = ArgonneConfig

    def __init__(self, config):
        super().__init__(config)
        # Create embeddings on CPU initially
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.dropout)

        # Build all blocks on CPU
        all_blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # Final LayerNorm + output head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        self.post_init()

        # Keep the blocks on CPU in a single ModuleList
        self.blocks = all_blocks
        # We'll defer pipeline splitting until later:
        self.pipeline_stages = None

    def distribute_model(self, device_ids=None):
        """
        Distribute the model blocks across multiple GPU devices in a pipeline style.
        If 'device_ids' is None, we'll discover all available GPUs.
        """
        if device_ids is None:
            num_gpus = torch.cuda.device_count()
            if num_gpus < 1:
                raise ValueError("No GPUs found—can't do pipeline parallel on CPU only.")
            device_ids = [f"cuda:{i}" for i in range(num_gpus)]
        
        # Store them so the training loop can keep referencing model.devices
        self.devices = [torch.device(d) for d in device_ids]

        self.pipeline_stages = nn.ModuleList()
        num_gpus = len(device_ids)
        blocks_per_gpu = math.ceil(len(self.blocks) / num_gpus)

        start_idx = 0
        for i in range(num_gpus):
            end_idx = min(start_idx + blocks_per_gpu, len(self.blocks))
            stage_blocks = self.blocks[start_idx:end_idx]
            stage = nn.Sequential(*stage_blocks).to(device_ids[i])
            self.pipeline_stages.append(stage)
            start_idx = end_idx
            if end_idx >= len(self.blocks):
                break

        # Move token_embedding + position_embedding to the first device
        self.token_embedding.to(device_ids[0])
        self.position_embedding.data = self.position_embedding.data.to(device_ids[0])
        self.drop.to(device_ids[0])

        # Move final LayerNorm + head to the last device
        self.ln_f.to(device_ids[-1])
        self.head.to(device_ids[-1])

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        If self.pipeline_stages is None, we do a normal single-device forward 
        (whatever device everything is currently on—CPU or a single GPU).
        Otherwise, we do a pipeline parallel forward.
        """
        if self.pipeline_stages is None:
            # Single-device forward pass
            device = self.token_embedding.weight.device
            idx = idx.to(device)
            b, t = idx.size()
            assert t <= self.config.block_size, "Sequence length exceeds block size"

            token_embeddings = self.token_embedding(idx)
            position_embeddings = self.position_embedding[:, :t, :]
            hidden_states = self.drop(token_embeddings + position_embeddings)

            for block in self.blocks:
                hidden_states = block(hidden_states)

            hidden_states = self.ln_f(hidden_states)
            logits = self.head(hidden_states)

            loss = None
            if targets is not None:
                targets = targets.to(device)
                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)
                loss = F.cross_entropy(logits, targets)

            return logits, loss
        else:
            # Pipeline parallel forward
            first_device = next(self.pipeline_stages[0].parameters()).device
            last_device = next(self.pipeline_stages[-1].parameters()).device

            x = idx.to(first_device)
            b, t = x.size()
            assert t <= self.config.block_size, "Sequence length exceeds block size"

            token_embeddings = self.token_embedding(x)
            position_embeddings = self.position_embedding[:, :t, :]
            hidden_states = self.drop(token_embeddings + position_embeddings)

            # Pass through each pipeline stage in sequence
            for stage in self.pipeline_stages:
                device_stage = next(stage.parameters()).device
                hidden_states = hidden_states.to(device_stage)
                hidden_states = stage(hidden_states)

            hidden_states = hidden_states.to(last_device)
            hidden_states = self.ln_f(hidden_states)
            logits = self.head(hidden_states)

            loss = None
            if targets is not None:
                targets = targets.to(last_device)
                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)
                loss = F.cross_entropy(logits, targets)

            return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=0.7, top_k=None):
        self.eval()
        if len(self.devices) == 0:
            raise ValueError("No GPUs available for model parallelism.")

        generated = input_ids.to(self.devices[0])
        for _ in range(max_new_tokens):
            if generated.shape[1] > self.config.block_size:
                generated = generated[:, -self.config.block_size:]

            logits, _ = self.forward(generated)
            logits = logits[:, -1, :].to(self.devices[-1])
            logits = logits / temperature

            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                logits[logits < values[:, -1:]] = float('-inf')

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_token = next_token.to(self.devices[0])
            generated = torch.cat((generated, next_token), dim=1)

        return generated

#####################################
# Training Loop (Streaming OR Full-Pass Non-Streaming)
#####################################

def train_model_parallel(data_files, use_streaming=False):
    """
    data_files should be a list of actual .arrow file paths, e.g.
    ["data/file1.arrow", "data/file2.arrow", ...]
    
    Includes automatic batch size adjustment when OOM errors occur.
    """
    # Initial batch size settings
    initial_batch_size = 128  # initial batch size
    min_batch_size = 12  # Minimum acceptable batch size
    batch_size = initial_batch_size  # Current working batch size
    
    # 1) If no tokenizer, train it on text extracted from Arrow
    if not os.path.exists("bpe_tokenizer/vocab.json"):
        print("No existing tokenizer found. Building a text file from Arrow and training one...")
        # Create a text file from Arrow files
        text_file_path = "all_text_for_tokenizer.txt"
        create_text_file_from_arrow(data_files, text_file_path)
        # Now train BPE on that text file
        train_bpe_tokenizer(text_file_path, vocab_size=12000)

    # Load the tokenizer we just created (or found)
    hf_tokenizer = load_bpe_tokenizer()

    block_size = 2048
    epochs = 5
    n_layer = 12
    n_head = 12
    n_embd = 1296
    dropout = 0.1

    config_model = ArgonneConfig(
        vocab_size=12000,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout
    )
    
    # Load non-streaming dataset once, outside the retry loop
    tokenized_data = None
    if not use_streaming:
        print("=== Loading dataset in memory for a full pass approach ===")
        tokenized_data = load_nonstream_data(data_files, hf_tokenizer, block_size, num_proc=128)
        total_samples = len(tokenized_data)
        print(f"Total tokenized samples: {total_samples}")
    
    # Main training loop with batch size adjustment
    while True:
        print(f"\n=== Attempting training with batch_size = {batch_size} ===")
        
        try:
            # Initialize a fresh model for each attempt
            model = ArgonneModelParallel(config_model)
            model.distribute_model()  # chunks across all visible GPUs 
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
            scaler = torch.amp.GradScaler("cuda")
            global_step = 0

            if use_streaming:
                ########################################################
                # STREAMING MODE
                ########################################################
                steps_per_epoch = 500
                
                for epoch in tqdm(range(epochs)):
                    print(f"==== Starting epoch {epoch} (STREAMING) with batch_size={batch_size} ====")
                    token_gen = streaming_token_generator(data_files, hf_tokenizer)
                    step_in_epoch = 0
                    token_batch = []

                    while step_in_epoch < steps_per_epoch:
                        try:
                            tokens = next(token_gen)
                            token_batch.append(tokens)

                            if len(token_batch) == batch_size:
                                x_tens, y_tens = collate_batch(token_batch, block_size)
                                token_batch.clear()
                                if x_tens is None:
                                    continue

                                first_device = model.devices[0]
                                x_tens, y_tens = x_tens.to(first_device), y_tens.to(first_device)

                                optimizer.zero_grad()
                                with torch.amp.autocast("cuda"):
                                    logits, loss = model(x_tens, y_tens)

                                scaler.scale(loss).backward()
                                scaler.step(optimizer)
                                scaler.update()

                                global_step += 1
                                step_in_epoch += 1
                                
                                if global_step % 50 == 0:
                                    print(f"Epoch {epoch} | Step {global_step} | Loss: {loss.item():.4f}")
                                    prompt_str = "Long long time ago, "
                                    token_ids = hf_tokenizer.encode(prompt_str)
                                    prompt_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
                                    generated = model.generate(prompt_tensor, max_new_tokens=50)
                                    generated_text = hf_tokenizer.decode(generated[0].tolist())
                                    print(f"\n--- Generated text at step {global_step} ---\n{generated_text}\n")

                                if global_step % 10000 == 0:
                                    checkpoint = {
                                        "epoch": epoch,
                                        "global_step": global_step,
                                        "batch_size": batch_size,  # Save the successful batch size
                                        "model_state_dict": model.state_dict(),
                                        "optimizer_state_dict": optimizer.state_dict(),
                                        "loss": loss.item()
                                    }
                                    os.makedirs("pretrained", exist_ok=True)
                                    torch.save(checkpoint, f"pretrained/checkpoint_step_{global_step}.pth")
                                    print(f"Checkpoint saved at step {global_step}")

                        except StopIteration:
                            print("Reached end of dataset (stream) before finishing this epoch.")
                            break

            else:
                ########################################################
                # NON-STREAMING MODE: full pass each epoch
                ########################################################
                batches_per_epoch = total_samples // batch_size

                for epoch in tqdm(range(epochs)):
                    print(f"==== Starting epoch {epoch} (NON-STREAMING) with batch_size={batch_size} ====")

                    for batch_idx in tqdm(range(batches_per_epoch)):
                        start_idx = batch_idx * batch_size
                        end_idx = start_idx + batch_size
                        batch_token_lists = tokenized_data[start_idx:end_idx]

                        x_tens, y_tens = collate_batch(batch_token_lists, block_size)
                        if x_tens is None:
                            continue

                        first_device = model.devices[0]
                        x_tens = x_tens.to(first_device)
                        y_tens = y_tens.to(first_device)

                        optimizer.zero_grad()
                        with torch.amp.autocast("cuda"):
                            logits, loss = model(x_tens, y_tens)

                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                        global_step += 1

                        if global_step % 100 == 0:
                            print(f"Epoch {epoch} | global_step {global_step} | Loss: {loss.item():.4f}")
                            prompt_str = "Long long time ago, "
                            token_ids = hf_tokenizer.encode(prompt_str)
                            prompt_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
                            generated = model.generate(prompt_tensor, max_new_tokens=50)
                            generated_text = hf_tokenizer.decode(generated[0].tolist())
                            print(f"\n--- Generated text at step {global_step} ---\n{generated_text}\n")

                        if global_step % 2000 == 0:
                            checkpoint = {
                                "epoch": epoch,
                                "global_step": global_step,
                                "batch_size": batch_size,  # Save the successful batch size
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "loss": loss.item()
                            }
                            os.makedirs("pretrained", exist_ok=True)
                            torch.save(checkpoint, f"pretrained/checkpoint_step_{global_step}.pth")
                            print(f"Checkpoint saved at step {global_step}")
            
            # If we reach here, training completed successfully
            print(f"Training completed successfully with batch_size={batch_size}")
            break
            
        except torch.cuda.OutOfMemoryError:
            # Free memory
            del model, optimizer, scaler
            torch.cuda.empty_cache()
            
            # Reduce batch size
            new_batch_size = max(batch_size - 12, min_batch_size)
            
            if new_batch_size == batch_size:
                print(f"⚠️ Already at minimum batch size ({min_batch_size}). Training failed.")
                break
                
            print(f"CUDA Out of Memory! Reducing batch size from {batch_size} to {new_batch_size}")
            batch_size = new_batch_size
            
            # Short pause to ensure memory is freed
            import time
            time.sleep(5)

    # Save final model and tokenizer
    try:
        model.save_pretrained("Argonne_LLM")
        hf_tokenizer.save_pretrained("Argonne_LLM")
        print("Model-parallel training complete; model and tokenizer saved successfully.")
    except:
        print("Failed to save final model, likely due to OOM issues.")

#####################################
# Register with Hugging Face Auto Classes
#####################################

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

# Register the model with Hugging Face's Auto classes
AutoConfig.register("argonne", ArgonneConfig)
AutoModel.register(ArgonneConfig, ArgonneModelParallel)
AutoModelForCausalLM.register(ArgonneConfig, ArgonneModelParallel)


def main():
    # Expand .arrow files via glob
    data_files = glob.glob("data/*.arrow")
    if not data_files:
        raise ValueError("No files matched the pattern 'data/*.arrow'")

    train_model_parallel(data_files=data_files, use_streaming=False)

if __name__ == "__main__":
    main()
