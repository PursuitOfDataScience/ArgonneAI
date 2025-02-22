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

#####################################
# BPE Tokenizer Utilities
#####################################

def train_bpe_tokenizer(file_path, vocab_size=12000):
    """Train a ByteLevel BPE tokenizer on the text and save it in Hugging Face format."""
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train([file_path], vocab_size=vocab_size, min_frequency=2,
                    special_tokens=[
                        "<|start_of_text|>", 
                        "<pad>", 
                        "<|end_of_text|>",
                        "<unk>", 
                        "<mask>"
                    ])
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
    """
    dataset = load_dataset(
        "arrow",
        data_files=data_files,
        streaming=True    
    )
    for example in dataset:
        text = example["text"] if "text" in example else ""
        token_ids = hf_tokenizer.encode(text)
        if len(token_ids) > 0:
            yield token_ids

#####################################
# NON-STREAMING: Full Pass
#####################################

from datasets import load_dataset, load_from_disk
import os

def load_nonstream_data(data_files, hf_tokenizer, block_size, num_proc=8):
    """
    Loads the entire dataset in memory either from a cached processed directory
    or processes it in parallel if not yet cached.
    Returns a list of token ID sequences.
    """

    # 1) Check if cached data exists
    processed_dir = "processed_data/tokenized_data"
    if os.path.exists(processed_dir):
        print(f"Loading cached dataset from '{processed_dir}'...")
        ds = load_from_disk(processed_dir)
        tokenized_data = ds["token_ids"]
        return tokenized_data

    # 2) Otherwise, read + process from scratch
    print("No cached dataset found. Processing in parallel...")

    # Load the raw dataset in memory
    ds_dict = load_dataset("arrow", data_files=data_files, streaming=False)
    if "train" in ds_dict:
        ds = ds_dict["train"]
    else:
        ds = ds_dict

    # Define the parallel tokenization function
    def tokenize_and_truncate(example):
        text = example["text"] if "text" in example else ""
        token_ids = hf_tokenizer.encode(text)
        # skip if < block_size+1
        if len(token_ids) < block_size + 1:
            return {"token_ids": None}
        # truncate if needed
        token_ids = token_ids[:block_size+1]
        return {"token_ids": token_ids}

    # Parallel map
    ds = ds.map(
        tokenize_and_truncate,
        batched=False,
        num_proc=num_proc
    )

    # Filter out rows where token_ids=None
    ds = ds.filter(lambda ex: ex["token_ids"] is not None)

    # (Optionally remove original text column if you want to save space)
    if "text" in ds.column_names:
        ds = ds.remove_columns(["text"])

    # 3) Save the processed dataset for next time
    os.makedirs(os.path.dirname(processed_dir), exist_ok=True)
    ds.save_to_disk(processed_dir)
    print(f"Processed dataset saved to '{processed_dir}'.")

    # 4) Convert to Python list of lists
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

        # Detect all available CUDA devices
        self.devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
        if len(self.devices) == 0:
            raise ValueError("No GPUs available for model parallelism. (torch.cuda.device_count() == 0)")

        # Create embeddings on CPU initially
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd, device=self.devices[0]))
        self.drop = nn.Dropout(config.dropout)

        # Build all blocks on CPU
        all_blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # Final LayerNorm + output head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        self.post_init()

        num_gpus = len(self.devices)
        blocks_per_gpu = math.ceil(config.n_layer / num_gpus)
        self.pipeline_stages = nn.ModuleList()

        start_idx = 0
        for i in range(num_gpus):
            end_idx = min(start_idx + blocks_per_gpu, config.n_layer)
            stage_blocks = all_blocks[start_idx:end_idx]
            stage = nn.Sequential(*stage_blocks).to(self.devices[i])
            self.pipeline_stages.append(stage)
            start_idx = end_idx
            if end_idx >= config.n_layer:
                break

        # embeddings on first GPU
        self.token_embedding.to(self.devices[0])
        self.position_embedding = self.position_embedding.to(self.devices[0])
        self.drop.to(self.devices[0])

        # final LN + head on last GPU
        self.ln_f.to(self.devices[-1])
        self.head.to(self.devices[-1])

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        x = idx.to(self.devices[0])
        b, t = x.size()
        assert t <= self.config.block_size, "Sequence length exceeds block size"

        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding[:, :t, :]
        hidden_states = self.drop(token_embeddings + position_embeddings)

        for stage_idx, stage in enumerate(self.pipeline_stages):
            hidden_states = hidden_states.to(self.devices[stage_idx])
            hidden_states = stage(hidden_states)

        hidden_states = hidden_states.to(self.devices[-1])
        hidden_states = self.ln_f(hidden_states)
        logits = self.head(hidden_states)

        loss = None
        if targets is not None:
            targets = targets.to(self.devices[-1])
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

def train_model_parallel(data_path, use_streaming=False):
    # 1) If no tokenizer, train it
    if not os.path.exists("bpe_tokenizer/vocab.json"):
        print("Training BPE tokenizer...")
        train_bpe_tokenizer(data_path, vocab_size=12000)
    hf_tokenizer = load_bpe_tokenizer()

    block_size = 2048
    epochs = 5
    n_layer = 24
    n_head = 24
    n_embd = 1296
    dropout = 0.1
    batch_size = 8

    config_model = ArgonneConfig(
        vocab_size=12000,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout
    )
    model = ArgonneModelParallel(config_model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    scaler = torch.amp.GradScaler("cuda")

    if use_streaming:
        ########################################################
        # STREAMING MODE
        ########################################################
        steps_per_epoch = 500  
        global_step = 0

        for epoch in tqdm(range(epochs)):
            print(f"==== Starting epoch {epoch} (STREAMING) ====")
            token_gen = streaming_token_generator(data_path, hf_tokenizer)  
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
                        print(f'current global step: {global_step}, current step in epoch: {step_in_epoch}')
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
        print("=== Loading dataset in memory for a full pass approach ===")
        tokenized_data = load_nonstream_data(data_path, hf_tokenizer, block_size, num_proc=8)
        total_samples = len(tokenized_data)
        print(f"Total tokenized samples: {total_samples}")

        batches_per_epoch = total_samples // batch_size
        global_step = 0

        for epoch in tqdm(range(epochs)):
            print(f"==== Starting epoch {epoch} (NON-STREAMING) ====")

            for batch_idx in tqdm(range(batches_per_epoch)):
                # slice out the portion of data for this batch
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

                # Logging
                if global_step % 50 == 0:
                    print(f"Epoch {epoch} | global_step {global_step} | Loss: {loss.item():.4f}")

                    # Quick generation
                    prompt_str = "Long long time ago, "
                    token_ids = hf_tokenizer.encode(prompt_str)
                    prompt_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
                    generated = model.generate(prompt_tensor, max_new_tokens=50)
                    generated_text = hf_tokenizer.decode(generated[0].tolist())
                    print(f"\n--- Generated text at step {global_step} ---\n{generated_text}\n")

                # Checkpointing
                if global_step % 5000 == 0:
                    checkpoint = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss.item()
                    }
                    os.makedirs("pretrained", exist_ok=True)
                    torch.save(checkpoint, f"pretrained/checkpoint_step_{global_step}.pth")
                    print(f"Checkpoint saved at step {global_step}")

    # Save final model and tokenizer
    model.save_pretrained("Argonne_LLM")
    hf_tokenizer.save_pretrained("Argonne_LLM")
    print("Model-parallel training complete; model and tokenizer saved successfully.")

def main():
    train_model_parallel(data_path="data/fineweb-edu-train-00144-of-00218.arrow",
                         use_streaming=False)

if __name__ == "__main__":
    main()
