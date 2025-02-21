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
from datasets import load_dataset 

#####################################
# BPE Tokenizer Utilities
#####################################

def train_bpe_tokenizer(file_path, vocab_size=12000):
    """Train a ByteLevel BPE tokenizer on the text and save it in Hugging Face format."""
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train([file_path], vocab_size=vocab_size, min_frequency=2,
                    special_tokens=["<|start_of_text|>", 
                                    "<pad>", 
                                    "<|end_of_text|>",
                                    "<unk>", 
                                    "<mask>"])
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


# We'll define a generator that iterates over a streaming dataset (Arrow files)
# and yields token lists. 
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
        # If your Arrow has a column "text", we do:
        text = example["text"] if "text" in example else ""
        # Tokenize
        token_ids = hf_tokenizer.encode(text)
        if len(token_ids) > 0:
            yield token_ids

def collate_batch(token_list_batch, block_size):
    """
    Given a list of tokenized sequences, each a Python list of ints,
    produce a single batch of shape (batch_size, block_size).
    We'll truncate or skip if too short.
    """
    x_list, y_list = [], []
    for tokens in token_list_batch:
        # Ensure at least block_size+1 tokens to form x,y
        if len(tokens) < block_size + 1:
            continue
        tokens = tokens[:block_size+1]  # truncate if longer
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

        # Final LayerNorm + output head (stay on CPU for now)
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Init position embedding
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        self.post_init()

        # Split blocks evenly across GPUs
        num_gpus = len(self.devices)
        blocks_per_gpu = math.ceil(config.n_layer / num_gpus)

        # We'll store these pipeline stages in a ModuleList
        self.pipeline_stages = nn.ModuleList()

        # Assign slices of blocks to each GPU
        start_idx = 0
        for i in range(num_gpus):
            end_idx = min(start_idx + blocks_per_gpu, config.n_layer)
            stage_blocks = all_blocks[start_idx:end_idx]
            stage = nn.Sequential(*stage_blocks).to(self.devices[i])
            self.pipeline_stages.append(stage)
            start_idx = end_idx
            if end_idx >= config.n_layer:
                break  # we assigned all blocks

        # Move embeddings to the first GPU
        self.token_embedding.to(self.devices[0])
        self.position_embedding = self.position_embedding.to(self.devices[0])
        self.drop.to(self.devices[0])

        # Move final LN + head to the last GPU
        self.ln_f.to(self.devices[-1])
        self.head.to(self.devices[-1])

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Pipeline forward pass:
          1) Move idx to GPU[0], compute input embeddings.
          2) Pass through pipeline stages 0..N-1. Transfer hidden states each time.
          3) On the final GPU, apply ln_f, head, compute loss if targets provided.
        """
        # Start on device[0]
        x = idx.to(self.devices[0])
        b, t = x.size()
        assert t <= self.config.block_size, "Sequence length exceeds block size"

        # Embeddings on GPU0
        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding[:, :t, :]
        hidden_states = self.drop(token_embeddings + position_embeddings)

        # Forward pass through each stage
        for stage_idx, stage in enumerate(self.pipeline_stages):
            # stage is already on self.devices[stage_idx]
            hidden_states = hidden_states.to(self.devices[stage_idx])
            hidden_states = stage(hidden_states)  # forward on this stage

        # Now hidden_states is on the last GPU
        hidden_states = hidden_states.to(self.devices[-1])
        hidden_states = self.ln_f(hidden_states)
        logits = self.head(hidden_states)

        loss = None
        if targets is not None:
            # Move targets to the final GPU as well
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
            # Crop context if longer than block_size
            if generated.shape[1] > self.config.block_size:
                generated = generated[:, -self.config.block_size:]

            logits, _ = self.forward(generated)
            # logits on last device
            logits = logits[:, -1, :].to(self.devices[-1])  # shape [batch, vocab_size]
            logits = logits / temperature

            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                logits[logits < values[:, -1:]] = float('-inf')

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            # Move next_token back to device[0] to append
            next_token = next_token.to(self.devices[0])
            generated = torch.cat((generated, next_token), dim=1)

        return generated

#####################################
# Training Loop (Streaming Version)
#####################################

def train_model_parallel(data_path="data/*.arrow"):

    # If the tokenizer doesn't exist, train it (but typically for streaming you'd pre-make it)
    if not os.path.exists("bpe_tokenizer/vocab.json"):
        print("Training BPE tokenizer...")
        # We pass a single file or pattern here. Possibly just the first arrow file or a text file.
        train_bpe_tokenizer(data_path, vocab_size=12000)
    hf_tokenizer = load_bpe_tokenizer()

    block_size = 2048
    epochs = 5
    n_layer = 24
    n_head = 24
    n_embd = 1296
    dropout = 0.1
    batch_size = 24

    # Build model config
    config_model = ArgonneConfig(
        vocab_size=12000,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout
    )
    # Build pipeline-parallel GPT model
    model = ArgonneModelParallel(config_model)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    scaler = torch.amp.GradScaler("cuda")

    steps_per_epoch = 1000  

    global_step = 0

    for epoch in tqdm(range(epochs)):
        print(f"==== Starting epoch {epoch} ====")

        # Re-initialize the generator for each epoch if you want multiple passes
        token_gen = streaming_token_generator(data_path, hf_tokenizer)  # returns token lists
        step_in_epoch = 0

        # We'll accumulate tokenized sequences to form a batch:
        token_batch = []

        while step_in_epoch < steps_per_epoch:
            try:
                tokens = next(token_gen)  # get next tokenized example
                token_batch.append(tokens)

                # Once we have 'batch_size' examples, do a training step
                if len(token_batch) == batch_size:
                    x_tens, y_tens = collate_batch(token_batch, block_size)
                    token_batch.clear()  # reset for next batch

                    # If x_tens is None, it means all examples were too short
                    if x_tens is None:
                        continue

                    # We'll always sample the batch on the first device
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
                    step_in_epoch += 1

                    if global_step % 100 == 0:
                        print(f"Epoch {epoch} | Step {global_step} | Loss: {loss.item():.4f}")

                        # Quick test generation
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
                # End of streaming dataset. If you want a single pass, break. 
                print("Reached end of dataset before finishing this epoch.")
                break

    # Save final model and tokenizer
    model.save_pretrained("Argonne_LLM")
    hf_tokenizer.save_pretrained("Argonne_LLM")
    print("Model-parallel streaming training complete; model and tokenizer saved successfully.")

def main():
    train_model_parallel()

if __name__ == "__main__":
    main()
