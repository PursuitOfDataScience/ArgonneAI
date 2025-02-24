'''
    This script trains the LLM acorss multiple nodes using DDP + oneCCL. 
    Within each of the nodes, the model is split across multiple XPUs using pipeline parallelism. The script can be run with the following command:
    mpiexec -n 32 -ppn 1 python ddp_mp_pretrain.py
'''

import os
import math
import json
import socket
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import intel_extension_for_pytorch as ipex
import oneccl_bindings_for_pytorch as torch_ccl

from mpi4py import MPI

from tokenizers import ByteLevelBPETokenizer
from transformers import (
    PreTrainedTokenizerFast,
    PretrainedConfig,
    PreTrainedModel
)
from tqdm import tqdm
from datasets import load_dataset

#####################################
# DDP Setup with MPI + oneCCL
#####################################

# MPI info
SIZE = MPI.COMM_WORLD.Get_size()
RANK = MPI.COMM_WORLD.Get_rank()
LOCAL_RANK = os.environ.get('PALS_LOCAL_RANKID', 0)

os.environ['RANK'] = str(RANK)
os.environ['WORLD_SIZE'] = str(SIZE)

# For ccl backend, we need a MASTER_ADDR + MASTER_PORT
MASTER_ADDR = socket.gethostname() if RANK == 0 else None
MASTER_ADDR = MPI.COMM_WORLD.bcast(MASTER_ADDR, root=0)
os.environ['MASTER_ADDR'] = f"{MASTER_ADDR}.hsn.cm.aurora.alcf.anl.gov"
os.environ['MASTER_PORT'] = str(2345)

print(f"DDP: Hi from rank={RANK}/{SIZE} local_rank={LOCAL_RANK}  on host={MASTER_ADDR}")

# Initialize process group using oneCCL
dist.init_process_group(
    backend='ccl',
    init_method='env://',
    rank=int(RANK),
    world_size=int(SIZE)
)

# Pin the XPU device to local rank
torch.xpu.set_device(int(LOCAL_RANK))
device = torch.device(f"xpu:{LOCAL_RANK}")

#####################################
# Create Text File from Arrow
#####################################

def create_text_file_from_arrow(arrow_files, output_file="all_text_for_tokenizer.txt"):
    """
    Load each Arrow file in streaming mode, extract the 'text' field,
    and write it (one example per line) to a plain-text file.
    """
    print(f"[Rank {RANK}] Creating text file '{output_file}' from Arrow files: {arrow_files}")
    with open(output_file, "w", encoding="utf-8") as wf:
        for arrow_file in arrow_files:
            ds = load_dataset("arrow", data_files=[arrow_file], streaming=True)
            # If there's a "train" split, use ds["train"], else ds is the dataset
            if "train" in ds:
                ds = ds["train"]
            for example in ds:
                text = example["text"] if "text" in example else ""
                # Replace newlines with spaces to keep one line per example
                text = text.replace("\n", " ")
                wf.write(text + "\n")


#####################################
# BPE Tokenizer Utilities
#####################################

def train_bpe_tokenizer(text_file_path, vocab_size=12000):
    """
    Train a ByteLevel BPE tokenizer on a *plain text* file in UTF-8.
    """
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        [text_file_path],
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

    with open(os.path.join("bpe_tokenizer", "tokenizer.json"), "w", encoding="utf-8") as f:
        f.write(tokenizer._tokenizer.to_str())

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
    """
    Load a previously trained BPE tokenizer in Hugging Face format.
    """
    return PreTrainedTokenizerFast.from_pretrained("bpe_tokenizer", use_fast=True)

#####################################
# Some Data Utils
#####################################

def streaming_token_generator(data_files, hf_tokenizer):
    dataset = load_dataset("arrow", data_files=data_files, streaming=True)
    # If dataset has 'train' split, use dataset["train"]
    # Otherwise, dataset is the entire data
    if "train" in dataset:
        dataset = dataset["train"]

    for example in dataset:
        text = example["text"] if "text" in example else ""
        token_ids = hf_tokenizer.encode(text)
        if len(token_ids) > 0:
            yield token_ids

def collate_batch(token_list_batch, block_size):
    x_list, y_list = []
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
# Model Config & Blocks
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
        assert config.n_embd % config.n_head == 0
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

#####################################
# Pipeline Parallel Model
# (We'll wrap in DDP for multi-node data parallel)
#####################################

class ArgonneModelParallel(PreTrainedModel):
    config_class = ArgonneConfig

    def __init__(self, config):
        super().__init__(config)

        # We assume each process is pinned to a single XPU device.
        xpu_count = torch.xpu.device_count()
        if xpu_count == 0:
            raise ValueError("No XPUs available?")

        self.devices = [torch.device(f"xpu:{i}") for i in range(xpu_count)]

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.dropout)

        all_blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        self.post_init()

        # If you truly want pipeline parallel across multiple XPUs *in one process*,
        # the code below splits the blocks among XPUs in self.devices.
        blocks_per_xpu = math.ceil(config.n_layer / xpu_count)
        self.pipeline_stages = nn.ModuleList()

        start_idx = 0
        for i in range(xpu_count):
            end_idx = min(start_idx + blocks_per_xpu, config.n_layer)
            stage_blocks = all_blocks[start_idx:end_idx]
            stage = nn.Sequential(*stage_blocks).to(self.devices[i])
            self.pipeline_stages.append(stage)
            start_idx = end_idx
            if end_idx >= config.n_layer:
                break

        # Move embeddings + final layers
        self.token_embedding.to(self.devices[0])
        self.position_embedding = nn.Parameter(
            self.position_embedding.to(self.devices[0])
        )
        self.drop.to(self.devices[0])

        self.ln_f.to(self.devices[-1])
        self.head.to(self.devices[-1])

    def forward(self, idx, targets=None):
        x = idx.to(self.devices[0])
        b, t = x.size()

        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding[:, :t, :]
        hidden_states = self.drop(token_emb + pos_emb)

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
    def generate(self, input_ids, max_new_tokens=50, temperature=0.7, top_k=None):
        self.eval()
        generated = input_ids.to(self.devices[0])
        for _ in range(max_new_tokens):
            if generated.shape[1] > self.config.block_size:
                generated = generated[:, -self.config.block_size:]

            logits, _ = self.forward(generated)
            logits = logits[:, -1, :].to(self.devices[-1])
            logits /= temperature

            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                logits[logits < values[:, -1:]] = float('-inf')

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).to(self.devices[0])
            generated = torch.cat((generated, next_token), dim=1)
        return generated

#####################################
# Training Loop with DDP
#####################################

from torch.nn.parallel import DistributedDataParallel as DDP

def train_model_ddp(data_path="data/*.arrow", use_streaming=False, epochs=3):
    """
    Main training function using DistributedDataParallel on XPUs + pipeline parallel.
    data_path can be a wildcard or list of Arrow files.
    """

    # 1) Build / load tokenizer
    #    => We cannot pass arrow files directly to ByteLevelBPETokenizer (it's binary).
    #    => Instead, we create a text file from arrow if needed.
    if not os.path.exists("bpe_tokenizer/vocab.json"):
        print("[Rank 0] No tokenizer found, creating text file from Arrow and training a new one...")
        if RANK == 0:
            # Only rank 0 performs the text-file generation + training
            # Then we barrier so other ranks wait until it's done
            import glob
            arrow_files = glob.glob(data_path)  # expand wildcard
            if len(arrow_files) == 0:
                raise ValueError(f"No files match the pattern {data_path}")

            create_text_file_from_arrow(arrow_files, "all_text_for_tokenizer.txt")
            train_bpe_tokenizer("all_text_for_tokenizer.txt", vocab_size=12000)

        dist.barrier()  # all ranks wait here

    # Now, each rank can load the tokenizer from disk
    hf_tokenizer = load_bpe_tokenizer()

    # 2) Prepare model
    block_size = 2048
    batch_size = 24

    config_model = ArgonneConfig(
        vocab_size=12000,
        block_size=block_size,
        n_layer=24,
        n_head=24,
        n_embd=1296,
        dropout=0.1
    )
    base_model = ArgonneModelParallel(config_model).to(device)

    lr = 3e-5
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=lr)

    # IPEX optimization
    base_model, optimizer = ipex.optimize(base_model, optimizer=optimizer)

    # Wrap in DDP
    model = DDP(base_model, device_ids=None)

    scaler = torch.amp.GradScaler()

    # 3) Streaming vs. Non-streaming
    if use_streaming:
        steps_per_epoch = 1000
        global_step = 0

        for epoch in range(epochs):
            print(f"[Rank {RANK}] Starting epoch {epoch}, streaming mode.")
            # Expand the Arrow pattern for streaming
            import glob
            arrow_files = glob.glob(data_path)
            token_gen = streaming_token_generator(arrow_files, hf_tokenizer)

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

                        # Move data to local XPU
                        x_tens = x_tens.to(device)
                        y_tens = y_tens.to(device)

                        optimizer.zero_grad()
                        with torch.amp.autocast(device_type="xpu"):
                            logits, loss = model(x_tens, y_tens)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                        global_step += 1
                        step_in_epoch += 1

                        if global_step % 100 == 0 and RANK == 0:
                            print(f"[Rank 0] Epoch={epoch}, step={global_step}, loss={loss.item():.4f}")

                except StopIteration:
                    print(f"[Rank {RANK}] Reached end of dataset stream early.")
                    break

    else:
        print(f"[Rank {RANK}] Non-streaming mode with DDP + full pass approach.")

        # 1) Expand arrow files
        import glob
        arrow_files = glob.glob(data_path)
        if len(arrow_files) == 0:
            raise ValueError(f"No files match the pattern {data_path}")

        # 2) Convert dataset to a Torch Dataset
        #    We'll do a minimal approach: load them all into a single HuggingFace dataset
        from torch.utils.data import Dataset, DataLoader
        from torch.utils.data.distributed import DistributedSampler

        hf_ds = load_dataset("arrow", data_files=arrow_files, streaming=False)
        if "train" in hf_ds:
            hf_ds = hf_ds["train"]

        class ArrowWrapDataset(Dataset):
            def __init__(self, ds):
                self.ds = ds
            def __len__(self):
                return len(self.ds)
            def __getitem__(self, idx):
                example = self.ds[idx]
                text = example["text"] if "text" in example else ""
                token_ids = hf_tokenizer.encode(text)
                return token_ids

        dataset = ArrowWrapDataset(hf_ds)
        sampler = DistributedSampler(dataset=dataset, num_replicas=SIZE, rank=RANK, shuffle=True)

        def collate_fn(batch):
            x_list, y_list = [], []
            for token_ids in batch:
                if len(token_ids) < block_size + 1:
                    continue
                token_ids = token_ids[: block_size + 1]
                x_list.append(token_ids[:-1])
                y_list.append(token_ids[1:])
            if len(x_list) == 0:
                return None, None
            x_tensor = torch.tensor(x_list, dtype=torch.long)
            y_tensor = torch.tensor(y_list, dtype=torch.long)
            return x_tensor, y_tensor

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_fn
        )

        # 3) Train for multiple epochs
        global_step = 0
        for epoch_idx in range(epochs):
            sampler.set_epoch(epoch_idx)  # ensure consistent shuffling across ranks

            print(f"[Rank {RANK}] Starting epoch {epoch_idx}, dataset size ~ {len(dataset)}")

            for batch_idx, (x_tens, y_tens) in enumerate(dataloader):
                if x_tens is None:
                    continue

                x_tens = x_tens.to(device)
                y_tens = y_tens.to(device)

                optimizer.zero_grad()
                with torch.amp.autocast(device_type="xpu"):
                    logits, loss = model(x_tens, y_tens)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                global_step += 1
                if global_step % 100 == 0 and RANK == 0:
                    print(f"[Rank 0] Epoch={epoch_idx}, global_step={global_step}, Loss={loss.item():.4f}")

                # Save checkpoint occasionally from rank 0
                if global_step % 5000 == 0 and RANK == 0:
                    os.makedirs("pretrained", exist_ok=True)
                    ckpt_path = f"pretrained/ckpt_step_{global_step}.pth"
                    checkpoint = {
                        "epoch": epoch_idx,
                        "global_step": global_step,
                        # For DDP, the underlying model is in model.module
                        "model_state_dict": model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss.item()
                    }
                    torch.save(checkpoint, ckpt_path)
                    print(f"[Rank 0] Saved checkpoint at step {global_step} to {ckpt_path}")

    # 4) Save final model + tokenizer (rank 0)
    if RANK == 0:
        os.makedirs("Argonne_LLM", exist_ok=True)
        model.module.save_pretrained("Argonne_LLM")
        hf_tokenizer.save_pretrained("Argonne_LLM")
        print("[Rank 0] Model + tokenizer saved successfully!")

    # 5) Cleanup
    dist.destroy_process_group()
    print(f"[Rank {RANK}] Training complete. Process group destroyed.")

def main():
    train_model_ddp(
        data_path="data/*.arrow",  
        use_streaming=False,
        epochs=3
    )

if __name__ == "__main__":
    main()
