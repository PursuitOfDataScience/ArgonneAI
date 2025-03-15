import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    AutoConfig, 
    AutoModel, 
    AutoModelForCausalLM
)
from transformers.modeling_outputs import CausalLMOutput

from typing import Optional

class ArgonneConfig(PretrainedConfig):
    model_type = "argonne"
    def __init__(self, vocab_size=12000, block_size=2048, n_layer=24, n_head=24, n_embd=1296, dropout=0.1, use_flash_attn=True, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.use_flash_attn = use_flash_attn

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
        self.use_flash_attn = getattr(config, 'use_flash_attn', True)
        
        # Register the causal mask for the traditional attention path
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

        if hasattr(F, 'scaled_dot_product_attention') and self.use_flash_attn:
            # When using is_causal=True, don't provide an attention mask
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=True  # Let PyTorch handle the causal mask internally
            )
            attn_output = attn_output.transpose(1, 2).contiguous().view(b, t, c)
            y = self.resid_drop(self.proj(attn_output))
            return y
        else:
            # Original attention implementation (fallback)
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

class ArgonneModel(PreTrainedModel):
    config_class = ArgonneConfig

    # for map_device = "auto"
    _no_split_modules = ["Block"]

    def __init__(self, config, device_map=None):
        super().__init__(config)
        # Create embeddings on CPU initially
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.dropout)

        # Build all blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # Final LayerNorm + output head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        self.post_init()

        # For pipeline parallelism
        self.pipeline_stages = None
        self.devices = []
        
        # Handle device_map="auto" for inference
        if device_map is not None:
            self.setup_device_map(device_map)

    def setup_device_map(self, device_map):
        """
        Set up the model on devices according to device_map.
        If device_map="auto", use accelerate to automatically assign model parts to devices.
        """
        if device_map == "auto":
            try:
                from accelerate import dispatch_model
                from accelerate.utils import infer_auto_device_map
                
                # Get device map automatically
                auto_device_map = infer_auto_device_map(self)
                # Dispatch model across devices
                dispatch_model(self, device_map=auto_device_map)
                
                print(f"Model automatically distributed across devices with device_map: {auto_device_map}")
                
            except ImportError:
                print("The 'accelerate' library is required for device_map='auto'. Please install it with 'pip install accelerate'.")
                print("Continuing with model on CPU or default device.")
        else:
            # Handle custom device map
            # This would be a more complex implementation where the user provides a specific mapping
            # of model components to devices
            pass

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

        # Move embeddings to the first device
        first_device = device_ids[0]
        self.token_embedding = self.token_embedding.to(first_device)
        # For nn.Parameter, we need to move the data, not replace the parameter
        self.position_embedding.data = self.position_embedding.data.to(first_device)
        self.drop = self.drop.to(first_device)

        # Move final LayerNorm + head to the last device
        last_device = device_ids[-1]
        self.ln_f = self.ln_f.to(last_device)
        self.head = self.head.to(last_device)

        print(f"Model distributed across {len(device_ids)} devices")
        print(f"First device: {first_device}, Last device: {last_device}")
        print(f"Transformer layers per device: ~{blocks_per_gpu}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def prepare_for_compile(self):
        """
        Prepare model for torch.compile() by ensuring all components
        are compatible with the compiler.
        """
        # Some models may need special handling for compilation
        # For now, we'll just return self since our model structure should be compatible
        return self

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs
    ):
        """
        HF-friendly forward method.

        Args:
            input_ids (torch.LongTensor): Tokens to be fed to the model. [batch_size, seq_len].
            attention_mask (torch.LongTensor, optional): Mask of shape [batch_size, seq_len],
                with 1 for actual tokens and 0 for padding, if you want to incorporate it. 
                Currently ignored in this minimal example.
            labels (torch.LongTensor, optional): Targets for language modeling, same shape as `input_ids`.
            **kwargs: Catch-all for any additional arguments (e.g. past_key_values) so we don't crash.
        """
        # 1) We'll rename the parameters from the old code
        if input_ids is None:
            raise ValueError("`input_ids` must be provided.")

        # We used to call it 'idx'
        idx = input_ids
        # We used to call it 'targets'
        targets = labels

        # [Optional] If we want to handle single-dim input_ids
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)

        # 2) Now the rest of your old forward logic remains, just replacing references
        #    to "idx" and "targets" with these new variables.

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

            return CausalLMOutput(
                loss=loss,
                logits=logits,
                )

        else:
            # Pipeline parallel forward
            first_device = next(self.token_embedding.parameters()).device
            last_device = next(self.ln_f.parameters()).device

            x = idx.to(first_device)
            b, t = x.size()
            assert t <= self.config.block_size, "Sequence length exceeds block size"

            token_embeddings = self.token_embedding(x)
            position_embeddings = self.position_embedding[:, :t, :]
            hidden_states = self.drop(token_embeddings + position_embeddings)

            # Pass through each pipeline stage in sequence
            for stage_idx, stage in enumerate(self.pipeline_stages):
                device_stage = next(stage.parameters()).device
                hidden_states = hidden_states.to(device_stage)
                hidden_states = stage(hidden_states)

            # Move to last device before final ops
            hidden_states = hidden_states.to(last_device)
            hidden_states = self.ln_f(hidden_states)
            logits = self.head(hidden_states)

            loss = None
            if targets is not None:
                targets = targets.to(last_device)
                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)
                loss = F.cross_entropy(logits, targets)

            return CausalLMOutput(
                loss=loss,
                logits=logits,
                )

    

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        max_length: int = 50,            # Standard HF param
        do_sample: bool = True,          # Replaces "sample=True/False"
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: float = 0.7,
        attention_mask: Optional[torch.Tensor] = None,
        # Catch-all for additional HF params (e.g. num_beams) so it doesn't crash:
        **kwargs
    ):
        """
        A bridging generate method that accepts common HF arguments
        but uses your custom GPT-style generation loop.

        Args:
            input_ids (Tensor): Starting prompt tokens [batch_size, seq_len].
            max_length (int): The total length of the final sequence (seq_len + new tokens).
            do_sample (bool): If True, sample from distribution; if False, do greedy.
            top_k (int): Top-k filtering threshold.
            top_p (float): Nucleus sampling threshold.
            temperature (float): Sampling temperature.
            attention_mask (Tensor): If you want to handle padding (unused in this minimal example).
            **kwargs: Ignored extra arguments (e.g. num_beams) so they don't cause an error.
        Returns:
            Tensor of shape [batch_size, total_seq_len] with the generated tokens.
        """
        self.eval()

        # 1) Figure out device
        if self.pipeline_stages is not None and len(self.devices) > 0:
            device = self.devices[0]
        else:
            device = next(self.parameters()).device

        # 2) Sanity checks
        if input_ids is None:
            raise ValueError("`input_ids` must be provided for generation.")

        batch_size, current_length = input_ids.shape
        if current_length >= max_length:
            raise ValueError(f"Current sequence length {current_length} >= max_length={max_length}")

        # 3) Move to the correct device
        generated = input_ids.to(device)

        # We'll generate new tokens until length == max_length
        total_new_tokens = max_length - current_length

        for _ in range(total_new_tokens):
            # Truncate if necessary to fit within the model's context window
            if generated.shape[1] > self.config.block_size:
                generated = generated[:, -self.config.block_size:]

            # Forward pass
            outputs = self.forward(generated)
            logits = outputs.logits            # outputs is a CausalLMOutput
            logits = logits[:, -1, :]          # get the last token's logits

            # Temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Greedy decode if do_sample=False
            if not do_sample:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                # top-k filtering
                if top_k is not None:
                    threshold = torch.topk(logits, top_k)[0][..., -1, None]
                    filter_mask = logits < threshold
                    logits = logits.masked_fill(filter_mask, float('-inf'))

                # top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    sorted_indices_to_remove = cumulative_probs > top_p
                    # shift right to retain the first token above threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    filter_mask = sorted_indices_to_remove.scatter(
                        dim=1, index=sorted_indices, src=sorted_indices_to_remove
                    )
                    logits = logits.masked_fill(filter_mask, float('-inf'))

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            # Append new token
            generated = torch.cat([generated, next_token.to(device)], dim=1)

        return generated

# Register the model with Hugging Face's Auto classes
AutoConfig.register("argonne", ArgonneConfig)
AutoModel.register(ArgonneConfig, ArgonneModel)
AutoModelForCausalLM.register(ArgonneConfig, ArgonneModel)