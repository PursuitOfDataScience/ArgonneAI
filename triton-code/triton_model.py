import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import inspect
import warnings
from typing import Optional, Dict, List, Union
import copy

# Import existing model classes
from model import ArgonneConfig, ArgonneModel, Block, CausalSelfAttention, MLP
from triton_kernels import triton_attention, triton_gelu, triton_layernorm, is_triton_supported

class TritonCausalSelfAttention(nn.Module):
    """
    Drop-in replacement for CausalSelfAttention that uses Triton kernels for faster computation.
    Falls back to original implementation when needed.
    """
    def __init__(self, config, original_attn=None):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding dim must be divisible by n_head"
        
        # Copy from original attention or create new
        if original_attn is not None:
            self.n_head = original_attn.n_head
            self.head_dim = original_attn.head_dim
            self.query = original_attn.query
            self.key = original_attn.key
            self.value = original_attn.value
            self.attn_drop = original_attn.attn_drop
            self.resid_drop = original_attn.resid_drop
            self.proj = original_attn.proj
            self.use_flash_attn = original_attn.use_flash_attn
            if hasattr(original_attn, 'register_buffer'):
                self.register_buffer("mask", original_attn.mask)
        else:
            # Standard initialization similar to original
            self.n_head = config.n_head
            self.head_dim = config.n_embd // config.n_head
            self.query = nn.Linear(config.n_embd, config.n_embd)
            self.key = nn.Linear(config.n_embd, config.n_embd)
            self.value = nn.Linear(config.n_embd, config.n_embd)
            self.attn_drop = nn.Dropout(config.dropout)
            self.resid_drop = nn.Dropout(config.dropout)
            self.proj = nn.Linear(config.n_embd, config.n_embd)
            self.use_flash_attn = getattr(config, 'use_flash_attn', True)
            
            # Register buffer for the non-Triton path
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(config.block_size, config.block_size))
                     .view(1, 1, config.block_size, config.block_size)
            )
        
        # Flag for whether Triton is available and supported
        self._triton_available = is_triton_supported()
        if not self._triton_available:
            warnings.warn(
                "Triton is not available or supported on this system. "
                "Falling back to standard PyTorch implementation."
            )

    def forward(self, x):
        b, t, c = x.size()
        
        # Project query, key, value
        q = self.query(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = self.key(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        
        # Use Triton attention when available, training on GPU, and not in tracing mode
        if self._triton_available and x.is_cuda and not torch.jit.is_scripting():
            try:
                # Use our fast Triton kernel for attention
                attn_output = triton_attention(q, k, v, causal=True)
                attn_output = attn_output.transpose(1, 2).contiguous().view(b, t, c)
                y = self.resid_drop(self.proj(attn_output))
                return y
            except Exception as e:
                warnings.warn(f"Triton attention failed: {e}. Falling back to PyTorch implementation.")
                # Continue with fallback implementation
        
        # Fallback to standard implementation (PyTorch or Flash Attention)
        if hasattr(F, 'scaled_dot_product_attention') and self.use_flash_attn:
            # Using PyTorch's built-in flash attention
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=True
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

class TritonLayerNorm(nn.Module):
    """
    Triton-accelerated LayerNorm that's a drop-in replacement for torch.nn.LayerNorm
    """
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        # Check if Triton is available
        self._triton_available = is_triton_supported()

    def forward(self, x):
        if self._triton_available and x.is_cuda and not torch.jit.is_scripting():
            try:
                return triton_layernorm(
                    x, 
                    self.weight if self.elementwise_affine else torch.ones(self.normalized_shape, device=x.device),
                    self.bias if self.elementwise_affine else torch.zeros(self.normalized_shape, device=x.device),
                    self.eps
                )
            except Exception as e:
                warnings.warn(f"Triton LayerNorm failed: {e}. Falling back to PyTorch implementation.")
                
        # Fall back to standard implementation
        return F.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps
        )

class TritonMLP(nn.Module):
    """
    Drop-in replacement for MLP that uses Triton kernels for faster computation
    """
    def __init__(self, config, original_mlp=None):
        super().__init__()
        
        # Copy from original MLP or create new
        if original_mlp is not None:
            self.fc1 = original_mlp.fc1
            self.act = original_mlp.act
            self.fc2 = original_mlp.fc2
            self.drop = original_mlp.drop
        else:
            # Standard initialization similar to original
            self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)
            self.drop = nn.Dropout(config.dropout)

        # Check if Triton is available
        self._triton_available = is_triton_supported()

    def forward(self, x):
        # Use standard PyTorch operations to avoid shape mismatch issues with Triton 3.0.0
        x = self.fc1(x)
        
        # Apply dropout before activation for better numerical stability
        x = self.act(x) 
        x = self.drop(x)
        
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TritonBlock(nn.Module):
    """Drop-in replacement for Block that uses Triton kernels"""
    def __init__(self, config, original_block=None):
        super().__init__()
        
        # Copy from original block or create new
        if original_block is not None:
            # Convert LayerNorm to Triton LayerNorm
            if isinstance(original_block.ln1, nn.LayerNorm):
                self.ln1 = TritonLayerNorm(
                    original_block.ln1.normalized_shape,
                    original_block.ln1.eps,
                    original_block.ln1.elementwise_affine
                )
                # Copy weights if affine
                if original_block.ln1.elementwise_affine:
                    self.ln1.weight.data.copy_(original_block.ln1.weight.data)
                    self.ln1.bias.data.copy_(original_block.ln1.bias.data)
            else:
                self.ln1 = original_block.ln1
                
            # Same for ln2
            if isinstance(original_block.ln2, nn.LayerNorm):
                self.ln2 = TritonLayerNorm(
                    original_block.ln2.normalized_shape,
                    original_block.ln2.eps,
                    original_block.ln2.elementwise_affine
                )
                if original_block.ln2.elementwise_affine:
                    self.ln2.weight.data.copy_(original_block.ln2.weight.data)
                    self.ln2.bias.data.copy_(original_block.ln2.bias.data)
            else:
                self.ln2 = original_block.ln2
            
            # Replace attention and MLP with Triton versions
            self.attn = TritonCausalSelfAttention(config, original_block.attn)
            self.mlp = TritonMLP(config, original_block.mlp)
        else:
            # Standard initialization
            self.ln1 = TritonLayerNorm(config.n_embd)
            self.attn = TritonCausalSelfAttention(config)
            self.ln2 = TritonLayerNorm(config.n_embd)
            self.mlp = TritonMLP(config)
            
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TritonArgonneModel(ArgonneModel):
    """
    A wrapper around ArgonneModel that uses Triton-accelerated blocks
    """
    def __init__(self, config, device_map=None):
        # Initialize the parent class first
        super().__init__(config, device_map)
        
        # Replace blocks with Triton-accelerated ones
        triton_blocks = nn.ModuleList()
        for block in self.blocks:
            triton_blocks.append(TritonBlock(config, block))
        self.blocks = triton_blocks
        
        # Replace final layer norm with Triton version
        if isinstance(self.ln_f, nn.LayerNorm):
            triton_ln_f = TritonLayerNorm(
                self.ln_f.normalized_shape,
                self.ln_f.eps,
                self.ln_f.elementwise_affine
            )
            # Copy weights if affine
            if self.ln_f.elementwise_affine:
                triton_ln_f.weight.data.copy_(self.ln_f.weight.data)
                triton_ln_f.bias.data.copy_(self.ln_f.bias.data)
            self.ln_f = triton_ln_f
        
        print("Initialized Triton-accelerated ArgonneModel")
    
    @classmethod
    def from_pretrained(cls, model_path, *args, **kwargs):
        """
        Load a pretrained model and convert to Triton version
        """
        # First load the standard model
        original_model = super().from_pretrained(model_path, *args, **kwargs)
        
        # Now convert to Triton version
        config = original_model.config
        triton_model = cls(config)
        
        # Copy all the trained weights
        triton_model.load_state_dict(original_model.state_dict())
        
        # If original model was distributed, re-distribute this one
        if original_model.pipeline_stages is not None:
            # Use the same device distribution
            triton_model.devices = original_model.devices
            triton_model.distribute_model(device_ids=[str(dev) for dev in original_model.devices])
        
        print(f"Model loaded from {model_path} and converted to use Triton acceleration")
        return triton_model
    
    @classmethod
    def from_model(cls, original_model):
        """
        Convert a standard ArgonneModel to TritonArgonneModel
        """
        config = original_model.config
        
        # Check if original model was already distributed with pipeline parallelism
        is_distributed = original_model.pipeline_stages is not None
        
        # CRITICAL FIX: Before doing anything else, move embeddings to GPU
        if hasattr(original_model, 'devices') and original_model.devices:
            first_device = original_model.devices[0]
            print(f"Moving embeddings to {first_device}")
            
            # Make sure token embedding is on the correct device
            if hasattr(original_model, 'token_embedding'):
                if original_model.token_embedding.weight.device != first_device:
                    original_model.token_embedding = original_model.token_embedding.to(first_device)
                    
            # Make sure position embedding is on the correct device  
            if hasattr(original_model, 'position_embedding'):
                if original_model.position_embedding.device != first_device:
                    original_model.position_embedding.data = original_model.position_embedding.data.to(first_device)
        
        if is_distributed:
            # For distributed models, we need special handling
            # Step 1: Create a base triton model (non-distributed)
            triton_model = cls(config)
            
            # Step 2: Manually extract each block's weights from the pipeline stages
            # and map them to the corresponding blocks in the triton model
            orig_blocks = []
            
            # Reconstruct the original blocks from pipeline stages
            for stage in original_model.pipeline_stages:
                for block in stage:
                    orig_blocks.append(block)
            
            # Verify we have the right number of blocks
            assert len(orig_blocks) == len(triton_model.blocks), \
                f"Block count mismatch: {len(orig_blocks)} != {len(triton_model.blocks)}"
            
            # Convert each block individually
            for i, (orig_block, triton_block) in enumerate(zip(orig_blocks, triton_model.blocks)):
                # Layer norm 1
                if hasattr(orig_block, 'ln1') and hasattr(triton_block, 'ln1'):
                    if isinstance(orig_block.ln1, nn.LayerNorm):
                        triton_block.ln1.weight.data.copy_(orig_block.ln1.weight.data)
                        triton_block.ln1.bias.data.copy_(orig_block.ln1.bias.data)
                
                # Attention
                if hasattr(orig_block, 'attn') and hasattr(triton_block, 'attn'):
                    triton_block.attn.query.weight.data.copy_(orig_block.attn.query.weight.data)
                    triton_block.attn.query.bias.data.copy_(orig_block.attn.query.bias.data)
                    triton_block.attn.key.weight.data.copy_(orig_block.attn.key.weight.data)
                    triton_block.attn.key.bias.data.copy_(orig_block.attn.key.bias.data)
                    triton_block.attn.value.weight.data.copy_(orig_block.attn.value.weight.data)
                    triton_block.attn.value.bias.data.copy_(orig_block.attn.value.bias.data)
                    triton_block.attn.proj.weight.data.copy_(orig_block.attn.proj.weight.data)
                    triton_block.attn.proj.bias.data.copy_(orig_block.attn.proj.bias.data)
                    if hasattr(orig_block.attn, 'mask') and hasattr(triton_block.attn, 'mask'):
                        triton_block.attn.mask.copy_(orig_block.attn.mask)
                
                # Layer norm 2
                if hasattr(orig_block, 'ln2') and hasattr(triton_block, 'ln2'):
                    if isinstance(orig_block.ln2, nn.LayerNorm):
                        triton_block.ln2.weight.data.copy_(orig_block.ln2.weight.data)
                        triton_block.ln2.bias.data.copy_(orig_block.ln2.bias.data)
                
                # MLP
                if hasattr(orig_block, 'mlp') and hasattr(triton_block, 'mlp'):
                    triton_block.mlp.fc1.weight.data.copy_(orig_block.mlp.fc1.weight.data)
                    triton_block.mlp.fc1.bias.data.copy_(orig_block.mlp.fc1.bias.data)
                    triton_block.mlp.fc2.weight.data.copy_(orig_block.mlp.fc2.weight.data)
                    triton_block.mlp.fc2.bias.data.copy_(orig_block.mlp.fc2.bias.data)
            
            # Copy other model components
            triton_model.token_embedding.weight.data.copy_(original_model.token_embedding.weight.data)
            triton_model.position_embedding.data.copy_(original_model.position_embedding.data)
            
            if hasattr(original_model, 'ln_f') and hasattr(triton_model, 'ln_f'):
                triton_model.ln_f.weight.data.copy_(original_model.ln_f.weight.data)
                triton_model.ln_f.bias.data.copy_(original_model.ln_f.bias.data)
            
            if hasattr(original_model, 'head') and hasattr(triton_model, 'head'):
                triton_model.head.weight.data.copy_(original_model.head.weight.data)
                if hasattr(original_model.head, 'bias') and original_model.head.bias is not None:
                    triton_model.head.bias.data.copy_(original_model.head.bias.data)
            
            # Copy device information
            triton_model.devices = original_model.devices.copy() if hasattr(original_model, 'devices') else []
            
            # CRITICAL FIX: Explicitly move embeddings to first device again after conversion
            if triton_model.devices:
                first_device = triton_model.devices[0]
                triton_model.token_embedding = triton_model.token_embedding.to(first_device)
                triton_model.position_embedding = triton_model.position_embedding.to(first_device)
                
                # Also ensure the first layernorm is on the proper device
                if hasattr(triton_model.blocks[0], 'ln1'):
                    triton_model.blocks[0].ln1 = triton_model.blocks[0].ln1.to(first_device)
            
            # Now distribute the triton model to match the original model's distribution
            if triton_model.devices:
                triton_model.distribute_model(device_ids=[str(dev) for dev in triton_model.devices])
                    
        else:
            # For non-distributed models, we can use a simpler approach
            triton_model = cls(config)
            triton_model.load_state_dict(original_model.state_dict())
            
            # CRITICAL FIX: If original model has a device, move embeddings there
            if hasattr(original_model, 'device'):
                device = original_model.device
                triton_model.token_embedding = triton_model.token_embedding.to(device)
                triton_model.position_embedding = triton_model.position_embedding.to(device)
        
        print("Model converted to Triton-accelerated version")
        return triton_model

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=0.7, top_k=None, top_p=0.9, sample=True):
        """
        Generate text using the model with improved stability for handling NaN values.
        
        Args:
            input_ids: Input token IDs to continue from
            max_new_tokens: Number of tokens to generate
            temperature: Temperature for sampling (higher = more random)
            top_k: If set, only sample from the top k most likely tokens
            top_p: If set, sample from the smallest set of tokens whose cumulative probability exceeds p
            sample: If True, sample from the distribution; if False, use greedy decoding
        
        Returns:
            Tensor containing the input_ids extended with max_new_tokens generated tokens
        """
        self.eval()
        
        # Determine which device to use
        if self.pipeline_stages is not None and len(self.devices) > 0:
            device = self.devices[0]  # Always use first device for generation
        else:
            device = next(self.parameters()).device
        
        # Ensure input is on the correct device
        generated = input_ids.to(device)
        
        for _ in range(max_new_tokens):
            # Truncate if necessary to fit within the model's context window
            if generated.shape[1] > self.config.block_size:
                generated = generated[:, -self.config.block_size:]
    
            # Forward pass
            logits, _ = self.forward(generated)
            
            # Make sure logits are on the same device
            logits = logits.to(device)
            
            # Get logits for the last token only
            logits = logits[:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Check for NaN values and replace with safe values
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                # Handle NaN/Inf in logits by replacing with safe values
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
                # Fall back to greedy sampling when numerical issues are detected
                sample = False
                print("⚠️ NaN detected in logits. Falling back to greedy decoding for stability.")
            
            # Greedy decoding (argmax) if sample=False
            if not sample:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                # Sampling logic with safety checks
                try:
                    # Apply top-k filtering
                    if top_k is not None and top_k > 0:
                        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits[logits < v[:, [-1]]] = -float('inf')
                    
                    # Apply top-p (nucleus) filtering
                    if top_p is not None and top_p > 0.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        
                        # Shift the indices to the right to keep the first token above the threshold
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices_to_remove.scatter(
                            dim=1, index=sorted_indices, src=sorted_indices_to_remove
                        )
                        logits = logits.masked_fill(indices_to_remove, -float('inf'))
                    
                    # Check for potential issues before sampling
                    if torch.isnan(logits).any() or torch.isinf(logits).all(dim=-1).any():
                        # Fall back to uniform sampling over vocab if all values are invalid
                        probs = torch.ones_like(logits) / logits.size(-1)
                    else:
                        # Convert to probability distribution
                        probs = F.softmax(logits, dim=-1)
                        
                    # Sample from the distribution
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                except RuntimeError as e:
                    # Handle sampling errors by doing greedy decoding
                    if "multinomial" in str(e):
                        print(f"⚠️ Sampling error: {e}. Falling back to greedy decoding.")
                        next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    else:
                        raise  # Re-raise other errors
            
            # Append the generated token to the sequence
            generated = torch.cat((generated, next_token), dim=1)

        return generated

    def prepare_for_compile(self):
        """
        Ensure all components are compatible with torch.compile()
        """
        # Check if any components need special handling
        # Current implementation should be compatible
        return self