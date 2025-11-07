# Day 26 – Attention Mechanisms

## QKV (Query-Key-Value)

### What is it?

Query-Key-Value (QKV) is the foundational formulation of modern attention mechanisms in deep learning. It draws inspiration from information retrieval systems:

- **Query (Q)**: Represents what we are looking for (e.g., the current word in a sentence).
- **Key (K)**: Represents what we can offer or match against (e.g., all words in the sentence).
- **Value (V)**: Contains the actual information or content associated with each key.

The attention mechanism computes how much focus (attention weight) should be placed on each value by comparing the query with all keys. This allows models to dynamically focus on the most relevant parts of the input sequence, regardless of distance, solving the long-range dependency problem in traditional RNNs.

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Basic QKV attention implementation
class QKVAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=1):
        super(QKVAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Final output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch_size, seq_len_q, embed_dim)
            key: (batch_size, seq_len_k, embed_dim)
            value: (batch_size, seq_len_v, embed_dim)
            mask: (batch_size, seq_len_q, seq_len_k)
        """
        batch_size = query.size(0)
        
        # Project Q, K, V
        Q = self.q_proj(query)  # (batch_size, seq_len_q, embed_dim)
        K = self.k_proj(key)    # (batch_size, seq_len_k, embed_dim)
        V = self.v_proj(value)  # (batch_size, seq_len_v, embed_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        # Q: (batch_size, num_heads, seq_len_q, head_dim)
        # K: (batch_size, num_heads, seq_len_k, head_dim)
        # scores: (batch_size, num_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        # attention_output: (batch_size, num_heads, seq_len_q, head_dim)
        attention_output = torch.matmul(attention_weights, V)
        
        # Reshape back to original dimensions
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, -1, self.embed_dim)
        
        # Final linear projection
        output = self.out_proj(attention_output)
        
        return output, attention_weights

# Demonstrate QKV attention
def demonstrate_qkv_attention():
    # Parameters
    batch_size = 2
    seq_len = 5
    embed_dim = 64
    num_heads = 4
    
    # Create random input tensors
    query = torch.randn(batch_size, seq_len, embed_dim)
    key = torch.randn(batch_size, seq_len, embed_dim)
    value = torch.randn(batch_size, seq_len, embed_dim)
    
    # Create attention module
    attention = QKVAttention(embed_dim, num_heads)
    
    # Forward pass
    output, weights = attention(query, key, value)
    
    print(f"Input shapes:")
    print(f"  Query: {query.shape}")
    print(f"  Key: {key.shape}")
    print(f"  Value: {value.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"Parameters: {sum(p.numel() for p in attention.parameters()):,}")
    
    return attention, output, weights

attention_model, output, weights = demonstrate_qkv_attention()
```

### When and where we use QKV

- **Transformer models**: The core mechanism in encoder-decoder architectures
- **Language modeling**: Understanding context in text generation
- **Machine translation**: Aligning source and target language tokens
- **Image captioning**: Focusing on relevant image regions for text generation
- **Speech recognition**: Attending to relevant audio segments
- **Recommendation systems**: Understanding user-item interactions

```python
# When to use QKV - Transformer encoder example
def transformer_encoder_example():
    """QKV attention in transformer encoder"""
    
    class TransformerEncoderLayer(nn.Module):
        def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
            super().__init__()
            
            # Multi-head attention
            self.attention = QKVAttention(embed_dim, num_heads)
            self.norm1 = nn.LayerNorm(embed_dim)
            self.dropout1 = nn.Dropout(dropout)
            
            # Feed-forward network
            self.ff = nn.Sequential(
                nn.Linear(embed_dim, ff_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, embed_dim)
            )
            self.norm2 = nn.LayerNorm(embed_dim)
            self.dropout2 = nn.Dropout(dropout)
        
        def forward(self, x, mask=None):
            # Self-attention (Q=K=V=x)
            attn_output, _ = self.attention(x, x, x, mask)
            x = self.norm1(x + self.dropout1(attn_output))
            
            # Feed-forward
            ff_output = self.ff(x)
            x = self.norm2(x + self.dropout2(ff_output))
            
            return x
    
    return TransformerEncoderLayer

encoder_layer = transformer_encoder_example()
```

### Real-world examples

- **Google Translate**: Uses transformer architecture with QKV attention
- **ChatGPT**: Leverages multi-head attention for context understanding
- **Image analysis**: Visual attention for object detection and segmentation
- **Financial forecasting**: Attending to relevant market indicators
- **Medical diagnosis**: Focusing on relevant symptoms in patient records

```python
# Real-world example: Medical text analysis with attention
def medical_attention_example():
    """QKV attention for medical diagnosis"""
    
    class MedicalDiagnosisAttention(nn.Module):
        def __init__(self, vocab_size, embed_dim, num_heads, num_classes):
            super().__init__()
            
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.attention = QKVAttention(embed_dim, num_heads)
            
            # Classification head
            self.classifier = nn.Sequential(
                nn.Linear(embed_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )
        
        def forward(self, x):
            # x: (batch_size, seq_len)
            embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
            
            # Self-attention
            attn_output, attention_weights = self.attention(embedded, embedded, embedded)
            
            # Global average pooling
            pooled = torch.mean(attn_output, dim=1)
            
            # Classification
            output = self.classifier(pooled)
            
            return output, attention_weights
    
    return MedicalDiagnosisAttention

medical_model = medical_attention_example()
```

### Problems and Advancements

**Problem**: Standard attention has quadratic complexity O(n²) with sequence length.
**Advancement**: Efficient attention mechanisms like sparse attention, linear attention, and FlashAttention.

```python
# Problem: Quadratic complexity in attention
def quadratic_complexity_demo():
    """Demonstrate quadratic complexity issue"""
    
    def compute_attention_complexity(seq_len, embed_dim):
        # QK^T computation: seq_len * seq_len * embed_dim
        attention_ops = seq_len * seq_len * embed_dim
        return attention_ops
    
    seq_lengths = [10, 50, 100, 500, 1000]
    complexities = [compute_attention_complexity(seq_len, 64) for seq_len in seq_lengths]
    
    print("Attention Complexity (QK^T operations):")
    for seq_len, comp in zip(seq_lengths, complexities):
        print(f"Sequence length {seq_len}: {comp:,} operations")
    
    # Plot complexity
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lengths, complexities, 'o-')
    plt.title('Attention Complexity vs Sequence Length')
    plt.xlabel('Sequence Length')
    plt.ylabel('Operations (QK^T)')
    plt.yscale('log')
    plt.grid(True)
    plt.show()

quadratic_complexity_demo()

# Solution: Efficient attention with sparse patterns
class SparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, sparsity_pattern='local'):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.sparsity_pattern = sparsity_pattern
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    
    def create_sparse_mask(self, seq_len, device):
        """Create sparse attention mask"""
        if self.sparsity_pattern == 'local':
            # Only attend to nearby tokens (window of 3)
            mask = torch.zeros(seq_len, seq_len, device=device)
            for i in range(seq_len):
                start = max(0, i - 1)
                end = min(seq_len, i + 2)
                mask[i, start:end] = 1
            return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        elif self.sparsity_pattern == 'global':
            # Attend to first and last tokens globally
            mask = torch.zeros(seq_len, seq_len, device=device)
            mask[0, :] = 1  # First token attends to all
            mask[:, 0] = 1  # All tokens attend to first
            mask[-1, :] = 1  # Last token attends to all
            mask[:, -1] = 1  # All tokens attend to last
            return mask.unsqueeze(0).unsqueeze(0)
    
    def forward(self, query, key, value):
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Create sparse mask
        sparse_mask = self.create_sparse_mask(seq_len, query.device)
        
        # Compute attention with sparse mask
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        scores = scores.masked_fill(sparse_mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, -1, self.embed_dim)
        output = self.out_proj(attention_output)
        
        return output, attention_weights

# Compare standard vs sparse attention
def compare_attention_efficiency():
    batch_size, seq_len, embed_dim = 1, 100, 64
    query = torch.randn(batch_size, seq_len, embed_dim)
    
    standard_attn = QKVAttention(embed_dim, num_heads=1)
    sparse_attn = SparseAttention(embed_dim, num_heads=1, sparsity_pattern='local')
    
    # Count non-zero attention weights
    _, std_weights = standard_attn(query, query, query)
    _, sparse_weights = sparse_attn(query, query, query)
    
    std_nonzero = (std_weights > 1e-6).sum().item()
    sparse_nonzero = (sparse_weights > 1e-6).sum().item()
    
    print(f"Standard attention non-zero weights: {std_nonzero}")
    print(f"Sparse attention non-zero weights: {sparse_nonzero}")
    print(f"Reduction: {(std_nonzero - sparse_nonzero) / std_nonzero * 100:.1f}%")

compare_attention_efficiency()
```

## Scaled Dot Product

### What is it?

Scaled dot-product attention is the specific mathematical formulation used to compute attention weights in the QKV framework. It computes the dot product between queries and keys, then scales the result by the square root of the key dimension (√dₖ) before applying softmax. The scaling factor prevents the dot products from becoming too large in magnitude when the dimension is large, which would push the softmax function into regions where it has extremely small gradients.

The formula is:
**Attention(Q, K, V) = softmax(QKᵀ / √dₖ)V**

Where:
- Q, K, V are the query, key, and value matrices
- dₖ is the dimension of the key vectors
- The division by √dₖ is the "scaling" that gives the method its name

### PyTorch Implementation

```python
# Manual scaled dot-product attention implementation
class ScaledDotProductAttention(nn.Module):
    def __init__(self, scale=True):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = scale
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch_size, num_heads, seq_len_q, d_k)
            key: (batch_size, num_heads, seq_len_k, d_k)
            value: (batch_size, num_heads, seq_len_v, d_v)
            mask: (batch_size, 1, seq_len_q, seq_len_k) or similar
        """
        d_k = query.size(-1)
        
        # Compute attention scores: QK^T
        scores = torch.matmul(query, key.transpose(-2, -1))
        
        # Scale by sqrt(d_k)
        if self.scale:
            scores = scores / np.sqrt(d_k)
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values: softmax(QK^T/sqrt(d_k)) * V
        attention_output = torch.matmul(attention_weights, value)
        
        return attention_output, attention_weights

# Comparison: with and without scaling
def demonstrate_scaling_effect():
    """Demonstrate the effect of scaling in dot-product attention"""
    
    # Parameters
    batch_size, num_heads, seq_len, d_k = 1, 1, 10, 64
    
    # Create random Q, K, V
    query = torch.randn(batch_size, num_heads, seq_len, d_k)
    key = torch.randn(batch_size, num_heads, seq_len, d_k)
    value = torch.randn(batch_size, num_heads, seq_len, d_k)
    
    # Without scaling
    attn_no_scale = ScaledDotProductAttention(scale=False)
    output_no_scale, weights_no_scale = attn_no_scale(query, key, value)
    
    # With scaling
    attn_with_scale = ScaledDotProductAttention(scale=True)
    output_with_scale, weights_with_scale = attn_with_scale(query, key, value)
    
    # Analyze attention weights
    print("Attention weights statistics:")
    print(f"Without scaling - Mean: {weights_no_scale.mean().item():.4f}, Std: {weights_no_scale.std().item():.4f}")
    print(f"With scaling - Mean: {weights_with_scale.mean().item():.4f}, Std: {weights_with_scale.std().item():.4f}")
    
    # Check for softmax saturation
    max_weights_no_scale = weights_no_scale.max().item()
    max_weights_with_scale = weights_with_scale.max().item()
    
    print(f"Without scaling - Max attention weight: {max_weights_no_scale:.4f}")
    print(f"With scaling - Max attention weight: {max_weights_with_scale:.4f}")
    
    # Visualize attention weights
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(weights_no_scale[0, 0].detach().numpy(), cmap='viridis')
    plt.title('Attention Weights (No Scaling)')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(weights_with_scale[0, 0].detach().numpy(), cmap='viridis')
    plt.title('Attention Weights (With Scaling)')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    
    return weights_no_scale, weights_with_scale

no_scale_weights, with_scale_weights = demonstrate_scaling_effect()
```

### When and where we use scaled dot product

- **Transformer architectures**: The standard attention mechanism in all transformer variants
- **Self-attention**: When Q=K=V for understanding sequence context
- **Cross-attention**: When Q comes from one sequence and K,V from another (e.g., encoder-decoder)
- **Multi-head attention**: Multiple parallel scaled dot-product attention layers
- **Vision transformers**: Applied to image patches instead of tokens

```python
# When to use scaled dot product - Multi-head attention example
def multi_head_attention_example():
    """Complete multi-head scaled dot-product attention"""
    
    class MultiHeadAttention(nn.Module):
        def __init__(self, embed_dim, num_heads):
            super().__init__()
            assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
            
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads
            
            # Combined projection for efficiency
            self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
            self.out_proj = nn.Linear(embed_dim, embed_dim)
            
            self.attention = ScaledDotProductAttention(scale=True)
        
        def forward(self, x, mask=None):
            batch_size, seq_len, embed_dim = x.shape
            
            # Project to Q, K, V in one operation
            qkv = self.qkv_proj(x)  # (batch_size, seq_len, 3 * embed_dim)
            qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
            
            Q, K, V = qkv[0], qkv[1], qkv[2]
            
            # Apply scaled dot-product attention
            attention_output, attention_weights = self.attention(Q, K, V, mask)
            
            # Reshape and project output
            attention_output = attention_output.transpose(1, 2).contiguous()
            attention_output = attention_output.reshape(batch_size, seq_len, embed_dim)
            output = self.out_proj(attention_output)
            
            return output, attention_weights
    
    return MultiHeadAttention

mha_model = multi_head_attention_example()
```

### Real-world examples

- **BERT**: Uses scaled dot-product attention for bidirectional context understanding
- **GPT**: Employs causal scaled dot-product attention for autoregressive generation
- **ViT (Vision Transformer)**: Applies attention to image patches for computer vision
- **Whisper**: Uses attention mechanisms for speech recognition
- **AlphaFold**: Leverages attention for protein structure prediction

```python
# Real-world example: Vision Transformer attention
def vision_transformer_attention():
    """Scaled dot-product attention for image patches"""
    
    class PatchEmbedding(nn.Module):
        def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
            super().__init__()
            self.patch_size = patch_size
            self.num_patches = (img_size // patch_size) ** 2
            
            self.projection = nn.Conv2d(
                in_channels, embed_dim, 
                kernel_size=patch_size, 
                stride=patch_size
            )
        
        def forward(self, x):
            # x: (batch_size, in_channels, img_size, img_size)
            x = self.projection(x)  # (batch_size, embed_dim, num_patches_h, num_patches_w)
            x = x.flatten(2)  # (batch_size, embed_dim, num_patches)
            x = x.transpose(1, 2)  # (batch_size, num_patches, embed_dim)
            return x
    
    class ViTAttention(nn.Module):
        def __init__(self, embed_dim, num_heads):
            super().__init__()
            self.patch_embed = PatchEmbedding(embed_dim=embed_dim)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embedding = nn.Parameter(torch.zeros(1, 197, embed_dim))  # 196 patches + 1 cls token
            
            self.mha = MultiHeadAttention(embed_dim, num_heads)
            self.norm = nn.LayerNorm(embed_dim)
        
        def forward(self, x):
            # x: (batch_size, 3, 224, 224)
            batch_size = x.shape[0]
            
            # Patch embedding
            patches = self.patch_embed(x)  # (batch_size, 196, embed_dim)
            
            # Add class token
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_tokens, patches), dim=1)  # (batch_size, 197, embed_dim)
            
            # Add positional embedding
            x = x + self.pos_embedding
            
            # Apply attention
            x = self.norm(x)
            attention_output, weights = self.mha(x)
            
            return attention_output, weights
    
    return ViTAttention

vit_attention = vision_transformer_attention()
```

### Problems and Advancements

**Problem**: Dot products can have high variance in high dimensions, leading to softmax saturation.
**Advancement**: Advanced attention mechanisms like relative positional encoding, learned positional embeddings, and attention refinements.

```python
# Problem: Softmax saturation without scaling
def softmax_saturation_demo():
    """Demonstrate softmax saturation problem"""
    
    def simulate_dot_products(d_k, num_samples=1000):
        # Simulate random dot products
        q = torch.randn(num_samples, d_k)
        k = torch.randn(num_samples, d_k)
        dot_products = torch.sum(q * k, dim=1)
        return dot_products
    
    dimensions = [16, 64, 256, 1024]
    
    plt.figure(figsize=(12, 8))
    
    for i, d_k in enumerate(dimensions):
        dot_products = simulate_dot_products(d_k)
        scaled_dot_products = dot_products / np.sqrt(d_k)
        
        plt.subplot(2, 2, i+1)
        plt.hist(dot_products.numpy(), alpha=0.7, label='Raw dot products', bins=50)
        plt.hist(scaled_dot_products.numpy(), alpha=0.7, label='Scaled dot products', bins=50)
        plt.title(f'd_k = {d_k}')
        plt.xlabel('Dot product values')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Show softmax gradients
    x = torch.linspace(-10, 10, 1000, requires_grad=True)
    softmax_vals = F.softmax(x.unsqueeze(0), dim=1).squeeze()
    softmax_grad = torch.autograd.grad(softmax_vals.sum(), x, retain_graph=True)[0]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x.detach().numpy(), softmax_grad.numpy())
    plt.title('Softmax Gradient vs Input Value')
    plt.xlabel('Input Value')
    plt.ylabel('Gradient Magnitude')
    plt.grid(True)
    plt.show()

softmax_saturation_demo()

# Solution: Advanced attention with relative positional encoding
class RelativePositionAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, max_rel_pos=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_rel_pos = max_rel_pos
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.r_proj = nn.Linear(embed_dim, embed_dim)  # Relative position projection
        
        # Relative position embeddings
        self.rel_pos_embeddings = nn.Embedding(2 * max_rel_pos + 1, self.head_dim)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = np.sqrt(self.head_dim)
    
    def relative_position_indices(self, seq_len):
        """Create relative position indices"""
        indices = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
        indices = torch.clamp(indices + self.max_rel_pos, 0, 2 * self.max_rel_pos)
        return indices
    
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Standard attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Relative position attention
        rel_indices = self.relative_position_indices(seq_len).to(x.device)
        rel_pos_embeds = self.rel_pos_embeddings(rel_indices)  # (seq_len, seq_len, head_dim)
        
        # Add relative position scores
        rel_scores = torch.einsum('bnid,jkd->bnij', Q, rel_pos_embeds) / self.scale
        attn_scores = attn_scores + rel_scores
        
        attention_weights = F.softmax(attn_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, embed_dim)
        output = self.out_proj(attention_output)
        
        return output, attention_weights

# Compare standard vs relative position attention
def compare_attention_types():
    batch_size, seq_len, embed_dim = 2, 8, 64
    
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    standard_attn = MultiHeadAttention(embed_dim, num_heads=2)
    relative_attn = RelativePositionAttention(embed_dim, num_heads=2)
    
    _, std_weights = standard_attn(x)
    _, rel_weights = relative_attn(x)
    
    print("Comparing attention patterns...")
    print(f"Standard attention shape: {std_weights.shape}")
    print(f"Relative attention shape: {rel_weights.shape}")
    
    # Visualize attention patterns
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(std_weights[0, 0].detach().numpy(), cmap='viridis')
    plt.title('Standard Attention')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(rel_weights[0, 0].detach().numpy(), cmap='viridis')
    plt.title('Relative Position Attention')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

compare_attention_types()
```

## 🧪 Exercise: Manual attention

```python
def manual_attention_exercise():
    """Complete manual implementation of attention mechanism"""
    
    print("Manual Attention Exercise")
    print("=" * 40)
    
    # Step 1: Create input embeddings
    batch_size, seq_len, embed_dim = 1, 4, 8
    
    # Simple embeddings for demonstration
    embeddings = torch.tensor([
        [[1.0, 0.8, 0.2, 0.1, 0.9, 0.3, 0.7, 0.4],   # Token 1
         [0.2, 0.9, 0.8, 0.7, 0.1, 0.6, 0.3, 0.8],   # Token 2  
         [0.8, 0.1, 0.9, 0.2, 0.7, 0.4, 0.6, 0.5],   # Token 3
         [0.4, 0.6, 0.3, 0.9, 0.2, 0.8, 0.1, 0.7]]   # Token 4
    ])
    
    print("Input embeddings:")
    print(f"Shape: {embeddings.shape}")
    print(f"Values:\n{embeddings}")
    
    # Step 2: Create Q, K, V weight matrices
    W_q = torch.randn(embed_dim, embed_dim)
    W_k = torch.randn(embed_dim, embed_dim)
    W_v = torch.randn(embed_dim, embed_dim)
    
    print(f"\nQ, K, V weight matrices shape: {W_q.shape}")
    
    # Step 3: Compute Q, K, V
    Q = torch.matmul(embeddings, W_q)
    K = torch.matmul(embeddings, W_k)
    V = torch.matmul(embeddings, W_v)
    
    print(f"\nQ shape: {Q.shape}")
    print(f"K shape: {K.shape}")
    print(f"V shape: {V.shape}")
    
    # Step 4: Compute attention scores (QK^T)
    attention_scores = torch.matmul(Q, K.transpose(-2, -1))
    print(f"\nAttention scores (QK^T) shape: {attention_scores.shape}")
    print(f"Attention scores:\n{attention_scores}")
    
    # Step 5: Scale attention scores
    d_k = embed_dim
    scaled_attention_scores = attention_scores / np.sqrt(d_k)
    print(f"\nScaled attention scores:\n{scaled_attention_scores}")
    
    # Step 6: Apply softmax to get attention weights
    attention_weights = F.softmax(scaled_attention_scores, dim=-1)
    print(f"\nAttention weights (after softmax):\n{attention_weights}")
    print(f"Row sums (should be 1.0): {attention_weights.sum(dim=-1)}")
    
    # Step 7: Apply attention weights to values
    attention_output = torch.matmul(attention_weights, V)
    print(f"\nFinal attention output shape: {attention_output.shape}")
    print(f"Attention output:\n{attention_output}")
    
    # Step 8: Verify with PyTorch implementation
    print("\n" + "="*40)
    print("Verification with PyTorch implementation")
    
    # Create QKV attention module
    class SimpleQKVAttention(nn.Module):
        def __init__(self, embed_dim):
            super().__init__()
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        def forward(self, x):
            Q = self.q_proj(x)
            K = self.k_proj(x)
            V = self.v_proj(x)
            
            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(embed_dim)
            weights = F.softmax(scores, dim=-1)
            output = torch.matmul(weights, V)
            
            return output, weights
    
    # Initialize with our manual weights
    pytorch_attn = SimpleQKVAttention(embed_dim)
    with torch.no_grad():
        pytorch_attn.q_proj.weight.copy_(W_q.T)
        pytorch_attn.k_proj.weight.copy_(W_k.T)
        pytorch_attn.v_proj.weight.copy_(W_v.T)
    
    pytorch_output, pytorch_weights = pytorch_attn(embeddings)
    
    # Compare results
    print(f"\nManual vs PyTorch output difference: {torch.abs(attention_output - pytorch_output).max().item():.6f}")
    print(f"Manual vs PyTorch weights difference: {torch.abs(attention_weights - pytorch_weights).max().item():.6f}")
    
    # Step 9: Create visualization of attention
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(attention_weights[0].detach().numpy(), cmap='viridis', vmin=0, vmax=1)
    plt.title('Attention Weights')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.colorbar()
    
    # Show what each query attends to
    plt.subplot(1, 2, 2)
    for i in range(seq_len):
        plt.plot(range(seq_len), attention_weights[0, i].detach().numpy(), 'o-', label=f'Query {i+1}')
    plt.title('Attention Distribution')
    plt.xlabel('Key Position')
    plt.ylabel('Attention Weight')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'embeddings': embeddings,
        'Q': Q, 'K': K, 'V': V,
        'attention_weights': attention_weights,
        'attention_output': attention_output
    }

# Run the manual attention exercise
manual_results = manual_attention_exercise()
```

## 📌 Project: Attention visualizer

```python
class AttentionVisualizer:
    """Complete attention visualization project"""
    
    def __init__(self, vocab_size=1000, embed_dim=64, num_heads=4):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Create embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Create multi-head attention
        self.mha = MultiHeadAttention(embed_dim, num_heads)
        
        # Create vocabulary for demonstration
        self.vocab = {f"token_{i}": i for i in range(vocab_size)}
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def create_sample_sequence(self, tokens):
        """Create tensor from token list"""
        indices = [self.vocab.get(token, 0) for token in tokens]
        return torch.tensor([indices], dtype=torch.long)
    
    def visualize_attention(self, input_tokens, save_path=None):
        """Visualize attention weights for given input"""
        # Convert tokens to indices
        input_tensor = self.create_sample_sequence(input_tokens)
        
        # Get embeddings
        embedded = self.embedding(input_tensor)
        
        # Get attention weights
        with torch.no_grad():
            _, attention_weights = self.mha(embedded)
        
        # Create visualization
        self._plot_attention_heatmap(attention_weights[0], input_tokens, save_path)
        self._plot_attention_patterns(attention_weights[0], input_tokens)
        
        return attention_weights[0]
    
    def _plot_attention_heatmap(self, attention_weights, tokens, save_path=None):
        """Create heatmap visualization of attention weights"""
        num_heads = attention_weights.shape[0]
        seq_len = len(tokens)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for head in range(min(num_heads, 4)):
            im = axes[head].imshow(attention_weights[head].cpu().numpy(), 
                                   cmap='viridis', aspect='auto', vmin=0, vmax=1)
            axes[head].set_title(f'Attention Head {head + 1}')
            axes[head].set_xlabel('Key Position')
            axes[head].set_ylabel('Query Position')
            
            # Set tick labels
            axes[head].set_xticks(range(seq_len))
            axes[head].set_yticks(range(seq_len))
            axes[head].set_xticklabels(tokens, rotation=45, ha='right')
            axes[head].set_yticklabels(tokens)
            
            # Add colorbar
            plt.colorbar(im, ax=axes[head])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_attention_patterns(self, attention_weights, tokens):
        """Plot attention patterns for each query position"""
        seq_len = len(tokens)
        num_heads = attention_weights.shape[0]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for head in range(min(num_heads, 4)):
            for query_pos in range(seq_len):
                weights = attention_weights[head, query_pos].cpu().numpy()
                axes[head].plot(range(seq_len), weights, 'o-', 
                               label=f'Query: {tokens[query_pos]}')
            
            axes[head].set_title(f'Attention Patterns - Head {head + 1}')
            axes[head].set_xlabel('Key Position')
            axes[head].set_ylabel('Attention Weight')
            axes[head].set_xticks(range(seq_len))
            axes[head].set_xticklabels(tokens, rotation=45, ha='right')
            axes[head].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[head].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_attention(self, attention_weights, tokens):
        """Analyze attention patterns"""
        seq_len = len(tokens)
        num_heads = attention_weights.shape[0]
        
        analysis = {}
        
        for head in range(num_heads):
            head_analysis = {}
            
            # Find most attended tokens for each query
            for query_pos in range(seq_len):
                weights = attention_weights[head, query_pos]
                max_weight_pos = torch.argmax(weights).item()
                head_analysis[f"query_{tokens[query_pos]}"] = {
                    'most_attended': tokens[max_weight_pos],
                    'max_weight': weights[max_weight_pos].item(),
                    'attention_distribution': weights.cpu().numpy()
                }
            
            analysis[f'head_{head + 1}'] = head_analysis
        
        return analysis
    
    def interactive_demo(self):
        """Create an interactive attention demo"""
        # Sample sentences for demonstration
        sentences = [
            ["The", "cat", "sat", "on", "the", "mat"],
            ["Attention", "mechanism", "works", "well", "for", "NLP"],
            ["Deep", "learning", "models", "require", "lots", "of", "data"],
            ["Transformers", "revolutionized", "natural", "language", "processing"]
        ]
        
        print("Attention Visualizer Demo")
        print("=" * 40)
        
        for i, sentence in enumerate(sentences):
            print(f"\nAnalyzing sentence {i+1}: {' '.join(sentence)}")
            
            # Visualize attention
            attention_weights = self.visualize_attention(sentence)
            
            # Analyze patterns
            analysis = self.analyze_attention(attention_weights, sentence)
            
            print(f"\nAttention Analysis for Head 1:")
            for query, info in analysis['head_1'].items():
                print(f"  {query.split('_')[1]} -> {info['most_attended']} (weight: {info['max_weight']:.3f})")

# Create and run the attention visualizer
def create_attention_visualizer():
    """Create and demonstrate the attention visualizer"""
    
    print("Creating Attention Visualizer...")
    visualizer = AttentionVisualizer(vocab_size=100, embed_dim=32, num_heads=4)
    
    # Initialize with some meaningful weights (optional)
    with torch.no_grad():
        # Initialize embedding to have some semantic meaning
        visualizer.embedding.weight.normal_(0, 0.1)
    
    # Run interactive demo
    visualizer.interactive_demo()
    
    return visualizer

# Run the attention visualizer project
attention_visualizer = create_attention_visualizer()

# Additional utility: Attention gradient visualization
def visualize_attention_gradients():
    """Visualize how attention weights change during training"""
    
    class SimpleAttentionModel(nn.Module):
        def __init__(self, vocab_size, embed_dim):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.attention = MultiHeadAttention(embed_dim, num_heads=1)
            self.classifier = nn.Linear(embed_dim, 2)
        
        def forward(self, x):
            embedded = self.embedding(x)
            attn_output, _ = self.attention(embedded)
            pooled = attn_output.mean(dim=1)
            return self.classifier(pooled)
    
    # Create model and data
    model = SimpleAttentionModel(100, 32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Sample data
    input_data = torch.randint(0, 100, (4, 6))
    target = torch.randint(0, 2, (4,))
    
    # Track attention weights during training
    attention_history = []
    
    for epoch in range(50):
        optimizer.zero_grad()
        
        # Get attention weights
        embedded = model.embedding(input_data)
        _, attn_weights = model.attention(embedded)
        
        if epoch % 10 == 0:
            attention_history.append(attn_weights[0, 0].detach().cpu().numpy())
        
        # Forward and backward pass
        output = model(input_data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    # Visualize attention evolution
    plt.figure(figsize=(15, 10))
    for i, attn in enumerate(attention_history):
        plt.subplot(2, 3, i+1)
        plt.imshow(attn, cmap='viridis', vmin=0, vmax=1)
        plt.title(f'Epoch {i*10}')
        plt.colorbar()
    
    plt.tight_layout()
    plt.show()

visualize_attention_gradients()
```