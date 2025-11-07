

# Day 27 – Transformer Architecture

## What is the Transformer Architecture and Why is it Important for Deep Learning?

The Transformer is a deep learning architecture introduced in the 2017 Google paper "Attention is All You Need." It was originally designed for machine translation but has since become the dominant architecture for a vast range of sequence-to-sequence tasks and beyond. Its core innovation is the complete abandonment of recurrence and convolution, relying instead entirely on a mechanism called **self-attention** to draw global dependencies between input and output.

Before the Transformer, the state-of-the-art for sequence tasks like translation was the encoder-decoder LSTM (or GRU) architecture. While powerful, these RNN-based models had two major limitations:
1.  **Sequential Computation:** They had to process sequences one element at a time, which prevented parallelization and made training on large datasets very slow.
2.  **Long-Range Dependencies:** Despite improvements like LSTMs, they still struggled to connect information from very distant parts of a long sequence due to the vanishing gradient problem and the bottleneck of the final hidden state.

The Transformer solves these problems:
1.  **Massive Parallelization:** Since it doesn't use recurrence, it can process all input tokens simultaneously, making it highly efficient and scalable on modern hardware like GPUs and TPUs.
2.  **Superior Long-Range Dependencies:** Self-attention allows every token in the sequence to directly attend to every other token. The path length between any two positions is a constant O(1), making it trivial to model relationships between distant words.

In deep learning, the Transformer is fundamentally important because:

-   **It is the foundation of modern Large Language Models (LLMs):** Models like BERT (and its variants like RoBERTa, DistilBERT), GPT (and its successors like GPT-3, GPT-4), T5, and BART are all based on the Transformer architecture. The recent explosion in AI capabilities is directly attributable to the scalability and performance of the Transformer.
-   **It has achieved state-of-the-art results across NLP:** It dominates leaderboards for machine translation, text summarization, question answering, sentiment analysis, and more.
-   **It has expanded beyond NLP:** The principles of self-attention have been successfully applied to other domains, including computer vision (Vision Transformer - ViT), protein folding (AlphaFold), and music generation.
-   **It enables transfer learning at scale:** Pre-trained Transformer models (like BERT and GPT) learn rich representations of language from massive text corpora. These can then be fine-tuned on smaller, task-specific datasets with great success, a paradigm that has become the standard in NLP.

## Essential Subtopics

### 1. Encoder-Decoder

**What is the Encoder-Decoder Architecture?**
The original Transformer, designed for machine translation, follows an encoder-decoder (or seq2seq) structure. This architecture consists of two main components:

1.  **The Encoder:** Its job is to process the entire input sequence (e.g., an English sentence) and compress it into a rich, context-aware numerical representation. It's composed of a stack of identical encoder layers. Each layer has two sub-layers: a multi-head self-attention mechanism and a simple, position-wise feed-forward network. The encoder reads the whole input sequence at once and, for each word, produces a vector that blends information about that word with information about all other words in the sentence.

2.  **The Decoder:** Its job is to take the encoded representation from the encoder and generate the output sequence one token at a time (e.g., the French translation). It's also a stack of identical decoder layers, but each decoder layer has three sub-layers: a multi-head self-attention mechanism (masked to prevent it from "cheating" by looking at future tokens), a multi-head attention mechanism that attends to the *output* of the encoder, and a position-wise feed-forward network. The decoder is auto-regressive, meaning it uses the tokens it has already generated as input to predict the next token.

The flow is: `Input Sequence -> Encoder -> Contextual Representation -> Decoder -> Output Sequence`

**Implementation (Conceptual Encoder-Decoder Structure in PyTorch):**

This code outlines the high-level structure of a Transformer model. We will implement the internal blocks in the next section.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. The Encoder Layer ---
# (We will implement the actual MultiHeadAttention and FeedForward blocks later)
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads) # Placeholder for now
        self.ffn = FeedForward(d_model, d_ff)                 # Placeholder for now
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Self-Attention sub-layer with residual connection and layer norm
        attn_output = self.self_attn(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # Feed-Forward sub-layer with residual connection and layer norm
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)
        return x

# --- 2. The Decoder Layer ---
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads) # Masked self-attention
        self.cross_attn = MultiHeadAttention(d_model, num_heads) # Cross-attention to encoder output
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask, src_mask):
        # Masked Self-Attention sub-layer
        attn_output = self.self_attn(x, x, x, tgt_mask) # tgt_mask prevents looking ahead
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # Cross-Attention sub-layer (attends to encoder's output)
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = x + self.dropout2(attn_output)
        x = self.norm2(x)
        
        # Feed-Forward sub-layer
        ffn_output = self.ffn(x)
        x = x + self.dropout3(ffn_output)
        x = self.norm3(x)
        return x

# --- 3. The Full Encoder and Decoder Stacks ---
class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x, enc_output, tgt_mask, src_mask):
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, src_mask)
        return x

# --- 4. The Full Transformer Model (High-Level View) ---
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_layers, num_heads, d_ff, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model) # Placeholder for now
        
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout)
        
        self.final_layer = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # Embedding and Positional Encoding for source and target
        src = self.encoder_embedding(src)
        src = self.positional_encoding(src)
        
        tgt = self.decoder_embedding(tgt)
        tgt = self.positional_encoding(tgt)
        
        # Pass through Encoder and Decoder
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, tgt_mask, src_mask)
        
        # Final linear layer for prediction
        output = self.final_layer(dec_output)
        return output

print("Conceptual structure of the Encoder-Decoder Transformer defined.")
print("We will implement the core components (MultiHeadAttention, FeedForward, PositionalEncoding) next.")
```

**When and where we use the encoder-decoder structure:**
-   **Machine Translation:** The canonical use case. The encoder processes the source language sentence, and the decoder generates the target language sentence.
-   **Text Summarization:** The encoder reads the full document, and the decoder generates a concise summary.
-   **Question Answering:** The encoder processes the context (e.g., a paragraph), and the decoder generates the answer to the question.
-   **Speech Recognition:** The encoder processes the audio features, and the decoder generates the text transcript.

**Real-world examples:**
-   **Google Translate:** The original application that popularized the Transformer. It uses a massive Transformer-based model to translate between over 100 languages.
-   **Facebook's M2M-100:** A state-of-the-art multilingual machine translation model that can translate directly between any pair of 100 languages without relying on English as an intermediary, all based on the Transformer architecture.
-   **Text-to-SQL models:** Models that take a natural language question as input and generate a SQL query as output. The encoder understands the question, and the decoder generates the structured query.

**Problems and Advancements:**
-   **Problem:** The standard encoder-decoder architecture requires processing the entire input sequence before the decoder can start generating the output. This makes the total latency equal to the sum of the encoding and decoding times, which can be problematic for real-time applications like live conversation translation.
  -   **Advancement:** **Non-Autoregressive Transformers (NAT)**. These models attempt to generate the entire output sequence in parallel, rather than one token at a time. Techniques include iterative refinement (generating a rough draft and then refining it in parallel passes) and using techniques from flow-based models to predict all tokens simultaneously. While they are much faster, they often lag behind autoregressive models in terms of generation quality.
-   **Problem:** The encoder produces a single fixed-size contextual representation for the entire input sequence. For very long documents (e.g., books), this can become a bottleneck, and the decoder might struggle to retrieve specific details from the beginning of the document.
  -   **Advancement:** **Hierarchical Transformers and Memory-Augmented Models**. To handle long documents, models like the **Longformer** or **BigBird** use sparse attention patterns to reduce the quadratic complexity of self-attention. Another approach is to first segment the long document, encode each segment with a Transformer, and then use a higher-level Transformer to summarize and combine these segment representations, creating a hierarchical understanding of the text.

### 2. Positional Encoding

**What is Positional Encoding?**
A key feature of the Transformer architecture is that it is not inherently sequential or recursive—it processes all input tokens in parallel. While this is great for parallelization, it means the model has no built-in sense of word order. The sentence "The cat sat on the mat" and "Mat the on sat cat the" would look identical to the self-attention mechanism if we only fed it word embeddings.

Positional Encoding is the solution to this problem. It's a clever way to inject information about the position of each token in the sequence into the model. The original Transformer uses a sinusoidal positional encoding. For each position `pos` and each dimension `i` of the embedding, the positional encoding is calculated as:

-   `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`
-   `PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`

where `d_model` is the embedding dimension.

This has several advantages:
-   The outputs are bounded between -1 and 1.
-   It produces a unique encoding for each position.
-   For any fixed offset `k`, `PE(pos+k)` can be represented as a linear function of `PE(pos)`, which makes it easy for the model to learn to attend by relative positions.

The final input to the encoder is the sum of the word embedding and the positional encoding.

**Implementation (Sinusoidal Positional Encoding):**

```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a long enough 'pe' matrix
        pe = torch.zeros(max_len, d_model)
        
        # Create a column vector of positions [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create the denominator term: 10000^(2i/d_model)
        # We use log space for numerical stability
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Calculate the sine and cosine terms
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices
        
        # Add a batch dimension and register it as a buffer (not a model parameter)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # Add positional encoding to the input embedding
        # x.shape[1] is the sequence length
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# --- Demonstration ---
# Let's visualize the positional encoding
import matplotlib.pyplot as plt

d_model = 128
pos_encoder = PositionalEncoding(d_model, dropout=0)

# Create a dummy input (batch_size=1, seq_len=50, d_model=128)
dummy_input = torch.zeros(1, 50, d_model)
encoded_input = pos_encoder(dummy_input)

# Plot the positional encoding for the first 20 positions and 20 dimensions
plt.figure(figsize=(12, 8))
plt.pcolormesh(encoded_input[0, :20, :20].T, cmap='viridis')
plt.xlabel('Position')
plt.ylabel('Dimension')
plt.title('Sinusoidal Positional Encoding')
plt.colorbar()
plt.show()

# The plot shows how each position has a unique sinusoidal pattern across the embedding dimensions.
# This allows the model to distinguish between tokens based on their position.
```

**When and where we use positional encoding:**
-   **In every Transformer model.** It is a non-negotiable component of the architecture. Without it, the model has no sense of word order.
-   **In any parallel sequence model.** If you are building a new architecture that processes sequences in parallel and needs to understand order, you will need some form of positional encoding.
-   **In Vision Transformers (ViT).** When an image is split into patches, these patches are treated as a sequence. Positional encoding is added to the patch embeddings to give the model information about where each patch was located in the original image.

**Real-world examples:**
-   **BERT (Bidirectional Encoder Representations from Transformers):** BERT uses the same sinusoidal positional encoding in its encoder stack to understand the order of words in a sentence.
-   **GPT (Generative Pre-trained Transformer):** GPT models also use positional encoding (though later versions like GPT-2 and GPT-3 switched to learned positional embeddings for better performance on very long sequences).
-   **T5 (Text-to-Text Transfer Transformer):** T5, which frames all NLP tasks as text-to-text, uses relative positional encoding, where the model learns to attend based on the relative distance between tokens rather than their absolute positions.

**Problems and Advancements:**
-   **Problem:** The fixed sinusoidal encoding has a limited ability to generalize to sequence lengths longer than those seen during training. The model may struggle to understand positions it has never encountered before.
  -   **Advancement:** **Learned Positional Embeddings**. Instead of using a fixed formula, we can create an embedding matrix where each position index (0, 1, 2, ...) has a corresponding learnable vector, just like word embeddings. This matrix is trained end-to-end with the rest of the model. This is the approach used in models like GPT-2 and BERT. It's more flexible and often performs better, but it may not generalize as well to extreme sequence lengths far outside the training distribution.
    ```python
    # Conceptual Learned Positional Embedding
    # class LearnedPositionalEncoding(nn.Module):
    #     def __init__(self, max_len, d_model):
    #         super().__init__()
    #         self.pos_embedding = nn.Embedding(max_len, d_model)
    #
    #     def forward(self, x):
    #         # Create position indices
    #         pos_indices = torch.arange(0, x.size(1), dtype=torch.long, device=x.device)
    #         return x + self.pos_embedding(pos_indices)
    ```
-   **Problem:** Both sinusoidal and learned absolute positional encodings provide information about a token's absolute position in the sequence, but they don't explicitly tell the model about the *relative distance* between tokens. For many tasks, knowing that token A is 5 positions before token B is more important than knowing that A is at position 10 and B is at position 15.
  -   **Advancement:** **Relative Positional Encoding**. Introduced in models like Transformer-XL and T5, this method modifies the self-attention mechanism to directly incorporate the relative distance between the query and key tokens. Instead of adding positional information to the input embeddings, it's integrated into the attention score calculation. This allows the model to more easily learn patterns based on relative positions, which is more robust and generalizes better to varying sequence lengths.

## 🧪 Exercise: Mini Transformer Block

Now, let's build the core components of a Transformer from scratch: Multi-Head Attention and the Feed-Forward Network. We'll then assemble them into a complete Transformer block.

**Implementation:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 1. Scaled Dot-Product Attention (The core of Multi-Head Attention) ---
def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Computes Scaled Dot-Product Attention.
    Args:
        q: query tensor, shape: (..., seq_len_q, d_k)
        k: key tensor, shape: (..., seq_len_k, d_k)
        v: value tensor, shape: (..., seq_len_v, d_v)
        mask: optional mask tensor, shape: (..., seq_len_q, seq_len_k)
    """
    # Compute dot product of q and k, scaled by sqrt(d_k)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    
    # Apply mask (if provided)
    # Mask is added to scores. For positions we want to mask, the mask value is -1e9.
    # After softmax, these positions will have a score near 0.
    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

    # Apply softmax to get attention weights
    attn_weights = F.softmax(attn_scores, dim=-1)
    
    # Multiply weights by values
    output = torch.matmul(attn_weights, v)
    
    return output, attn_weights

# --- 2. Multi-Head Attention ---
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear layers for Q, K, V projections
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        
        # Output linear layer
        self.wo = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 1. Linear projections and split into heads
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, d_k)
        q = self.wq(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.wk(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.wv(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. Apply scaled dot-product attention to each head
        # attn_output shape: (batch_size, num_heads, seq_len_q, d_k)
        attn_output, attn_weights = scaled_dot_product_attention(q, k, v, mask)
        
        # 3. Concatenate heads
        # (batch_size, num_heads, seq_len_q, d_k) -> (batch_size, seq_len_q, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 4. Final linear projection
        output = self.wo(attn_output)
        
        return output, attn_weights

# --- 3. Position-wise Feed-Forward Network ---
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# --- 4. Assemble a Mini Transformer Block (Encoder Layer) ---
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 1. Multi-Head Attention with Add & Norm
        attn_output, _ = self.mha(x, x, x, mask) # Self-attention: Q, K, V all come from x
        x = x + self.dropout1(attn_output)  # Residual connection
        x = self.norm1(x)                  # Layer normalization
        
        # 2. Feed-Forward Network with Add & Norm
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)  # Residual connection
        x = self.norm2(x)                  # Layer normalization
        
        return x

# --- 5. Demonstration ---
if __name__ == "__main__":
    # Hyperparameters
    d_model = 512
    num_heads = 8
    d_ff = 2048
    batch_size = 32
    seq_len = 10

    # Create a dummy input tensor (e.g., a batch of embedded sentences)
    dummy_input = torch.randn(batch_size, seq_len, d_model)
    print(f"Dummy input shape: {dummy_input.shape}")

    # Instantiate our mini Transformer block
    transformer_block = TransformerEncoderBlock(d_model, num_heads, d_ff)
    
    # Pass the input through the block
    output = transformer_block(dummy_input)
    
    print(f"\nOutput shape from Transformer block: {output.shape}")
    print("\nSuccess! The output shape matches the input shape, allowing for stacking.")
    print("This single block contains the core components of a Transformer encoder:")
    print("- Multi-Head Self-Attention")
    print("- Position-wise Feed-Forward Network")
    print("- Residual Connections")
    print("- Layer Normalization")
```

## 📌 Project: Build Transformer from Scratch

Now for the main event. We will combine all the pieces we've built—positional encoding, multi-head attention, and feed-forward networks—to construct a complete, functional Transformer model from the ground up, just like in the "Attention is All You Need" paper.

**Implementation (`transformer_from_scratch.py`):**

This script will define all the necessary components and assemble them into a full `Transformer` class. We'll then test it on a dummy sequence-to-sequence task to ensure it runs correctly.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

# ===================================================================
# 1. CORE COMPONENTS (Copied from the exercise for completeness)
# ===================================================================

def scaled_dot_product_attention(q, k, v, mask=None):
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
    attn_weights = F.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    return output, attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.num_heads = num_heads

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q = self.wq(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.wk(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.wv(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        attn_output, _ = scaled_dot_product_attention(q, k, v, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.wo(attn_output)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# ===================================================================
# 2. ENCODER AND DECODER LAYERS
# ===================================================================

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask, src_mask):
        # Masked self-attention
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        # Cross-attention with encoder output
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout2(attn_output))
        # Feed-forward
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_output))
        return x

# ===================================================================
# 3. FULL ENCODER AND DECODER STACKS
# ===================================================================

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(EncoderLayer(d_model, num_heads, d_ff, dropout)) for _ in range(num_layers)])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(DecoderLayer(d_model, num_heads, d_ff, dropout)) for _ in range(num_layers)])

    def forward(self, x, enc_output, tgt_mask, src_mask):
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, src_mask)
        return x

# ===================================================================
# 4. THE COMPLETE TRANSFORMER MODEL
# ===================================================================

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048, dropout=0.1):
        super().__init__()
        # Define hyperparameters
        self.d_model = d_model
        
        # Embedding layers
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        # Encoder and Decoder stacks
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout)
        
        # Final linear layer for prediction
        self.final_layer = nn.Linear(d_model, tgt_vocab_size)
        
        # Initialize parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_padding_mask(self, seq):
        # seq: (batch_size, seq_len)
        mask = (seq != 0).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
        return mask

    def create_look_ahead_mask(self, size):
        # Creates a mask to hide future tokens
        mask = torch.triu(torch.ones(size, size) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt):
        # src, tgt: (batch_size, seq_len)
        
        # Create masks
        src_mask = self.create_padding_mask(src)
        tgt_mask = self.create_padding_mask(tgt)
        tgt_look_ahead_mask = self.create_look_ahead_mask(tgt.size(1))
        tgt_mask = tgt_mask & tgt_look_ahead_mask # Combine padding and look-ahead masks
        
        # Embedding and Positional Encoding
        src = self.encoder_embedding(src) * math.sqrt(self.d_model)
        src = self.positional_encoding(src)
        
        tgt = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.positional_encoding(tgt)
        
        # Pass through Encoder and Decoder
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, tgt_mask, src_mask)
        
        # Final prediction
        output = self.final_layer(dec_output)
        return output

# ===================================================================
# 5. DEMONSTRATION
# ===================================================================
if __name__ == "__main__":
    # Hyperparameters
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    batch_size = 32
    seq_len = 10

    print("Building a complete Transformer from scratch...")
    
    # Instantiate the model
    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff)
    print("Transformer model instantiated successfully.")
    print(f"Model has {sum(p.numel() for p in transformer.parameters()):,} trainable parameters.")
    
    # Create dummy input data (token indices)
    # We use 0 as padding token
    dummy_src = torch.randint(1, src_vocab_size, (batch_size, seq_len))
    dummy_tgt = torch.randint(1, tgt_vocab_size, (batch_size, seq_len))
    
    # For demonstration, let's make the first sequence shorter to test padding mask
    dummy_src[0, 5:] = 0
    dummy_tgt[0, 7:] = 0

    print(f"\nDummy source shape: {dummy_src.shape}")
    print(f"Dummy target shape: {dummy_tgt.shape}")
    
    # Forward pass
    print("\nPerforming a forward pass...")
    output = transformer(dummy_src, dummy_tgt)
    
    print(f"\nOutput shape from the full Transformer: {output.shape}")
    print("\nSuccess! The full Transformer model is working.")
    print("The output can be passed through a softmax to get probabilities for the next token.")
    
    # Example of getting the next token prediction for the first batch, last sequence position
    # This is what happens during autoregressive decoding
    next_token_logits = output[0, -1, :]
    predicted_token_id = torch.argmax(next_token_logits, dim=-1)
    print(f"\nPredicted next token ID for the first sequence: {predicted_token_id.item()}")
```

**How to Run and Understand the Project:**

1.  **Save the Code:** Save the code above as `transformer_from_scratch.py`.
2.  **Install Dependencies:**
    ```bash
    pip install torch
    ```
3.  **Run the Script:**
    ```bash
    python transformer_from_scratch.py
    ```

**What the Output Means:**

*   You will see the model being instantiated, with a printout of its total number of parameters (which should be in the tens of millions, similar to the original "base" Transformer).
*   It will create dummy source and target sequences, including some padding (zeros) to demonstrate the masking functionality.
*   It will perform a full forward pass through the entire model.
*   The final output shape will be `(batch_size, seq_len, tgt_vocab_size)`. This is the correct shape. For each token in the target sequence, the model outputs a vector of logits (un-normalized scores) over the entire target vocabulary.
*   To get a prediction, you would apply a `softmax` function to these logits to get a probability distribution and then take the `argmax` to find the most likely next token. The script demonstrates this for the very last token of the first sequence in the batch.
