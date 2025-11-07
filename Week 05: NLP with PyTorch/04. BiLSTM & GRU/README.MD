# Day 24 – BiLSTM & GRU

## BiLSTM (Bidirectional LSTM)

### What is it?

Bidirectional Long Short-Term Memory (BiLSTM) is an extension of the standard LSTM architecture that processes input sequences in both forward and backward directions simultaneously. This allows the model to access both past and future context for each time step, making it particularly effective for tasks where the entire sequence context is important, such as named entity recognition, sentiment analysis, and machine translation.

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Basic BiLSTM implementation
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=1, dropout=0.3):
        super(BiLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = 2  # Forward and backward
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,  # Key parameter for bidirectional
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim * self.num_directions, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # lstm_out shape: (batch_size, seq_len, hidden_dim * 2) - *2 for bidirectional
        
        # Take the output from the last time step
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim * 2)
        
        # Apply dropout and final linear layer
        output = self.dropout(last_output)
        output = self.fc(output)
        
        return output

# Demonstrate BiLSTM functionality
def demonstrate_bilstm():
    # Sample parameters
    vocab_size = 1000
    embedding_dim = 128
    hidden_dim = 64
    output_dim = 2  # Binary classification
    
    # Create model
    model = BiLSTM(vocab_size, embedding_dim, hidden_dim, output_dim)
    
    # Sample input (batch_size=4, seq_len=10)
    sample_input = torch.randint(1, vocab_size, (4, 10))
    
    # Forward pass
    output = model(sample_input)
    
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

model = demonstrate_bilstm()

# Advanced BiLSTM with attention mechanism
class BiLSTMWithAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=1, dropout=0.3):
        super(BiLSTMWithAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = 2
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, 1)  # 2 for bidirectional
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Apply attention
        attention_weights = torch.tanh(lstm_out)
        attention_weights = self.attention(attention_weights).squeeze(-1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention to LSTM output
        attended_output = torch.bmm(lstm_out.transpose(1, 2), attention_weights.unsqueeze(-1)).squeeze(-1)
        
        # Final classification
        output = self.dropout(attended_output)
        output = self.fc(output)
        
        return output, attention_weights

def demonstrate_bilstm_attention():
    vocab_size = 1000
    embedding_dim = 128
    hidden_dim = 64
    output_dim = 2
    
    model = BiLSTMWithAttention(vocab_size, embedding_dim, hidden_dim, output_dim)
    sample_input = torch.randint(1, vocab_size, (4, 10))
    
    output, attention_weights = model(sample_input)
    
    print(f"BiLSTM with Attention - Input: {sample_input.shape}, Output: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    return model

attention_model = demonstrate_bilstm_attention()
```

### When and where we use BiLSTM

- **Named Entity Recognition (NER)**: Accessing both left and right context for entity classification
- **Sentiment Analysis**: Understanding sentiment with full sentence context
- **Machine Translation**: Processing source sentences with bidirectional context
- **Part-of-speech Tagging**: Using both past and future words for accurate tagging
- **Question Answering**: Understanding questions and context with full sequence information
- **Text Classification**: Capturing comprehensive sequence patterns

```python
# When to use BiLSTM - Sentiment analysis example
def sentiment_analysis_bilstm():
    """BiLSTM for sentiment analysis with bidirectional context"""
    
    class SentimentBiLSTM(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, max_len=100):
            super().__init__()
            
            self.max_len = max_len
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            
            # Bidirectional LSTM for context
            self.lstm = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers=2,
                bidirectional=True,
                dropout=0.3,
                batch_first=True
            )
            
            # Self-attention for important word focus
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim * 2,  # 2 for bidirectional
                num_heads=8,
                dropout=0.1
            )
            
            # Classification layers
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, output_dim)
            )
        
        def forward(self, x):
            # Embedding
            embedded = self.embedding(x)  # (batch, seq, embed_dim)
            
            # BiLSTM
            lstm_out, (hidden, cell) = self.lstm(embedded)  # (batch, seq, hidden*2)
            
            # Self-attention
            attn_out, _ = self.attention(
                lstm_out.transpose(0, 1),  # (seq, batch, hidden*2)
                lstm_out.transpose(0, 1),
                lstm_out.transpose(0, 1)
            )
            attn_out = attn_out.transpose(0, 1)  # (batch, seq, hidden*2)
            
            # Global max pooling
            pooled = torch.max(attn_out, dim=1)[0]  # (batch, hidden*2)
            
            # Classification
            output = self.classifier(pooled)
            return output
    
    return SentimentBiLSTM

sentiment_model_class = sentiment_analysis_bilstm()
```

### Real-world examples

- **Google Translate**: Using bidirectional context for accurate translations
- **Twitter sentiment analysis**: Understanding tweets with full context
- **Medical text analysis**: Processing clinical notes with bidirectional understanding
- **Legal document classification**: Analyzing contracts with full document context
- **Chatbot understanding**: Processing user queries with bidirectional context

```python
# Real-world example: Medical text analysis with BiLSTM
def medical_text_analyzer():
    """BiLSTM for medical text analysis"""
    
    class MedicalBiLSTM(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
            super().__init__()
            
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            
            # Multiple BiLSTM layers for complex medical text
            self.bilstm_layers = nn.ModuleList([
                nn.LSTM(
                    embedding_dim if i == 0 else hidden_dim * 2,
                    hidden_dim,
                    num_layers=1,
                    bidirectional=True,
                    batch_first=True,
                    dropout=0.2 if i < 2 else 0
                )
                for i in range(3)
            ])
            
            # Classification head
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )
        
        def forward(self, x):
            embedded = self.embedding(x)
            lstm_out = embedded
            
            # Stack BiLSTM layers
            for i, lstm_layer in enumerate(self.bilstm_layers):
                lstm_out, _ = lstm_layer(lstm_out)
            
            # Global average pooling
            pooled = torch.mean(lstm_out, dim=1)
            
            # Classification
            output = self.classifier(pooled)
            return output
    
    return MedicalBiLSTM

medical_analyzer = medical_text_analyzer()
```

### Problems and Advancements

**Problem**: Computational complexity increases due to processing in both directions.
**Advancement**: Efficient implementations and optimized architectures like Transformer-XL.

```python
# Problem: Computational overhead in BiLSTM
def computational_comparison():
    """Compare computational requirements"""
    
    class UniLSTM(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)
        
        def forward(self, x):
            embedded = self.embedding(x)
            lstm_out, _ = self.lstm(embedded)
            return self.fc(lstm_out[:, -1, :])
    
    class BiLSTM(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
            self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional
        
        def forward(self, x):
            embedded = self.embedding(x)
            lstm_out, _ = self.lstm(embedded)
            return self.fc(lstm_out[:, -1, :])
    
    # Compare parameters
    unilstm = UniLSTM(1000, 128, 64, 2)
    bilstm = BiLSTM(1000, 128, 64, 2)
    
    print(f"UniLSTM parameters: {sum(p.numel() for p in unilstm.parameters()):,}")
    print(f"BiLSTM parameters: {sum(p.numel() for p in bilstm.parameters()):,}")
    print(f"Increase: {((sum(p.numel() for p in bilstm.parameters()) - sum(p.numel() for p in unilstm.parameters())) / sum(p.numel() for p in unilstm.parameters()) * 100):.1f}%")

computational_comparison()

# Solution: Optimized BiLSTM with layer normalization
class OptimizedBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Multiple layers with layer normalization
        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            input_dim = embedding_dim if i == 0 else hidden_dim * 2
            lstm = nn.LSTM(
                input_dim,
                hidden_dim,
                bidirectional=True,
                batch_first=True
            )
            ln = nn.LayerNorm(hidden_dim * 2)
            self.lstm_layers.extend([lstm, ln])
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out = embedded
        
        # Apply LSTM layers with layer normalization
        for i in range(0, len(self.lstm_layers), 2):
            lstm_layer = self.lstm_layers[i]
            ln_layer = self.lstm_layers[i + 1]
            
            lstm_out, _ = lstm_layer(lstm_out)
            lstm_out = ln_layer(lstm_out)
        
        output = self.dropout(lstm_out[:, -1, :])
        return self.fc(output)

optimized_model = OptimizedBiLSTM(1000, 128, 64, 2)
```

## GRU (Gated Recurrent Unit)

### What is it?

Gated Recurrent Unit (GRU) is a simplified version of LSTM that uses fewer gates and parameters while maintaining similar performance. GRU combines the forget and input gates into a single "update gate" and merges the cell state and hidden state. This makes GRU computationally more efficient than LSTM while still effectively handling long-term dependencies in sequential data.

### PyTorch Implementation

```python
# Basic GRU implementation
class GRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=1, dropout=0.3):
        super(GRU, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # GRU layer
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # GRU forward pass
        gru_out, hidden = self.gru(embedded)
        # gru_out shape: (batch_size, seq_len, hidden_dim)
        # hidden shape: (num_layers, batch_size, hidden_dim)
        
        # Take the output from the last time step
        last_output = gru_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # Apply dropout and final linear layer
        output = self.dropout(last_output)
        output = self.fc(output)
        
        return output

# Demonstrate GRU functionality
def demonstrate_gru():
    vocab_size = 1000
    embedding_dim = 128
    hidden_dim = 64
    output_dim = 2
    
    model = GRU(vocab_size, embedding_dim, hidden_dim, output_dim)
    sample_input = torch.randint(1, vocab_size, (4, 10))
    
    output = model(sample_input)
    
    print(f"GRU Input shape: {sample_input.shape}")
    print(f"GRU Output shape: {output.shape}")
    print(f"GRU Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

gru_model = demonstrate_gru()

# Bidirectional GRU implementation
class BiGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=1, dropout=0.3):
        super(BiGRU, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = 2  # Forward and backward
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Bidirectional GRU
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,  # Key parameter for bidirectional
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim * self.num_directions, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, hidden = self.gru(embedded)
        
        # Take the output from the last time step
        last_output = gru_out[:, -1, :]  # (batch_size, hidden_dim * 2)
        
        # Apply dropout and final linear layer
        output = self.dropout(last_output)
        output = self.fc(output)
        
        return output

def demonstrate_bigru():
    vocab_size = 1000
    embedding_dim = 128
    hidden_dim = 64
    output_dim = 2
    
    model = BiGRU(vocab_size, embedding_dim, hidden_dim, output_dim)
    sample_input = torch.randint(1, vocab_size, (4, 10))
    
    output = model(sample_input)
    
    print(f"BiGRU Input shape: {sample_input.shape}")
    print(f"BiGRU Output shape: {output.shape}")
    print(f"BiGRU Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

bigru_model = demonstrate_bigru()

# Advanced GRU with attention mechanism
class GRUWithAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=1, dropout=0.3):
        super(GRUWithAttention, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, hidden = self.gru(embedded)
        
        # Apply attention
        attention_weights = torch.tanh(gru_out)
        attention_weights = self.attention(attention_weights).squeeze(-1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention to GRU output
        attended_output = torch.bmm(gru_out.transpose(1, 2), attention_weights.unsqueeze(-1)).squeeze(-1)
        
        # Final classification
        output = self.dropout(attended_output)
        output = self.fc(output)
        
        return output, attention_weights

def demonstrate_gru_attention():
    vocab_size = 1000
    embedding_dim = 128
    hidden_dim = 64
    output_dim = 2
    
    model = GRUWithAttention(vocab_size, embedding_dim, hidden_dim, output_dim)
    sample_input = torch.randint(1, vocab_size, (4, 10))
    
    output, attention_weights = model(sample_input)
    
    print(f"GRU with Attention - Input: {sample_input.shape}, Output: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    return model

gru_attention_model = demonstrate_gru_attention()
```

### When and where we use GRU

- **Speech recognition**: Processing audio sequences efficiently
- **Machine translation**: Handling long sequences with fewer parameters than LSTM
- **Text generation**: Generating coherent text with manageable computational cost
- **Time series prediction**: Modeling temporal dependencies with efficiency
- **Chatbots**: Processing conversation history with minimal computational overhead
- **Real-time applications**: When speed is crucial and LSTM is too slow

```python
# When to use GRU - Time series prediction example
def timeseries_gru():
    """GRU for time series prediction"""
    
    class TimeSeriesGRU(nn.Module):
        def __init__(self, input_size, hidden_dim, output_size, num_layers=2):
            super().__init__()
            
            self.gru = nn.GRU(
                input_size,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.2
            )
            
            self.fc = nn.Linear(hidden_dim, output_size)
            self.dropout = nn.Dropout(0.2)
        
        def forward(self, x):
            # x shape: (batch_size, seq_len, input_size)
            gru_out, hidden = self.gru(x)
            
            # Use last output for prediction
            output = self.dropout(gru_out[:, -1, :])
            output = self.fc(output)
            
            return output
    
    return TimeSeriesGRU

timeseries_model = timeseries_gru()
```

### Real-world examples

- **Google Assistant**: Processing voice commands with GRU for efficiency
- **Netflix recommendations**: Modeling user behavior sequences with GRU
- **Stock price prediction**: Predicting prices based on historical data
- **Customer behavior analysis**: Understanding purchase sequences
- **IoT sensor data**: Processing time-series sensor readings efficiently

```python
# Real-world example: Stock price prediction with GRU
def stock_prediction_gru():
    """GRU for stock price prediction"""
    
    class StockGRU(nn.Module):
        def __init__(self, input_size, hidden_dim, output_size, num_layers=2):
            super().__init__()
            
            self.gru = nn.GRU(
                input_size,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.1
            )
            
            # Multiple heads for different predictions
            self.price_head = nn.Linear(hidden_dim, 1)  # Next price
            self.volume_head = nn.Linear(hidden_dim, 1)  # Next volume
            self.direction_head = nn.Linear(hidden_dim, 2)  # Up/Down classification
            
        def forward(self, x):
            gru_out, hidden = self.gru(x)
            last_hidden = gru_out[:, -1, :]  # (batch_size, hidden_dim)
            
            price_pred = self.price_head(last_hidden)
            volume_pred = self.volume_head(last_hidden)
            direction_pred = self.direction_head(last_hidden)
            
            return {
                'price': price_pred,
                'volume': volume_pred,
                'direction': direction_pred
            }
    
    return StockGRU

stock_model = stock_prediction_gru()
```

### Problems and Advancements

**Problem**: GRU can still struggle with very long sequences and vanishing gradients.
**Advancement**: Advanced architectures like Transformer models and improved GRU variants.

```python
# Problem: Vanishing gradients in long sequences
def long_sequence_problem():
    """Demonstrate the challenge with long sequences"""
    
    class SimpleGRU(nn.Module):
        def __init__(self, input_size, hidden_dim, output_size):
            super().__init__()
            self.gru = nn.GRU(input_size, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_size)
    
    class ImprovedGRU(nn.Module):
        def __init__(self, input_size, hidden_dim, output_size, num_layers=2):
            super().__init__()
            
            # Multiple layers with residual connections
            self.gru_layers = nn.ModuleList([
                nn.GRU(
                    input_size if i == 0 else hidden_dim,
                    hidden_dim,
                    batch_first=True
                )
                for i in range(num_layers)
            ])
            
            self.fc = nn.Linear(hidden_dim, output_size)
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in range(num_layers)
            ])
        
        def forward(self, x):
            gru_out = x
            
            for i, (gru_layer, ln_layer) in enumerate(zip(self.gru_layers, self.layer_norms)):
                gru_out, _ = gru_layer(gru_out)
                gru_out = ln_layer(gru_out)
            
            return self.fc(gru_out[:, -1, :])
    
    simple_gru = SimpleGRU(50, 128, 1)
    improved_gru = ImprovedGRU(50, 128, 1, num_layers=3)
    
    print(f"Simple GRU parameters: {sum(p.numel() for p in simple_gru.parameters()):,}")
    print(f"Improved GRU parameters: {sum(p.numel() for p in improved_gru.parameters()):,}")

long_sequence_problem()

# Solution: GRU with highway connections
class HighwayGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # GRU with highway connections
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        
        # Highway network for information flow
        self.highway_transform = nn.Linear(hidden_dim, hidden_dim)
        self.highway_gate = nn.Linear(hidden_dim, hidden_dim)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, _ = self.gru(embedded)
        
        # Apply highway connections
        highway_input = gru_out[:, -1, :]  # Last time step
        transform = torch.tanh(self.highway_transform(highway_input))
        gate = torch.sigmoid(self.highway_gate(highway_input))
        
        # Highway connection: gate * transform + (1 - gate) * input
        highway_output = gate * transform + (1 - gate) * highway_input
        
        output = self.dropout(highway_output)
        return self.fc(output)

highway_gru = HighwayGRU(1000, 128, 64, 2)
```

## 🧪 Exercise: Sentiment classifier

```python
def sentiment_classifier_exercise():
    """Complete sentiment classifier exercise with BiLSTM and GRU"""
    
    # Sample vocabulary and data
    vocab_size = 5000
    embedding_dim = 100
    hidden_dim = 128
    output_dim = 2  # Positive/Negative
    
    # BiLSTM sentiment classifier
    class BiLSTMSentimentClassifier(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
            super().__init__()
            
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            
            # Bidirectional LSTM
            self.lstm = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers=2,
                bidirectional=True,
                dropout=0.3,
                batch_first=True
            )
            
            # Attention mechanism
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim * 2,
                num_heads=8,
                dropout=0.1
            )
            
            # Classification layers
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, output_dim)
            )
        
        def forward(self, x):
            embedded = self.embedding(x)
            lstm_out, (hidden, cell) = self.lstm(embedded)
            
            # Self-attention
            attn_out, _ = self.attention(
                lstm_out.transpose(0, 1),
                lstm_out.transpose(0, 1),
                lstm_out.transpose(0, 1)
            )
            attn_out = attn_out.transpose(0, 1)
            
            # Global max pooling
            pooled = torch.max(attn_out, dim=1)[0]
            
            return self.classifier(pooled)
    
    # GRU sentiment classifier
    class GRUSentimentClassifier(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
            super().__init__()
            
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            
            # GRU
            self.gru = nn.GRU(
                embedding_dim,
                hidden_dim,
                num_layers=2,
                dropout=0.3,
                batch_first=True
            )
            
            # Classification layers
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, output_dim)
            )
        
        def forward(self, x):
            embedded = self.embedding(x)
            gru_out, hidden = self.gru(embedded)
            
            # Global max pooling
            pooled = torch.max(gru_out, dim=1)[0]
            
            return self.classifier(pooled)
    
    # Create models
    bilstm_model = BiLSTMSentimentClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
    gru_model = GRUSentimentClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
    
    # Sample input (batch_size=8, seq_len=20)
    sample_input = torch.randint(1, vocab_size, (8, 20))
    
    print("Sentiment Classifier Exercise:")
    print("=" * 40)
    
    # Test BiLSTM
    bilstm_output = bilstm_model(sample_input)
    print(f"BiLSTM - Input: {sample_input.shape}, Output: {bilstm_output.shape}")
    print(f"BiLSTM Parameters: {sum(p.numel() for p in bilstm_model.parameters()):,}")
    
    # Test GRU
    gru_output = gru_model(sample_input)
    print(f"GRU - Input: {sample_input.shape}, Output: {gru_output.shape}")
    print(f"GRU Parameters: {sum(p.numel() for p in gru_model.parameters()):,}")
    
    # Compare architectures
    print(f"\nParameter comparison:")
    print(f"BiLSTM vs GRU: {((sum(p.numel() for p in bilstm_model.parameters()) - sum(p.numel() for p in gru_model.parameters())) / sum(p.numel() for p in gru_model.parameters()) * 100):.1f}% more parameters in BiLSTM")
    
    return bilstm_model, gru_model

bilstm_model, gru_model = sentiment_classifier_exercise()
```

## 📌 Project: IMDB sentiment analyzer

```python
class IMDBSentimentAnalyzer:
    """Complete IMDB sentiment analyzer project using BiLSTM and GRU"""
    
    def __init__(self, vocab_size=10000, embedding_dim=128, hidden_dim=128, max_len=500):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        
        # BiLSTM model
        self.bilstm_model = self._create_bilstm_model()
        
        # GRU model
        self.gru_model = self._create_gru_model()
        
        # Optimizers
        self.bilstm_optimizer = optim.Adam(self.bilstm_model.parameters(), lr=0.001)
        self.gru_optimizer = optim.Adam(self.gru_model.parameters(), lr=0.001)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
    def _create_bilstm_model(self):
        """Create BiLSTM model for sentiment analysis"""
        class BiLSTMModel(nn.Module):
            def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim=2):
                super().__init__()
                
                self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
                
                # Bidirectional LSTM
                self.lstm = nn.LSTM(
                    embedding_dim,
                    hidden_dim,
                    num_layers=2,
                    bidirectional=True,
                    dropout=0.3,
                    batch_first=True
                )
                
                # Attention mechanism
                self.attention = nn.Linear(hidden_dim * 2, 1)
                
                # Classification layers
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim * 2, 128),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, output_dim)
                )
            
            def forward(self, x):
                embedded = self.embedding(x)
                lstm_out, (hidden, cell) = self.lstm(embedded)
                
                # Apply attention
                attention_weights = torch.tanh(lstm_out)
                attention_weights = self.attention(attention_weights).squeeze(-1)
                attention_weights = F.softmax(attention_weights, dim=1)
                
                # Apply attention to LSTM output
                attended_output = torch.bmm(lstm_out.transpose(1, 2), attention_weights.unsqueeze(-1)).squeeze(-1)
                
                return self.classifier(attended_output)
        
        return BiLSTMModel(self.vocab_size, self.embedding_dim, self.hidden_dim)
    
    def _create_gru_model(self):
        """Create GRU model for sentiment analysis"""
        class GRUModel(nn.Module):
            def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim=2):
                super().__init__()
                
                self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
                
                # GRU
                self.gru = nn.GRU(
                    embedding_dim,
                    hidden_dim,
                    num_layers=2,
                    dropout=0.3,
                    batch_first=True
                )
                
                # Classification layers
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim, 128),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, output_dim)
                )
            
            def forward(self, x):
                embedded = self.embedding(x)
                gru_out, hidden = self.gru(embedded)
                
                # Global max pooling
                pooled = torch.max(gru_out, dim=1)[0]
                
                return self.classifier(pooled)
        
        return GRUModel(self.vocab_size, self.embedding_dim, self.hidden_dim)
    
    def preprocess_text(self, text, word_to_idx, max_len=500):
        """Preprocess text for model input"""
        # Simple tokenization and conversion to indices
        tokens = text.lower().split()
        indices = [word_to_idx.get(token, 1) for token in tokens]  # 1 for unknown
        
        # Pad or truncate
        if len(indices) < max_len:
            indices.extend([0] * (max_len - len(indices)))  # 0 for padding
        else:
            indices = indices[:max_len]
        
        return torch.tensor(indices, dtype=torch.long)
    
    def train_step(self, model, optimizer, batch_x, batch_y):
        """Single training step"""
        model.train()
        optimizer.zero_grad()
        
        outputs = model(batch_x)
        loss = self.criterion(outputs, batch_y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        return loss.item()
    
    def evaluate(self, model, test_loader):
        """Evaluate model performance"""
        model.eval()
        correct = 0
        total = 0
        total_loss = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                
                predicted = torch.argmax(outputs, dim=1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(test_loader)
        
        return accuracy, avg_loss
    
    def predict_sentiment(self, model, text, word_to_idx):
        """Predict sentiment for a single text"""
        model.eval()
        with torch.no_grad():
            # Preprocess text
            input_tensor = self.preprocess_text(text, word_to_idx, self.max_len).unsqueeze(0)
            
            # Get prediction
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return predicted_class, confidence
    
    def save_model(self, model, filepath):
        """Save model to file"""
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__
        }, filepath)
    
    def load_model(self, model, filepath):
        """Load model from file"""
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

# Example usage and demonstration
def demonstrate_imdb_analyzer():
    """Demonstrate the IMDB sentiment analyzer"""
    
    print("IMDB Sentiment Analyzer Project")
    print("=" * 40)
    
    # Create analyzer
    analyzer = IMDBSentimentAnalyzer()
    
    print(f"BiLSTM Model Parameters: {sum(p.numel() for p in analyzer.bilstm_model.parameters()):,}")
    print(f"GRU Model Parameters: {sum(p.numel() for p in analyzer.gru_model.parameters()):,}")
    
    # Sample vocabulary mapping (in real application, this would be built from training data)
    sample_vocab = {f"word_{i}": i for i in range(1, 1000)}  # Sample vocabulary
    
    # Sample texts for prediction
    sample_texts = [
        "This movie is absolutely fantastic! Great acting and amazing plot.",
        "Terrible film, waste of time. Poor acting and boring story.",
        "The movie was okay, nothing special but not bad either.",
        "Outstanding performance by the lead actor, highly recommended!"
    ]
    
    print("\nSample Predictions:")
    print("-" * 30)
    
    for text in sample_texts:
        # Predict with BiLSTM
        bilstm_pred, bilstm_conf = analyzer.predict_sentiment(analyzer.bilstm_model, text, sample_vocab)
        # Predict with GRU
        gru_pred, gru_conf = analyzer.predict_sentiment(analyzer.gru_model, text, sample_vocab)
        
        sentiment_bilstm = "Positive" if bilstm_pred == 1 else "Negative"
        sentiment_gru = "Positive" if gru_pred == 1 else "Negative"
        
        print(f"Text: {text[:50]}...")
        print(f"  BiLSTM: {sentiment_bilstm} (confidence: {bilstm_conf:.3f})")
        print(f"  GRU: {sentiment_gru} (confidence: {gru_conf:.3f})")
        print()
    
    # Model comparison
    print("Model Architecture Comparison:")
    print("- BiLSTM: Better at capturing long-range dependencies with bidirectional context")
    print("- GRU: More efficient with fewer parameters, faster training")
    print("- Both: Effective for sentiment analysis with proper attention mechanisms")
    
    return analyzer

# Run the demonstration
imdb_analyzer = demonstrate_imdb_analyzer()

# Additional utility functions
def model_comparison_summary():
    """Provide a summary of BiLSTM vs GRU comparison"""
    
    print("\nBiLSTM vs GRU: Detailed Comparison")
    print("=" * 40)
    
    comparison = {
        "Parameters": {
            "BiLSTM": "More parameters due to separate input, forget, and output gates",
            "GRU": "Fewer parameters with combined reset and update gates"
        },
        "Computational Efficiency": {
            "BiLSTM": "Slower due to more complex gating mechanism",
            "GRU": "Faster with simpler architecture"
        },
        "Memory Usage": {
            "BiLSTM": "Higher memory usage with separate cell state",
            "GRU": "Lower memory usage with combined hidden state"
        },
        "Performance": {
            "BiLSTM": "Better for tasks requiring precise long-term memory",
            "GRU": "Competitive performance with better efficiency"
        },
        "Use Cases": {
            "BiLSTM": "NER, machine translation, complex sequence tasks",
            "GRU": "Real-time applications, resource-constrained environments"
        }
    }
    
    for aspect, details in comparison.items():
        print(f"\n{aspect}:")
        print(f"  BiLSTM: {details['BiLSTM']}")
        print(f"  GRU: {details['GRU']}")

model_comparison_summary()
```