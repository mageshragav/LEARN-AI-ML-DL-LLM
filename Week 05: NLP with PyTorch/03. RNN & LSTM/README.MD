

# Day 23 – RNN & LSTM

## What are RNNs and LSTMs and Why are they Important for Deep Learning?

Recurrent Neural Networks (RNNs) are a class of neural networks designed to recognize patterns in sequences of data, such as text, genomes, handwriting, or spoken words. Unlike standard feedforward neural networks, RNNs have a "memory" that captures information about what has been calculated so far. They achieve this by having loops in their architecture, allowing information to persist.

In a traditional neural network, we assume that all inputs (and outputs) are independent of each other. But for many tasks, this is a poor assumption. If you want to predict the next word in a sentence, it's helpful to know the words that came before it. RNNs address this by processing sequences one element at a time, maintaining a hidden state that acts as a "memory" of the information they have seen so far.

However, standard RNNs suffer from a critical problem: the **vanishing gradient problem**. When training RNNs using backpropagation through time (BPTT), the gradients can become exponentially small as they are propagated back through many time steps. This means the network effectively "forgets" information from the distant past and is unable to learn long-range dependencies.

**Long Short-Term Memory (LSTM) networks** are a special kind of RNN, capable of learning long-term dependencies. They were explicitly designed to combat the vanishing gradient problem. LSTMs have a more complex unit than a standard RNN, called a "memory cell," which is regulated by three "gates":
1.  **Forget Gate:** Decides what information to throw away from the cell state.
2.  **Input Gate:** Decides what new information to store in the cell state.
3.  **Output Gate:** Decides what to output based on the cell state.

This gated mechanism allows LSTMs to selectively remember or forget information over long sequences, making them incredibly powerful for sequence modeling tasks.

In deep learning, RNNs and LSTMs are fundamentally important because:

-   **They are the primary architecture for sequential data:** Before Transformers, LSTMs were the state-of-the-art for almost all NLP tasks.
-   **They handle variable-length inputs:** Unlike CNNs or standard feedforward networks that require fixed-size inputs, RNNs can process sequences of any length.
-   **They model temporal dependencies:** They are essential for any task where the order of data matters, such as time-series forecasting or language modeling.
-   **They form the foundation for more advanced models:** Understanding RNNs and LSTMs is crucial for grasping the innovations introduced by attention mechanisms and Transformers.

## Essential Subtopics

### 1. Sequence Modeling

**What is Sequence Modeling?**
Sequence modeling is the task of predicting the next element in a sequence of data. The data can be anything from words in a sentence, stock prices over time, musical notes, or sensor readings. The core idea is that the elements in a sequence are not independent; the value of an element at position `t` is dependent on the elements at positions `t-1, t-2, ...`.

RNNs are naturally suited for sequence modeling because their recurrent connection allows them to maintain a state or "memory" of previous elements in the sequence. This hidden state `h_t` is a function of the current input `x_t` and the previous hidden state `h_{t-1}`: `h_t = f(W_x * x_t + W_h * h_{t-1} + b)`.

**PyTorch Implementation (A simple RNN for sequence modeling):**

Let's create a simple RNN to model a sequence of numbers. Our goal is to train it to predict the next number in a simple arithmetic sequence.

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Set device and seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# --- 1. Data Preparation ---
# Create a simple sequence: 0, 1, 2, 3, 4, 5, ...
# We will create sequences of length 5 to predict the 6th number.
sequence_length = 5
data = np.arange(0, 100) # Our data source

# Create input (X) and target (y) sequences
X = []
y = []
for i in range(len(data) - sequence_length):
    X.append(data[i:i+sequence_length])
    y.append(data[i+sequence_length])

X = np.array(X)
y = np.array(y)

# Reshape X for PyTorch RNN: (batch_size, sequence_length, input_size)
# Here, input_size is 1 because each element in the sequence is a single number.
X = X.reshape((X.shape[0], X.shape[1], 1))

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X).to(device)
y_tensor = torch.FloatTensor(y).to(device)

print("Input shape (batch_size, seq_len, input_size):", X_tensor.shape)
print("Target shape:", y_tensor.shape)

# --- 2. Define the RNN Model ---
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        # nn.RNN layer
        # batch_first=True means input/output tensors have (batch, seq, feature) shape
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        # Fully connected layer to map RNN output to our desired output
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        # Shape: (num_layers * num_directions, batch_size, hidden_size)
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)
        
        # Pass input through RNN
        # out contains the output features (h_t) from the last layer of the RNN for each t
        # hn contains the final hidden state
        out, _ = self.rnn(x, h0)
        
        # We only need the output from the last time step
        # out shape: (batch_size, seq_length, hidden_size)
        # We take the last time step output: out[:, -1, :]
        out = self.fc(out[:, -1, :])
        return out

# --- 3. Training the Model ---
input_size = 1
hidden_size = 32
output_size = 1
learning_rate = 0.01
num_epochs = 100

model = SimpleRNN(input_size, hidden_size, output_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("\nTraining the RNN model...")
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor.view(-1, 1))
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

print("Training finished.")

# --- 4. Evaluation and Prediction ---
model.eval()
with torch.no_grad():
    # Let's predict the next number after the sequence [95, 96, 97, 98, 99]
    test_seq = torch.FloatTensor([[[95], [96], [97], [98], [99]]]).to(device)
    predicted_num = model(test_seq).item()
    print(f"\nPrediction for sequence [95, 96, 97, 98, 99]: {predicted_num:.2f}")

    # Let's see how it performs on a few training examples
    print("\n--- Sample Predictions on Training Data ---")
    for i in range(5):
        input_seq = X_tensor[i].unsqueeze(0) # Add batch dimension
        true_val = y_tensor[i].item()
        predicted_val = model(input_seq).item()
        print(f"Input: {X[i].flatten()}, True: {true_val}, Predicted: {predicted_val:.2f}")
```

**When and where we use sequence modeling:**
-   **Natural Language Processing (NLP):** Language modeling, machine translation, text generation, sentiment analysis.
-   **Time Series Analysis:** Stock price prediction, weather forecasting, sales forecasting.
-   **Speech Recognition:** Converting audio signals into text.
-   **Music Generation:** Creating new musical sequences.
-   **Video Analysis:** Action recognition in video clips.

**Real-world examples:**
-   **Google Translate (older versions):** Used an encoder-decoder LSTM architecture to translate text from one language to another. The encoder reads the source sentence and the decoder generates the translation.
-   **Apple's Siri and Google Assistant:** Use RNNs/LSTMs to process your spoken words (a sequence of audio features) and understand the intent behind your command.
-   **Stock Market Prediction:** Financial institutions use sequence models to analyze historical stock prices (a sequence of numbers) and predict future trends.

**Problems and Advancements:**
-   **Problem:** Standard RNNs struggle with **long-term dependencies** due to the vanishing gradient problem. They can't remember information from many steps back in the sequence.
  -   **Advancement:** **Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRUs)**. These architectures use gating mechanisms to regulate the flow of information, allowing them to learn dependencies over hundreds of time steps. GRUs are a simplified version of LSTMs with fewer parameters.
    ```python
    # Using an LSTM instead of an RNN is a simple change in PyTorch
    # class SimpleLSTM(nn.Module):
    #     def __init__(self, input_size, hidden_size, output_size):
    #         super(SimpleLSTM, self).__init__()
    #         self.hidden_size = hidden_size
    #         self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
    #         self.fc = nn.Linear(hidden_size, output_size)
    #
    #     def forward(self, x):
    #         # LSTM also has a cell state (c0)
    #         h0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)
    #         c0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)
    #         out, _ = self.lstm(x, (h0, c0))
    #         out = self.fc(out[:, -1, :])
    #         return out
    ```
-   **Problem:** RNNs are inherently **sequential**, which means they cannot be parallelized across time steps. This makes them slow to train on modern hardware like GPUs that excel at parallel computation.
  -   **Advancement:** **Transformer Architectures and Attention Mechanisms**. The Transformer model, introduced in the "Attention is All You Need" paper, completely does away with recurrence. It uses a self-attention mechanism to process all input tokens in parallel, calculating the relevance of every other token for each token. This allows for much greater parallelization and has become the dominant architecture for NLP, forming the basis for models like BERT and GPT.

### 2. Time Steps

**What are Time Steps?**
In the context of RNNs, a "time step" refers to a single position in the input sequence. An RNN processes a sequence iteratively, one time step at a time. At each time step `t`, the RNN takes two inputs:
1.  The current element of the sequence, `x_t`.
2.  The hidden state from the previous time step, `h_{t-1}`.

It then computes the new hidden state `h_t` and, optionally, an output `y_t`. This process is repeated for the entire length of the sequence.

**Explanation:**
-   **Unfolding an RNN:** Conceptually, an RNN can be "unfolded" over time. This means we can visualize the single RNN cell being copied for each time step in the sequence. The hidden state `h` is the connection that links these copies together, passing information from one step to the next.
-   **Input Shape:** The input to an RNN layer in PyTorch is typically a 3D tensor of shape `(batch_size, sequence_length, input_size)`.
    -   `batch_size`: The number of sequences processed in parallel.
    -   `sequence_length`: The number of time steps in each sequence.
    -   `input_size`: The number of features for each element at a time step.
-   **Backpropagation Through Time (BPTT):** This is the algorithm used to train RNNs. It's essentially backpropagation applied to the unfolded computational graph of the RNN over all time steps. The gradients are calculated at the final time step and propagated backward through each time step, which is where the vanishing/exploding gradient problem occurs.

**Implementation (Visualizing time steps in an LSTM):**

Let's build a simple character-level LSTM and inspect the hidden and cell states at each time step to see how information flows.

```python
import torch
import torch.nn as nn
import numpy as np

# Set device and seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# --- 1. Data Preparation ---
# A simple sentence to process
sentence = "hello_world"
chars = list(sentence)
vocab_size = len(set(chars)) # Unique characters
char_to_idx = {ch: i for i, ch in enumerate(set(chars))}
idx_to_char = {i: ch for ch, i in char_to_idx.items()}

# Convert sentence to indices
sentence_indices = [char_to_idx[c] for c in sentence]

# One-hot encode the characters
def one_hot_encode(sequence, vocab_size):
    encoding = np.zeros((len(sequence), vocab_size))
    for i, idx in enumerate(sequence):
        encoding[i, idx] = 1
    return encoding

one_hot_sentence = one_hot_encode(sentence_indices, vocab_size)

# Reshape for PyTorch: (batch_size=1, sequence_length, input_size=vocab_size)
input_tensor = torch.FloatTensor(one_hot_sentence).unsqueeze(0).to(device)
print("Input shape:", input_tensor.shape)
print("Input sentence:", sentence)

# --- 2. Define the LSTM Model ---
class CharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        # We want to get the output at each time step, not just the last one
        # So we don't pre-initialize h0 and c0, PyTorch will do it with zeros
        out, (hn, cn) = self.lstm(x)
        # out: (batch_size, seq_len, hidden_size)
        # hn: (num_layers * num_directions, batch_size, hidden_size) -> final hidden state
        # cn: (num_layers * num_directions, batch_size, hidden_size) -> final cell state
        return out, (hn, cn)

input_size = vocab_size
hidden_size = 16
model = CharLSTM(input_size, hidden_size).to(device)

# --- 3. Forward Pass and Inspecting States ---
# We are not training here, just doing a forward pass to see the states
model.eval()
with torch.no_grad():
    outputs, (final_hidden, final_cell) = model(input_tensor)

print("\n--- Inspecting Outputs and States ---")
print("Output tensor shape (all time steps):", outputs.shape)
print("Final hidden state shape:", final_hidden.shape)
print("Final cell state shape:", final_cell.shape)

# Let's look at the hidden state at each time step
# The `outputs` tensor contains the hidden state h_t for each time step t
print("\nHidden state at each time step (first 3 dimensions):")
for i, char in enumerate(sentence):
    # outputs[0, i, :] gets the hidden state for the i-th character in the batch
    state_at_t = outputs[0, i, :].cpu().numpy()
    print(f"Time Step {i} (char '{char}'): h_t = {state_at_t[:3]}...")

# --- 4. Understanding the Flow ---
print("\n--- Understanding the Flow ---")
print("At each time step t:")
print("1. Input x_t (one-hot encoded char) is fed into the LSTM cell.")
print("2. The cell takes the previous hidden state h_{t-1} and cell state c_{t-1}.")
print("3. It calculates the new hidden state h_t and cell state c_t using its gates.")
print("4. h_t is also the output y_t for that time step.")
print("5. h_t and c_t are passed to the next time step t+1.")
print("\nThe final hidden state `final_hidden` is the h_t of the last time step.")
print("The final cell state `final_cell` is the c_t of the last time step.")
```

**When and where we use the concept of time steps:**
-   **When designing the input data pipeline:** You must structure your data into sequences of a fixed or variable length, where each element corresponds to a time step.
-   **When debugging RNNs:** Understanding the flow of information through time steps is crucial for diagnosing issues like vanishing gradients or the model failing to capture long-range dependencies.
-   **When working with variable-length sequences:** You often use padding to make all sequences in a batch the same length. You then use mechanisms like `pack_padded_sequence` in PyTorch to ensure the RNN only processes the actual, non-padded time steps.
-   **When building encoder-decoder models:** In machine translation, the encoder processes the source sentence one time step at a time to create a context vector. The decoder then generates the translation one time step at a time.

**Real-world examples:**
-   **Sentiment Analysis:** A sentence is a sequence of words. Each word is a time step. The RNN processes the sentence word-by-word, and the final hidden state is often used to make the sentiment prediction (positive/negative).
-   **Stock Price Prediction:** A sequence of daily stock prices over the last 30 days can be used to predict the price on day 31. Each day's data (e.g., open, high, low, close prices) is a time step.
-   **Video Classification:** A video clip is a sequence of image frames. Each frame is a time step. An RNN (often combined with a CNN) processes the frames sequentially to classify the action in the video (e.g., "running," "swimming").

**Problems and Advancements:**
-   **Problem:** Handling **variable-length sequences** is cumbersome. Padding adds noise, and standard RNNs can be inefficient if sequences are very long.
  -   **Advancement:** **Masking and Packing**. PyTorch provides `pack_padded_sequence` and `pad_packed_sequence` to handle this efficiently. Packing removes the padded time steps from computation, so the RNN only does work on the real data, significantly speeding up training on batches of variable-length sequences.
    ```python
    # Example of packing a padded sequence
    # from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    # seqs = ['hello', 'world', 'pytorch'] # Variable length sequences
    # lengths = [5, 5, 7] # Actual lengths
    # padded_seqs = pad_sequence(seqs_tensor, batch_first=True) # Padded to max_len=7
    # packed_input = pack_padded_sequence(padded_seqs, lengths, batch_first=True, enforce_sorted=False)
    # output, (hn, cn) = lstm(packed_input)
    # padded_output, _ = pad_packed_sequence(output, batch_first=True)
    ```
-   **Problem:** The hidden state at the final time step of an RNN is a **fixed-size vector** that must summarize the entire input sequence. For very long sequences, this "bottleneck" can cause information loss, as the final state might not retain all the important details from the beginning.
  -   **Advancement:** **Attention Mechanisms**. Instead of relying solely on the final hidden state, attention allows the decoder (in a seq2seq model) to "look back" at all the hidden states of the encoder. It learns to assign a weight or "attention score" to each encoder hidden state, creating a custom context vector for each step of the decoding process. This directly addresses the bottleneck problem and was a key innovation that led to the Transformer architecture.
    ```python
    # Conceptual: Attention creates a weighted sum of all encoder hidden states
    # context_t = sum(alpha_ti * h_i_encoder) for i from 0 to T_encoder
    # where alpha_ti is the attention weight for encoder state i at decoder step t.
    ```

## 🧪 Exercise: Next Word Prediction

This exercise will build a simple next-word prediction model using an LSTM. We'll train it on a small corpus and then use it to generate text by feeding its predictions back as input.

**Implementation:**

```python
import torch
import torch.nn as nn
import numpy as np
import random

# Set device and seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

# --- 1. Data Preparation ---
# A small corpus of text
corpus = """
The quick brown fox jumps over the lazy dog.
A journey of a thousand miles begins with a single step.
To be or not to be, that is the question.
All that is gold does not glitter,
Not all those who wander are lost;
The old that is strong does not wither,
Deep roots are not reached by the frost.
"""

# Preprocessing: lowercase and tokenize
corpus = corpus.lower().replace('\n', ' ').split()
vocab = sorted(list(set(corpus)))
vocab_size = len(vocab)
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for i, word in enumerate(vocab)}

print(f"Vocabulary Size: {vocab_size}")
print("Vocabulary:", vocab)

# Create sequences of words and their corresponding targets (next word)
sequence_length = 4 # Use 4 words to predict the 5th
sequences = []
targets = []
for i in range(len(corpus) - sequence_length):
    sequences.append(corpus[i:i+sequence_length])
    targets.append(corpus[i+sequence_length])

print(f"\nNumber of sequences: {len(sequences)}")
print("Example sequence:", sequences[0], "-> Target:", targets[0])

# Convert sequences and targets to numerical indices
X = [[word_to_idx[word] for word in seq] for seq in sequences]
y = [word_to_idx[word] for word in targets]

# Convert to PyTorch tensors
X_tensor = torch.LongTensor(X).to(device)
y_tensor = torch.LongTensor(y).to(device)

# --- 2. Define the LSTM Model ---
class NextWordLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(NextWordLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x) # (batch_size, seq_len, embedding_dim)
        
        # We don't need to initialize h0, c0, PyTorch does it with zeros
        lstm_out, _ = self.lstm(embedded) # (batch_size, seq_len, hidden_size)
        
        # We only need the output from the last time step
        last_time_step_out = lstm_out[:, -1, :] # (batch_size, hidden_size)
        
        output = self.fc(last_time_step_out) # (batch_size, vocab_size)
        return output

# --- 3. Training the Model ---
embedding_dim = 64
hidden_size = 128
learning_rate = 0.005
num_epochs = 200

model = NextWordLSTM(vocab_size, embedding_dim, hidden_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("\nTraining the Next-Word Prediction model...")
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training finished.")

# --- 4. Prediction and Text Generation ---
def predict_next_word(model, input_sequence, word_to_idx, idx_to_word, device):
    model.eval()
    with torch.no_grad():
        # Convert input sequence to indices
        input_indices = [word_to_idx.get(word, 0) for word in input_sequence] # Use 0 for unknown words
        
        # Pad or truncate to the correct sequence length
        if len(input_indices) < sequence_length:
            input_indices = [0] * (sequence_length - len(input_indices)) + input_indices
        else:
            input_indices = input_indices[-sequence_length:]
            
        input_tensor = torch.LongTensor(input_indices).unsqueeze(0).to(device) # Add batch dim
        
        # Get prediction
        output = model(input_tensor)
        
        # Get the index of the highest probability
        predicted_idx = torch.argmax(output, dim=1).item()
        
        return idx_to_word[predicted_idx]

def generate_text(model, seed_text, num_words, word_to_idx, idx_to_word, device):
    model.eval()
    current_sequence = seed_text.lower().split()
    generated_text = current_sequence.copy()
    
    for _ in range(num_words):
        next_word = predict_next_word(model, current_sequence, word_to_idx, idx_to_word, device)
        generated_text.append(next_word)
        # Update the sequence for the next prediction
        current_sequence.append(next_word)
        if len(current_sequence) > sequence_length:
            current_sequence = current_sequence[-sequence_length:]
            
    return " ".join(generated_text)

# --- 5. Test the Model ---
print("\n--- Testing Next-Word Prediction ---")
test_seq = "the quick brown"
predicted_word = predict_next_word(model, test_seq.split(), word_to_idx, idx_to_word, device)
print(f"Sequence: '{test_seq}' -> Predicted next word: '{predicted_word}'")

test_seq_2 = "to be or"
predicted_word_2 = predict_next_word(model, test_seq_2.split(), word_to_idx, idx_to_word, device)
print(f"Sequence: '{test_seq_2}' -> Predicted next word: '{predicted_word_2}'")

print("\n--- Text Generation ---")
seed_text = "the journey"
generated_text = generate_text(model, seed_text, 10, word_to_idx, idx_to_word, device)
print(f"Seed: '{seed_text}'")
print(f"Generated: '{generated_text}'")

seed_text_2 = "all that is"
generated_text_2 = generate_text(model, seed_text_2, 12, word_to_idx, idx_to_word, device)
print(f"\nSeed: '{seed_text_2}'")
print(f"Generated: '{generated_text_2}'")
```

## 📌 Project: Text Generator from Corpus

Now, let's build a more robust text generator. We'll use a larger corpus (a collection of Shakespeare's sonnets), implement a more sophisticated training loop, and create a Streamlit app for an interactive experience.

**Implementation:**

**Part 1: The Training Script (`train_text_generator.py`)**

This script will download the data, train the LSTM model, and save it.

```python
import torch
import torch.nn as nn
import numpy as np
import requests
import os
from torch.utils.data import Dataset, DataLoader

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQUENCE_LENGTH = 100
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.001
EMBEDDING_DIM = 256
HIDDEN_SIZE = 1024
NUM_LAYERS = 2 # Stacked LSTM

# Create models directory
os.makedirs("models", exist_ok=True)
MODEL_PATH = "models/shakespeare_generator.pth"

# --- 1. Data Loading and Preprocessing ---
def load_shakespeare_data():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    response = requests.get(url)
    text = response.text
    return text

text = load_shakespeare_data()
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

print(f"Vocabulary Size: {vocab_size}")
print("First 500 characters of corpus:")
print(text[:500])

# --- 2. Dataset and DataLoader ---
class ShakespeareDataset(Dataset):
    def __init__(self, text, char_to_idx, seq_length):
        self.text = text
        self.char_to_idx = char_to_idx
        self.seq_length = seq_length

    def __len__(self):
        return len(self.text) - self.seq_length

    def __getitem__(self, idx):
        chunk = self.text[idx:idx + self.seq_length + 1]
        dix = [self.char_to_idx[c] for c in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

dataset = ShakespeareDataset(text, char_to_idx, SEQUENCE_LENGTH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- 3. Define the LSTM Model ---
class CharLSTMGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
        super(CharLSTMGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        output = self.fc(output)
        # Reshape output to be (batch_size * seq_length, vocab_size) for CrossEntropyLoss
        output = output.reshape(output.size(0) * output.size(1), output.size(2))
        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_(),
                  weight.new(self.num_layers, batch_size, self.hidden_size).zero_())
        return hidden

# --- 4. Training Loop ---
model = CharLSTMGenerator(vocab_size, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("\nStarting training...")
for epoch in range(EPOCHS):
    model.train()
    hidden = model.init_hidden(BATCH_SIZE)
    total_loss = 0
    for batch_idx, (data, targets) in enumerate(dataloader):
        # Move data to device
        data, targets = data.to(DEVICE), targets.to(DEVICE)
        
        # Detach hidden state from its history to prevent backprop through entire training history
        hidden = tuple([each.data for each in hidden])
        
        optimizer.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output, targets.view(-1))
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        
        optimizer.step()
        total_loss += loss.item()
        
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

# --- 5. Save the Model ---
torch.save(model.state_dict(), MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")
```

**Part 2: The Streamlit App (`app/streamlit_text_generator_app.py`)**

This script will load the trained model and provide a UI for text generation.

```python
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import requests
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Shakespearean Text Generator",
    page_icon="📜",
    layout="centered"
)

# --- Configuration (must match training) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 256
HIDDEN_SIZE = 1024
NUM_LAYERS = 2
MODEL_PATH = "../models/shakespeare_generator.pth" # Path relative to the app file

# --- Load Data and Model (with Caching) ---
@st.cache_resource
def load_data_and_model():
    """Loads the Shakespeare text and the trained model."""
    # Load text to get vocabulary
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = requests.get(url).text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    # Define model structure
    class CharLSTMGenerator(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
            super(CharLSTMGenerator, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden_size, vocab_size)

        def forward(self, x, hidden):
            embedded = self.embedding(x)
            output, hidden = self.lstm(embedded, hidden)
            output = self.fc(output)
            return output, hidden

        def init_hidden(self, batch_size):
            weight = next(self.parameters()).data
            hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_(),
                      weight.new(self.num_layers, batch_size, self.hidden_size).zero_())
            return hidden
    
    # Instantiate and load model
    model = CharLSTMGenerator(vocab_size, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print("Model loaded successfully.")
    else:
        st.error(f"Model file not found at {MODEL_PATH}. Please run the training script first.")
        return None, None, None, None
        
    return model, char_to_idx, idx_to_char, vocab_size

model, char_to_idx, idx_to_char, vocab_size = load_data_and_model()

# --- Text Generation Function ---
def generate_text(model, prime_str, predict_len, temperature, char_to_idx, idx_to_char, device):
    model.eval()
    with torch.no_grad():
        # Initialize hidden state
        hidden = model.init_hidden(1)
        
        # Prime the model with the starting string
        prime_str = prime_str.lower()
        current_seq = [char_to_idx.get(c, 0) for c in prime_str]
        predicted_text = prime_str
        
        for p in range(predict_len):
            # Prepare input for the model
            x = torch.tensor([[current_seq[-1]]], dtype=torch.long).to(device)
            
            # Get prediction
            output, hidden = model(x, hidden)
            
            # Apply temperature
            output_dist = output.data.view(-1).div(temperature).exp()
            top_i = torch.multinomial(output_dist, 1)[0].item()
            
            # Add predicted character to text and sequence
            predicted_char = idx_to_char[top_i]
            predicted_text += predicted_char
            current_seq.append(top_i)
            
    return predicted_text

# --- App UI ---
st.title("📜 Shakespearean Text Generator")
st.markdown("""
This app uses a **Stacked LSTM** trained on the complete works of Shakespeare to generate text in the style of the Bard.
Enter a starting phrase (a "prime"), adjust the creativity slider, and click generate!
""")

if model:
    # --- Input Widgets ---
    prime_str = st.text_input("Enter a starting phrase (prime):", "To be or not to be")
    predict_len = st.slider("Number of characters to generate:", 50, 1000, 200)
    temperature = st.slider("Creativity (Temperature):", 0.1, 2.0, 0.8, step=0.05)
    st.markdown("""
    **Temperature** controls the randomness of predictions:
    - **Low (e.g., 0.1):** More conservative and predictable text.
    - **High (e.g., 1.5+):** More creative and surprising text, but may be less coherent.
    """)

    # --- Generation Button ---
    if st.button("Generate Text"):
        with st.spinner("Generating text..."):
            generated_text = generate_text(model, prime_str, predict_len, temperature, char_to_idx, idx_to_char, DEVICE)
        
        st.subheader("Generated Text:")
        st.text_area("", generated_text, height=300)

        # Add a download button
        st.download_button(
            label="Download Generated Text",
            data=generated_text,
            file_name="shakespeare_generated.txt",
            mime="text/plain"
        )

else:
    st.warning("The application cannot run because the model failed to load. Please ensure the training script has been run and the model file is in the correct location.")

# --- Sidebar with Information ---
st.sidebar.header("About the Model")
st.sidebar.info("""
This text generator is based on a **Long Short-Term Memory (LSTM)** network, a type of Recurrent Neural Network (RNN).

**Key Features:**
- **Stacked LSTM:** Uses two LSTM layers on top of each other to learn more complex patterns.
- **Character-Level:** Predicts one character at a time, allowing it to generate new words and spellings.
- **Temperature Sampling:** Uses a temperature parameter to control the trade-off between predictable and creative text.
- **Trained on Shakespeare:** The model's style and vocabulary are derived from a large corpus of Shakespeare's plays and sonnets.
""")
```

**How to Run the Project:**

1.  **Project Structure:**
    ```
    text_generator_project/
    ├── models/
    │   └── shakespeare_generator.pth  (created by training script)
    ├── app/
    │   └── streamlit_text_generator_app.py
    └── train_text_generator.py
    ```
2.  **Train the Model:**
    Open your terminal, navigate to the `text_generator_project` directory, and run the training script. This will take some time.
    ```bash
    python train_text_generator.py
    ```
3.  **Run the Streamlit App:**
    Once the training is complete and `models/shakespeare_generator.pth` is created, run the app.
    ```bash
    streamlit run app/streamlit_text_generator_app.py
    ```

