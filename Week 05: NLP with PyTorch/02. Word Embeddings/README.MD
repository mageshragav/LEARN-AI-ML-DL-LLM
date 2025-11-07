

# Day 22 – Word Embeddings

## What are Word Embeddings and Why are they Important for Deep Learning?

Word embeddings are a type of word representation that allows words with similar meaning to have a similar representation. They are a set of techniques in NLP where words or phrases from the vocabulary are mapped to vectors of real numbers. These vectors are called "embeddings" because they embed the meaning of a word into a high-dimensional space.

Before embeddings, words were often represented using sparse vectors like one-hot encoding, where each word was a unique vector with a '1' in its position and '0's everywhere else. This approach had two major flaws:
1.  **Massive Dimensionality:** The vector size is equal to the entire vocabulary size (which can be hundreds of thousands or millions), making it computationally inefficient.
2.  **No Semantic Information:** The representation captures no information about the word's meaning. "King" and "queen" are as different from each other as "king" is from "car."

Word embeddings solve these problems by creating dense, low-dimensional vectors (e.g., 100-300 dimensions) where the geometric relationships between vectors correspond to semantic relationships between words.

In deep learning, word embeddings are fundamentally important because:

-   **They Capture Semantic Meaning:** They are the primary way deep learning models "understand" the meaning of words. Words that appear in similar contexts (like "coffee" and "tea") will have vectors that are close to each other in the embedding space.
-   **They Enable Transfer Learning:** Pre-trained word embeddings (like Word2Vec or GloVe) can be used in downstream NLP tasks (like sentiment analysis or text classification). This allows a model to leverage knowledge from a massive text corpus (like all of Wikipedia) even if the task-specific dataset is small.
-   **They are the Input Layer for NLP Models:** In almost all modern deep learning architectures for NLP (RNNs, LSTMs, Transformers), the first step is to convert input words into their corresponding embedding vectors. These vectors are then fed into the subsequent layers of the network.
-   **They Enable Complex Reasoning:** The vector space allows for analogical reasoning. The classic example is: `vector('King') - vector('Man') + vector('Woman')` results in a vector very close to `vector('Queen')`.

## Essential Subtopics

### 1. TF-IDF (Term Frequency-Inverse Document Frequency)

**What is TF-IDF?**
TF-IDF is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus. While it's not a "deep learning" embedding in the sense of Word2Vec or GloVe (as it's not learned by a neural network and doesn't produce dense vectors), it's a crucial concept to understand as a precursor to modern embeddings. It's a method for vectorizing text that goes beyond simple word counts.

It's composed of two terms:
1.  **Term Frequency (TF):** Measures how frequently a term appears in a document. A term that appears more often is likely more important.
    `TF(t, d) = (Number of times term t appears in document d) / (Total number of terms in document d)`
2.  **Inverse Document Frequency (IDF):** Measures how important a term is across the entire corpus. It diminishes the weight of terms that are very common across all documents (like "the", "a", "is") and increases the weight of terms that are rare.
    `IDF(t, D) = log_e(Total number of documents in corpus D / Number of documents with term t in them)`

The final TF-IDF score is the product of these two: `TF-IDF(t, d, D) = TF(t, d) * IDF(t, D)`

**PyTorch/Numpy/Scikit-learn Implementation:**

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import torch

# --- 1. Manual Calculation for Understanding ---
# Let's define a small corpus of documents
corpus = [
    "the cat sat on the mat",
    "the dog ate the cat",
    "the cat and the dog are friends"
]

# Step 1: Preprocess and get vocabulary
vocab = {}
documents = []
for i, doc in enumerate(corpus):
    tokens = doc.split()
    documents.append(tokens)
    for token in tokens:
        if token not in vocab:
            vocab[token] = len(vocab)

print("Vocabulary:", vocab)
print("Vocabulary Size:", len(vocab))

# Step 2: Calculate TF for each document
def calculate_tf(doc_tokens, vocab):
    tf_vector = np.zeros(len(vocab))
    term_count = len(doc_tokens)
    for token in doc_tokens:
        tf_vector[vocab[token]] += 1
    tf_vector = tf_vector / term_count
    return tf_vector

tf_matrix = np.array([calculate_tf(doc, vocab) for doc in documents])
print("\nTF Matrix:\n", tf_matrix)

# Step 3: Calculate IDF for the corpus
def calculate_idf(corpus_tokens, vocab):
    idf_vector = np.zeros(len(vocab))
    num_docs = len(corpus_tokens)
    for token, idx in vocab.items():
        doc_count = sum(1 for doc_tokens in corpus_tokens if token in doc_tokens)
        idf_vector[idx] = np.log(num_docs / doc_count)
    return idf_vector

idf_vector = calculate_idf(documents, vocab)
print("\nIDF Vector:\n", idf_vector)

# Step 4: Calculate TF-IDF Matrix
tfidf_matrix_manual = tf_matrix * idf_vector
print("\nManual TF-IDF Matrix:\n", np.round(tfidf_matrix_manual, 2))

# --- 2. Using Scikit-learn for Practical Application ---
# Scikit-learn's TfidfVectorizer handles all the steps automatically
vectorizer = TfidfVectorizer()
tfidf_matrix_sklearn = vectorizer.fit_transform(corpus)

# The result is a sparse matrix, let's convert it to a dense array for viewing
print("\nScikit-learn TF-IDF Matrix (Dense):\n", np.round(tfidf_matrix_sklearn.toarray(), 2))

# Get the feature names (vocabulary)
print("\nScikit-learn Vocabulary:", vectorizer.get_feature_names_out())

# --- 3. Converting to PyTorch Tensor ---
# If you wanted to use this in a PyTorch model
tfidf_tensor = torch.from_numpy(tfidf_matrix_sklearn.toarray()).float()
print("\nTF-IDF PyTorch Tensor Shape:", tfidf_tensor.shape)
print("TF-IDF PyTorch Tensor:\n", tfidf_tensor)
```

**When and where we use TF-IDF:**
-   **As a baseline model** for text classification, information retrieval, and document similarity tasks.
-   **In traditional machine learning models** like Naive Bayes, Support Vector Machines (SVMs), and Logistic Regression for NLP.
-   **For keyword extraction** to identify the most important words in a document.
-   **In search engines** to rank documents based on a user's query.

**Real-world examples:**
-   **Spam Detection:** A classifier can be trained where the features are the TF-IDF scores of words in emails. Words like "free," "viagra," and "offer" will have high scores in spam emails.
-   **Document Search:** When you search for a document in a large database (like a legal discovery system), the system can convert your query into a TF-IDF vector and compare it to the TF-IDF vectors of all documents to find the most relevant ones.
-   **Content Recommendation:** News aggregators can use TF-IDF to find articles similar to the one a user is currently reading by comparing their TF-IDF vectors.

**Problems and Advancements:**
-   **Problem:** TF-IDF produces very high-dimensional and sparse vectors (mostly zeros). It also fails to capture semantic meaning, word order, or context. The words "good" and "great" are treated as completely different, and it doesn't understand that "car" and "automobile" are synonyms.
  -   **Advancement:** **Dense Word Embeddings (Word2Vec, GloVe)**. These methods solve the sparsity and semantic limitations by learning dense, low-dimensional vectors where similar words have similar vectors. They capture semantic relationships through the context in which words appear.
    ```python
    # This is the conceptual leap from TF-IDF to Word2Vec/GloVe
    # TF-IDF: "car" -> [0, 0, 5.2, 0, ..., 0] (sparse, no meaning)
    # Word2Vec: "car" -> [0.21, -0.88, 0.45, ..., 0.12] (dense, semantic meaning)
    ```
-   **Problem:** TF-IDF is a "bag-of-words" model, meaning it completely ignores word order. The phrases "dog bites man" and "man bites dog" have the exact same TF-IDF representation.
  -   **Advancement:** **Contextual Embeddings (BERT, GPT)**. These are the next generation of embeddings. Unlike Word2Vec where a word has one fixed vector, contextual embeddings generate a vector for a word based on the specific sentence it appears in. This allows them to distinguish between "bank" in "river bank" and "bank" in "money bank," solving the polysemy problem and capturing word order and syntax at a much deeper level.

### 2. Word2Vec

**What is Word2Vec?**
Word2Vec is a group of related models that are used to produce word embeddings. These models are shallow, two-layer neural networks that are trained to reconstruct linguistic contexts of words. Word2Vec takes as its input a large corpus of text and produces a vector space, typically of several hundred dimensions, with each unique word in the corpus being assigned a corresponding vector in the space.

There are two main architectures for Word2Vec:
1.  **Continuous Bag-of-Words (CBOW):** The model predicts the current word from a window of surrounding context words. The context is the "bag-of-words" because the order of context words doesn't matter. For example, given the context `["the", "cat", "on", "mat"]`, the model tries to predict the target word `sat`.
2.  **Skip-gram:** The model does the inverse of CBOW. It uses the current word to predict the surrounding window of context words. For example, given the target word `sat`, the model tries to predict the context words `["the", "cat", "on", "mat"]`. Skip-gram tends to work better with smaller datasets and can better represent rare words.

**PyTorch Implementation (Training a simple Word2Vec model):**

*Note: In practice, you would almost always use a highly optimized library like `gensim` to train or load Word2Vec models. This PyTorch implementation is for educational purposes to understand the underlying mechanics.*

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# --- 1. Data Preparation ---
corpus = [
    "the quick brown fox jumps over the lazy dog",
    "i love dogs and cats",
    "dogs are loyal animals",
    "cats are independent pets",
    "the fox is a cunning animal"
]

# Tokenize and create vocabulary
def tokenize_and_create_vocab(corpus):
    vocab = {}
    for sentence in corpus:
        tokens = sentence.split()
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab

vocab = tokenize_and_create_vocab(corpus)
vocab_size = len(vocab)
word_to_idx = vocab
idx_to_word = {i: w for w, i in word_to_idx.items()}

print("Vocabulary:", word_to_idx)
print("Vocabulary Size:", vocab_size)

# Generate (target, context) pairs for Skip-gram
def generate_skipgram_data(corpus, word_to_idx, window_size=2):
    data = []
    for sentence in corpus:
        tokens = sentence.split()
        for i, target_word in enumerate(tokens):
            target_idx = word_to_idx[target_word]
            # Define context window
            start = max(0, i - window_size)
            end = min(len(tokens), i + window_size + 1)
            for j in range(start, end):
                if i == j:
                    continue
                context_word = tokens[j]
                context_idx = word_to_idx[context_word]
                data.append((target_idx, context_idx))
    return data

training_data = generate_skipgram_data(corpus, word_to_idx)
print("\nSample training data (target, context) pairs:", training_data[:5])

# --- 2. Define the Word2Vec Model (Skip-gram) ---
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        # Embedding layer for target words
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Embedding layer for context words (this will be our output layer)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, target_word, context_word):
        # Get embeddings for target and context words
        target_embeds = self.target_embeddings(target_word)
        context_embeds = self.context_embeddings(context_word)
        
        # The goal is to make the dot product of target and context embeddings high
        # for true pairs and low for negative samples (not shown in this simple version)
        # For simplicity, we'll use a basic approach here.
        # A full implementation would use Negative Sampling or Hierarchical Softmax.
        score = torch.mul(target_embeds, context_embeds).sum(dim=1)
        return score

# --- 3. Training the Model ---
EMBEDDING_DIM = 10
LEARNING_RATE = 0.01
EPOCHS = 100

model = Word2Vec(vocab_size, EMBEDDING_DIM)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# For simplicity, we'll use a simple loss function.
# In a real Word2Vec, you'd use Negative Sampling Loss.
# We will treat it as a binary classification: is this a valid (target, context) pair?
# We assume all our generated pairs are positive (score=1).
criterion = nn.BCEWithLogitsLoss()

for epoch in range(EPOCHS):
    total_loss = 0
    for target_idx, context_idx in training_data:
        target_tensor = torch.LongTensor([target_idx])
        context_tensor = torch.LongTensor([context_idx])
        
        optimizer.zero_grad()
        
        score = model(target_tensor, context_tensor)
        
        # Our target is 1 (positive pair)
        loss = criterion(score, torch.ones_like(score))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

# --- 4. Extracting and Visualizing the Embeddings ---
# The learned word embeddings are in the target_embeddings layer
learned_embeddings = model.target_embeddings.weight.data.numpy()
print("\nLearned Embeddings Shape:", learned_embeddings.shape)

# Visualize the embeddings using t-SNE
def plot_embeddings(embeddings, idx_to_word):
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(idx_to_word)-1))
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    for i, word in idx_to_word.items():
        x, y = embeddings_2d[i]
        plt.scatter(x, y)
        plt.annotate(word, (x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.title("Word2Vec Embeddings Visualized with t-SNE")
    plt.show()

plot_embeddings(learned_embeddings, idx_to_word)
```

**When and where we use Word2Vec:**
-   **As a pre-processing step** for almost any NLP task using deep learning (e.g., sentiment analysis, named entity recognition, machine translation).
-   **When you need to find semantically similar words** or measure the semantic distance between words.
-   **For feature engineering** in machine learning models, converting text into meaningful numerical vectors.
-   **In recommendation systems** to find similar items based on their textual descriptions.

**Real-world examples:**
-   **Google News:** The original Word2Vec paper famously demonstrated that `vector('King') - vector('Man') + vector('Woman')` was closest to `vector('Queen')`, showing that the model captures analogical relationships.
-   **Product Recommendations:** An e-commerce site can train Word2Vec on product descriptions. The resulting embeddings can then be used to find products that are "similar" to each other, even if they don't share many keywords. For example, "wireless headphones" and "Bluetooth earbuds" would have close vectors.
-   **Resume Screening:** A company can train Word2Vec on a large corpus of job descriptions and resumes. This can help match candidates to jobs by finding resumes that use semantically similar language to the job description, even if the exact keywords aren't present.

**Problems and Advancements:**
-   **Problem:** Word2Vec produces **static embeddings**. A word has the same vector regardless of its context. This means it cannot handle polysemy (words with multiple meanings). For example, "bank" in "river bank" and "bank account" would have the exact same vector.
  -   **Advancement:** **Contextualized Language Models (ELMo, BERT)**. These models use deep Transformer architectures to generate a dynamic embedding for a word based on the specific sentence it appears in. ELMo generates context-aware embeddings by running a bidirectional LSTM over the sentence. BERT goes further by using a Transformer's self-attention mechanism to create deeply contextualized representations.
    ```python
    # Conceptual difference
    # Word2Vec: vector("bank") is always the same.
    # BERT: vector("bank") in "river bank" is different from vector("bank") in "money bank".
    ```
-   **Problem:** Word2Vec requires a large amount of text data to train effectively. Training it from scratch for a small, specific domain can be computationally expensive and may not yield high-quality embeddings.
  -   **Advancement:** **Pre-trained Embeddings and Transfer Learning**. Instead of training from scratch, it's now standard practice to use embeddings that have been pre-trained on enormous, general-domain corpora (like Google News, Wikipedia, or Common Crawl). These pre-trained vectors capture a vast amount of world knowledge and can be used directly or fine-tuned for a specific downstream task, even with a small amount of task-specific data.

### 3. GloVe (Global Vectors for Word Representation)

**What is GloVe?**
GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Developed by researchers at Stanford, it combines the advantages of two major model families: the global matrix factorization methods (like Latent Semantic Analysis, which is related to TF-IDF) and the local context window methods (like Word2Vec).

While Word2Vec is a predictive model (it predicts context words given a target word, or vice-versa), GloVe is a count-based model. It starts by constructing a large co-occurrence matrix `X`, where `X_ij` is the number of times word `j` appears in the context of word `i` in the entire corpus. The core idea of GloVe is that the ratio of co-occurrence probabilities is a more effective way to encode meaning than raw probabilities themselves.

The model then learns word vectors such that their dot product equals the logarithm of the words' probability of co-occurrence. The loss function is designed to make the dot product of two word vectors equal to the log of their co-occurrence probability.

**Implementation (Loading Pre-trained GloVe with `gensim`):**

Training GloVe from scratch is complex and requires constructing a massive co-occurrence matrix. In practice, everyone uses pre-trained GloVe vectors. The `gensim` library makes this very easy.

First, you need to download the pre-trained vectors. A popular choice is the `glove.6B.zip` file from the Stanford GloVe project page, which contains 50d, 100d, 200d, and 300d vectors trained on 6 billion tokens from Wikipedia and Gigaword.

```python
# You might need to install gensim and download the GloVe file first
# pip install gensim
# Download glove.6B.100d.txt from https://nlp.stanford.edu/projects/glove/

import numpy as np
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import torch

# --- 1. Convert GloVe format to Word2Vec format (one-time task) ---
# Gensim can load GloVe files directly, but converting them to the
# Word2Vec format first can be faster for repeated use.
# glove_input_file = "path/to/glove.6B.100d.txt"
# word2vec_output_file = "glove.6B.100d.word2vec.txt"
# glove2word2vec(glove_input_file, word2vec_output_file)

# --- 2. Load the GloVe Model ---
# For this example, let's assume you have the converted file
# If you don't, gensim can also load the original GloVe file
try:
    # Load the converted Word2Vec format
    glove_model = KeyedVectors.load_word2vec_format('glove.6B.100d.word2vec.txt', binary=False)
except FileNotFoundError:
    print("glove.6B.100d.word2vec.txt not found.")
    print("Please download glove.6B.zip from https://nlp.stanford.edu/projects/glove/")
    print("And run the conversion step, or use the code below to load the .txt file directly.")
    # Alternative: Load the original GloVe file directly
    # from gensim.models.keyedvectors import KeyedVectors
    # glove_model = KeyedVectors.load_word2vec_format('path/to/glove.6B.100d.txt', binary=False)
    glove_model = None # Placeholder

if glove_model:
    # --- 3. Exploring the GloVe Model ---
    print("GloVe model loaded successfully.")
    print("Vector dimension:", glove_model.vector_size)
    print("Vocabulary size:", len(glove_model.key_to_index))

    # Get the vector for a specific word
    word = 'computer'
    vector = glove_model[word]
    print(f"\nVector for '{word}':\n", vector[:10], "...") # Print first 10 dimensions

    # --- 4. Convert to PyTorch Embedding Layer ---
    # This is how you would use it in a PyTorch model
    vocab = list(glove_model.key_to_index.keys())
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    idx_to_word = {i: word for i, word in enumerate(vocab)}

    # Create a weight matrix for the embedding layer
    embedding_dim = glove_model.vector_size
    weights = torch.FloatTensor(glove_model.vectors)

    # Create an nn.Embedding layer
    embedding_layer = nn.Embedding.from_pretrained(weights, freeze=True)
    # freeze=True means the embeddings won't be updated during training of your model
    # Set freeze=False if you want to fine-tune the embeddings

    print("\nPyTorch Embedding Layer Shape:", embedding_layer.weight.shape)

    # Example of using the layer
    word_idx = torch.LongTensor([word_to_idx['king'], word_to_idx['queen']])
    word_embeddings = embedding_layer(word_idx)
    print("Embeddings for 'king' and 'queen':\n", word_embeddings)
```

**When and where we use GloVe:**
-   **Interchangeably with Word2Vec** for most NLP pre-processing tasks. The choice between Word2Vec and GloVe often comes down to empirical performance on a specific task.
-   **When you want embeddings that capture both local context (like Word2Vec) and global co-occurrence statistics (like matrix factorization)**. This can sometimes lead to better performance on tasks that require understanding broader semantic relationships.
-   **In academic research**, GloVe is often cited for its strong performance on word analogy tasks and its elegant theoretical foundation.

**Real-world examples:**
-   **Sentiment Analysis:** A model can be trained where the input is the average of the GloVe vectors of all words in a sentence. This average vector is then fed into a classifier to predict sentiment (positive/negative).
-   **Information Retrieval:** Documents and queries can be represented as the average of their word vectors. Retrieval then becomes a matter of finding the document vectors that are closest to the query vector in the embedding space.
-   **Text Generation:** In simple RNN-based language models, GloVe vectors can be used to initialize the embedding layer, giving the model a strong starting point for understanding word meaning.

**Problems and Advancements:**
-   **Problem:** Like Word2Vec, GloVe produces **static embeddings** and cannot handle polysemy. The meaning of a word is fixed, regardless of context.
  -   **Advancement:** **Transformer-based Contextual Embeddings (BERT, RoBERTa)**. As mentioned before, this is the primary advancement that addresses the static nature of Word2Vec and GloVe. Models like BERT generate a vector for a word that is a function of the entire input sentence, allowing them to capture nuanced, context-dependent meanings.
    ```python
    # BERT's attention mechanism allows it to weigh the importance of all other words
    # when creating the embedding for a specific word.
    # Sentence: "The robot reached for the bank to withdraw cash."
    # BERT's embedding for "bank" will be heavily influenced by "withdraw" and "cash".
    ```
-   **Problem:** Both Word2Vec and GloVe are **agnostic to word order** at the training level. They only care about which words appear near each other, not their sequence. This limits their ability to understand syntax.
  -   **Advancement:** **Positional Encodings in Transformers**. The Transformer architecture, which powers models like BERT and GPT, explicitly injects information about the position of each word in the sequence. These "positional encodings" are added to the input embeddings, allowing the model to differentiate between "dog bites man" and "man bites dog" even before any processing happens. This is a fundamental reason why Transformers excel at tasks requiring an understanding of grammar and syntax.

## 🧪 Exercise: Word Similarity Using Vectors

This exercise will demonstrate the power of word embeddings by finding semantically similar words and performing word analogies. We'll use the pre-trained GloVe model from the previous section.

**Implementation:**

```python
import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import torch

# --- 1. Load the Pre-trained GloVe Model ---
# (Assuming you have 'glove.6B.100d.word2vec.txt' from the previous section)
try:
    glove_model = KeyedVectors.load_word2vec_format('glove.6B.100d.word2vec.txt', binary=False)
except FileNotFoundError:
    print("GloVe model file not found. Please ensure you have downloaded and converted it.")
    glove_model = None

if glove_model:
    # --- 2. Finding Similar Words ---
    # Gensim provides a convenient function for this
    def find_similar_words(word, model, topn=5):
        if word not in model:
            return f"'{word}' not in vocabulary."
        similar_words = model.most_similar(word, topn=topn)
        return similar_words

    word_to_test = 'computer'
    print(f"Words most similar to '{word_to_test}':")
    print(find_similar_words(word_to_test, glove_model))

    word_to_test_2 = 'king'
    print(f"\nWords most similar to '{word_to_test_2}':")
    print(find_similar_words(word_to_test_2, glove_model))

    # --- 3. Manual Similarity Calculation with Cosine Similarity ---
    # Cosine similarity measures the cosine of the angle between two vectors.
    # A value of 1 means identical, 0 means orthogonal, -1 means opposite.
    def manual_cosine_similarity(vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)

    word1 = 'car'
    word2 = 'automobile'
    word3 = 'cloud'

    vec1 = glove_model[word1]
    vec2 = glove_model[word2]
    vec3 = glove_model[word3]

    sim_car_auto = manual_cosine_similarity(vec1, vec2)
    sim_car_cloud = manual_cosine_similarity(vec1, vec3)

    print(f"\nManual Cosine Similarity between '{word1}' and '{word2}': {sim_car_auto:.4f}")
    print(f"Manual Cosine Similarity between '{word1}' and '{word3}': {sim_car_cloud:.4f}")

    # --- 4. Word Analogies ---
    # The classic example: king - man + woman = queen
    # Gensim's most_similar function can do this directly with positive/negative lists
    def solve_analogy(pos_list, neg_list, model):
        """
        Solves an analogy of the form: pos_list[0] is to neg_list[0] as pos_list[1] is to ?
        e.g., king - man + woman -> queen
        pos_list = ['king', 'woman']
        neg_list = ['man']
        """
        try:
            result = model.most_similar(positive=pos_list, negative=neg_list, topn=1)
            return result[0]
        except KeyError as e:
            return f"Error: {e}. One of the words is not in the vocabulary."

    print("\n--- Solving Word Analogies ---")
    # Analogy: king - man + woman = ?
    analogy_result = solve_analogy(['king', 'woman'], ['man'], glove_model)
    print("king - man + woman ≈", analogy_result)

    # Analogy: france - paris + rome = ?
    analogy_result_2 = solve_analogy(['france', 'rome'], ['paris'], glove_model)
    print("france - paris + rome ≈", analogy_result_2)

    # Analogy: walking - walked + swimming = ?
    analogy_result_3 = solve_analogy(['walking', 'swam'], ['walked'], glove_model)
    print("walking - walked + swam ≈", analogy_result_3) # Note: 'swam' is past tense of 'swim', not 'swimming'

    # --- 5. Visualization of Word Relationships ---
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    words_to_visualize = ['king', 'queen', 'man', 'woman', 'prince', 'princess',
                         'car', 'truck', 'bus', 'bicycle', 'airplane',
                         'france', 'paris', 'germany', 'berlin', 'italy', 'rome']

    word_vectors = np.array([glove_model[word] for word in words_to_visualize])

    # Use t-SNE to reduce to 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    vectors_2d = tsne.fit_transform(word_vectors)

    # Plot the words
    plt.figure(figsize=(12, 10))
    for i, word in enumerate(words_to_visualize):
        x, y = vectors_2d[i]
        plt.scatter(x, y)
        plt.annotate(word, (x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.title("t-SNE Visualization of Word Clusters")
    plt.grid(True)
    plt.show()
```

## 📌 Project: Similar Word Search Tool

Now let's build a simple but functional web application using Streamlit that allows a user to type in a word and get back a list of the most similar words based on our GloVe embeddings.

**Implementation (`streamlit_word_search_app.py`):**

First, make sure you have your pre-trained GloVe file (`glove.6B.100d.word2vec.txt`) in the same directory as your app script.

**`streamlit_word_search_app.py`:**
```python
import streamlit as st
import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import torch
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Word Similarity Search",
    page_icon=":mag:",
    layout="centered"
)

# --- Load GloVe Model (with Caching) ---
# This is a large file, so we load it once and cache it.
@st.cache_resource
def load_glove_model(filepath):
    """
    Loads the GloVe model from the specified file path.
    Uses caching to avoid reloading on every script run.
    """
    try:
        print("Loading GloVe model...")
        model = KeyedVectors.load_word2vec_format(filepath, binary=False)
        print("GloVe model loaded successfully.")
        return model
    except FileNotFoundError:
        st.error(f"Error: GloVe model file not found at '{filepath}'.")
        st.error("Please download 'glove.6B.100d.txt' from https://nlp.stanford.edu/projects/glove/")
        st.error("And convert it to 'glove.6B.100d.word2vec.txt' format using gensim's glove2word2vec script.")
        return None

GLOVE_FILE = "glove.6B.100d.word2vec.txt"
glove_model = load_glove_model(GLOVE_FILE)

# --- App UI and Logic ---
st.title(":mag: Word Similarity Search Tool")
st.markdown("""
This tool uses **GloVe (Global Vectors for Word Representation)** to find words that are semantically similar to a given word.
Simply type a word into the box below and press Enter to see its closest neighbors in the vector space.
""")

# --- Input Widget ---
# Only show the input if the model loaded successfully
if glove_model:
    input_word = st.text_input("Enter a word:", "computer")

    if input_word:
        # --- Process Input and Find Similar Words ---
        # Convert to lowercase to match the GloVe vocabulary
        input_word_lower = input_word.lower()

        if input_word_lower in glove_model:
            # Get the vector for the input word
            input_vector = glove_model[input_word_lower].reshape(1, -1)

            # Get the full vocabulary and vectors
            vocab_list = list(glove_model.key_to_index.keys())
            all_vectors = glove_model.vectors

            # Calculate cosine similarity between the input word and all other words
            # This is more efficient than looping, but can be memory intensive for huge vocabs
            # For a very large vocab, you might use Annoy or FAISS for approximate nearest neighbors
            similarities = cosine_similarity(input_vector, all_vectors)

            # Get the top N most similar words (excluding the input word itself)
            top_n = 10
            # Get indices of the top N similarity scores
            top_indices = similarities.argsort()[0][-top_n-1:-1][::-1] # Exclude the word itself

            # Display results
            st.subheader(f"Top {top_n} words most similar to '{input_word_lower}':")

            # Create a DataFrame for nice display
            results_data = []
            for idx in top_indices:
                similar_word = vocab_list[idx]
                similarity_score = similarities[0][idx]
                results_data.append({"Word": similar_word, "Cosine Similarity": f"{similarity_score:.4f}"})

            st.dataframe(results_data, use_container_width=True)

            # --- Visualization ---
            st.subheader("Similarity Score Distribution")
            st.bar_chart(similarities[0])

        else:
            st.warning(f"The word '{input_word_lower}' was not found in the GloVe vocabulary.")
            st.info("Try another word, e.g., 'king', 'computer', 'france', 'happiness'.")

else:
    st.warning("The application cannot run because the GloVe model failed to load.")

# --- Sidebar with Information ---
st.sidebar.header("About GloVe")
st.sidebar.info("""
**GloVe** (Global Vectors for Word Representation) is an unsupervised learning algorithm for obtaining vector representations for words.

Key features:
- **Trained on global co-occurrence statistics:** It captures how often words appear together across a large corpus.
- **Dense vectors:** Each word is represented by a dense vector (e.g., 100 dimensions).
- **Semantic relationships:** Words with similar meanings have similar vectors. This allows for finding similar words and solving analogies (e.g., king - man + woman ≈ queen).

This app uses the `glove.6B.100d.txt` model, which was trained on 6 billion tokens from Wikipedia and Gigaword.
""")

st.sidebar.header("How it Works")
st.sidebar.markdown("""
1. **Input:** You enter a word.
2. **Lookup:** The app finds the corresponding 100-dimensional vector for your word from the pre-trained GloVe model.
3. **Comparison:** It calculates the **cosine similarity** between your word's vector and the vectors of all 400,000 other words in the vocabulary.
4. **Ranking:** It ranks all words by their similarity score.
5. **Display:** It shows you the top 10 words with the highest similarity scores.
""")
```

**How to Run the Project:**

1.  **Save the Code:** Save the code above as `streamlit_word_search_app.py`.
2.  **Get the GloVe File:** Make sure you have `glove.6B.100d.word2vec.txt` in the same directory. If you only have the original `.txt` file, you can convert it using the `gensim` script shown in the GloVe section.
3.  **Install Dependencies:**
    ```bash
    pip install streamlit gensim numpy scikit-learn torch
    ```
4.  **Run the App:**
    ```bash
    streamlit run streamlit_word_search_app.py
    ```
