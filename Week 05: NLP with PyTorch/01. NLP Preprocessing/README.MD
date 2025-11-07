# Day 21 – NLP Preprocessing

## Tokenization

### What is it?

Tokenization is the process of breaking down text into smaller units called tokens, which can be words, phrases, or symbols. It's the first and most fundamental step in Natural Language Processing (NLP), converting raw text into a structured format that machine learning models can process. Tokens serve as the basic building blocks for further NLP tasks like text analysis, sentiment analysis, and language modeling.

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import re
import string
from collections import Counter, defaultdict
import numpy as np

# Basic tokenization implementation
class BasicTokenizer:
    def __init__(self, lowercase=True, remove_punctuation=True):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.vocab = {}
        self.reverse_vocab = {}
        self.vocab_size = 0
    
    def tokenize(self, text):
        """Basic word tokenization"""
        if self.lowercase:
            text = text.lower()
        
        if self.remove_punctuation:
            # Remove punctuation
            text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split by whitespace
        tokens = text.split()
        return tokens
    
    def build_vocab(self, texts):
        """Build vocabulary from a list of texts"""
        all_tokens = []
        for text in texts:
            tokens = self.tokenize(text)
            all_tokens.extend(tokens)
        
        # Count token frequencies
        token_counts = Counter(all_tokens)
        
        # Create vocab mappings
        self.vocab = {'<PAD>': 0, '<UNK>': 1}  # Special tokens
        self.reverse_vocab = {0: '<PAD>', 1: '<UNK>'}
        
        for token, _ in token_counts.most_common():
            if token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[token] = idx
                self.reverse_vocab[idx] = token
        
        self.vocab_size = len(self.vocab)
        return self.vocab_size

# Advanced tokenization with subword techniques
class SubwordTokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        self.reverse_vocab = {0: '<PAD>', 1: '<UNK>', 2: '<SOS>', 3: '<EOS>'}
        self.char_vocab = {}
    
    def simple_bpe(self, texts, merges=1000):
        """Simple Byte-Pair Encoding implementation"""
        # Get all characters as initial vocabulary
        all_chars = set()
        for text in texts:
            all_chars.update(text.lower())
        
        # Initialize character vocabulary
        for char in sorted(all_chars):
            if char not in self.vocab:
                idx = len(self.vocab)
                self.vocab[char] = idx
                self.reverse_vocab[idx] = char
        
        # Create initial word tokens
        word_freq = Counter()
        for text in texts:
            words = text.lower().split()
            word_freq.update(words)
        
        # Perform BPE merges
        for merge in range(merges):
            pairs = Counter()
            
            # Find most frequent pairs
            for word, freq in word_freq.items():
                chars = list(word)
                for i in range(len(chars) - 1):
                    pair = (chars[i], chars[i+1])
                    pairs[pair] += freq
            
            if not pairs:
                break
                
            # Get most common pair
            best_pair = max(pairs, key=pairs.get)
            
            # Create new token
            new_token = best_pair[0] + best_pair[1]
            if new_token not in self.vocab and len(self.vocab) < self.vocab_size:
                idx = len(self.vocab)
                self.vocab[new_token] = idx
                self.reverse_vocab[idx] = new_token
            
            # Update word frequencies
            new_word_freq = Counter()
            for word, freq in word_freq.items():
                new_word = self._merge_word(word, best_pair, new_token)
                new_word_freq[new_word] += freq
            
            word_freq = new_word_freq
        
        self.vocab_size = len(self.vocab)
        return self.vocab_size
    
    def _merge_word(self, word, pair, new_token):
        """Merge a pair in a word"""
        chars = list(word)
        i = 0
        while i < len(chars) - 1:
            if chars[i] == pair[0] and chars[i+1] == pair[1]:
                chars[i] = new_token
                chars.pop(i+1)
            else:
                i += 1
        return ''.join(chars)
    
    def tokenize(self, text):
        """Tokenize text using subword units"""
        text = text.lower()
        words = text.split()
        tokens = []
        
        for word in words:
            # Simple greedy approach for subword tokenization
            subwords = self._tokenize_word(word)
            tokens.extend(subwords)
        
        return tokens
    
    def _tokenize_word(self, word):
        """Tokenize a single word into subwords"""
        # Simple greedy longest-match-first approach
        subwords = []
        start = 0
        
        while start < len(word):
            end = len(word)
            found = False
            
            while end > start:
                subword = word[start:end]
                if subword in self.vocab:
                    subwords.append(subword)
                    start = end
                    found = True
                    break
                end -= 1
            
            if not found:
                subwords.append('<UNK>')
                start += 1
        
        return subwords

# Demonstrate tokenization
def demonstrate_tokenization():
    # Sample texts
    texts = [
        "Hello world! This is a sample text.",
        "Natural language processing is fascinating.",
        "Machine learning models need tokenized input."
    ]
    
    # Basic tokenization
    basic_tokenizer = BasicTokenizer()
    basic_tokenizer.build_vocab(texts)
    
    print("Basic Tokenization Example:")
    for text in texts:
        tokens = basic_tokenizer.tokenize(text)
        print(f"Text: {text}")
        print(f"Tokens: {tokens}")
        print()
    
    # Subword tokenization
    subword_tokenizer = SubwordTokenizer(vocab_size=50)
    subword_tokenizer.simple_bpe(texts, merges=20)
    
    print("Subword Tokenization Example:")
    for text in texts:
        tokens = subword_tokenizer.tokenize(text)
        print(f"Text: {text}")
        print(f"Subword tokens: {tokens}")
        print()
    
    print(f"Vocabulary size - Basic: {basic_tokenizer.vocab_size}, Subword: {subword_tokenizer.vocab_size}")

demonstrate_tokenization()
```

### When and where we use tokenization

- **Text preprocessing pipelines**: First step before any NLP task
- **Machine translation**: Breaking down source and target texts
- **Text classification**: Converting documents to token sequences
- **Language modeling**: Training models on token sequences
- **Information retrieval**: Processing queries and documents
- **Sentiment analysis**: Analyzing token-level sentiment patterns

```python
# When to use tokenization - Text classification example
def text_classification_pipeline():
    """Example: Tokenization in text classification"""
    
    class TextClassifier(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
            self.classifier = nn.Linear(hidden_dim, num_classes)
        
        def forward(self, x):
            embedded = self.embedding(x)
            lstm_out, (hidden, _) = self.lstm(embedded)
            # Use the last hidden state
            output = self.classifier(hidden[-1])
            return output
    
    # Tokenizer for classification
    tokenizer = BasicTokenizer()
    sample_texts = ["This movie is great!", "Terrible experience.", "Amazing product!"]
    tokenizer.build_vocab(sample_texts)
    
    # Convert texts to sequences
    def text_to_sequence(text, vocab, max_length=10):
        tokens = tokenizer.tokenize(text)
        sequence = [vocab.get(token, vocab['<UNK>']) for token in tokens]
        # Pad or truncate
        if len(sequence) < max_length:
            sequence.extend([vocab['<PAD>']] * (max_length - len(sequence)))
        else:
            sequence = sequence[:max_length]
        return sequence
    
    # Example usage
    sequences = [text_to_sequence(text, tokenizer.vocab) for text in sample_texts]
    print("Text sequences for classification:", sequences)
    
    return TextClassifier(len(tokenizer.vocab), 128, 64, 2)

text_classifier = text_classification_pipeline()
```

### Real-world examples

- **Search engines**: Tokenizing queries and documents for indexing
- **Chatbots**: Breaking down user inputs for intent recognition
- **Spam detection**: Tokenizing emails for classification
- **Social media analysis**: Processing tweets, posts, and comments
- **Document summarization**: Tokenizing text for extractive methods

```python
# Real-world example: Social media text processing
def social_media_tokenizer():
    """Tokenization for social media text with special handling"""
    
    class SocialMediaTokenizer:
        def __init__(self):
            self.emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+', flags=re.UNICODE)
            self.hashtag_pattern = re.compile(r'#\w+')
            self.mention_pattern = re.compile(r'@\w+')
            self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        
        def tokenize(self, text):
            tokens = []
            
            # Extract hashtags, mentions, URLs separately
            hashtags = self.hashtag_pattern.findall(text)
            mentions = self.mention_pattern.findall(text)
            urls = self.url_pattern.findall(text)
            
            # Remove special patterns and tokenize remaining text
            clean_text = self.hashtag_pattern.sub(' ', text)
            clean_text = self.mention_pattern.sub(' ', clean_text)
            clean_text = self.url_pattern.sub(' ', clean_text)
            
            # Remove emojis
            clean_text = self.emoji_pattern.sub(' ', clean_text)
            
            # Basic tokenization
            basic_tokens = re.findall(r'\b\w+\b', clean_text.lower())
            
            # Combine all tokens
            tokens.extend(hashtags)
            tokens.extend(mentions)
            tokens.extend(['URL'] * len(urls))  # Replace URLs with generic token
            tokens.extend(basic_tokens)
            
            return tokens
    
    # Example usage
    sm_tokenizer = SocialMediaTokenizer()
    social_texts = [
        "Just had an amazing lunch! #foodie @restaurant #delicious",
        "Check out this link: https://example.com #tech news",
        "This product is 😍 absolutely fantastic!"
    ]
    
    for text in social_texts:
        tokens = sm_tokenizer.tokenize(text)
        print(f"Text: {text}")
        print(f"Tokens: {tokens}")
        print()
    
    return sm_tokenizer

social_tokenizer = social_media_tokenizer()
```

### Problems and Advancements

**Problem**: Traditional word-level tokenization can lead to large vocabularies and out-of-vocabulary (OOV) issues.
**Advancement**: Subword tokenization methods like BPE, WordPiece, and SentencePiece.

```python
# Problem: Large vocabulary with rare words
def problem_example():
    """Example of vocabulary explosion problem"""
    rare_words = ["antidisestablishmentarianism", "pneumonoultramicroscopicsilicovolcanoconiosis", "floccinaucinihilipilification"]
    
    basic_tokenizer = BasicTokenizer()
    basic_tokenizer.build_vocab(rare_words)
    
    print(f"Vocabulary size with rare words: {basic_tokenizer.vocab_size}")
    print("Problem: Each rare word creates a new vocabulary entry")

problem_example()

# Solution: Subword tokenization reduces vocabulary size
def solution_example():
    """Subword tokenization solution"""
    rare_words = ["antidisestablishmentarianism", "pneumonoultramicroscopicsilicovolcanoconiosis", "floccinaucinihilipilification"]
    
    subword_tokenizer = SubwordTokenizer(vocab_size=100)
    subword_tokenizer.simple_bpe(rare_words, merges=50)
    
    print(f"Vocabulary size with subword tokenization: {subword_tokenizer.vocab_size}")
    
    # Show how rare words are broken down
    for word in rare_words:
        tokens = subword_tokenizer.tokenize(word)
        print(f"Word: {word}")
        print(f"Subword tokens: {tokens}")
        print()

solution_example()
```

## Stopwords

### What is it?

Stopwords are common words in a language (like "the", "is", "at", "which", "on") that typically carry little semantic meaning and are often removed during text preprocessing. These words appear frequently but contribute minimally to the overall meaning of a text, making them noise for many NLP tasks. Removing stopwords can reduce dimensionality, improve model performance, and speed up processing.

### PyTorch Implementation

```python
# Stopwords implementation
class StopwordsRemover:
    def __init__(self, language='english'):
        # Common English stopwords
        self.stopwords = {
            'english': {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
                'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
                'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
                'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
                'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
                'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
                'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after', 'above', 
                'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
                'further', 'then', 'once'
            }
        }
        self.language = language
    
    def remove_stopwords(self, tokens):
        """Remove stopwords from token list"""
        if self.language not in self.stopwords:
            return tokens
        
        return [token for token in tokens if token.lower() not in self.stopwords[self.language]]
    
    def add_stopwords(self, new_stopwords):
        """Add custom stopwords"""
        if self.language in self.stopwords:
            self.stopwords[self.language].update(new_stopwords)
    
    def remove_stopwords_from_text(self, text):
        """Remove stopwords from raw text"""
        # First tokenize the text
        basic_tokenizer = BasicTokenizer()
        tokens = basic_tokenizer.tokenize(text)
        
        # Remove stopwords
        filtered_tokens = self.remove_stopwords(tokens)
        
        # Join back to text (optional)
        return ' '.join(filtered_tokens)

# Advanced stopwords with domain-specific lists
class DomainSpecificStopwords:
    def __init__(self):
        self.general_stopwords = StopwordsRemover('english')
        self.domain_stopwords = {
            'tech': {'api', 'sdk', 'json', 'xml', 'http', 'url', 'html', 'css', 'js', 'py', 'java'},
            'medical': {'patient', 'doctor', 'hospital', 'medicine', 'treatment', 'symptoms', 'diagnosis'},
            'finance': {'dollar', 'euro', 'stock', 'bond', 'portfolio', 'investment', 'trading', 'market'}
        }
    
    def remove_stopwords(self, tokens, domain=None):
        """Remove both general and domain-specific stopwords"""
        # Remove general stopwords
        filtered_tokens = self.general_stopwords.remove_stopwords(tokens)
        
        # Remove domain-specific stopwords if specified
        if domain and domain in self.domain_stopwords:
            filtered_tokens = [token for token in filtered_tokens 
                              if token.lower() not in self.domain_stopwords[domain]]
        
        return filtered_tokens

# Demonstrate stopwords removal
def demonstrate_stopwords():
    # Sample text
    texts = [
        "The quick brown fox jumps over the lazy dog in the park.",
        "This is a sample text with many common words that may not be useful.",
        "Natural language processing is a fascinating field of study."
    ]
    
    # Basic stopwords removal
    stopwords_remover = StopwordsRemover()
    
    print("Stopwords Removal Example:")
    for text in texts:
        tokens = text.lower().split()
        filtered_tokens = stopwords_remover.remove_stopwords(tokens)
        
        print(f"Original: {text}")
        print(f"Tokens: {tokens}")
        print(f"Filtered: {filtered_tokens}")
        print()
    
    # Domain-specific stopwords
    domain_stopwords = DomainSpecificStopwords()
    
    tech_texts = [
        "API documentation and JSON parsing are important in web development.",
        "The SDK provides tools for HTTP requests and URL handling."
    ]
    
    print("Domain-Specific Stopwords Example (Tech):")
    for text in tech_texts:
        tokens = text.lower().split()
        filtered_tokens = domain_stopwords.remove_stopwords(tokens, domain='tech')
        
        print(f"Original: {text}")
        print(f"Filtered: {filtered_tokens}")
        print()

demonstrate_stopwords()
```

### When and where we use stopwords

- **Information retrieval**: Removing common words from search queries
- **Text classification**: Reducing noise in feature vectors
- **Document clustering**: Focusing on meaningful terms
- **Sentiment analysis**: Removing neutral words
- **Machine translation**: Filtering out non-essential words
- **Topic modeling**: Identifying important terms

```python
# When to use stopwords - Text classification example
def classification_with_stopwords():
    """Example: Stopwords in text classification"""
    
    # Sample documents with different classes
    documents = [
        ("The movie was absolutely fantastic and entertaining", "positive"),
        ("This film is terrible and boring", "negative"),
        ("The book was okay, nothing special", "neutral"),
        ("Amazing experience, highly recommended", "positive")
    ]
    
    # Without stopwords removal
    basic_tokenizer = BasicTokenizer()
    all_texts = [doc[0] for doc in documents]
    basic_tokenizer.build_vocab(all_texts)
    
    # With stopwords removal
    stopwords_remover = StopwordsRemover()
    processed_texts = []
    
    for text, label in documents:
        tokens = basic_tokenizer.tokenize(text)
        filtered_tokens = stopwords_remover.remove_stopwords(tokens)
        processed_text = ' '.join(filtered_tokens)
        processed_texts.append((processed_text, label))
    
    print("Text Classification with Stopwords:")
    for original, processed in zip(documents, processed_texts):
        print(f"Original: {original[0]} -> {original[1]}")
        print(f"Processed: {processed[0]} -> {processed[1]}")
        print()

classification_with_stopwords()
```

### Real-world examples

- **Search engines**: Removing common words from queries
- **Recommendation systems**: Filtering out generic terms
- **Social media monitoring**: Focusing on meaningful content
- **Legal document analysis**: Removing boilerplate language
- **Medical records processing**: Removing common medical terms

```python
# Real-world example: Search query processing
def search_query_processor():
    """Stopwords removal for search queries"""
    
    class SearchQueryProcessor:
        def __init__(self):
            self.stopwords_remover = StopwordsRemover()
        
        def process_query(self, query):
            """Process search query by removing stopwords"""
            tokens = query.lower().split()
            filtered_tokens = self.stopwords_remover.remove_stopwords(tokens)
            return ' '.join(filtered_tokens)
        
        def expand_query(self, query):
            """Expand query with synonyms (simplified)"""
            processed_query = self.process_query(query)
            
            # Simple synonym expansion (in practice, use WordNet or embeddings)
            synonym_map = {
                'buy': ['purchase', 'acquire', 'obtain'],
                'car': ['vehicle', 'automobile', 'auto'],
                'house': ['home', 'property', 'residence']
            }
            
            expanded_terms = []
            for term in processed_query.split():
                expanded_terms.append(term)
                if term in synonym_map:
                    expanded_terms.extend(synonym_map[term])
            
            return ' '.join(expanded_terms)
    
    # Example usage
    processor = SearchQueryProcessor()
    queries = [
        "the best car for sale in my area",
        "house for rent with garden and parking",
        "how to buy a new smartphone online"
    ]
    
    print("Search Query Processing:")
    for query in queries:
        processed = processor.process_query(query)
        expanded = processor.expand_query(query)
        
        print(f"Original: {query}")
        print(f"Processed: {processed}")
        print(f"Expanded: {expanded}")
        print()

search_query_processor()
```

### Problems and Advancements

**Problem**: Removing stopwords can sometimes remove context that's important for certain tasks.
**Advancement**: Task-specific stopwords lists and conditional removal.

```python
# Problem: Context loss with aggressive stopwords removal
def problem_example():
    """Example of context loss with stopwords"""
    sentences = [
        "What is the capital of France?",
        "How to use the best practices in Python?"
    ]
    
    stopwords_remover = StopwordsRemover()
    
    print("Problem: Context loss with stopwords removal")
    for sentence in sentences:
        tokens = sentence.lower().split()
        filtered = stopwords_remover.remove_stopwords(tokens)
        
        print(f"Original: {sentence}")
        print(f"Filtered: {' '.join(filtered)}")
        print("Context lost: 'what', 'the', 'how' provide important context")
        print()

problem_example()

# Solution: Task-specific stopwords
def solution_example():
    """Task-specific stopwords solution"""
    
    class TaskSpecificStopwords:
        def __init__(self):
            self.stopwords = {
                'classification': {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'},
                'question_answering': {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'for', 'of', 'with', 'by'},  # Keep question words
                'summarization': {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
            }
        
        def remove_stopwords(self, tokens, task='classification'):
            if task in self.stopwords:
                return [token for token in tokens if token.lower() not in self.stopwords[task]]
            else:
                # Use default stopwords
                default_remover = StopwordsRemover()
                return default_remover.remove_stopwords(tokens)
    
    # Example usage
    task_stopwords = TaskSpecificStopwords()
    
    question = "What is the capital of France?"
    tokens = question.lower().split()
    
    print("Solution: Task-specific stopwords")
    print(f"Original: {question}")
    print(f"Classification task: {' '.join(task_stopwords.remove_stopwords(tokens, 'classification'))}")
    print(f"Question answering: {' '.join(task_stopwords.remove_stopwords(tokens, 'question_answering'))}")
    print(f"Summarization task: {' '.join(task_stopwords.remove_stopwords(tokens, 'summarization'))}")

solution_example()
```

## Lemmatization

### What is it?

Lemmatization is the process of reducing words to their base or dictionary form (lemma) by considering the word's morphological analysis. Unlike stemming, which simply removes suffixes, lemmatization uses vocabulary and morphological analysis to correctly identify the base form of a word. For example, "running" becomes "run", "better" becomes "good", and "mice" becomes "mouse".

### PyTorch Implementation

```python
# Basic lemmatization implementation
class BasicLemmatizer:
    def __init__(self):
        # Common word mappings
        self.lemma_dict = {
            # Irregular verbs
            'am': 'be', 'is': 'be', 'are': 'be', 'was': 'be', 'were': 'be', 'been': 'be', 'being': 'be',
            'have': 'have', 'has': 'have', 'had': 'have', 'having': 'have',
            'do': 'do', 'does': 'do', 'did': 'do', 'doing': 'do',
            
            # Comparative/superlative forms
            'better': 'good', 'best': 'good',
            'worse': 'bad', 'worst': 'bad',
            'further': 'far', 'farther': 'far', 'furthest': 'far', 'farthest': 'far',
            
            # Plural forms
            'children': 'child', 'mice': 'mouse', 'feet': 'foot', 'teeth': 'tooth',
            'geese': 'goose', 'men': 'man', 'women': 'woman', 'people': 'person',
            
            # Common verb forms
            'running': 'run', 'ran': 'run', 'runs': 'run',
            'walking': 'walk', 'walked': 'walk', 'walks': 'walk',
            'eating': 'eat', 'ate': 'eat', 'eaten': 'eat', 'eats': 'eat',
            'singing': 'sing', 'sang': 'sing', 'sung': 'sing', 'sings': 'sing',
            
            # Common adjective forms
            'happier': 'happy', 'happiest': 'happy', 'happily': 'happy',
            'easier': 'easy', 'easiest': 'easy', 'easily': 'easy',
            'angrier': 'angry', 'angriest': 'angry', 'angrily': 'angry'
        }
        
        # Suffix rules for automatic lemmatization
        self.suffix_rules = [
            # Verb forms
            ('ies', 'y'), ('es', ''), ('s', ''),
            ('ing', ''), ('ed', ''), ('er', ''), ('est', ''),
            ('ly', ''), ('tion', 'te'), ('sion', 'se')
        ]
    
    def lemmatize_word(self, word, pos='noun'):
        """Lemmatize a single word"""
        word_lower = word.lower()
        
        # Check if word is in dictionary
        if word_lower in self.lemma_dict:
            return self.lemma_dict[word_lower]
        
        # Apply suffix rules
        for suffix, replacement in self.suffix_rules:
            if word_lower.endswith(suffix):
                # Simple replacement (in practice, check if it's a valid word)
                candidate = word_lower[:-len(suffix)] + replacement
                if len(candidate) > 2:  # Basic validation
                    return candidate
        
        # Return original word if no rule applies
        return word_lower
    
    def lemmatize_sentence(self, sentence):
        """Lemmatize all words in a sentence"""
        tokens = sentence.lower().split()
        lemmatized_tokens = [self.lemmatize_word(token) for token in tokens]
        return ' '.join(lemmatized_tokens)

# Advanced lemmatization with part-of-speech awareness
class POSAwareLemmatizer:
    def __init__(self):
        # POS-specific lemmatization rules
        self.pos_rules = {
            'noun': {
                'children': 'child', 'mice': 'mouse', 'feet': 'foot', 'teeth': 'tooth',
                'geese': 'goose', 'men': 'man', 'women': 'woman', 'people': 'person',
                'geese': 'goose', 'oxen': 'ox', 'phenomena': 'phenomenon', 'criteria': 'criterion'
            },
            'verb': {
                'am': 'be', 'is': 'be', 'are': 'be', 'was': 'be', 'were': 'be', 'been': 'be', 'being': 'be',
                'have': 'have', 'has': 'have', 'had': 'have', 'having': 'have',
                'do': 'do', 'does': 'do', 'did': 'do', 'doing': 'do',
                'running': 'run', 'ran': 'run', 'runs': 'run', 'run': 'run',
                'walking': 'walk', 'walked': 'walk', 'walks': 'walk', 'walk': 'walk',
                'eating': 'eat', 'ate': 'eat', 'eaten': 'eat', 'eats': 'eat', 'eat': 'eat',
                'singing': 'sing', 'sang': 'sing', 'sung': 'sing', 'sings': 'sing', 'sing': 'sing'
            },
            'adj': {
                'better': 'good', 'best': 'good', 'good': 'good',
                'worse': 'bad', 'worst': 'bad', 'bad': 'bad',
                'further': 'far', 'farther': 'far', 'furthest': 'far', 'farthest': 'far', 'far': 'far',
                'happier': 'happy', 'happiest': 'happy', 'happily': 'happy', 'happy': 'happy',
                'easier': 'easy', 'easiest': 'easy', 'easily': 'easy', 'easy': 'easy'
            }
        }
        
        # Default suffix rules
        self.suffix_rules = {
            'noun': [('ies', 'y'), ('es', ''), ('s', '')],
            'verb': [('ing', ''), ('ed', ''), ('s', '')],
            'adj': [('er', ''), ('est', ''), ('ly', '')]
        }
    
    def lemmatize_word(self, word, pos='noun'):
        """Lemmatize word with POS information"""
        word_lower = word.lower()
        
        # Check POS-specific dictionary
        if pos in self.pos_rules and word_lower in self.pos_rules[pos]:
            return self.pos_rules[pos][word_lower]
        
        # Apply POS-specific suffix rules
        if pos in self.suffix_rules:
            for suffix, replacement in self.suffix_rules[pos]:
                if word_lower.endswith(suffix):
                    candidate = word_lower[:-len(suffix)] + replacement
                    if len(candidate) > 2:
                        return candidate
        
        return word_lower
    
    def simple_pos_tag(self, word):
        """Simple POS tagging (in practice, use proper POS tagger)"""
        # Basic heuristic for demonstration
        if word.endswith('ing') or word.endswith('ed') or word in ['am', 'is', 'are', 'was', 'were', 'have', 'has', 'had', 'do', 'does', 'did']:
            return 'verb'
        elif word.endswith('ly') or word in ['better', 'best', 'worse', 'worst', 'happier', 'happiest']:
            return 'adj'
        else:
            return 'noun'

# Demonstrate lemmatization
def demonstrate_lemmatization():
    # Basic lemmatization
    basic_lemmatizer = BasicLemmatizer()
    
    test_words = [
        'running', 'ran', 'runs', 'children', 'better', 'best',
        'walking', 'walked', 'walking', 'mice', 'feet', 'easier'
    ]
    
    print("Basic Lemmatization:")
    for word in test_words:
        lemma = basic_lemmatizer.lemmatize_word(word)
        print(f"{word} -> {lemma}")
    
    print("\nSentence Lemmatization:")
    sentences = [
        "The children were running and playing in the park.",
        "She is better than her sister at singing.",
        "The mice ran quickly across the floor."
    ]
    
    for sentence in sentences:
        lemmatized = basic_lemmatizer.lemmatize_sentence(sentence)
        print(f"Original: {sentence}")
        print(f"Lemmatized: {lemmatized}")
        print()
    
    # POS-aware lemmatization
    pos_lemmatizer = POSAwareLemmatizer()
    
    print("POS-Aware Lemmatization:")
    test_pairs = [
        ('better', 'adj'),
        ('running', 'verb'),
        ('children', 'noun'),
        ('ran', 'verb')
    ]
    
    for word, pos in test_pairs:
        lemma = pos_lemmatizer.lemmatize_word(word, pos)
        print(f"{word} ({pos}) -> {lemma}")

demonstrate_lemmatization()
```

### When and where we use lemmatization

- **Text preprocessing**: Before feeding text to NLP models
- **Information retrieval**: Improving search accuracy
- **Sentiment analysis**: Grouping related words
- **Text classification**: Reducing feature space
- **Machine translation**: Normalizing source text
- **Question answering**: Matching question and answer patterns

```python
# When to use lemmatization - Text classification example
def lemmatization_for_classification():
    """Lemmatization in text classification pipeline"""
    
    # Sample documents
    documents = [
        "The cat is running in the garden.",
        "Cats run through gardens quickly.",
        "A cat ran across the yard.",
        "Dogs are barking loudly.",
        "The dog barked all night.",
        "Dogs bark frequently."
    ]
    
    # Without lemmatization
    basic_tokenizer = BasicTokenizer()
    basic_tokenizer.build_vocab(documents)
    
    # With lemmatization
    lemmatizer = BasicLemmatizer()
    processed_docs = []
    
    for doc in documents:
        tokens = doc.lower().split()
        lemmatized_tokens = [lemmatizer.lemmatize_word(token) for token in tokens]
        processed_docs.append(' '.join(lemmatized_tokens))
    
    print("Lemmatization for Text Classification:")
    for original, processed in zip(documents, processed_docs):
        print(f"Original: {original}")
        print(f"Processed: {processed}")
        print()
    
    # Show vocabulary reduction
    all_processed_tokens = ' '.join(processed_docs).split()
    unique_tokens = set(all_processed_tokens)
    
    all_original_tokens = ' '.join(documents).split()
    unique_original = set(all_original_tokens)
    
    print(f"Original unique tokens: {len(unique_original)}")
    print(f"Lemmatized unique tokens: {len(unique_tokens)}")
    print(f"Reduction: {len(unique_original) - len(unique_tokens)} tokens")

lemmatization_for_classification()
```

### Real-world examples

- **Search engines**: Matching "running" and "ran" as the same concept
- **Chatbots**: Understanding different forms of user queries
- **Document clustering**: Grouping documents with similar concepts
- **Sentiment analysis**: Treating "happy", "happier", "happiest" as related
- **Legal document analysis**: Normalizing legal terminology

```python
# Real-world example: Customer review processing
def customer_review_lemmatizer():
    """Lemmatization for customer reviews"""
    
    class ReviewProcessor:
        def __init__(self):
            self.lemmatizer = BasicLemmatizer()
            self.stopwords_remover = StopwordsRemover()
        
        def process_review(self, review):
            """Process customer review with lemmatization and stopwords removal"""
            # Tokenize
            tokens = review.lower().split()
            
            # Lemmatize
            lemmatized_tokens = [self.lemmatizer.lemmatize_word(token) for token in tokens]
            
            # Remove stopwords
            filtered_tokens = self.stopwords_remover.remove_stopwords(lemmatized_tokens)
            
            return ' '.join(filtered_tokens)
        
        def extract_sentiment_words(self, review):
            """Extract sentiment-related words after processing"""
            processed = self.process_review(review)
            sentiment_indicators = {
                'positive': ['good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'best', 'wonderful'],
                'negative': ['bad', 'terrible', 'awful', 'hate', 'worst', 'poor', 'disappointing', 'horrible']
            }
            
            tokens = processed.split()
            positive_words = [word for word in tokens if word in sentiment_indicators['positive']]
            negative_words = [word for word in tokens if word in sentiment_indicators['negative']]
            
            return positive_words, negative_words
    
    # Example usage
    processor = ReviewProcessor()
    reviews = [
        "The product is absolutely amazing and I love it!",
        "Terrible quality, worst purchase ever made.",
        "The service was excellent and staff were very helpful.",
        "Disappointing experience, would not recommend."
    ]
    
    print("Customer Review Processing:")
    for review in reviews:
        processed = processor.process_review(review)
        pos_words, neg_words = processor.extract_sentiment_words(review)
        
        print(f"Original: {review}")
        print(f"Processed: {processed}")
        print(f"Positive words: {pos_words}")
        print(f"Negative words: {neg_words}")
        print()

customer_review_lemmatizer()
```

### Problems and Advancements

**Problem**: Lemmatization requires POS tagging and large dictionaries, making it computationally expensive.
**Advancement**: Neural lemmatizers and pre-trained models for faster processing.

```python
# Problem: Computational complexity of lemmatization
def problem_example():
    """Example of computational complexity in lemmatization"""
    
    import time
    
    # Large text for demonstration
    large_text = "The running children were happily playing in the garden. The children ran around the garden. Better children play better games."
    
    lemmatizer = BasicLemmatizer()
    
    start_time = time.time()
    for _ in range(1000):  # Simulate multiple processing
        tokens = large_text.lower().split()
        lemmatized = [lemmatizer.lemmatize_word(token) for token in tokens]
    end_time = time.time()
    
    print(f"Time taken for basic lemmatization: {end_time - start_time:.4f} seconds")
    print("Problem: Basic lemmatization is slow for large texts")

problem_example()

# Solution: Optimized lemmatization with caching
def solution_example():
    """Optimized lemmatization with caching"""
    
    class OptimizedLemmatizer:
        def __init__(self):
            self.lemma_dict = {
                'am': 'be', 'is': 'be', 'are': 'be', 'was': 'be', 'were': 'be', 'been': 'be', 'being': 'be',
                'have': 'have', 'has': 'have', 'had': 'have', 'having': 'have',
                'do': 'do', 'does': 'do', 'did': 'do', 'doing': 'do',
                'running': 'run', 'ran': 'run', 'runs': 'run', 'run': 'run',
                'children': 'child', 'mice': 'mouse', 'feet': 'foot', 'teeth': 'tooth',
                'better': 'good', 'best': 'good', 'good': 'good'
            }
            self.cache = {}  # Cache for previously lemmatized words
        
        def lemmatize_word(self, word):
            """Lemmatize with caching"""
            word_lower = word.lower()
            
            # Check cache first
            if word_lower in self.cache:
                return self.cache[word_lower]
            
            # Check dictionary
            if word_lower in self.lemma_dict:
                lemma = self.lemma_dict[word_lower]
            else:
                # Apply simple rules (simplified)
                if word_lower.endswith('ing'):
                    lemma = word_lower[:-3]  # Remove 'ing'
                elif word_lower.endswith('ed'):
                    lemma = word_lower[:-2]  # Remove 'ed'
                else:
                    lemma = word_lower
            
            # Store in cache
            self.cache[word_lower] = lemma
            return lemma
    
    import time
    
    # Test optimized version
    large_text = "The running children were happily playing in the garden. The children ran around the garden. Better children play better games."
    optimized_lemmatizer = OptimizedLemmatizer()
    
    start_time = time.time()
    for _ in range(1000):  # Simulate multiple processing
        tokens = large_text.lower().split()
        lemmatized = [optimized_lemmatizer.lemmatize_word(token) for token in tokens]
    end_time = time.time()
    
    print(f"Time taken for optimized lemmatization: {end_time - start_time:.4f} seconds")
    print("Solution: Caching reduces repeated computation")

solution_example()
```

## 🧪 Exercise: Clean and process raw text

```python
def clean_and_process_raw_text():
    """Complete exercise: Clean and process raw text"""
    
    # Sample raw text with various issues
    raw_texts = [
        "This is a sample text with PUNCTUATION!!! And CAPITALS???",
        "The children were running and playing happily in the garden.",
        "We've got the BEST products at AMAZING prices. Buy now!!!",
        "Machine learning is fascinating. It's changing everything rapidly.",
        "The running dogs were barking loudly in the park."
    ]
    
    # Initialize components
    tokenizer = BasicTokenizer(lowercase=True, remove_punctuation=True)
    stopwords_remover = StopwordsRemover()
    lemmatizer = BasicLemmatizer()
    
    print("Raw Text Processing Exercise:")
    print("=" * 50)
    
    processed_results = []
    
    for i, text in enumerate(raw_texts, 1):
        print(f"\nText {i}: {text}")
        
        # Step 1: Tokenization
        tokens = tokenizer.tokenize(text)
        print(f"Tokens: {tokens}")
        
        # Step 2: Stopwords removal
        filtered_tokens = stopwords_remover.remove_stopwords(tokens)
        print(f"After stopwords removal: {filtered_tokens}")
        
        # Step 3: Lemmatization
        lemmatized_tokens = [lemmatizer.lemmatize_word(token) for token in filtered_tokens]
        print(f"After lemmatization: {lemmatized_tokens}")
        
        # Final processed text
        processed_text = ' '.join(lemmatized_tokens)
        print(f"Final: {processed_text}")
        
        processed_results.append({
            'original': text,
            'tokens': tokens,
            'filtered': filtered_tokens,
            'lemmatized': lemmatized_tokens,
            'final': processed_text
        })
    
    # Build vocabulary from processed texts
    all_texts = [result['final'] for result in processed_results]
    tokenizer.build_vocab(all_texts)
    
    print(f"\nVocabulary size: {tokenizer.vocab_size}")
    print(f"Sample vocabulary: {dict(list(tokenizer.vocab.items())[:10])}")
    
    # Convert to numerical sequences
    def text_to_sequence(text, vocab, max_length=20):
        tokens = text.split()
        sequence = [vocab.get(token, vocab['<UNK>']) for token in tokens]
        # Pad or truncate
        if len(sequence) < max_length:
            sequence.extend([vocab['<PAD>']] * (max_length - len(sequence)))
        else:
            sequence = sequence[:max_length]
        return sequence
    
    # Convert all processed texts to sequences
    sequences = [text_to_sequence(result['final'], tokenizer.vocab) for result in processed_results]
    
    print(f"\nNumerical sequences:")
    for i, seq in enumerate(sequences):
        print(f"Text {i+1}: {seq}")
    
    return processed_results, sequences

# Run the exercise
exercise_results, sequences = clean_and_process_raw_text()
```

## 📌 Project: Text cleaner utility

```python
class TextCleanerUtility:
    """A comprehensive text cleaning utility"""
    
    def __init__(self):
        self.tokenizer = BasicTokenizer()
        self.stopwords_remover = StopwordsRemover()
        self.lemmatizer = BasicLemmatizer()
        self.domain_stopwords = DomainSpecificStopwords()
        
        # Common text cleaning patterns
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'url': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'number': r'\b\d+\b',
            'whitespace': r'\s+'
        }
    
    def clean_text(self, text, remove_emails=True, remove_urls=True, remove_numbers=False, 
                   remove_stopwords=True, lemmatize=True, domain=None, custom_stopwords=None):
        """Comprehensive text cleaning"""
        
        # Remove specific patterns
        if remove_emails:
            text = re.sub(self.patterns['email'], ' EMAIL ', text)
        if remove_urls:
            text = re.sub(self.patterns['url'], ' URL ', text)
        if remove_numbers:
            text = re.sub(self.patterns['number'], ' NUMBER ', text)
        
        # Normalize whitespace
        text = re.sub(self.patterns['whitespace'], ' ', text).strip()
        
        # Tokenization
        tokens = self.tokenizer.tokenize(text)
        
        # Remove custom stopwords if provided
        if custom_stopwords:
            tokens = [token for token in tokens if token.lower() not in custom_stopwords]
        
        # Remove domain-specific stopwords
        if domain:
            tokens = self.domain_stopwords.remove_stopwords(tokens, domain)
        elif remove_stopwords:
            tokens = self.stopwords_remover.remove_stopwords(tokens)
        
        # Lemmatization
        if lemmatize:
            tokens = [self.lemmatizer.lemmatize_word(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def batch_clean(self, texts, **kwargs):
        """Clean multiple texts at once"""
        cleaned_texts = []
        for text in texts:
            cleaned = self.clean_text(text, **kwargs)
            cleaned_texts.append(cleaned)
        return cleaned_texts
    
    def analyze_text(self, text):
        """Analyze text statistics"""
        tokens = self.tokenizer.tokenize(text)
        
        stats = {
            'original_length': len(text),
            'token_count': len(tokens),
            'unique_tokens': len(set(tokens)),
            'avg_token_length': np.mean([len(token) for token in tokens]) if tokens else 0,
            'most_common_tokens': Counter(tokens).most_common(5)
        }
        
        return stats
    
    def save_cleaned_text(self, text, filepath):
        """Save cleaned text to file"""
        cleaned = self.clean_text(text)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(cleaned)
        print(f"Cleaned text saved to {filepath}")
    
    def create_vocabulary(self, texts, min_freq=2, max_vocab_size=10000):
        """Create vocabulary from texts"""
        all_tokens = []
        for text in texts:
            cleaned = self.clean_text(text)
            tokens = cleaned.split()
            all_tokens.extend(tokens)
        
        # Count frequencies
        token_counts = Counter(all_tokens)
        
        # Filter by frequency and create vocab
        filtered_tokens = {token: count for token, count in token_counts.items() 
                          if count >= min_freq}
        
        # Limit vocabulary size
        if len(filtered_tokens) > max_vocab_size:
            filtered_tokens = dict(Counter(filtered_tokens).most_common(max_vocab_size))
        
        # Create vocab mappings
        vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        for token in filtered_tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
        
        return vocab

# Example usage of the text cleaner utility
def demonstrate_text_cleaner():
    """Demonstrate the text cleaner utility"""
    
    cleaner = TextCleanerUtility()
    
    # Sample texts with various issues
    sample_texts = [
        "Contact us at john@example.com or visit https://example.com for more info!",
        "The children were running and playing in the garden. It was amazing!",
        "Call 555-123-4567 for support. We offer great services at low prices.",
        "Machine learning algorithms are powerful tools for data analysis.",
        "The running dogs were barking loudly. It was quite disturbing."
    ]
    
    print("Text Cleaner Utility Demo:")
    print("=" * 50)
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\nOriginal Text {i}: {text}")
        
        # Clean with different options
        cleaned_basic = cleaner.clean_text(text, remove_stopwords=True, lemmatize=True)
        print(f"Basic cleaning: {cleaned_basic}")
        
        cleaned_advanced = cleaner.clean_text(
            text, 
            remove_emails=True, 
            remove_urls=True, 
            remove_numbers=True,
            remove_stopwords=True, 
            lemmatize=True,
            domain='tech'  # Example domain
        )
        print(f"Advanced cleaning: {cleaned_advanced}")
        
        # Analyze text
        stats = cleaner.analyze_text(text)
        print(f"Text stats: {stats}")
    
    # Batch processing
    print(f"\nBatch Processing:")
    batch_cleaned = cleaner.batch_clean(sample_texts, remove_emails=True, remove_urls=True)
    for i, cleaned in enumerate(batch_cleaned, 1):
        print(f"Text {i}: {cleaned}")
    
    # Create vocabulary
    print(f"\nCreating vocabulary...")
    vocab = cleaner.create_vocabulary(sample_texts, min_freq=1, max_vocab_size=100)
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Sample vocab: {dict(list(vocab.items())[:10])}")
    
    # Save example
    example_text = "This is an example text that will be cleaned and saved."
    cleaned_example = cleaner.clean_text(example_text)
    print(f"\nExample cleaned text: {cleaned_example}")
    
    return cleaner

# Create and demonstrate the text cleaner
text_cleaner = demonstrate_text_cleaner()

# Additional utility functions
def text_preprocessing_pipeline(texts, pipeline_config=None):
    """Configurable text preprocessing pipeline"""
    
    if pipeline_config is None:
        pipeline_config = {
            'remove_emails': True,
            'remove_urls': True,
            'remove_numbers': False,
            'remove_stopwords': True,
            'lemmatize': True,
            'domain': None
        }
    
    cleaner = TextCleanerUtility()
    processed_texts = cleaner.batch_clean(texts, **pipeline_config)
    
    return processed_texts

# Example of different preprocessing configurations
def pipeline_examples():
    """Examples of different preprocessing pipelines"""
    
    texts = [
        "The running children were playing in the park.",
        "Machine learning algorithms are very powerful.",
        "Contact: john@example.com or visit https://example.com"
    ]
    
    print("\nDifferent Preprocessing Pipelines:")
    print("=" * 40)
    
    # Minimal preprocessing
    minimal_config = {
        'remove_emails': False,
        'remove_urls': False,
        'remove_numbers': False,
        'remove_stopwords': False,
        'lemmatize': False
    }
    minimal_result = text_preprocessing_pipeline(texts, minimal_config)
    print("Minimal preprocessing:", minimal_result[0])
    
    # Standard preprocessing
    standard_config = {
        'remove_emails': True,
        'remove_urls': True,
        'remove_numbers': False,
        'remove_stopwords': True,
        'lemmatize': True
    }
    standard_result = text_preprocessing_pipeline(texts, standard_config)
    print("Standard preprocessing:", standard_result[0])
    
    # Aggressive preprocessing
    aggressive_config = {
        'remove_emails': True,
        'remove_urls': True,
        'remove_numbers': True,
        'remove_stopwords': True,
        'lemmatize': True
    }
    aggressive_result = text_preprocessing_pipeline(texts, aggressive_config)
    print("Aggressive preprocessing:", aggressive_result[0])

pipeline_examples()
```