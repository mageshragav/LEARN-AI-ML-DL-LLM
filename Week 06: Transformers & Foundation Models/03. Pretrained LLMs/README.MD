# Day 28 – Pretrained LLMs: BERT, GPT, T5 with HuggingFace

## What are Pretrained LLMs?

Pretrained Large Language Models (LLMs) are neural networks that have been trained on massive amounts of text data to understand and generate human-like text. These models learn patterns, relationships, and structures in language during pretraining, and can then be fine-tuned for specific tasks with relatively little additional training data.

The key advantage of pretrained LLMs is transfer learning - the knowledge gained during pretraining can be applied to a wide variety of downstream tasks, from text classification to question answering and content generation.

## Essential Subtopics

### 1. BERT (Bidirectional Encoder Representations from Transformers)

BERT, introduced by Google in 2018, is a transformer-based model that's designed to pretrain deep bidirectional representations from unlabeled text. Unlike previous models that looked at text sequences from left-to-right or right-to-left, BERT simultaneously considers context from both directions.

**Key Concepts:**
- Bidirectional context understanding
- Masked Language Modeling (MLM) pretraining objective
- Next Sentence Prediction (NSP) pretraining objective
- Fine-tuning approach for downstream tasks

**Coding Example - Using BERT for text classification:**

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load pre-trained model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Example text
text = "BERT is a powerful language model for understanding context."

# Tokenize input
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Forward pass
outputs = model(**inputs)
logits = outputs.logits
predictions = torch.argmax(logits, dim=1)

print(f"Predicted class: {predictions.item()}")
```

### 2. GPT (Generative Pre-trained Transformer)

GPT, developed by OpenAI, is a series of autoregressive language models that use the transformer architecture. Unlike BERT, GPT is designed primarily for text generation, predicting the next token in a sequence based on previous context.

**Key Concepts:**
- Autoregressive generation (left-to-right)
- Unsupervised pretraining on large text corpora
- Zero-shot, one-shot, and few-shot learning capabilities
- Scaling laws showing improved performance with increased model size

**Coding Example - Using GPT-2 for text generation:**

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load pre-trained model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set pad token to eos token
tokenizer.pad_token = tokenizer.eos_token

# Example prompt
prompt = "The future of artificial intelligence is"

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text
output = model.generate(
    inputs.input_ids, 
    max_length=100, 
    num_return_sequences=1,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

# Decode and print generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

### 3. T5 (Text-to-Text Transfer Transformer)

T5, introduced by Google in 2020, frames all NLP tasks as a text-to-text problem. This means both the input and output are always text strings, making the model highly flexible for various tasks without needing task-specific architectures.

**Key Concepts:**
- Text-to-text framework for all NLP tasks
- Prefix-based task specification
- Encoder-decoder architecture
- Multitask pretraining on C4 dataset

**Coding Example - Using T5 for text summarization:**

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load pre-trained model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Example text to summarize
text = """The Apollo 11 mission was the first manned mission to land on the Moon. 
The mission was carried out by NASA and launched on July 16, 1969. 
On July 20, 1969, astronauts Neil Armstrong and Buzz Aldrin became the first humans to walk on the lunar surface."""

# Prepare input with task prefix
input_text = f"summarize: {text}"

# Tokenize input
inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

# Generate summary
outputs = model.generate(
    inputs.input_ids,
    max_length=100,
    min_length=30,
    length_penalty=2.0,
    num_beams=4,
    early_stopping=True
)

# Decode and print summary
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(summary)
```

### 4. HuggingFace Transformers Library

HuggingFace Transformers is an open-source library that provides thousands of pretrained models for various NLP tasks. It offers a unified API for using these models, making it easy to switch between different architectures like BERT, GPT, T5, and many others.

**Key Concepts:**
- Model hub with thousands of pretrained models
- Unified API for different models and tasks
- Tokenizers for preprocessing text
- Pipelines for easy inference
- Training and fine-tuning utilities

**Coding Example - Using HuggingFace pipelines for easy inference:**

```python
from transformers import pipeline

# Text classification pipeline
classifier = pipeline("sentiment-analysis")
result = classifier("I love using pretrained language models!")
print(f"Sentiment: {result[0]['label']}, Score: {result[0]['score']:.4f}")

# Question answering pipeline
qa_pipeline = pipeline("question-answering")
context = "Hugging Face is a company that builds tools for the machine learning community."
question = "What does Hugging Face build?"
result = qa_pipeline(question=question, context=context)
print(f"Answer: {result['answer']}, Score: {result['score']:.4f}")

# Text generation pipeline
generator = pipeline("text-generation", model="gpt2")
result = generator("The key to artificial intelligence is", max_length=50)
print(f"Generated text: {result[0]['generated_text']}")
```

## When and Where We Use Pretrained LLMs

### When to Use Pretrained LLMs:

1. **When you have limited labeled data**: Pretrained LLMs leverage knowledge from large-scale pretraining, reducing the need for task-specific labeled data.

2. **When you need to understand complex language patterns**: These models capture nuanced linguistic patterns that would be difficult to learn from scratch.

3. **When you need to quickly prototype NLP applications**: Pretrained models provide strong baselines with minimal setup.

4. **When computational resources for training from scratch are limited**: Training large models from scratch requires significant computational resources.

### Where We Use Pretrained LLMs:

1. **Search Engines**: For understanding user queries and matching them to relevant content.

2. **Virtual Assistants**: For understanding user commands and generating appropriate responses.

3. **Content Creation**: For generating articles, social media posts, and marketing copy.

4. **Customer Support**: For automating responses to common customer inquiries.

5. **Healthcare**: For analyzing medical literature and assisting with diagnosis.

**Coding Example - Using BERT for semantic search:**

```python
from transformers import BertModel, BertTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Function to get BERT embeddings
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the [CLS] token embedding as the sentence representation
    return outputs.last_hidden_state[:, 0, :].numpy()

# Example documents
documents = [
    "The sun is the star at the center of the Solar System.",
    "Python is a high-level programming language.",
    "Machine learning is a subset of artificial intelligence."
]

# Query
query = "What is at the center of our solar system?"

# Get embeddings
query_embedding = get_bert_embedding(query)
document_embeddings = [get_bert_embedding(doc) for doc in documents]

# Calculate similarities
similarities = [cosine_similarity(query_embedding, doc_emb)[0][0] for doc_emb in document_embeddings]

# Find most similar document
most_similar_idx = similarities.index(max(similarities))
print(f"Query: {query}")
print(f"Most similar document: {documents[most_similar_idx]}")
print(f"Similarity score: {similarities[most_similar_idx]:.4f}")
```

## Real-World Applications and Examples

### 1. Question Answering Systems

**Application**: Customer support chatbots that can answer user questions based on product documentation.

**Coding Example - Building a QA pipeline with BERT:**

```python
from transformers import pipeline

# Create a question-answering pipeline
qa_pipeline = pipeline("question-answering")

# Context (product documentation)
context = """
Our premium smartphone features a 6.7-inch OLED display with a 120Hz refresh rate.
It has a triple camera system with a 50MP main sensor, 12MP ultra-wide, and 10MP telephoto lens.
The battery capacity is 4500mAh with support for 65W fast charging and 15W wireless charging.
The device is water-resistant with an IP68 rating and comes with 8GB of RAM and 256GB of storage.
"""

# Example questions
questions = [
    "What is the refresh rate of the display?",
    "How many cameras does the phone have?",
    "Is the phone water-resistant?"
]

# Answer questions
for question in questions:
    result = qa_pipeline(question=question, context=context)
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['score']:.4f}\n")
```

### 2. Text Summarization

**Application**: Automatically generating summaries of news articles, research papers, or business reports.

**Coding Example - Summarization with T5:**

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Example news article
article = """
NASA's Perseverance rover has successfully collected its first sample of Martian rock, 
a key step in the mission to search for signs of ancient microbial life on Mars. 
The rover drilled into a rock nicknamed "Rochette" in a region called Jezero Crater, 
which scientists believe was once home to a river delta and lake. 
The sample, slightly thicker than a pencil, was sealed in a titanium tube and will be 
stored in the rover's belly until it can be retrieved by a future mission and returned 
to Earth for detailed analysis. This marks a historic achievement in Mars exploration, 
as it's the first time a sample has been collected on another planet with the intention 
of bringing it back to Earth.
"""

# Prepare input with task prefix
input_text = f"summarize: {article}"

# Tokenize input
inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

# Generate summary
outputs = model.generate(
    inputs.input_ids,
    max_length=100,
    min_length=30,
    length_penalty=2.0,
    num_beams=4,
    early_stopping=True
)

# Decode and print summary
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Original Article Length:", len(article.split()))
print("Summary Length:", len(summary.split()))
print("\nSummary:")
print(summary)
```

### 3. Chatbot Development

**Application**: Creating conversational agents for customer service, personal assistants, or interactive entertainment.

**Coding Example - Simple chatbot using GPT-2:**

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load model and tokenizer
model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set pad token
tokenizer.pad_token = tokenizer.eos_token

# Chat function
def chat(prompt, max_length=100):
    # Format the prompt
    formatted_prompt = f"User: {prompt}\nAssistant:"
    
    # Tokenize input
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    # Generate response
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode and extract the assistant's response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("Assistant:")[1].strip()
    
    return response

# Example conversation
print("Chatbot: Hello! I'm an AI assistant. How can I help you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Chatbot: Goodbye!")
        break
    
    response = chat(user_input)
    print(f"Chatbot: {response}")
```

## Common Problems and How Advancements Have Solved Them

### 1. Problem: Computational Requirements

**Issue**: Pretrained LLMs require significant computational resources for both training and inference, making them inaccessible to many researchers and developers.

**Solution**: Model optimization techniques like quantization, pruning, and knowledge distillation have reduced the computational requirements.

**Problem Code (Inefficient model loading):**

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Loading the full model without optimization
model_name = "gpt2-large"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)  # This loads the full model

# This uses more memory and compute than necessary for many applications
```

**Enhancement Code (Using quantization):**

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load model with quantization
model_name = "gpt2-large"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Load model with 8-bit quantization
model = GPT2LMHeadModel.from_pretrained(
    model_name,
    load_in_8bit=True,  # This reduces memory usage significantly
    device_map="auto"   # This automatically handles device placement
)

# The quantized model uses less memory with minimal impact on performance
```

### 2. Problem: Context Length Limitations

**Issue**: Most pretrained LLMs have a fixed maximum context length (e.g., 512 or 1024 tokens), limiting their ability to process long documents.

**Solution**: Newer models and techniques have extended context windows, and methods like sliding window attention and hierarchical processing have been developed.

**Problem Code (Standard approach with context limitations):**

```python
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Long document that exceeds BERT's 512 token limit
long_document = "..."  # A very long text

# This will truncate the document to 512 tokens
inputs = tokenizer(long_document, return_tensors="pt", truncation=True, max_length=512)
outputs = model(**inputs)

# We lose information from the truncated part
```

**Enhancement Code (Using Longformer for extended context):**

```python
from transformers import LongformerTokenizer, LongformerModel

# Longformer can handle up to 4096 tokens
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
model = LongformerModel.from_pretrained("allenai/longformer-base-4096")

# Same long document
long_document = "..."  # A very long text

# Now we can process much longer documents
inputs = tokenizer(long_document, return_tensors="pt", truncation=True, max_length=4096)
outputs = model(**inputs)

# We preserve much more of the original context
```

### 3. Problem: Hallucination and Factual Inaccuracy

**Issue**: LLMs can generate plausible-sounding but factually incorrect information, which is problematic for applications requiring accuracy.

**Solution**: Retrieval-augmented generation (RAG) combines LLMs with external knowledge bases to improve factual accuracy.

**Problem Code (Pure generation without verification):**

```python
from transformers import pipeline

# Basic text generation without fact checking
generator = pipeline("text-generation", model="gpt2")
prompt = "The capital of Australia is"
result = generator(prompt, max_length=20)

# This might generate incorrect information
print(result[0]['generated_text'])
```

**Enhancement Code (Retrieval-augmented generation):**

```python
from transformers import pipeline, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import torch

# Initialize models
generator = pipeline("text-generation", model="gpt2")
question_encoder = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
question_model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

# Simple knowledge base (in practice, this would be a vector database)
knowledge_base = [
    {"text": "Canberra is the capital city of Australia.", "embedding": None},
    {"text": "Sydney is the largest city in Australia.", "embedding": None},
    {"text": "Melbourne is known for its street art and coffee culture.", "embedding": None}
]

# Function to retrieve relevant knowledge
def retrieve_relevant_knowledge(query, top_k=1):
    # Encode the query
    inputs = question_encoder(query, return_tensors="pt")
    query_embedding = question_model(**inputs).pooler_output
    
    # In a real implementation, we would compute similarities with pre-computed embeddings
    # For simplicity, we'll just return the first item
    return knowledge_base[0]["text"]

# Generate with retrieval
query = "What is the capital of Australia?"
relevant_knowledge = retrieve_relevant_knowledge(query)

# Augment the prompt with retrieved knowledge
augmented_prompt = f"Based on the information '{relevant_knowledge}', answer: {query}"

# Generate response
result = generator(augmented_prompt, max_length=50)
print(result[0]['generated_text'])
```

## 🧪 Exercise: QA and Summarization Pipelines

Let's create a combined pipeline for question answering and summarization using HuggingFace:

```python
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
import torch

# Initialize pipelines
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
summarizer = pipeline("summarization", model="t5-small")

# Function to answer questions about a document
def answer_questions(document, questions):
    print("=== Question Answering ===")
    for question in questions:
        result = qa_pipeline(question=question, context=document)
        print(f"Question: {question}")
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['score']:.4f}\n")

# Function to summarize a document
def summarize_document(document, max_length=150, min_length=30):
    print("=== Document Summarization ===")
    summary = summarizer(document, max_length=max_length, min_length=min_length)
    print(f"Original length: {len(document.split())} words")
    print(f"Summary length: {len(summary[0]['summary_text'].split())} words")
    print(f"\nSummary:\n{summary[0]['summary_text']}")

# Example document
document = """
The James Webb Space Telescope (JWST) is a space telescope designed primarily to conduct infrared astronomy. 
As the largest optical telescope in space, its high resolution and sensitivity allow it to view objects too old, 
distant, or faint for the Hubble Space Telescope. The U.S. National Aeronautics and Space Administration (NASA) 
led JWST's development in collaboration with the European Space Agency (ESA) and the Canadian Space Agency (CSA). 
The telescope was launched in December 2021 and is intended to succeed the Hubble as NASA's flagship mission in astrophysics. 
JWST's primary mirror consists of 18 hexagonal mirror segments made of gold-plated beryllium, which together create a 
6.5-meter-diameter (21 ft) mirror—considerably larger than Hubble's 2.4-meter (7.9 ft) mirror. Unlike Hubble, 
which observes in the near ultraviolet, visible, and near infrared spectra, JWST observes in a lower frequency range, 
from long-wavelength visible light through mid-infrared (0.6–28.3 μm). This allows it to observe high-redshift objects 
that are too old and too distant for Hubble and other earlier instruments to observe.
"""

# Example questions
questions = [
    "When was the James Webb Space Telescope launched?",
    "Which agencies collaborated on JWST's development?",
    "How does JWST's mirror compare to Hubble's?"
]

# Run QA and summarization
answer_questions(document, questions)
summarize_document(document)
```

## 📌 Project: LLM Chatbot

Now, let's build a more sophisticated chatbot that combines multiple LLM capabilities:

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    T5ForConditionalGeneration,
    T5Tokenizer
)
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class LLMChatbot:
    def __init__(self):
        # Initialize models
        self.qa_model_name = "distilbert-base-cased-distilled-squad"
        self.summarizer_model_name = "t5-small"
        self.generator_model_name = "microsoft/DialoGPT-medium"
        
        # Initialize pipelines
        self.qa_pipeline = pipeline("question-answering", model=self.qa_model_name)
        self.summarizer = pipeline("summarization", model=self.summarizer_model_name)
        
        # Initialize conversation model
        self.tokenizer = AutoTokenizer.from_pretrained(self.generator_model_name)
        self.generator = AutoModelForCausalLM.from_pretrained(self.generator_model_name)
        
        # Initialize conversation history
        self.chat_history_ids = None
        
        # Knowledge base (in a real application, this would be a vector database)
        self.knowledge_base = [
            "The James Webb Space Telescope was launched in December 2021.",
            "JWST is a collaboration between NASA, ESA, and CSA.",
            "JWST's primary mirror is 6.5 meters in diameter, larger than Hubble's 2.4-meter mirror.",
            "JWST observes in the infrared spectrum, unlike Hubble which observes in visible and near-infrared."
        ]
        
        # Command handlers
        self.commands = {
            "/help": self.show_help,
            "/summarize": self.summarize_text,
            "/qa": self.answer_question,
            "/reset": self.reset_conversation
        }
    
    def show_help(self):
        help_text = """
        Available commands:
        /help - Show this help message
        /summarize [text] - Summarize the provided text
        /qa [question] - Answer a question based on the knowledge base
        /reset - Reset the conversation history
        """
        return help_text
    
    def summarize_text(self, text):
        try:
            summary = self.summarizer(text, max_length=100, min_length=30)
            return summary[0]['summary_text']
        except Exception as e:
            return f"Error summarizing text: {str(e)}"
    
    def answer_question(self, question):
        try:
            # Find the most relevant knowledge base entry
            best_answer = None
            best_score = 0
            
            for knowledge in self.knowledge_base:
                result = self.qa_pipeline(question=question, context=knowledge)
                if result['score'] > best_score:
                    best_score = result['score']
                    best_answer = result['answer']
            
            if best_answer:
                return f"Answer: {best_answer} (confidence: {best_score:.2f})"
            else:
                return "I couldn't find a relevant answer in my knowledge base."
        except Exception as e:
            return f"Error answering question: {str(e)}"
    
    def reset_conversation(self):
        self.chat_history_ids = None
        return "Conversation history has been reset."
    
    def generate_response(self, input_text):
        # Check for commands
        if input_text.startswith("/"):
            parts = input_text.split(" ", 1)
            command = parts[0]
            args = parts[1] if len(parts) > 1 else ""
            
            if command in self.commands:
                if command == "/summarize" and args:
                    return self.summarize_text(args)
                elif command == "/qa" and args:
                    return self.answer_question(args)
                else:
                    return self.commands[command]()
            else:
                return f"Unknown command: {command}. Type /help for available commands."
        
        # Regular conversation
        try:
            # Encode the new user input, add the eos_token and return a tensor in PyTorch
            new_user_input_ids = self.tokenizer.encode(input_text + self.tokenizer.eos_token, return_tensors='pt')
            
            # Append the new user input tokens to the chat history
            bot_input_ids = torch.cat([self.chat_history_ids, new_user_input_ids], dim=-1) if self.chat_history_ids is not None else new_user_input_ids
            
            # Generate a response
            self.chat_history_ids = self.generator.generate(
                bot_input_ids,
                max_length=1000,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                do_sample=True,
                top_k=100,
                top_p=0.7,
                temperature=0.8
            )
            
            # Decode and return the response
            response = self.tokenizer.decode(self.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def chat(self):
        print("LLM Chatbot initialized. Type /help for available commands.")
        print("Chatbot: Hello! I'm an AI assistant. How can I help you today?")
        
        while True:
            user_input = input("You: ")
            
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Chatbot: Goodbye!")
                break
            
            response = self.generate_response(user_input)
            print(f"Chatbot: {response}")

# Initialize and run the chatbot
if __name__ == "__main__":
    chatbot = LLMChatbot()
    chatbot.chat()
```

This chatbot combines multiple LLM capabilities:
1. Conversational ability using DialoGPT
2. Question answering using a distilled BERT model
3. Text summarization using T5
4. Command-based interface for specific tasks
