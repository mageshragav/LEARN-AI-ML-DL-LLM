# Day 37 – Ollama + Groq Integration

## What is Ollama and Groq?

### Ollama
Ollama is an open-source framework that allows you to run large language models (LLMs) locally on your machine. It provides a simple command-line interface and API for running models like Llama, Mistral, and others without needing to set up complex infrastructure. Ollama handles model downloading, management, and execution in a user-friendly way.

### Groq
Groq is a company that has developed a specialized hardware processing unit called the Language Processing Unit (LPU), designed specifically for high-speed inference of large language models. Groq's cloud platform provides extremely fast inference speeds for LLMs through their API, offering an alternative to local execution when speed is critical.

## Essential Subtopics

### 1. Running Models Locally with Ollama

**What it is:**
Running models locally with Ollama means executing LLMs directly on your hardware without relying on cloud services. This provides privacy, offline capability, and potentially lower costs for high-volume usage.

**Explanation:**
Ollama simplifies the process of running LLMs locally by:
1. Managing model downloads and updates
2. Providing a unified API for different models
3. Handling resource allocation and optimization
4. Offering both CLI and REST API interfaces

**Coding Example:**
```python
import requests
import json

class OllamaClient:
    def __init__(self, base_url="http://localhost:11434"):
        """
        Initialize the Ollama client.
        
        Args:
            base_url: Base URL for the Ollama API (default: http://localhost:11434)
        """
        self.base_url = base_url
    
    def list_models(self):
        """
        List all available models.
        
        Returns:
            List of model names
        """
        response = requests.get(f"{self.base_url}/api/tags")
        if response.status_code == 200:
            models = response.json()
            return [model["name"] for model in models["models"]]
        else:
            return []
    
    def pull_model(self, model_name):
        """
        Pull a model from the Ollama registry.
        
        Args:
            model_name: Name of the model to pull
            
        Returns:
            Success status
        """
        response = requests.post(
            f"{self.base_url}/api/pull",
            json={"name": model_name},
            stream=True
        )
        
        for line in response.iter_lines():
            if line:
                json_line = json.loads(line)
                print(json_line.get("status", ""))
        
        return response.status_code == 200
    
    def generate(self, model_name, prompt, system=None, stream=False):
        """
        Generate text using a model.
        
        Args:
            model_name: Name of the model to use
            prompt: Input prompt
            system: Optional system prompt
            stream: Whether to stream the response
            
        Returns:
            Generated text
        """
        data = {
            "model": model_name,
            "prompt": prompt,
            "stream": stream
        }
        
        if system:
            data["system"] = system
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=data,
            stream=stream
        )
        
        if stream:
            return response
        else:
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                return f"Error: {response.status_code}"
    
    def chat(self, model_name, messages, stream=False):
        """
        Chat with a model.
        
        Args:
            model_name: Name of the model to use
            messages: List of message dictionaries with "role" and "content"
            stream: Whether to stream the response
            
        Returns:
            Model response
        """
        data = {
            "model": model_name,
            "messages": messages,
            "stream": stream
        }
        
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=data,
            stream=stream
        )
        
        if stream:
            return response
        else:
            if response.status_code == 200:
                result = response.json()
                return result.get("message", {}).get("content", "")
            else:
                return f"Error: {response.status_code}"

# Example usage
ollama = OllamaClient()

# List available models
print("Available models:", ollama.list_models())

# Pull a model (if not already available)
# print("Pulling model...")
# ollama.pull_model("mistral:7b")

# Generate text
response = ollama.generate("mistral:7b", "What is the capital of France?")
print("Response:", response)

# Chat with the model
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
]
chat_response = ollama.chat("mistral:7b", messages)
print("Chat response:", chat_response)
```

### 2. Using Groq API for High-Speed Inference

**What it is:**
Groq provides cloud-based LLM inference with extremely low latency through their specialized Language Processing Units (LPUs). This allows for real-time applications that require fast response times.

**Explanation:**
Groq's key advantages include:
1. Extremely fast inference speeds (hundreds of tokens per second)
2. Consistent latency regardless of input length
3. Simple API interface
4. Support for popular models like Llama and Mixtral

**Coding Example:**
```python
import requests
import json

class GroqClient:
    def __init__(self, api_key, base_url="https://api.groq.com/openai/v1"):
        """
        Initialize the Groq client.
        
        Args:
            api_key: Groq API key
            base_url: Base URL for the Groq API
        """
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def list_models(self):
        """
        List all available models.
        
        Returns:
            List of model information
        """
        response = requests.get(
            f"{self.base_url}/models",
            headers=self.headers
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Status code: {response.status_code}"}
    
    def chat_completion(self, model, messages, temperature=0.7, max_tokens=1024, stream=False):
        """
        Create a chat completion.
        
        Args:
            model: Model to use (e.g., "llama3-8b-8192")
            messages: List of message dictionaries with "role" and "content"
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Completion response
        """
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=data,
            stream=stream
        )
        
        if stream:
            return response
        else:
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Status code: {response.status_code}", "details": response.text}

# Example usage (would require actual API key)
# groq = GroqClient(api_key="your-groq-api-key")

# List available models
# models = groq.list_models()
# print("Available models:", models)

# Create a chat completion
# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "What is the capital of France?"}
# ]
# response = groq.chat_completion("llama3-8b-8192", messages)
# print("Response:", response["choices"][0]["message"]["content"])
```

## When and Where We Use Ollama and Groq

### When to Use Ollama:
- When you need to run models offline
- When privacy is a concern and data cannot leave your infrastructure
- When you have limited internet connectivity
- When you want to avoid ongoing API costs for high-volume usage
- When you need to customize or fine-tune models locally

### When to Use Groq:
- When you need extremely fast inference speeds
- When building real-time applications that require low latency
- When you don't have the hardware to run models locally
- When you need to scale quickly without managing infrastructure

### Where to Use Ollama:
- Local development environments
- On-premises enterprise applications
- Edge computing devices
- Research and experimentation
- Privacy-sensitive applications

### Where to Use Groq:
- Customer support chatbots
- Real-time translation services
- Interactive applications
- High-traffic websites
- Time-sensitive analytical applications

**Coding Example:**
```python
class HybridLLMClient:
    """
    A hybrid client that can use either Ollama (local) or Groq (cloud)
    based on availability and requirements.
    """
    def __init__(self, ollama_base_url="http://localhost:11434", groq_api_key=None):
        """
        Initialize the hybrid client.
        
        Args:
            ollama_base_url: Base URL for Ollama
            groq_api_key: API key for Groq
        """
        self.ollama = OllamaClient(ollama_base_url)
        self.groq = GroqClient(groq_api_key) if groq_api_key else None
        self.use_local = True  # Default to local
    
    def check_ollama_available(self):
        """Check if Ollama is available."""
        try:
            models = self.ollama.list_models()
            return len(models) > 0
        except:
            return False
    
    def set_preferred_backend(self, use_local=True):
        """
        Set the preferred backend.
        
        Args:
            use_local: True to use Ollama (local), False to use Groq (cloud)
        """
        self.use_local = use_local
    
    def generate(self, prompt, model=None, system=None):
        """
        Generate text using the preferred backend.
        
        Args:
            prompt: Input prompt
            model: Model name (if None, uses default for the backend)
            system: Optional system prompt
            
        Returns:
            Generated text
        """
        # Check if preferred backend is available
        if self.use_local and self.check_ollama_available():
            # Use Ollama
            if model is None:
                model = "mistral:7b"  # Default model
            return self.ollama.generate(model, prompt, system)
        elif self.groq:
            # Use Groq
            if model is None:
                model = "llama3-8b-8192"  # Default model
            
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            
            response = self.groq.chat_completion(model, messages)
            return response["choices"][0]["message"]["content"]
        else:
            return "Error: No available backend"
    
    def benchmark(self, prompt, model=None, system=None, iterations=5):
        """
        Benchmark both backends.
        
        Args:
            prompt: Input prompt
            model: Model name
            system: Optional system prompt
            iterations: Number of iterations for each backend
            
        Returns:
            Benchmark results
        """
        import time
        
        results = {"ollama": [], "groq": []}
        
        # Benchmark Ollama
        if self.check_ollama_available():
            print("Benchmarking Ollama...")
            for i in range(iterations):
                start_time = time.time()
                self.ollama.generate(model or "mistral:7b", prompt, system)
                end_time = time.time()
                results["ollama"].append(end_time - start_time)
        
        # Benchmark Groq
        if self.groq:
            print("Benchmarking Groq...")
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            
            for i in range(iterations):
                start_time = time.time()
                self.groq.chat_completion(model or "llama3-8b-8192", messages)
                end_time = time.time()
                results["groq"].append(end_time - start_time)
        
        # Calculate averages
        for backend in results:
            if results[backend]:
                avg_time = sum(results[backend]) / len(results[backend])
                print(f"{backend} average time: {avg_time:.2f} seconds")
        
        return results

# Example usage
# hybrid_client = HybridLLMClient(groq_api_key="your-groq-api-key")

# Generate text using preferred backend
# response = hybrid_client.generate("What is the capital of France?")
# print("Response:", response)

# Benchmark both backends
# hybrid_client.benchmark("Explain the concept of machine learning in one paragraph.")
```

## Real-World Applications and Examples

### 1. Real-Time Customer Support Chatbot

**Application:**
A customer support chatbot that needs to respond quickly to customer inquiries while maintaining privacy for sensitive information.

**Coding Example:**
```python
class CustomerSupportChatbot:
    """
    A customer support chatbot that can use either local or cloud models.
    """
    def __init__(self, knowledge_base, hybrid_client):
        """
        Initialize the chatbot.
        
        Args:
            knowledge_base: List of documents for the knowledge base
            hybrid_client: HybridLLMClient instance
        """
        self.knowledge_base = knowledge_base
        self.hybrid_client = hybrid_client
        self.conversation_history = []
    
    def retrieve_relevant_info(self, query):
        """
        Retrieve relevant information from the knowledge base.
        
        Args:
            query: User query
            
        Returns:
            Relevant information
        """
        # Simple keyword matching for demonstration
        relevant_docs = []
        for doc in self.knowledge_base:
            if any(keyword.lower() in doc.lower() for keyword in query.lower().split()):
                relevant_docs.append(doc)
        
        return "\n\n".join(relevant_docs[:3])  # Return top 3 docs
    
    def respond(self, query, use_fast_mode=True):
        """
        Respond to a user query.
        
        Args:
            query: User query
            use_fast_mode: Whether to use fast mode (Groq) or local mode
            
        Returns:
            Response to the user
        """
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        # Retrieve relevant information
        relevant_info = self.retrieve_relevant_info(query)
        
        # Set backend based on mode
        self.hybrid_client.set_preferred_backend(not use_fast_mode)
        
        # Create prompt
        system_prompt = f"""You are a customer support assistant. Answer the user's question based on the provided information.
        
        Relevant Information:
        {relevant_info}
        
        If the information doesn't contain the answer, say that you don't have enough information to answer.
        Be helpful, concise, and professional."""
        
        # Generate response
        response = self.hybrid_client.generate(
            query,
            system=system_prompt
        )
        
        # Add to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response

# Example usage
knowledge_base = [
    "Our return policy allows returns within 30 days of purchase with a valid receipt.",
    "To reset your device, hold the power button for 10 seconds until the LED flashes.",
    "Our premium support plan includes 24/7 phone support and guaranteed 1-hour response times.",
    "Standard shipping takes 3-5 business days. Express shipping is available for an additional fee."
]

# hybrid_client = HybridLLMClient(groq_api_key="your-groq-api-key")
# chatbot = CustomerSupportChatbot(knowledge_base, hybrid_client)

# Simulate a conversation
# print("User: How do I return a product?")
# print("Bot:", chatbot.respond("How do I return a product?"))

# print("\nUser: How long does shipping take?")
# print("Bot:", chatbot.respond("How long does shipping take?"))

# print("\nUser: How do I reset my device?")
# print("Bot:", chatbot.respond("How do I reset my device?"))
```

### 2. Real-Time Document Summarization

**Application:**
A system that can summarize long documents in real-time, useful for professionals who need to quickly understand the content of reports, articles, or papers.

**Coding Example:**
```python
class RealTimeSummarizer:
    """
    A real-time document summarization system.
    """
    def __init__(self, hybrid_client):
        """
        Initialize the summarizer.
        
        Args:
            hybrid_client: HybridLLMClient instance
        """
        self.hybrid_client = hybrid_client
    
    def summarize_text(self, text, max_length=500, use_fast_mode=True):
        """
        Summarize a text.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of the summary
            use_fast_mode: Whether to use fast mode (Groq) or local mode
            
        Returns:
            Summary of the text
        """
        # Set backend based on mode
        self.hybrid_client.set_preferred_backend(not use_fast_mode)
        
        # Create prompt
        system_prompt = f"""You are an expert summarizer. Summarize the following text in under {max_length} characters.
        
        The summary should:
        - Capture the main points
        - Be well-structured and coherent
        - Maintain the original meaning
        - Exclude unnecessary details"""
        
        # Generate summary
        summary = self.hybrid_client.generate(
            f"Summarize this text:\n\n{text}",
            system=system_prompt
        )
        
        return summary
    
    def summarize_file(self, file_path, max_length=500, use_fast_mode=True):
        """
        Summarize a text file.
        
        Args:
            file_path: Path to the file
            max_length: Maximum length of the summary
            use_fast_mode: Whether to use fast mode (Groq) or local mode
            
        Returns:
            Summary of the file
        """
        # Read file
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Summarize
        return self.summarize_text(text, max_length, use_fast_mode)

# Example usage
# hybrid_client = HybridLLMClient(groq_api_key="your-groq-api-key")
# summarizer = RealTimeSummarizer(hybrid_client)

# Create a sample document
sample_text = """
Machine learning is a subset of artificial intelligence that focuses on building systems that can learn from data. 
These systems improve their performance over time without being explicitly programmed. 
Machine learning algorithms build a model based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to do so.

There are several types of machine learning:
1. Supervised learning: The algorithm learns from labeled training data, and makes predictions based on that data.
2. Unsupervised learning: The algorithm learns from unlabeled data, and finds patterns in the data.
3. Semi-supervised learning: The algorithm learns from a mix of labeled and unlabeled data.
4. Reinforcement learning: The algorithm learns through trial and error, receiving rewards or penalties for actions.

Machine learning has many applications, including:
- Image and speech recognition
- Natural language processing
- Recommendation systems
- Fraud detection
- Medical diagnosis
- Financial trading

The field of machine learning is rapidly evolving, with new algorithms and techniques being developed regularly.
Deep learning, a subset of machine learning that uses neural networks with many layers, has been particularly successful in recent years.
"""

# Summarize the text
# summary = summarizer.summarize_text(sample_text, max_length=300)
# print("Summary:", summary)
```

## Common Problems and How Advancements Have Solved Them

### Problem 1: Limited Computational Resources for Local Inference

**Problem:**
Running large language models locally requires significant computational resources, making it inaccessible for users without high-end hardware.

**Solution:**
Ollama addresses this by:
1. Providing optimized model implementations
2. Supporting quantization to reduce memory requirements
3. Offering a range of model sizes to match different hardware capabilities
4. Efficient resource management

**Problem Code:**
```python
# Old approach - trying to run a large model without optimization
def run_large_model_unoptimized(prompt):
    """
    Attempt to run a large model without optimization.
    This would likely fail on most consumer hardware.
    """
    # This is a conceptual example - in reality, you'd need a lot of RAM and GPU power
    model = load_huge_model("llama-70b")  # This would require ~140GB of RAM
    
    # Generate response
    response = model.generate(prompt)
    return response

# This would likely fail on most consumer machines
# response = run_large_model_unoptimized("What is the capital of France?")
```

**Enhancement Code:**
```python
# New approach with Ollama - optimized for local execution
def run_model_with_ollama(prompt, model_name="llama3:8b"):
    """
    Run a model using Ollama with optimizations.
    
    Args:
        prompt: Input prompt
        model_name: Name of the model to use
        
    Returns:
        Model response
    """
    # Initialize Ollama client
    ollama = OllamaClient()
    
    # Check if model is available, pull if not
    available_models = ollama.list_models()
    if model_name not in available_models:
        print(f"Pulling model {model_name}...")
        ollama.pull_model(model_name)
    
    # Generate response
    response = ollama.generate(model_name, prompt)
    return response

# This can run on consumer hardware
# response = run_model_with_ollama("What is the capital of France?")
# print(response)
```

### Problem 2: High Latency in Cloud-Based Inference

**Problem:**
Traditional cloud-based LLM inference can have high latency, making it unsuitable for real-time applications.

**Solution:**
Groq addresses this with:
1. Specialized hardware (Language Processing Units) designed for LLM inference
2. Optimized software stack that eliminates bottlenecks
3. Consistent latency regardless of input length
4. High throughput (hundreds of tokens per second)

**Problem Code:**
```python
# Old approach - using standard cloud API with potential high latency
import time

def standard_cloud_inference(prompt, api_key):
    """
    Use a standard cloud API for inference.
    This might have high latency.
    """
    start_time = time.time()
    
    # Initialize standard API client
    client = StandardAPIClient(api_key)
    
    # Make request
    response = client.generate(prompt)
    
    end_time = time.time()
    latency = end_time - start_time
    
    return {
        "response": response,
        "latency": latency
    }

# This might have high latency
# result = standard_cloud_inference("What is the capital of France?", "your-api-key")
# print(f"Response: {result['response']}")
# print(f"Latency: {result['latency']:.2f} seconds")
```

**Enhancement Code:**
```python
# New approach with Groq - optimized for low latency
def groq_inference(prompt, api_key):
    """
    Use Groq API for low-latency inference.
    
    Args:
        prompt: Input prompt
        api_key: Groq API key
        
    Returns:
        Response and latency information
    """
    start_time = time.time()
    
    # Initialize Groq client
    groq = GroqClient(api_key)
    
    # Create messages
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # Make request
    response = groq.chat_completion("llama3-8b-8192", messages)
    text_response = response["choices"][0]["message"]["content"]
    
    end_time = time.time()
    latency = end_time - start_time
    
    # Get tokens per second from response metadata if available
    tokens_per_second = response.get("usage", {}).get("tokens_per_second", "N/A")
    
    return {
        "response": text_response,
        "latency": latency,
        "tokens_per_second": tokens_per_second
    }

# This should have much lower latency
# result = groq_inference("What is the capital of France?", "your-groq-api-key")
# print(f"Response: {result['response']}")
# print(f"Latency: {result['latency']:.2f} seconds")
# print(f"Tokens per second: {result['tokens_per_second']}")
```

## 🧪 Exercise: Use Mistral/LLama3 with Ollama

Let's create a simple exercise to use Mistral or Llama3 with Ollama.

```python
def ollama_exercise():
    """
    Exercise: Use Mistral/Llama3 with Ollama.
    """
    # Initialize Ollama client
    ollama = OllamaClient()
    
    # Check available models
    print("Checking available models...")
    available_models = ollama.list_models()
    print(f"Available models: {available_models}")
    
    # Choose a model (prefer Mistral or Llama3 if available)
    model_name = None
    for model in available_models:
        if "mistral" in model.lower() or "llama3" in model.lower():
            model_name = model
            break
    
    if not model_name:
        # If Mistral or Llama3 is not available, pull one
        print("Mistral or Llama3 not found. Pulling Mistral...")
        success = ollama.pull_model("mistral:7b")
        if success:
            model_name = "mistral:7b"
        else:
            print("Failed to pull model. Using first available model.")
            model_name = available_models[0] if available_models else None
    
    if not model_name:
        print("No models available. Please pull a model first using the Ollama CLI.")
        return
    
    print(f"Using model: {model_name}")
    
    # Test basic generation
    print("\nTesting basic generation...")
    prompt = "What is artificial intelligence?"
    response = ollama.generate(model_name, prompt)
    print(f"Response: {response}")
    
    # Test chat functionality
    print("\nTesting chat functionality...")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What are the main types of machine learning?"}
    ]
    chat_response = ollama.chat(model_name, messages)
    print(f"Chat response: {chat_response}")
    
    # Test streaming
    print("\nTesting streaming...")
    stream_response = ollama.generate(model_name, "Explain quantum computing in simple terms.", stream=True)
    print("Streaming response:")
    for line in stream_response.iter_lines():
        if line:
            json_line = json.loads(line)
            print(json_line.get("response", ""), end="", flush=True)
    print()

# Run the exercise
# ollama_exercise()
```

## 📌 Project: Summarizer with Ollama

Now, let's create a more comprehensive project - a document summarizer using Ollama.

```python
import os
import glob
import argparse
from typing import List, Dict, Any

class OllamaSummarizer:
    """
    A document summarizer using Ollama.
    """
    def __init__(self, model_name="mistral:7b", ollama_base_url="http://localhost:11434"):
        """
        Initialize the summarizer.
        
        Args:
            model_name: Name of the model to use
            ollama_base_url: Base URL for the Ollama API
        """
        self.model_name = model_name
        self.ollama = OllamaClient(ollama_base_url)
        self._ensure_model_available()
    
    def _ensure_model_available(self):
        """Ensure the specified model is available."""
        available_models = self.ollama.list_models()
        if self.model_name not in available_models:
            print(f"Model {self.model_name} not found. Pulling...")
            success = self.ollama.pull_model(self.model_name)
            if not success:
                raise Exception(f"Failed to pull model {self.model_name}")
    
    def summarize_text(self, text: str, max_length: int = 500, style: str = "neutral") -> str:
        """
        Summarize a text.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of the summary in characters
            style: Style of the summary (neutral, formal, casual, academic)
            
        Returns:
            Summary of the text
        """
        # Create system prompt based on style
        style_prompts = {
            "neutral": "You are a neutral summarizer. Provide a balanced summary of the text.",
            "formal": "You are a formal summarizer. Provide a professional, formal summary of the text.",
            "casual": "You are a casual summarizer. Provide a relaxed, easy-to-understand summary of the text.",
            "academic": "You are an academic summarizer. Provide a scholarly, detailed summary of the text."
        }
        
        system_prompt = f"""{style_prompts.get(style, style_prompts["neutral"])}
        
        Your summary should:
        - Be under {max_length} characters
        - Capture the main points
        - Be well-structured and coherent
        - Maintain the original meaning"""
        
        # Create user prompt
        user_prompt = f"Summarize the following text:\n\n{text}"
        
        # Generate summary
        summary = self.ollama.generate(self.model_name, user_prompt, system_prompt)
        
        # Ensure the summary is within the length limit
        if len(summary) > max_length:
            # If it's too long, try again with a more specific instruction
            user_prompt = f"Summarize the following text in under {max_length} characters:\n\n{text}"
            summary = self.ollama.generate(self.model_name, user_prompt, system_prompt)
        
        return summary
    
    def summarize_file(self, file_path: str, max_length: int = 500, style: str = "neutral") -> str:
        """
        Summarize a file.
        
        Args:
            file_path: Path to the file
            max_length: Maximum length of the summary in characters
            style: Style of the summary
            
        Returns:
            Summary of the file
        """
        # Read file
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Summarize
        return self.summarize_text(text, max_length, style)
    
    def summarize_directory(self, directory: str, pattern: str = "*.txt", max_length: int = 500, style: str = "neutral") -> Dict[str, str]:
        """
        Summarize all files in a directory.
        
        Args:
            directory: Path to the directory
            pattern: File pattern to match
            max_length: Maximum length of each summary
            style: Style of the summaries
            
        Returns:
            Dictionary mapping file paths to summaries
        """
        # Get all matching files
        file_paths = glob.glob(os.path.join(directory, pattern))
        
        # Summarize each file
        summaries = {}
        for file_path in file_paths:
            try:
                summary = self.summarize_file(file_path, max_length, style)
                summaries[file_path] = summary
                print(f"Summarized {file_path}")
            except Exception as e:
                print(f"Error summarizing {file_path}: {e}")
                summaries[file_path] = f"Error: {e}"
        
        return summaries
    
    def save_summaries(self, summaries: Dict[str, str], output_dir: str):
        """
        Save summaries to files.
        
        Args:
            summaries: Dictionary of file paths to summaries
            output_dir: Directory to save summaries in
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each summary
        for file_path, summary in summaries.items():
            # Get base filename
            base_name = os.path.basename(file_path)
            name, ext = os.path.splitext(base_name)
            
            # Create output file path
            output_path = os.path.join(output_dir, f"{name}_summary.txt")
            
            # Write summary
            with open(output_path, 'w', encoding='utf-8') as file:
                file.write(f"Summary of {file_path}\n\n")
                file.write(summary)
            
            print(f"Saved summary to {output_path}")

def create_sample_documents():
    """Create sample documents for demonstration."""
    os.makedirs("sample_docs", exist_ok=True)
    
    # Document 1 - Technology
    with open("sample_docs/tech_trends.txt", "w") as f:
        f.write("""
        Artificial intelligence continues to be one of the most transformative technologies of our time. 
        In recent years, we've seen significant advances in large language models, which have demonstrated 
        remarkable capabilities in natural language understanding and generation. These models are now 
        being integrated into a wide range of applications, from customer service chatbots to content 
        creation tools.
        
        Another major trend is the development of more efficient AI hardware. Companies like NVIDIA, 
        AMD, and Intel are producing specialized chips designed specifically for AI workloads. 
        These chips offer improved performance and energy efficiency compared to general-purpose processors.
        
        Quantum computing is also making progress, though it remains in early stages. Companies like 
        IBM, Google, and startups like Rigetti are working on quantum processors that could eventually 
        solve problems that are intractable for classical computers.
        
        In the realm of consumer technology, we're seeing the continued proliferation of smart devices 
        and the Internet of Things (IoT). Smart homes, wearable devices, and connected appliances are 
        becoming increasingly common, creating new opportunities for automation and data collection.
        """)
    
    # Document 2 - Environment
    with open("sample_docs/climate_change.txt", "w") as f:
        f.write("""
        Climate change remains one of the most pressing challenges facing humanity. 
        The latest reports from the Intergovernmental Panel on Climate Change (IPCC) indicate that 
        global temperatures continue to rise, with the past decade being the warmest on record.
        
        The primary driver of climate change is the emission of greenhouse gases, particularly 
        carbon dioxide from the burning of fossil fuels. Other significant contributors include 
        methane from agriculture and waste, and nitrous oxide from industrial processes.
        
        The impacts of climate change are already being felt around the world. These include more 
        frequent and severe heatwaves, changes in precipitation patterns leading to both droughts 
        and floods, rising sea levels threatening coastal communities, and disruptions to ecosystems.
        
        In response to these challenges, there is growing momentum for climate action. Many countries 
        have committed to reducing their greenhouse gas emissions to net-zero by mid-century. 
        Renewable energy sources like solar and wind are becoming increasingly cost-competitive 
        with fossil fuels, leading to rapid deployment in many parts of the world.
        
        However, current efforts are still insufficient to meet the goals of the Paris Agreement, 
        which aims to limit global warming to well below 2 degrees Celsius above pre-industrial levels. 
        More ambitious action is needed to transition to a low-carbon economy and adapt to the 
        changes that are already underway.
        """)
    
    # Document 3 - Health
    with open("sample_docs/health_innovations.txt", "w") as f:
        f.write("""
        The field of healthcare is experiencing rapid innovation, driven by advances in technology 
        and a deeper understanding of human biology. One of the most significant developments in 
        recent years has been the application of artificial intelligence in medicine.
        
        AI algorithms are now being used to analyze medical images with accuracy that can match or 
        exceed human experts. For example, AI systems can detect signs of diabetic retinopathy in 
        retinal scans, identify tumors in mammograms, and find abnormalities in chest X-rays.
        
        Another area of innovation is personalized medicine. By analyzing a patient's genetic makeup, 
        doctors can tailor treatments to the individual characteristics of each patient. This approach 
        has been particularly successful in oncology, where targeted therapies have improved outcomes 
        for many cancer patients.
        
        Telemedicine has also seen rapid growth, accelerated by the COVID-19 pandemic. Virtual 
        consultations allow patients to receive care without traveling to a clinic, improving access 
        for those in rural areas or with mobility issues. Remote monitoring devices enable doctors to 
        track patients' health between visits, potentially catching problems before they become serious.
        
        Gene editing technologies like CRISPR are opening new possibilities for treating genetic 
        disorders. While still largely experimental, these approaches have shown promise in early 
        trials for conditions like sickle cell disease and certain types of inherited blindness.
        
        Despite these advances, significant challenges remain. Healthcare systems around the world 
        are struggling with rising costs, aging populations, and health disparities. Ensuring that 
        innovations benefit all segments of society will require thoughtful policy and investment.
        """)

def main():
    """Main function for the summarizer project."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Summarize documents using Ollama")
    parser.add_argument("--input", type=str, help="Input file or directory")
    parser.add_argument("--output", type=str, default="summaries", help="Output directory")
    parser.add_argument("--model", type=str, default="mistral:7b", help="Model to use")
    parser.add_argument("--max-length", type=int, default=500, help="Maximum summary length")
    parser.add_argument("--style", type=str, default="neutral", 
                       choices=["neutral", "formal", "casual", "academic"],
                       help="Summary style")
    
    args = parser.parse_args()
    
    # Create sample documents if no input is specified
    if not args.input:
        print("No input specified. Creating sample documents...")
        create_sample_documents()
        args.input = "sample_docs"
    
    # Initialize summarizer
    print(f"Initializing summarizer with model {args.model}...")
    summarizer = OllamaSummarizer(model_name=args.model)
    
    # Summarize
    print(f"Summarizing {args.input}...")
    if os.path.isfile(args.input):
        # Single file
        summary = summarizer.summarize_file(args.input, args.max_length, args.style)
        summaries = {args.input: summary}
    else:
        # Directory
        summaries = summarizer.summarize_directory(args.input, max_length=args.max_length, style=args.style)
    
    # Save summaries
    print(f"Saving summaries to {args.output}...")
    summarizer.save_summaries(summaries, args.output)
    
    print("Done!")

if __name__ == "__main__":
    main()
```
