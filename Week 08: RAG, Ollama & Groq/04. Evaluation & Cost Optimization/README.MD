# Day 39 – Evaluation & Cost Optimization

## What is Evaluation & Cost Optimization in LLM Applications?

### Evaluation
Evaluation in the context of LLM applications refers to the systematic process of assessing the performance, quality, and effectiveness of language models and the systems built around them. This includes measuring metrics like accuracy, relevance, coherence, latency, and user satisfaction.

### Cost Optimization
Cost optimization involves strategies and techniques to reduce the financial and computational expenses associated with running LLM applications. This includes optimizing API calls, implementing caching strategies, model selection, and efficient resource allocation.

## Essential Subtopics

### 1. LangSmith for Evaluation

**What it is:**
LangSmith is a platform developed by LangChain for evaluating, debugging, and monitoring LLM applications. It provides tools to track runs, visualize results, compare different model configurations, and measure performance against various metrics.

**Explanation:**
LangSmith addresses the challenge of evaluating LLM applications by:
1. Providing a centralized platform for tracking and comparing runs
2. Offering visualization tools to understand model behavior
3. Enabling A/B testing of different prompts and models
4. Facilitating collaboration among team members
5. Supporting custom evaluation metrics

**Coding Example:**
```python
import os
from langchain.smith import RunEvalConfig, run_on_dataset
from langchain.chat_models import ChatGroq
from langchain.evaluation import load_evaluator

def setup_langsmith_evaluation(api_key, langchain_api_key):
    """
    Set up LangSmith for evaluation.
    
    Args:
        api_key: Groq API key
        langchain_api_key: LangSmith API key
        
    Returns:
        Configured evaluation components
    """
    # Set API keys
    os.environ["GROQ_API_KEY"] = api_key
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    
    # Initialize model
    model = ChatGroq(model="llama3-8b-8192")
    
    # Define evaluation configuration
    eval_config = RunEvalConfig(
        evaluators=[
            "qa",  # Q&A evaluator
            "cot_qa",  # Chain of thought Q&A evaluator
            "criteria",  # Criteria-based evaluator
        ],
        custom_evaluators=[],
        eval_llm=model
    )
    
    # Create custom evaluator
    relevance_evaluator = load_evaluator("labeled_criteria", criteria="relevance")
    
    return {
        "model": model,
        "eval_config": eval_config,
        "relevance_evaluator": relevance_evaluator
    }

# Example usage
# groq_api_key = "your-groq-api-key"
# langchain_api_key = "your-langchain-api-key"
# eval_components = setup_langsmith_evaluation(groq_api_key, langchain_api_key)

# Define a simple chain to evaluate
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain

# prompt = PromptTemplate(
#     input_variables=["question"],
#     template="Answer the following question: {question}"
# )
# chain = LLMChain(llm=eval_components["model"], prompt=prompt)

# Run evaluation on a dataset
# dataset_name = "your-dataset-name"
# results = run_on_dataset(
#     dataset_name=dataset_name,
#     llm_or_chain_factory=lambda: chain,
#     evaluation=eval_components["eval_config"],
#     verbose=True
# )
```

### 2. Latency Optimization

**What it is:**
Latency optimization refers to techniques and strategies to reduce the time it takes for an LLM application to respond to user inputs. Lower latency improves user experience and enables real-time applications.

**Explanation:**
Key strategies for latency optimization include:
1. Using faster models (like Groq's high-speed inference)
2. Implementing caching for repeated queries
3. Optimizing prompts to reduce token count
4. Parallelizing independent operations
5. Using streaming responses for faster perceived performance
6. Reducing context window size when possible

**Coding Example:**
```python
import time
from langchain.cache import InMemoryCache
from langchain.schema import HumanMessage
from langchain.chat_models import ChatGroq

class LatencyOptimizer:
    """
    A class to demonstrate latency optimization techniques.
    """
    def __init__(self, api_key):
        """
        Initialize the latency optimizer.
        
        Args:
            api_key: Groq API key
        """
        # Set up model
        self.model = ChatGroq(model="llama3-8b-8192", groq_api_key=api_key)
        
        # Set up cache
        self.model.cache = InMemoryCache()
        
        # Track metrics
        self.latency_history = []
        self.cache_hits = 0
        self.total_requests = 0
    
    def measure_latency(self, func, *args, **kwargs):
        """
        Measure the latency of a function call.
        
        Args:
            func: Function to measure
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Tuple of (result, latency)
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        latency = end_time - start_time
        return result, latency
    
    def generate_response(self, prompt, use_cache=True):
        """
        Generate a response with latency tracking.
        
        Args:
            prompt: Input prompt
            use_cache: Whether to use caching
            
        Returns:
            Response and latency information
        """
        self.total_requests += 1
        
        # Generate response
        if use_cache:
            response, latency = self.measure_latency(
                self.model.invoke,
                [HumanMessage(content=prompt)]
            )
            
            # Check if this was a cache hit
            if latency < 0.1:  # Threshold for cache hit
                self.cache_hits += 1
        else:
            # Temporarily disable cache
            original_cache = self.model.cache
            self.model.cache = None
            
            response, latency = self.measure_latency(
                self.model.invoke,
                [HumanMessage(content=prompt)]
            )
            
            # Restore cache
            self.model.cache = original_cache
        
        # Track latency
        self.latency_history.append(latency)
        
        return {
            "response": response.content,
            "latency": latency,
            "cache_hit": use_cache and latency < 0.1
        }
    
    def optimize_prompt(self, original_prompt):
        """
        Optimize a prompt to reduce token count.
        
        Args:
            original_prompt: Original prompt
            
        Returns:
            Optimized prompt
        """
        # Simple optimization strategies
        optimized = original_prompt
        
        # Remove redundant whitespace
        optimized = " ".join(optimized.split())
        
        # Remove unnecessary pleasantries
        phrases_to_remove = [
            "please", "could you", "would you mind", "if you don't mind",
            "i was wondering if", "it would be great if"
        ]
        
        for phrase in phrases_to_remove:
            optimized = optimized.lower().replace(phrase, "")
        
        # Capitalize first letter
        if optimized:
            optimized = optimized[0].upper() + optimized[1:]
        
        return optimized
    
    def get_stats(self):
        """
        Get performance statistics.
        
        Returns:
            Dictionary of statistics
        """
        if not self.latency_history:
            return {"error": "No data available"}
        
        avg_latency = sum(self.latency_history) / len(self.latency_history)
        min_latency = min(self.latency_history)
        max_latency = max(self.latency_history)
        
        cache_hit_rate = (self.cache_hits / self.total_requests) * 100 if self.total_requests > 0 else 0
        
        return {
            "average_latency": avg_latency,
            "min_latency": min_latency,
            "max_latency": max_latency,
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": cache_hit_rate
        }

# Example usage
# optimizer = LatencyOptimizer("your-groq-api-key")

# Test with and without caching
# prompt = "What is the capital of France?"
# result_with_cache = optimizer.generate_response(prompt, use_cache=True)
# print(f"With cache - Latency: {result_with_cache['latency']:.4f}s, Response: {result_with_cache['response']}")

# Test again to see cache effect
# result_cached = optimizer.generate_response(prompt, use_cache=True)
# print(f"Cached - Latency: {result_cached['latency']:.4f}s, Response: {result_cached['response']}")

# Test without cache
# result_no_cache = optimizer.generate_response(prompt, use_cache=False)
# print(f"No cache - Latency: {result_no_cache['latency']:.4f}s, Response: {result_no_cache['response']}")

# Get statistics
# stats = optimizer.get_stats()
# print(f"Statistics: {stats}")
```

### 3. Caching Strategies

**What it is:**
Caching in LLM applications involves storing the results of previous model inferences to avoid recomputing them when the same or similar inputs are encountered again. This can significantly reduce latency and costs.

**Explanation:**
Common caching strategies include:
1. In-memory caching: Fastest option, but limited to a single process
2. Database caching: Persistent across sessions, but slower
3. Distributed caching: Scalable across multiple servers
4. Semantic caching: Stores similar prompts together, allowing for partial matches
5. Prompt fingerprinting: Creates unique identifiers for similar prompts

**Coding Example:**
```python
import hashlib
import json
import time
import sqlite3
from typing import Dict, Any, Optional
from langchain.cache import BaseCache
from langchain.schema import Generation

class SQLiteCache(BaseCache):
    """
    A SQLite-based cache for LLM responses.
    """
    def __init__(self, db_path="llm_cache.db"):
        """
        Initialize the SQLite cache.
        
        Args:
            db_path: Path to the SQLite database
        """
        self.db_path = db_path
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize the database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create cache table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS cache (
            prompt_hash TEXT PRIMARY KEY,
            prompt TEXT,
            response TEXT,
            timestamp REAL,
            model_name TEXT,
            token_count INTEGER
        )
        ''')
        
        # Create stats table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS stats (
            stat_name TEXT PRIMARY KEY,
            stat_value REAL
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def _hash_prompt(self, prompt: str) -> str:
        """
        Create a hash for a prompt.
        
        Args:
            prompt: The prompt to hash
            
        Returns:
            Hash string
        """
        return hashlib.sha256(prompt.encode()).hexdigest()
    
    def lookup(self, prompt: str, llm_string: str) -> Optional[Generation]:
        """
        Look up a prompt in the cache.
        
        Args:
            prompt: The prompt to look up
            llm_string: The LLM string (not used in this implementation)
            
        Returns:
            Generation if found, None otherwise
        """
        prompt_hash = self._hash_prompt(prompt)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT response FROM cache WHERE prompt_hash = ?",
            (prompt_hash,)
        )
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            # Update stats
            self._update_stat("cache_hits", 1)
            return Generation(text=result[0])
        
        return None
    
    def update(self, prompt: str, llm_string: str, return_val: Generation) -> None:
        """
        Update the cache with a new prompt-response pair.
        
        Args:
            prompt: The prompt
            llm_string: The LLM string (not used in this implementation)
            return_val: The generation to cache
        """
        prompt_hash = self._hash_prompt(prompt)
        response_text = return_val.text
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT OR REPLACE INTO cache VALUES (?, ?, ?, ?, ?, ?)",
            (
                prompt_hash,
                prompt,
                response_text,
                time.time(),
                "llama3-8b-8192",  # Default model name
                len(response_text.split())  # Approximate token count
            )
        )
        
        conn.commit()
        conn.close()
        
        # Update stats
        self._update_stat("cache_misses", 1)
    
    def _update_stat(self, stat_name: str, increment: float):
        """
        Update a statistic.
        
        Args:
            stat_name: Name of the statistic
            increment: Amount to increment by
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT OR REPLACE INTO stats VALUES (?, COALESCE((SELECT stat_value FROM stats WHERE stat_name = ?), 0) + ?)",
            (stat_name, stat_name, increment)
        )
        
        conn.commit()
        conn.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary of statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all stats
        cursor.execute("SELECT stat_name, stat_value FROM stats")
        stats_rows = cursor.fetchall()
        stats = {row[0]: row[1] for row in stats_rows}
        
        # Get cache size
        cursor.execute("SELECT COUNT(*) FROM cache")
        cache_size = cursor.fetchone()[0]
        stats["cache_size"] = cache_size
        
        # Calculate hit rate
        cache_hits = stats.get("cache_hits", 0)
        cache_misses = stats.get("cache_misses", 0)
        total_requests = cache_hits + cache_misses
        
        if total_requests > 0:
            hit_rate = (cache_hits / total_requests) * 100
            stats["hit_rate"] = hit_rate
        
        conn.close()
        return stats
    
    def clear(self):
        """Clear the cache."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM cache")
        
        conn.commit()
        conn.close()

class SemanticCache:
    """
    A semantic cache that stores similar prompts together.
    """
    def __init__(self, embedding_model, threshold=0.9):
        """
        Initialize the semantic cache.
        
        Args:
            embedding_model: Model to create embeddings
            threshold: Similarity threshold for cache hits
        """
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.cache = []  # List of (prompt, response, embedding) tuples
    
    def _get_embedding(self, text):
        """
        Get embedding for a text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # In a real implementation, this would use the embedding model
        # For demonstration, we'll use a simple hash-based approach
        import numpy as np
        hash_value = int(hashlib.sha256(text.encode()).hexdigest(), 16) % 1000
        return np.array([hash_value % 100, (hash_value // 100) % 100])
    
    def _cosine_similarity(self, vec1, vec2):
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity
        """
        import numpy as np
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)
    
    def lookup(self, prompt):
        """
        Look up a prompt in the semantic cache.
        
        Args:
            prompt: The prompt to look up
            
        Returns:
            Response if found, None otherwise
        """
        prompt_embedding = self._get_embedding(prompt)
        
        for cached_prompt, cached_response, cached_embedding in self.cache:
            similarity = self._cosine_similarity(prompt_embedding, cached_embedding)
            
            if similarity >= self.threshold:
                return cached_response
        
        return None
    
    def update(self, prompt, response):
        """
        Update the semantic cache.
        
        Args:
            prompt: The prompt
            response: The response
        """
        prompt_embedding = self._get_embedding(prompt)
        self.cache.append((prompt, response, prompt_embedding))
    
    def get_stats(self):
        """
        Get cache statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "cache_size": len(self.cache),
            "threshold": self.threshold
        }

# Example usage
# from langchain.chat_models import ChatGroq

# Set up models
# model = ChatGroq(model="llama3-8b-8192", groq_api_key="your-groq-api-key")

# Set up SQLite cache
# sqlite_cache = SQLiteCache("llm_cache.db")
# model.cache = sqlite_cache

# Test the cache
# prompt = "What is the capital of France?"
# response1 = model.invoke([HumanMessage(content=prompt)])
# print(f"First response: {response1.content}")

# Second call should be faster (from cache)
# response2 = model.invoke([HumanMessage(content=prompt)])
# print(f"Cached response: {response2.content}")

# Get cache stats
# cache_stats = sqlite_cache.get_stats()
# print(f"Cache stats: {cache_stats}")

# Test semantic cache
# semantic_cache = SemanticCache(embedding_model=None)  # Using default embedding
# semantic_response = semantic_cache.lookup(prompt)
# if not semantic_response:
#     semantic_response = model.invoke([HumanMessage(content=prompt)]).content
#     semantic_cache.update(prompt, semantic_response)
# print(f"Semantic cache response: {semantic_response}")
```

## When and Where We Use Evaluation & Cost Optimization

### When to Use Evaluation:
- During development to test different model configurations
- When deploying new versions of an application
- When monitoring production performance
- When troubleshooting issues or user complaints
- When comparing different models or prompts

### When to Use Cost Optimization:
- When operating at scale with high request volumes
- When working with budget constraints
- When building real-time applications with strict latency requirements
- When deploying to resource-constrained environments

### Where to Use Evaluation:
- In CI/CD pipelines for automated testing
- In production monitoring dashboards
- In A/B testing frameworks
- In development environments for iterative improvement

### Where to Use Cost Optimization:
- In API layers that handle LLM requests
- In client-side applications that make frequent calls
- In backend services that process multiple requests
- In edge computing environments with limited resources

**Coding Example:**
```python
class LLMOptimizationFramework:
    """
    A comprehensive framework for LLM evaluation and cost optimization.
    """
    def __init__(self, api_key, langchain_api_key=None):
        """
        Initialize the optimization framework.
        
        Args:
            api_key: Groq API key
            langchain_api_key: LangSmith API key (optional)
        """
        # Set up models
        self.models = {
            "llama3-8b": ChatGroq(model="llama3-8b-8192", groq_api_key=api_key),
            "mixtral-8x7b": ChatGroq(model="mixtral-8x7b-32768", groq_api_key=api_key)
        }
        
        # Set up caches
        self.sqlite_cache = SQLiteCache("llm_optimization_cache.db")
        self.semantic_cache = SemanticCache(embedding_model=None)
        
        # Set up LangSmith if API key provided
        self.langchain_api_key = langchain_api_key
        if langchain_api_key:
            os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
        
        # Track metrics
        self.metrics = {
            "requests": 0,
            "tokens": 0,
            "cost": 0,
            "latency": [],
            "cache_hits": 0,
            "model_usage": {model: 0 for model in self.models}
        }
    
    def evaluate_model(self, model_name, test_dataset):
        """
        Evaluate a model on a test dataset.
        
        Args:
            model_name: Name of the model to evaluate
            test_dataset: Dataset of (prompt, expected_response) pairs
            
        Returns:
            Evaluation results
        """
        if model_name not in self.models:
            return {"error": f"Model {model_name} not found"}
        
        model = self.models[model_name]
        results = []
        
        for prompt, expected_response in test_dataset:
            # Generate response
            start_time = time.time()
            response = model.invoke([HumanMessage(content=prompt)])
            end_time = time.time()
            
            # Calculate metrics
            latency = end_time - start_time
            
            # Simple accuracy check (in a real implementation, use more sophisticated metrics)
            accuracy = 1 if expected_response.lower() in response.content.lower() else 0
            
            results.append({
                "prompt": prompt,
                "expected": expected_response,
                "actual": response.content,
                "latency": latency,
                "accuracy": accuracy
            })
            
            # Update metrics
            self.metrics["requests"] += 1
            self.metrics["latency"].append(latency)
            self.metrics["model_usage"][model_name] += 1
        
        # Calculate aggregate metrics
        avg_latency = sum(r["latency"] for r in results) / len(results)
        accuracy = sum(r["accuracy"] for r in results) / len(results)
        
        return {
            "model_name": model_name,
            "avg_latency": avg_latency,
            "accuracy": accuracy,
            "total_requests": len(results),
            "detailed_results": results
        }
    
    def compare_models(self, test_dataset):
        """
        Compare multiple models on a test dataset.
        
        Args:
            test_dataset: Dataset of (prompt, expected_response) pairs
            
        Returns:
            Comparison results
        """
        comparison_results = {}
        
        for model_name in self.models:
            comparison_results[model_name] = self.evaluate_model(model_name, test_dataset)
        
        return comparison_results
    
    def optimize_request(self, prompt, model_name="llama3-8b", use_cache=True):
        """
        Make an optimized request to an LLM.
        
        Args:
            prompt: Input prompt
            model_name: Name of the model to use
            use_cache: Whether to use caching
            
        Returns:
            Response and optimization information
        """
        if model_name not in self.models:
            return {"error": f"Model {model_name} not found"}
        
        model = self.models[model_name]
        
        # Check caches
        cache_response = None
        cache_source = None
        
        if use_cache:
            # Check SQLite cache
            model.cache = self.sqlite_cache
            cache_response = self.sqlite_cache.lookup(prompt, "")
            
            if cache_response:
                cache_source = "sqlite"
            else:
                # Check semantic cache
                cache_response = self.semantic_cache.lookup(prompt)
                if cache_response:
                    cache_source = "semantic"
        
        # Generate response if not in cache
        if cache_response:
            response = cache_response
            self.metrics["cache_hits"] += 1
        else:
            start_time = time.time()
            response_obj = model.invoke([HumanMessage(content=prompt)])
            end_time = time.time()
            
            response = response_obj.content
            latency = end_time - start_time
            
            # Update caches
            if use_cache:
                self.sqlite_cache.update(prompt, response_obj)
                self.semantic_cache.update(prompt, response)
            
            # Update metrics
            self.metrics["requests"] += 1
            self.metrics["latency"].append(latency)
            self.metrics["model_usage"][model_name] += 1
        
        return {
            "response": response,
            "model_name": model_name,
            "cache_source": cache_source,
            "metrics": self.get_metrics()
        }
    
    def get_metrics(self):
        """
        Get current metrics.
        
        Returns:
            Dictionary of metrics
        """
        metrics = self.metrics.copy()
        
        # Calculate derived metrics
        if metrics["latency"]:
            metrics["avg_latency"] = sum(metrics["latency"]) / len(metrics["latency"])
            metrics["min_latency"] = min(metrics["latency"])
            metrics["max_latency"] = max(metrics["latency"])
        
        if metrics["requests"] > 0:
            metrics["cache_hit_rate"] = (metrics["cache_hits"] / metrics["requests"]) * 100
        
        # Get cache stats
        metrics["sqlite_cache"] = self.sqlite_cache.get_stats()
        metrics["semantic_cache"] = self.semantic_cache.get_stats()
        
        return metrics
    
    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics = {
            "requests": 0,
            "tokens": 0,
            "cost": 0,
            "latency": [],
            "cache_hits": 0,
            "model_usage": {model: 0 for model in self.models}
        }

# Example usage
# framework = LLMOptimizationFramework("your-groq-api-key", "your-langchain-api-key")

# Define test dataset
# test_dataset = [
#     ("What is the capital of France?", "Paris"),
#     ("Who wrote 'Romeo and Juliet'?", "Shakespeare"),
#     ("What is the largest planet in our solar system?", "Jupiter")
# ]

# Compare models
# comparison = framework.compare_models(test_dataset)
# print("Model comparison:")
# for model_name, results in comparison.items():
#     print(f"{model_name}: Avg Latency: {results['avg_latency']:.4f}s, Accuracy: {results['accuracy']:.2f}")

# Make optimized request
# optimized_result = framework.optimize_request("What is the capital of Spain?")
# print(f"Optimized result: {optimized_result['response']}")
# print(f"Cache source: {optimized_result['cache_source']}")

# Get metrics
# metrics = framework.get_metrics()
# print(f"Metrics: {metrics}")
```

## Real-World Applications and Examples

### 1. Customer Support Analytics Dashboard

**Application:**
A dashboard that monitors and evaluates the performance of a customer support chatbot, tracking metrics like response time, accuracy, and user satisfaction.

**Coding Example:**
```python
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json

class CustomerSupportAnalytics:
    """
    Analytics dashboard for customer support chatbot performance.
    """
    def __init__(self, api_key, langchain_api_key=None):
        """
        Initialize the analytics dashboard.
        
        Args:
            api_key: Groq API key
            langchain_api_key: LangSmith API key (optional)
        """
        self.framework = LLMOptimizationFramework(api_key, langchain_api_key)
        self.conversations = []
        self.feedback = []
    
    def log_conversation(self, user_input, bot_response, model_name="llama3-8b"):
        """
        Log a conversation for analytics.
        
        Args:
            user_input: User's input
            bot_response: Bot's response
            model_name: Name of the model used
        """
        self.conversations.append({
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "bot_response": bot_response,
            "model_name": model_name
        })
    
    def log_feedback(self, conversation_id, rating, comment=None):
        """
        Log user feedback for a conversation.
        
        Args:
            conversation_id: ID of the conversation
            rating: User rating (1-5)
            comment: Optional comment
        """
        self.feedback.append({
            "conversation_id": conversation_id,
            "rating": rating,
            "comment": comment,
            "timestamp": datetime.now().isoformat()
        })
    
    def generate_report(self, days=7):
        """
        Generate a performance report for the specified time period.
        
        Args:
            days: Number of days to include in the report
            
        Returns:
            Report data
        """
        # Filter conversations by date
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_conversations = [
            conv for conv in self.conversations
            if datetime.fromisoformat(conv["timestamp"]) >= cutoff_date
        ]
        
        # Calculate metrics
        total_conversations = len(recent_conversations)
        
        # Get model usage
        model_usage = {}
        for conv in recent_conversations:
            model = conv["model_name"]
            model_usage[model] = model_usage.get(model, 0) + 1
        
        # Calculate average response length
        response_lengths = [len(conv["bot_response"]) for conv in recent_conversations]
        avg_response_length = sum(response_lengths) / len(response_lengths) if response_lengths else 0
        
        # Get feedback metrics
        recent_feedback = [
            fb for fb in self.feedback
            if datetime.fromisoformat(fb["timestamp"]) >= cutoff_date
        ]
        
        if recent_feedback:
            avg_rating = sum(fb["rating"] for fb in recent_feedback) / len(recent_feedback)
        else:
            avg_rating = 0
        
        # Get framework metrics
        framework_metrics = self.framework.get_metrics()
        
        # Create report
        report = {
            "period_days": days,
            "total_conversations": total_conversations,
            "model_usage": model_usage,
            "avg_response_length": avg_response_length,
            "avg_rating": avg_rating,
            "framework_metrics": framework_metrics,
            "conversations": recent_conversations,
            "feedback": recent_feedback
        }
        
        return report
    
    def visualize_report(self, report):
        """
        Visualize the report data.
        
        Args:
            report: Report data from generate_report
        """
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Customer Support Analytics - Last {report['period_days']} Days", fontsize=16)
        
        # Model usage pie chart
        if report["model_usage"]:
            axs[0, 0].pie(
                report["model_usage"].values(),
                labels=report["model_usage"].keys(),
                autopct='%1.1f%%'
            )
            axs[0, 0].set_title("Model Usage")
        
        # Latency histogram
        if report["framework_metrics"]["latency"]:
            axs[0, 1].hist(report["framework_metrics"]["latency"], bins=20)
            axs[0, 1].set_title("Response Latency Distribution")
            axs[0, 1].set_xlabel("Latency (seconds)")
        
        # Daily conversation count
        daily_counts = {}
        for conv in report["conversations"]:
            date = datetime.fromisoformat(conv["timestamp"]).date().isoformat()
            daily_counts[date] = daily_counts.get(date, 0) + 1
        
        if daily_counts:
            axs[1, 0].bar(daily_counts.keys(), daily_counts.values())
            axs[1, 0].set_title("Daily Conversation Count")
            axs[1, 0].tick_params(axis='x', rotation=45)
        
        # Feedback rating distribution
        if report["feedback"]:
            ratings = [fb["rating"] for fb in report["feedback"]]
            axs[1, 1].hist(ratings, bins=5, range=(0.5, 5.5))
            axs[1, 1].set_title("Feedback Rating Distribution")
            axs[1, 1].set_xlabel("Rating")
        
        plt.tight_layout()
        plt.show()
    
    def save_report(self, report, file_path):
        """
        Save the report to a file.
        
        Args:
            report: Report data
            file_path: Path to save the report
        """
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to {file_path}")

# Example usage
# analytics = CustomerSupportAnalytics("your-groq-api-key", "your-langchain-api-key")

# Log some conversations
# analytics.log_conversation("What is your return policy?", "Our return policy allows returns within 30 days of purchase.")
# analytics.log_conversation("How do I reset my password?", "You can reset your password by clicking 'Forgot Password' on the login page.")
# analytics.log_conversation("What are your business hours?", "Our business hours are 9 AM to 5 PM, Monday to Friday.")

# Log some feedback
# analytics.log_feedback(0, 5, "Very helpful response!")
# analytics.log_feedback(1, 4, "Good, but could be more detailed.")
# analytics.log_feedback(2, 3, "Response was accurate but took too long.")

# Generate and visualize report
# report = analytics.generate_report(days=7)
# analytics.visualize_report(report)

# Save report
# analytics.save_report(report, "customer_support_report.json")
```

### 2. Cost Optimization Service

**Application:**
A service that automatically optimizes LLM usage by selecting the most cost-effective model for each request, implementing caching strategies, and providing detailed cost analytics.

**Coding Example:**
```python
class CostOptimizationService:
    """
    A service for optimizing LLM costs.
    """
    def __init__(self, api_key, pricing_info=None):
        """
        Initialize the cost optimization service.
        
        Args:
            api_key: Groq API key
            pricing_info: Dictionary with pricing information
        """
        # Set up models
        self.models = {
            "llama3-8b": ChatGroq(model="llama3-8b-8192", groq_api_key=api_key),
            "mixtral-8x7b": ChatGroq(model="mixtral-8x7b-32768", groq_api_key=api_key)
        }
        
        # Default pricing (per 1K tokens)
        self.pricing = pricing_info or {
            "llama3-8b": {"input": 0.0005, "output": 0.0005},
            "mixtral-8x7b": {"input": 0.001, "output": 0.001}
        }
        
        # Set up caches
        self.sqlite_cache = SQLiteCache("cost_optimization_cache.db")
        self.semantic_cache = SemanticCache(embedding_model=None)
        
        # Track usage and costs
        self.usage = {
            model: {
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost": 0
            }
            for model in self.models
        }
        
        self.total_cost = 0
        self.total_requests = 0
    
    def estimate_tokens(self, text):
        """
        Estimate the number of tokens in a text.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # Simple estimation: 4 characters per token on average
        return len(text) / 4
    
    def estimate_cost(self, prompt, model_name):
        """
        Estimate the cost of a request.
        
        Args:
            prompt: Input prompt
            model_name: Name of the model
            
        Returns:
            Estimated cost
        """
        if model_name not in self.pricing:
            return 0
        
        # Estimate input tokens
        input_tokens = self.estimate_tokens(prompt)
        
        # Estimate output tokens (rough estimate based on input)
        output_tokens = input_tokens * 0.75  # Assume output is 75% of input
        
        # Calculate cost
        input_cost = (input_tokens / 1000) * self.pricing[model_name]["input"]
        output_cost = (output_tokens / 1000) * self.pricing[model_name]["output"]
        
        return input_cost + output_cost
    
    def select_model(self, prompt, complexity="medium"):
        """
        Select the most cost-effective model for a prompt.
        
        Args:
            prompt: Input prompt
            complexity: Complexity of the task (simple, medium, complex)
            
        Returns:
            Selected model name
        """
        # For simple tasks, use the smaller model
        if complexity == "simple":
            return "llama3-8b"
        
        # For complex tasks, use the larger model
        if complexity == "complex":
            return "mixtral-8x7b"
        
        # For medium tasks, estimate costs and choose
        llama_cost = self.estimate_cost(prompt, "llama3-8b")
        mixtral_cost = self.estimate_cost(prompt, "mixtral-8x7b")
        
        # If mixtral is not much more expensive, use it for better quality
        if mixtral_cost < llama_cost * 1.5:
            return "mixtral-8x7b"
        else:
            return "llama3-8b"
    
    def optimized_request(self, prompt, complexity="medium", use_cache=True):
        """
        Make an optimized request to an LLM.
        
        Args:
            prompt: Input prompt
            complexity: Complexity of the task
            use_cache: Whether to use caching
            
        Returns:
            Response and cost information
        """
        # Select model
        model_name = self.select_model(prompt, complexity)
        
        # Check caches
        cache_response = None
        cache_source = None
        
        if use_cache:
            # Check SQLite cache
            self.models[model_name].cache = self.sqlite_cache
            cache_response = self.sqlite_cache.lookup(prompt, "")
            
            if cache_response:
                cache_source = "sqlite"
            else:
                # Check semantic cache
                cache_response = self.semantic_cache.lookup(prompt)
                if cache_response:
                    cache_source = "semantic"
        
        # Generate response if not in cache
        if cache_response:
            response = cache_response
            cost = 0  # No cost for cached responses
        else:
            # Make request
            start_time = time.time()
            response_obj = self.models[model_name].invoke([HumanMessage(content=prompt)])
            end_time = time.time()
            
            response = response_obj.content
            latency = end_time - start_time
            
            # Estimate tokens and cost
            input_tokens = self.estimate_tokens(prompt)
            output_tokens = self.estimate_tokens(response)
            
            input_cost = (input_tokens / 1000) * self.pricing[model_name]["input"]
            output_cost = (output_tokens / 1000) * self.pricing[model_name]["output"]
            cost = input_cost + output_cost
            
            # Update caches
            if use_cache:
                self.sqlite_cache.update(prompt, response_obj)
                self.semantic_cache.update(prompt, response)
            
            # Update usage
            self.usage[model_name]["requests"] += 1
            self.usage[model_name]["input_tokens"] += input_tokens
            self.usage[model_name]["output_tokens"] += output_tokens
            self.usage[model_name]["cost"] += cost
        
        # Update totals
        self.total_requests += 1
        self.total_cost += cost
        
        return {
            "response": response,
            "model_name": model_name,
            "cost": cost,
            "cache_source": cache_source,
            "latency": latency if not cache_response else 0
        }
    
    def get_cost_report(self):
        """
        Generate a cost report.
        
        Returns:
            Cost report data
        """
        report = {
            "total_cost": self.total_cost,
            "total_requests": self.total_requests,
            "avg_cost_per_request": self.total_cost / self.total_requests if self.total_requests > 0 else 0,
            "model_usage": {}
        }
        
        for model_name, usage_data in self.usage.items():
            if usage_data["requests"] > 0:
                report["model_usage"][model_name] = {
                    "requests": usage_data["requests"],
                    "input_tokens": usage_data["input_tokens"],
                    "output_tokens": usage_data["output_tokens"],
                    "cost": usage_data["cost"],
                    "avg_cost_per_request": usage_data["cost"] / usage_data["requests"]
                }
        
        return report
    
    def visualize_cost_report(self, report):
        """
        Visualize the cost report.
        
        Args:
            report: Cost report data
        """
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("LLM Cost Optimization Report", fontsize=16)
        
        # Model usage pie chart
        if report["model_usage"]:
            model_names = list(report["model_usage"].keys())
            model_costs = [report["model_usage"][model]["cost"] for model in model_names]
            
            axs[0, 0].pie(model_costs, labels=model_names, autopct='%1.1f%%')
            axs[0, 0].set_title("Cost by Model")
        
        # Cost per request bar chart
        if report["model_usage"]:
            model_names = list(report["model_usage"].keys())
            avg_costs = [report["model_usage"][model]["avg_cost_per_request"] for model in model_names]
            
            axs[0, 1].bar(model_names, avg_costs)
            axs[0, 1].set_title("Average Cost per Request")
            axs[0, 1].set_ylabel("Cost ($)")
        
        # Token usage
        if report["model_usage"]:
            model_names = list(report["model_usage"].keys())
            input_tokens = [report["model_usage"][model]["input_tokens"] for model in model_names]
            output_tokens = [report["model_usage"][model]["output_tokens"] for model in model_names]
            
            x = range(len(model_names))
            width = 0.35
            
            axs[1, 0].bar(x, input_tokens, width, label='Input Tokens')
            axs[1, 0].bar([i + width for i in x], output_tokens, width, label='Output Tokens')
            axs[1, 0].set_title("Token Usage by Model")
            axs[1, 0].set_xticks([i + width/2 for i in x])
            axs[1, 0].set_xticklabels(model_names)
            axs[1, 0].legend()
        
        # Cache hit rate
        cache_stats = self.sqlite_cache.get_stats()
        if "hit_rate" in cache_stats:
            hit_rate = cache_stats["hit_rate"]
            miss_rate = 100 - hit_rate
            
            axs[1, 1].pie([hit_rate, miss_rate], labels=["Cache Hits", "Cache Misses"], autopct='%1.1f%%')
            axs[1, 1].set_title("Cache Hit Rate")
        
        plt.tight_layout()
        plt.show()

# Example usage
# cost_service = CostOptimizationService("your-groq-api-key")

# Make some optimized requests
# result1 = cost_service.optimized_request("What is the capital of France?", complexity="simple")
# print(f"Response: {result1['response']}, Cost: ${result1['cost']:.6f}, Model: {result1['model_name']}")

# result2 = cost_service.optimized_request("Explain quantum computing in detail.", complexity="complex")
# print(f"Response: {result2['response']}, Cost: ${result2['cost']:.6f}, Model: {result2['model_name']}")

# Test cache
# result3 = cost_service.optimized_request("What is the capital of France?", complexity="simple")
# print(f"Cached response: {result3['response']}, Cost: ${result3['cost']:.6f}, Cache: {result3['cache_source']}")

# Generate and visualize cost report
# cost_report = cost_service.get_cost_report()
# cost_service.visualize_cost_report(cost_report)
```

## Common Problems and How Advancements Have Solved Them

### Problem 1: Lack of Comprehensive Evaluation Tools

**Problem:**
Evaluating LLM applications was challenging due to the lack of standardized tools and metrics. Developers had to build custom evaluation frameworks, which was time-consuming and often inconsistent.

**Solution:**
LangSmith and similar platforms provide comprehensive evaluation tools that:
1. Offer standardized metrics for different types of applications
2. Enable visualization of model behavior and performance
3. Support A/B testing of different configurations
4. Facilitate collaboration among team members
5. Provide detailed logging and debugging capabilities

**Problem Code:**
```python
# Old approach - manual evaluation
def manual_evaluation(model, test_questions, expected_answers):
    """
    Manually evaluate a model on test questions.
    
    Args:
        model: The model to evaluate
        test_questions: List of test questions
        expected_answers: List of expected answers
        
    Returns:
        Dictionary of evaluation results
    """
    results = []
    
    for i, question in enumerate(test_questions):
        # Generate response
        response = model.invoke([HumanMessage(content=question)])
        
        # Simple accuracy check
        is_correct = expected_answers[i].lower() in response.content.lower()
        
        # Record result
        results.append({
            "question": question,
            "expected": expected_answers[i],
            "actual": response.content,
            "is_correct": is_correct
        })
    
    # Calculate accuracy
    accuracy = sum(1 for r in results if r["is_correct"]) / len(results)
    
    return {
        "accuracy": accuracy,
        "results": results
    }

# This approach is limited and doesn't provide detailed insights
# model = ChatGroq(model="llama3-8b-8192", groq_api_key="your-groq-api-key")
# test_questions = ["What is the capital of France?", "Who wrote 'Romeo and Juliet'?"]
# expected_answers = ["Paris", "Shakespeare"]
# eval_results = manual_evaluation(model, test_questions, expected_answers)
# print(f"Accuracy: {eval_results['accuracy']:.2f}")
```

**Enhancement Code:**
```python
# New approach with LangSmith
def langsmith_evaluation(model, test_dataset, eval_config):
    """
    Evaluate a model using LangSmith.
    
    Args:
        model: The model to evaluate
        test_dataset: Dataset to evaluate on
        eval_config: Evaluation configuration
        
    Returns:
        Evaluation results
    """
    # Run evaluation on dataset
    results = run_on_dataset(
        dataset_name=test_dataset,
        llm_or_chain_factory=lambda: model,
        evaluation=eval_config,
        verbose=True
    )
    
    return results

# This approach provides comprehensive evaluation with multiple metrics
# from langchain.chat_models import ChatGroq
# from langchain.smith import RunEvalConfig

# model = ChatGroq(model="llama3-8b-8192", groq_api_key="your-groq-api-key")
# eval_config = RunEvalConfig(
#     evaluators=[
#         "qa",  # Q&A evaluator
#         "cot_qa",  # Chain of thought Q&A evaluator
#         "criteria",  # Criteria-based evaluator
#     ],
#     custom_evaluators=[],
#     eval_llm=model
# )

# eval_results = langsmith_evaluation(model, "your-dataset-name", eval_config)
# print(f"Evaluation results: {eval_results}")
```

### Problem 2: High Costs and Latency in Production

**Problem:**
Running LLM applications in production was expensive and slow, especially at scale. Each API call incurred costs, and response times could be inconsistent, leading to poor user experiences.

**Solution:**
Advanced optimization strategies address these issues by:
1. Implementing intelligent caching to avoid redundant API calls
2. Selecting the most cost-effective model for each task
3. Optimizing prompts to reduce token count
4. Using streaming for faster perceived performance
5. Monitoring and analyzing usage patterns to identify optimization opportunities

**Problem Code:**
```python
# Old approach - simple API calls without optimization
def simple_llm_api(prompt, api_key):
    """
    Make a simple API call to an LLM.
    
    Args:
        prompt: Input prompt
        api_key: API key
        
    Returns:
        Response from the LLM
    """
    # Initialize model
    model = ChatGroq(model="llama3-8b-8192", groq_api_key=api_key)
    
    # Make API call
    response = model.invoke([HumanMessage(content=prompt)])
    
    return response.content

# This approach is inefficient, especially for repeated or similar requests
# api_key = "your-groq-api-key"
# response1 = simple_llm_api("What is the capital of France?", api_key)
# response2 = simple_llm_api("What is the capital of France?", api_key)  # Same request, full cost again
```

**Enhancement Code:**
```python
# New approach with comprehensive optimization
def optimized_llm_api(prompt, api_key, complexity="medium"):
    """
    Make an optimized API call to an LLM.
    
    Args:
        prompt: Input prompt
        api_key: API key
        complexity: Complexity of the task
        
    Returns:
        Response and optimization information
    """
    # Initialize cost optimization service
    cost_service = CostOptimizationService(api_key)
    
    # Make optimized request
    result = cost_service.optimized_request(prompt, complexity=complexity)
    
    return {
        "response": result["response"],
        "model_used": result["model_name"],
        "cost": result["cost"],
        "cache_source": result["cache_source"],
        "latency": result["latency"]
    }

# This approach is much more efficient, especially for repeated or similar requests
# api_key = "your-groq-api-key"
# result1 = optimized_llm_api("What is the capital of France?", api_key)
# print(f"First response: {result1['response']}, Cost: ${result1['cost']:.6f}")

# Second request should be faster and cheaper (from cache)
# result2 = optimized_llm_api("What is the capital of France?", api_key)
# print(f"Cached response: {result2['response']}, Cost: ${result2['cost']:.6f}")
```

## 🧪 Exercise: Evaluate RAG Flows

Let's create an exercise to evaluate RAG (Retrieval-Augmented Generation) flows using LangSmith.

```python
def evaluate_rag_flows(api_key, langchain_api_key, knowledge_base_path, test_questions):
    """
    Exercise: Evaluate RAG flows using LangSmith.
    
    Args:
        api_key: Groq API key
        langchain_api_key: LangSmith API key
        knowledge_base_path: Path to the knowledge base
        test_questions: List of test questions
    """
    print("Setting up RAG flow evaluation...")
    
    # Set up LangSmith
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    
    # Initialize model
    model = ChatGroq(model="llama3-8b-8192", groq_api_key=api_key)
    
    # Set up knowledge base
    print("Loading knowledge base...")
    loader = DirectoryLoader(knowledge_base_path, glob="**/*.txt", recursive=True)
    documents = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    
    # Note: Using OpenAI embeddings for this example
    embeddings = OpenAIEmbeddings(openai_api_key="your-openai-api-key")
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    # Create RAG chain
    print("Creating RAG chain...")
    from langchain.chains import RetrievalQA
    rag_chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    
    # Create evaluation dataset
    print("Creating evaluation dataset...")
    dataset_name = "rag-evaluation-dataset"
    
    # In a real implementation, you would create a proper dataset
    # For this exercise, we'll use the test questions directly
    
    # Set up evaluation
    eval_config = RunEvalConfig(
        evaluators=[
            "qa",  # Q&A evaluator
            "cot_qa",  # Chain of thought Q&A evaluator
            "context_qa",  # Context-aware Q&A evaluator
        ],
        custom_evaluators=[],
        eval_llm=model
    )
    
    # Run evaluation
    print("Running evaluation...")
    results = []
    
    for question in test_questions:
        # Get RAG response
        rag_response = rag_chain.run(question)
        
        # Evaluate response
        eval_result = {
            "question": question,
            "answer": rag_response,
            "contexts": [doc.page_content for doc in vectorstore.similarity_search(question, k=3)]
        }
        
        results.append(eval_result)
    
    # Print results
    print("\nEvaluation Results:")
    for i, result in enumerate(results, 1):
        print(f"\nQuestion {i}: {result['question']}")
        print(f"Answer: {result['answer']}")
        print(f"Contexts: {len(result['contexts'])} documents retrieved")
    
    # Calculate metrics
    # In a real implementation, you would use LangSmith's evaluation metrics
    # For this exercise, we'll just count the results
    print(f"\nTotal questions evaluated: {len(results)}")
    
    return results

# Example usage
# api_key = "your-groq-api-key"
# langchain_api_key = "your-langchain-api-key"
# knowledge_base_path = "knowledge_base"  # Assuming this directory exists
# test_questions = [
#     "What products does TechCorp offer?",
#     "How much does the Pro plan cost?",
#     "What is the company's return policy?"
# ]

# evaluation_results = evaluate_rag_flows(api_key, langchain_api_key, knowledge_base_path, test_questions)
```

## 📌 Project: LLMOps Dashboard

Now, let's create a comprehensive project - an LLMOps dashboard for monitoring and optimizing LLM applications.

```python
import os
import time
import json
import argparse
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
from typing import Dict, List, Any, Optional

class LLMOpsDashboard:
    """
    A comprehensive LLMOps dashboard for monitoring and optimizing LLM applications.
    """
    def __init__(self, api_key, langchain_api_key=None, config_path=None):
        """
        Initialize the LLMOps dashboard.
        
        Args:
            api_key: Groq API key
            langchain_api_key: LangSmith API key (optional)
            config_path: Path to configuration file (optional)
        """
        # Load configuration
        self.config = self._load_config(config_path) if config_path else {}
        
        # Set up services
        self.cost_service = CostOptimizationService(api_key)
        self.analytics = CustomerSupportAnalytics(api_key, langchain_api_key)
        
        # Initialize data storage
        self.data_dir = self.config.get("data_dir", "llmops_data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize dashboard
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self._setup_layout()
        self._setup_callbacks()
    
    def _load_config(self, config_path):
        """
        Load configuration from a file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Configuration dictionary
        """
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _setup_layout(self):
        """Set up the dashboard layout."""
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("LLMOps Dashboard", className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            # Control panel
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Controls"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Date Range"),
                                    dcc.DatePickerRange(
                                        id='date-range',
                                        min_date_allowed=datetime.now() - timedelta(days=30),
                                        max_date_allowed=datetime.now(),
                                        start_date=datetime.now() - timedelta(days=7),
                                        end_date=datetime.now()
                                    )
                                ], width=6),
                                dbc.Col([
                                    html.Label("Model Filter"),
                                    dcc.Dropdown(
                                        id='model-filter',
                                        options=[
                                            {'label': 'All Models', 'value': 'all'},
                                            {'label': 'Llama3-8B', 'value': 'llama3-8b'},
                                            {'label': 'Mixtral-8x7B', 'value': 'mixtral-8x7b'}
                                        ],
                                        value='all'
                                    )
                                ], width=6)
                            ]),
                            html.Br(),
                            dbc.Button("Refresh Data", id="refresh-button", color="primary", className="mt-2")
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Metrics cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Total Cost", className="card-title"),
                            html.H2(id="total-cost", children="$0.00", className="text-primary")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Total Requests", className="card-title"),
                            html.H2(id="total-requests", children="0", className="text-success")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Avg. Latency", className="card-title"),
                            html.H2(id="avg-latency", children="0.00s", className="text-info")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Cache Hit Rate", className="card-title"),
                            html.H2(id="cache-hit-rate", children="0%", className="text-warning")
                        ])
                    ])
                ], width=3)
            ], className="mb-4"),
            
            # Charts
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Cost Over Time"),
                        dbc.CardBody([
                            dcc.Graph(id="cost-chart")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Request Volume"),
                        dbc.CardBody([
                            dcc.Graph(id="volume-chart")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Model Usage"),
                        dbc.CardBody([
                            dcc.Graph(id="model-usage-chart")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Latency Distribution"),
                        dbc.CardBody([
                            dcc.Graph(id="latency-chart")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Data table
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Recent Requests"),
                        dbc.CardBody([
                            html.Div(id="recent-requests-table")
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=30*1000,  # 30 seconds
                n_intervals=0
            )
        ], fluid=True)
    
    def _setup_callbacks(self):
        """Set up the dashboard callbacks."""
        @self.app.callback(
            [
                Output("total-cost", "children"),
                Output("total-requests", "children"),
                Output("avg-latency", "children"),
                Output("cache-hit-rate", "children"),
                Output("cost-chart", "figure"),
                Output("volume-chart", "figure"),
                Output("model-usage-chart", "figure"),
                Output("latency-chart", "figure"),
                Output("recent-requests-table", "children")
            ],
            [
                Input("refresh-button", "n_clicks"),
                Input("interval-component", "n_intervals")
            ],
            [
                State("date-range", "start_date"),
                State("date-range", "end_date"),
                State("model-filter", "value")
            ]
        )
        def update_dashboard(n_clicks, n_intervals, start_date, end_date, model_filter):
            """Update the dashboard with fresh data."""
            # Get cost report
            cost_report = self.cost_service.get_cost_report()
            
            # Get analytics report
            analytics_report = self.analytics.generate_report(days=7)
            
            # Update metrics
            total_cost = f"${cost_report['total_cost']:.4f}"
            total_requests = str(cost_report['total_requests'])
            
            if cost_report['latency']:
                avg_latency = f"{sum(cost_report['latency']) / len(cost_report['latency']):.2f}s"
            else:
                avg_latency = "0.00s"
            
            cache_stats = self.cost_service.sqlite_cache.get_stats()
            cache_hit_rate = f"{cache_stats.get('hit_rate', 0):.1f}%"
            
            # Create charts
            cost_chart = self._create_cost_chart(cost_report)
            volume_chart = self._create_volume_chart(analytics_report)
            model_usage_chart = self._create_model_usage_chart(cost_report)
            latency_chart = self._create_latency_chart(cost_report)
            
            # Create data table
            recent_requests_table = self._create_recent_requests_table(analytics_report)
            
            return (
                total_cost,
                total_requests,
                avg_latency,
                cache_hit_rate,
                cost_chart,
                volume_chart,
                model_usage_chart,
                latency_chart,
                recent_requests_table
            )
    
    def _create_cost_chart(self, cost_report):
        """
        Create a cost over time chart.
        
        Args:
            cost_report: Cost report data
            
        Returns:
            Plotly figure
        """
        # In a real implementation, you would have time-series cost data
        # For this example, we'll create a simple bar chart
        
        fig = go.Figure()
        
        if cost_report["model_usage"]:
            models = list(cost_report["model_usage"].keys())
            costs = [cost_report["model_usage"][model]["cost"] for model in models]
            
            fig.add_trace(go.Bar(
                x=models,
                y=costs,
                marker_color='rgb(55, 83, 109)'
            ))
        
        fig.update_layout(
            title="Cost by Model",
            xaxis_title="Model",
            yaxis_title="Cost ($)",
            height=300
        )
        
        return fig
    
    def _create_volume_chart(self, analytics_report):
        """
        Create a request volume chart.
        
        Args:
            analytics_report: Analytics report data
            
        Returns:
            Plotly figure
        """
        # In a real implementation, you would have time-series volume data
        # For this example, we'll create a simple line chart
        
        fig = go.Figure()
        
        # Get daily counts
        daily_counts = {}
        for conv in analytics_report["conversations"]:
            date = datetime.fromisoformat(conv["timestamp"]).date().isoformat()
            daily_counts[date] = daily_counts.get(date, 0) + 1
        
        if daily_counts:
            dates = list(daily_counts.keys())
            counts = list(daily_counts.values())
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=counts,
                mode='lines+markers',
                name='Requests'
            ))
        
        fig.update_layout(
            title="Request Volume Over Time",
            xaxis_title="Date",
            yaxis_title="Number of Requests",
            height=300
        )
        
        return fig
    
    def _create_model_usage_chart(self, cost_report):
        """
        Create a model usage pie chart.
        
        Args:
            cost_report: Cost report data
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        if cost_report["model_usage"]:
            models = list(cost_report["model_usage"].keys())
            requests = [cost_report["model_usage"][model]["requests"] for model in models]
            
            fig.add_trace(go.Pie(
                labels=models,
                values=requests,
                hole=0.3
            ))
        
        fig.update_layout(
            title="Model Usage Distribution",
            height=300
        )
        
        return fig
    
    def _create_latency_chart(self, cost_report):
        """
        Create a latency distribution chart.
        
        Args:
            cost_report: Cost report data
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        if cost_report["latency"]:
            fig.add_trace(go.Histogram(
                x=cost_report["latency"],
                nbinsx=20,
                marker_color='rgb(55, 83, 109)'
            ))
        
        fig.update_layout(
            title="Response Latency Distribution",
            xaxis_title="Latency (seconds)",
            yaxis_title="Frequency",
            height=300
        )
        
        return fig
    
    def _create_recent_requests_table(self, analytics_report):
        """
        Create a table of recent requests.
        
        Args:
            analytics_report: Analytics report data
            
        Returns:
            HTML table
        """
        # Get recent conversations
        recent_conversations = analytics_report["conversations"][-10:]  # Last 10
        
        # Create table
        table_header = [
            html.Thead([
                html.Tr([
                    html.Th("Time"),
                    html.Th("User Input"),
                    html.Th("Model"),
                    html.Th("Response Length")
                ])
            ])
        ]
        
        table_body = [
            html.Tbody([
                html.Tr([
                    html.Td(datetime.fromisoformat(conv["timestamp"]).strftime("%H:%M:%S")),
                    html.Td(conv["user_input"][:50] + "..." if len(conv["user_input"]) > 50 else conv["user_input"]),
                    html.Td(conv["model_name"]),
                    html.Td(str(len(conv["bot_response"])))
                ]) for conv in recent_conversations
            ])
        ]
        
        return dbc.Table(table_header + table_body, striped=True, bordered=True, hover=True)
    
    def run(self, debug=False, port=8050):
        """
        Run the dashboard.
        
        Args:
            debug: Whether to run in debug mode
            port: Port to run on
        """
        self.app.run_server(debug=debug, port=port)

def create_sample_data():
    """Create sample data for the dashboard."""
    # Create sample knowledge base
    os.makedirs("knowledge_base", exist_ok=True)
    
    with open("knowledge_base/product_info.txt", "w") as f:
        f.write("""
        Product Information
        
        CloudSync Features:
        - 10GB free storage space
        - Real-time synchronization across all devices
        - Advanced file sharing with customizable permissions
        - End-to-end encryption for security
        - Integration with popular productivity tools
        - 24/7 customer support
        
        Pricing:
        - Free Plan: 10GB storage, basic features
        - Pro Plan: $9.99/month, 100GB storage, advanced features
        - Business Plan: $19.99/month per user, 1TB storage, team features
        - Enterprise Plan: Custom pricing, unlimited storage, dedicated support
        """)
    
    with open("knowledge_base/support_info.txt", "w") as f:
        f.write("""
        Support Information
        
        Getting Help:
        Our customer support team is available 24/7 to assist you with any questions or issues.
        
        Contact Methods:
        - Email: support@techcorp.com
        - Phone: 1-800-TECHCORP (1-800-832-4267)
        - Live Chat: Available on our website
        - Help Center: https://help.techcorp.com
        
        Common Issues and Solutions:
        
        1. Forgot Password
           Solution: Click "Forgot Password" on the login page and follow the instructions.
        
        2. Can't Sync Files
           Solution: Check your internet connection, ensure you have enough storage space, 
           and verify that the CloudSync app is up to date.
        """)

def main():
    """Main function for the LLMOps dashboard project."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="LLMOps Dashboard")
    parser.add_argument("--api-key", type=str, required=True, help="Groq API key")
    parser.add_argument("--langchain-api-key", type=str, help="LangSmith API key")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--create-sample-data", action="store_true", help="Create sample data")
    parser.add_argument("--port", type=int, default=8050, help="Port to run on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    # Create sample data if requested
    if args.create_sample_data:
        print("Creating sample data...")
        create_sample_data()
    
    # Initialize and run dashboard
    print("Initializing LLMOps Dashboard...")
    dashboard = LLMOpsDashboard(
        api_key=args.api_key,
        langchain_api_key=args.langchain_api_key,
        config_path=args.config
    )
    
    print(f"Running dashboard on port {args.port}...")
    dashboard.run(debug=args.debug, port=args.port)

if __name__ == "__main__":
    main()
```

