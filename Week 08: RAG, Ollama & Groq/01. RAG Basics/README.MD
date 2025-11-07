# Day 36 – RAG Basics

## What is Retrieval-Augmented Generation (RAG)?

Retrieval-Augmented Generation (RAG) is a technique that combines the strengths of retrieval-based and generative AI systems. In a RAG system, an LLM is augmented with an external knowledge retrieval mechanism that allows it to access relevant information before generating responses. This approach helps overcome limitations of LLMs such as knowledge cutoff dates, hallucination, and lack of domain-specific knowledge.

The basic RAG pipeline works as follows:
1. A user query is received
2. Relevant documents/information is retrieved from a knowledge base
3. The retrieved information is provided as context to the LLM
4. The LLM generates a response based on both the query and the retrieved context

## Essential Subtopics

### 1. Context Stuffing

**What it is:**
Context stuffing refers to the process of including relevant information directly in the context window of an LLM. In RAG systems, this means adding retrieved documents to the prompt before sending it to the language model.

**Explanation:**
When we retrieve documents from our knowledge base, we need to include them in the prompt along with the user's question. This is called "stuffing" the context. The challenge is to maximize the amount of relevant information while staying within the context window limits of the model.

**Coding Example:**
```python
def stuff_context(query, retrieved_docs, model_context_limit=4096):
    """
    Stuff retrieved documents into the context while respecting model limits.
    
    Args:
        query: The user's question
        retrieved_docs: List of retrieved documents
        model_context_limit: Maximum context length for the model
        
    Returns:
        A prompt with stuffed context
    """
    # Start with the basic prompt structure
    prompt_template = """Answer the following question based on the provided context:
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:"""
    
    # Calculate available space for context
    query_length = len(query)
    template_length = len(prompt_template.format(context="", question=query))
    available_space = model_context_limit - template_length
    
    # Add documents until we run out of space
    context = ""
    for doc in retrieved_docs:
        doc_text = f"\n\n{doc['content']}"
        if len(context) + len(doc_text) <= available_space:
            context += doc_text
        else:
            break
    
    # Create the final prompt
    prompt = prompt_template.format(context=context, question=query)
    return prompt

# Example usage
query = "What is the capital of France?"
retrieved_docs = [
    {"content": "France is a country in Western Europe. Its capital is Paris."},
    {"content": "Paris is known for the Eiffel Tower and the Louvre Museum."},
    {"content": "France has a population of about 67 million people."}
]

prompt = stuff_context(query, retrieved_docs)
print(prompt)
```

### 2. Context Compression

**What it is:**
Context compression is a technique to reduce the size of retrieved information while preserving its essential meaning. This allows more relevant information to fit within the limited context window of an LLM.

**Explanation:**
Instead of including entire documents, compression techniques extract only the most relevant parts or summarize the information. This can be done through various methods including extractive summarization (selecting important sentences) or abstractive summarization (generating new summaries).

**Coding Example:**
```python
from transformers import pipeline

class ContextCompressor:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        """
        Initialize the context compressor with a summarization model.
        
        Args:
            model_name: Name of the summarization model to use
        """
        self.summarizer = pipeline("summarization", model=model_name)
    
    def compress_context(self, documents, max_length=150, min_length=30):
        """
        Compress multiple documents into a concise summary.
        
        Args:
            documents: List of document texts
            max_length: Maximum length of the summary
            min_length: Minimum length of the summary
            
        Returns:
            A compressed summary of the documents
        """
        # Combine documents
        combined_text = " ".join(documents)
        
        # Generate summary
        summary = self.summarizer(
            combined_text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )
        
        return summary[0]['summary_text']

# Example usage
documents = [
    "France is a country in Western Europe. Its capital is Paris. France has a population of about 67 million people.",
    "Paris is known for the Eiffel Tower and the Louvre Museum. The city is divided into 20 arrondissements.",
    "France has a diverse economy with sectors including tourism, manufacturing, and agriculture. It's one of the world's largest economies."
]

compressor = ContextCompressor()
compressed_context = compressor.compress_context(documents)
print(f"Compressed context: {compressed_context}")
```

## When and Where We Use RAG

### When to Use RAG:
- When you need up-to-date information beyond the LLM's training cutoff
- When dealing with domain-specific knowledge not well-represented in the LLM's training data
- When you need to reduce hallucinations by grounding responses in factual information
- When you need to provide citations or sources for the information provided

### Where to Use RAG:
- Customer support systems that need to access product documentation
- Enterprise knowledge bases for internal Q&A
- Research assistants that need to access academic papers
- Legal or medical systems that require accurate, sourced information

**Coding Example:**
```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

def setup_rag_system(documents, openai_api_key):
    """
    Set up a basic RAG system using LangChain.
    
    Args:
        documents: List of documents to use as knowledge base
        openai_api_key: OpenAI API key
        
    Returns:
        A configured RAG system
    """
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    # Create vector store
    vectorstore = FAISS.from_texts(documents, embeddings)
    
    # Initialize LLM
    llm = OpenAI(openai_api_key=openai_api_key)
    
    # Create RAG chain
    rag_system = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    
    return rag_system

# Example usage
documents = [
    "The company's return policy allows returns within 30 days of purchase.",
    "To process a return, customers must provide their order number and proof of purchase.",
    "Refunds are typically processed within 5-7 business days after the return is received."
]

# In a real scenario, you would provide your actual API key
# rag_system = setup_rag_system(documents, "your-api-key")
# query = "How long do I have to return an item?"
# response = rag_system.run(query)
# print(response)
```

## Real-World Applications and Examples

### 1. Customer Support Chatbots

**Application:**
Companies use RAG to power customer support chatbots that can answer questions about products, services, and policies by retrieving information from product documentation, FAQs, and knowledge bases.

**Coding Example:**
```python
class CustomerSupportRAG:
    def __init__(self, product_docs, faq_docs, policy_docs):
        """
        Initialize a customer support RAG system.
        
        Args:
            product_docs: Documentation about products
            faq_docs: Frequently asked questions
            policy_docs: Company policies
        """
        # In a real implementation, this would set up vector stores
        # and retrieval systems for each type of document
        self.product_docs = product_docs
        self.faq_docs = faq_docs
        self.policy_docs = policy_docs
        
        # Mock retrieval function
        self.retriever = self._mock_retriever
    
    def _mock_retriever(self, query, doc_type="all"):
        """Mock retrieval function for demonstration."""
        if doc_type == "product":
            return ["Product X has a battery life of 24 hours."]
        elif doc_type == "faq":
            return ["To reset Product X, hold the power button for 10 seconds."]
        elif doc_type == "policy":
            return ["Our return policy allows returns within 30 days."]
        else:
            return [
                "Product X has a battery life of 24 hours.",
                "To reset Product X, hold the power button for 10 seconds.",
                "Our return policy allows returns within 30 days."
            ]
    
    def answer_query(self, query):
        """
        Answer a customer query using RAG.
        
        Args:
            query: The customer's question
            
        Returns:
            Response to the customer
        """
        # Retrieve relevant documents
        retrieved_docs = self.retriever(query)
        
        # Stuff context into prompt
        context = "\n".join(retrieved_docs)
        
        # Mock LLM response (in reality, this would call an actual LLM)
        if "battery" in query.lower():
            return "Based on our product documentation, Product X has a battery life of 24 hours."
        elif "reset" in query.lower():
            return "According to our FAQ, to reset Product X, hold the power button for 10 seconds."
        elif "return" in query.lower():
            return "Our policy allows returns within 30 days of purchase."
        else:
            return f"I'm not sure about that. Here's what I found: {context}"

# Example usage
product_docs = ["Product X has a battery life of 24 hours.", "Product X is water-resistant."]
faq_docs = ["To reset Product X, hold the power button for 10 seconds."]
policy_docs = ["Our return policy allows returns within 30 days."]

support_bot = CustomerSupportRAG(product_docs, faq_docs, policy_docs)

print(support_bot.answer_query("How long does the battery last?"))
print(support_bot.answer_query("How do I reset my device?"))
print(support_bot.answer_query("Can I return this product?"))
```

### 2. Research Assistant

**Application:**
Research assistants use RAG to help researchers find and synthesize information from academic papers, research articles, and other scholarly documents.

**Coding Example:**
```python
import requests
from bs4 import BeautifulSoup

class ResearchAssistantRAG:
    def __init__(self, paper_urls):
        """
        Initialize a research assistant RAG system.
        
        Args:
            paper_urls: URLs of research papers to index
        """
        self.paper_urls = paper_urls
        self.paper_contents = self._fetch_papers()
        
    def _fetch_papers(self):
        """Fetch and parse paper contents (mock implementation)."""
        contents = {}
        for url in self.paper_urls:
            # In a real implementation, this would fetch and parse the actual paper
            contents[url] = f"Content of paper at {url}. This paper discusses neural networks and their applications."
        return contents
    
    def search_papers(self, query):
        """
        Search for relevant information in papers.
        
        Args:
            query: Research query
            
        Returns:
            Relevant information from papers
        """
        # Mock search - in reality, this would use semantic search
        results = []
        for url, content in self.paper_contents.items():
            if any(keyword in content.lower() for keyword in query.lower().split()):
                results.append({
                    "url": url,
                    "content": content,
                    "relevance_score": 0.8  # Mock relevance score
                })
        
        return results
    
    def synthesize_findings(self, query):
        """
        Synthesize findings from multiple papers.
        
        Args:
            query: Research query
            
        Returns:
            Synthesized response
        """
        # Search for relevant papers
        relevant_papers = self.search_papers(query)
        
        if not relevant_papers:
            return "No relevant information found in the indexed papers."
        
        # Extract content from relevant papers
        paper_contents = [paper["content"] for paper in relevant_papers]
        
        # Mock synthesis (in reality, this would use an LLM)
        synthesis = f"Based on {len(relevant_papers)} relevant papers, here's what I found:\n\n"
        for i, paper in enumerate(relevant_papers, 1):
            synthesis += f"{i}. Paper from {paper['url']}: {paper['content'][:100]}...\n"
        
        synthesis += "\nKey findings: Neural networks have wide applications in various fields."
        
        return synthesis

# Example usage
paper_urls = [
    "https://arxiv.org/abs/1234.56789",
    "https://arxiv.org/abs/9876.54321",
    "https://arxiv.org/abs/1111.22222"
]

research_assistant = ResearchAssistantRAG(paper_urls)

print(research_assistant.synthesize_findings("neural networks"))
print(research_assistant.synthesize_findings("quantum computing"))
```

## Common Problems and How Advancements Have Solved Them

### Problem 1: Limited Context Window

**Problem:**
Early LLMs had very limited context windows (e.g., 2048 tokens), making it difficult to include enough retrieved information for complex queries.

**Solution:**
Advancements in model architecture have led to larger context windows (e.g., GPT-4 with 32K tokens, Claude with 100K tokens). Additionally, techniques like context compression help maximize the utility of available space.

**Problem Code:**
```python
# Old approach with limited context window
def old_rag_approach(query, retrieved_docs):
    # Simple concatenation without considering context limits
    context = "\n\n".join([doc["content"] for doc in retrieved_docs])
    
    prompt = f"""Answer the following question based on the provided context:
    
    Context:
    {context}
    
    Question: {query}
    
    Answer:"""
    
    # This could easily exceed context limits of older models
    return prompt

# Example that would exceed context limits
query = "What is the history of artificial intelligence?"
retrieved_docs = [{"content": "A very long document about AI history..."} * 100]  # Mock long documents

# This would fail with older models due to context length
# old_prompt = old_rag_approach(query, retrieved_docs)
```

**Enhancement Code:**
```python
# New approach with context management
def enhanced_rag_approach(query, retrieved_docs, model_context_limit=4096):
    # Calculate available space
    query_length = len(query)
    template_length = len("Answer the following question based on the provided context:\n\nContext:\n\nQuestion:\n\nAnswer:")
    available_space = model_context_limit - query_length - template_length
    
    # Select documents that fit
    selected_docs = []
    current_length = 0
    
    for doc in retrieved_docs:
        doc_length = len(doc["content"])
        if current_length + doc_length <= available_space:
            selected_docs.append(doc)
            current_length += doc_length
        else:
            # Try to include a truncated version
            remaining_space = available_space - current_length
            if remaining_space > 100:  # Only include if we can have a meaningful chunk
                truncated_doc = {
                    "content": doc["content"][:remaining_space],
                    "truncated": True
                }
                selected_docs.append(truncated_doc)
            break
    
    # Create prompt with selected documents
    context_parts = []
    for doc in selected_docs:
        if doc.get("truncated"):
            context_parts.append(f"[TRUNCATED] {doc['content']}...")
        else:
            context_parts.append(doc["content"])
    
    context = "\n\n".join(context_parts)
    
    prompt = f"""Answer the following question based on the provided context:
    
    Context:
    {context}
    
    Question: {query}
    
    Answer:"""
    
    return prompt

# Example with context management
query = "What is the history of artificial intelligence?"
retrieved_docs = [{"content": "A very long document about AI history..."} * 100]  # Mock long documents

enhanced_prompt = enhanced_rag_approach(query, retrieved_docs)
print(f"Enhanced prompt length: {len(enhanced_prompt)}")
```

### Problem 2: Inaccurate Retrieval

**Problem:**
Early retrieval systems often returned irrelevant documents, leading to poor quality responses from the LLM.

**Solution:**
Advanced embedding models (like OpenAI's text-embedding-ada-002 or Cohere's embeddings) provide better semantic understanding. Additionally, reranking techniques can improve the relevance of retrieved documents.

**Problem Code:**
```python
# Old approach with simple keyword matching
def old_retrieval(query, documents):
    # Simple keyword-based retrieval
    query_words = set(query.lower().split())
    scored_docs = []
    
    for doc in documents:
        doc_words = set(doc.lower().split())
        # Simple overlap score
        score = len(query_words.intersection(doc_words))
        scored_docs.append((doc, score))
    
    # Sort by score and return top 3
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_docs[:3]]

# Example with poor semantic understanding
documents = [
    "The bank of the river was overflowing with water.",
    "I need to go to the bank to deposit money.",
    "The financial bank reported record profits this quarter."
]

query = "financial institution"
# This would likely return the river document first due to simple word overlap
old_results = old_retrieval(query, documents)
print("Old retrieval results:", old_results)
```

**Enhancement Code:**
```python
# New approach with semantic embeddings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class EnhancedRetriever:
    def __init__(self, documents):
        """
        Initialize with semantic embeddings.
        
        Args:
            documents: List of documents to index
        """
        self.documents = documents
        # In a real implementation, this would use actual embeddings
        self.doc_embeddings = self._mock_embed_documents(documents)
    
    def _mock_embed_documents(self, documents):
        """Mock embedding function for demonstration."""
        # In reality, this would use a real embedding model
        embeddings = []
        for doc in documents:
            # Create mock embeddings based on document content
            if "bank" in doc and "money" in doc:
                embeddings.append([0.9, 0.1, 0.1])  # Financial bank
            elif "bank" in doc and "river" in doc:
                embeddings.append([0.1, 0.9, 0.1])  # River bank
            elif "bank" in doc and "financial" in doc:
                embeddings.append([0.95, 0.05, 0.1])  # Strong financial bank
            else:
                embeddings.append([0.3, 0.3, 0.3])  # Neutral
        return np.array(embeddings)
    
    def retrieve(self, query, top_k=3):
        """
        Retrieve documents using semantic similarity.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        # Mock query embedding
        if "financial" in query.lower():
            query_embedding = np.array([[0.9, 0.1, 0.1]])
        else:
            query_embedding = np.array([[0.3, 0.3, 0.3]])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.doc_embeddings)[0]
        
        # Get top documents
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [self.documents[i] for i in top_indices]

# Example with better semantic understanding
documents = [
    "The bank of the river was overflowing with water.",
    "I need to go to the bank to deposit money.",
    "The financial bank reported record profits this quarter."
]

retriever = EnhancedRetriever(documents)
query = "financial institution"
enhanced_results = retriever.retrieve(query)
print("Enhanced retrieval results:", enhanced_results)
```

## 🧪 Exercise: Simple RAG Demo

Let's create a simple RAG system that can answer questions about a set of documents.

```python
import os
import openai
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

def simple_rag_demo(api_key, document_path, query):
    """
    Simple RAG demonstration.
    
    Args:
        api_key: OpenAI API key
        document_path: Path to the document to use as knowledge base
        query: Question to answer
        
    Returns:
        Answer to the query
    """
    # Set OpenAI API key
    openai.api_key = api_key
    
    # Load document
    loader = TextLoader(document_path)
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    # Create RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    
    # Answer query
    answer = qa_chain.run(query)
    return answer

# Example usage (would require actual API key and document)
# api_key = "your-openai-api-key"
# document_path = "knowledge_base.txt"
# query = "What is the main topic of the document?"
# answer = simple_rag_demo(api_key, document_path, query)
# print(f"Answer: {answer}")
```

## 📌 Project: Local File-based QA Bot

Now, let's create a local file-based QA bot that can answer questions about documents stored on your local machine.

```python
import os
import glob
from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai

class LocalFileQABot:
    def __init__(self, api_key: str, file_directory: str, file_pattern: str = "*.txt"):
        """
        Initialize a local file-based QA bot.
        
        Args:
            api_key: OpenAI API key
            file_directory: Directory containing documents
            file_pattern: Pattern to match files (default: "*.txt")
        """
        self.api_key = api_key
        self.file_directory = file_directory
        self.file_pattern = file_pattern
        self.documents = {}
        self.vectorizer = None
        self.doc_vectors = None
        self._load_documents()
        self._build_index()
    
    def _load_documents(self):
        """Load all documents from the specified directory."""
        file_paths = glob.glob(os.path.join(self.file_directory, self.file_pattern))
        
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                self.documents[file_path] = {
                    'content': content,
                    'filename': os.path.basename(file_path)
                }
    
    def _build_index(self):
        """Build a search index using TF-IDF vectorization."""
        # Extract document contents
        doc_contents = [doc_info['content'] for doc_info in self.documents.values()]
        
        # Create TF-IDF vectors
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.doc_vectors = self.vectorizer.fit_transform(doc_contents)
    
    def retrieve_relevant_docs(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant documents for a query.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant documents with metadata
        """
        # Vectorize the query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.doc_vectors)[0]
        
        # Get top document indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return relevant documents with similarity scores
        relevant_docs = []
        for idx in top_indices:
            file_path = list(self.documents.keys())[idx]
            doc_info = self.documents[file_path].copy()
            doc_info['similarity'] = similarities[idx]
            doc_info['file_path'] = file_path
            relevant_docs.append(doc_info)
        
        return relevant_docs
    
    def generate_answer(self, query: str, relevant_docs: List[Dict[str, Any]]) -> str:
        """
        Generate an answer using the OpenAI API.
        
        Args:
            query: User query
            relevant_docs: List of relevant documents
            
        Returns:
            Generated answer
        """
        # Prepare context from relevant documents
        context = ""
        for doc in relevant_docs:
            context += f"\n\nDocument: {doc['filename']}\n{doc['content'][:500]}..."  # Limit content length
        
        # Create prompt
        prompt = f"""Answer the following question based on the provided documents:
        
        Context:
        {context}
        
        Question: {query}
        
        Answer:"""
        
        # Call OpenAI API
        openai.api_key = self.api_key
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=500,
            temperature=0.5
        )
        
        return response.choices[0].text.strip()
    
    def ask(self, query: str) -> Dict[str, Any]:
        """
        Ask a question and get an answer.
        
        Args:
            query: User query
            
        Returns:
            Dictionary containing answer and relevant documents
        """
        # Retrieve relevant documents
        relevant_docs = self.retrieve_relevant_docs(query)
        
        # Generate answer
        answer = self.generate_answer(query, relevant_docs)
        
        return {
            'answer': answer,
            'relevant_documents': relevant_docs
        }

# Example usage
def create_sample_documents():
    """Create sample documents for demonstration."""
    os.makedirs("sample_docs", exist_ok=True)
    
    # Document 1 - About Machine Learning
    with open("sample_docs/ml_intro.txt", "w") as f:
        f.write("""
        Machine Learning is a subset of artificial intelligence that focuses on building systems that can learn from data.
        There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.
        In supervised learning, the model is trained on labeled data. In unsupervised learning, the model finds patterns in unlabeled data.
        Reinforcement learning involves training agents to make decisions in an environment to maximize rewards.
        """)
    
    # Document 2 - About Deep Learning
    with open("sample_docs/deep_learning.txt", "w") as f:
        f.write("""
        Deep Learning is a specialized form of machine learning that uses neural networks with multiple layers.
        These deep neural networks can automatically learn features from raw data, eliminating the need for manual feature extraction.
        Common deep learning architectures include Convolutional Neural Networks (CNNs) for image processing,
        Recurrent Neural Networks (RNNs) for sequence data, and Transformers for natural language processing.
        Deep learning has achieved state-of-the-art results in many domains including computer vision, speech recognition, and natural language processing.
        """)
    
    # Document 3 - About Neural Networks
    with open("sample_docs/neural_networks.txt", "w") as f:
        f.write("""
        Neural Networks are computing systems inspired by the biological neural networks in animal brains.
        A neural network consists of layers of interconnected nodes or neurons. Each connection has a weight that is adjusted during training.
        The basic structure includes an input layer, one or more hidden layers, and an output layer.
        Neural networks learn by adjusting the weights of connections based on the error of the output compared to the expected result.
        This process is called backpropagation and is typically done using optimization algorithms like gradient descent.
        """)

# Create sample documents
create_sample_documents()

# Initialize the QA bot (would require actual API key)
# qa_bot = LocalFileQABot(api_key="your-openai-api-key", file_directory="sample_docs")

# Ask a question
# query = "What is the difference between machine learning and deep learning?"
# response = qa_bot.ask(query)
# print(f"Answer: {response['answer']}")
# print("\nRelevant Documents:")
# for doc in response['relevant_documents']:
#     print(f"- {doc['filename']} (Similarity: {doc['similarity']:.2f})")
```
