# Day 32 – Vector Databases & Embeddings

## FAISS

### What is it?

FAISS (Facebook AI Similarity Search) is a library developed by Meta (formerly Facebook) for efficient similarity search and clustering of dense vectors. It is optimized for high-performance retrieval in large-scale vector spaces, making it ideal for applications like semantic search, recommendation systems, and image/text retrieval. FAISS supports various indexing strategies, GPU acceleration, and approximate nearest neighbor (ANN) search algorithms to handle billions of vectors efficiently.

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt

# Text embedding model using Hugging Face transformers
class TextEmbedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
    
    def embed(self, texts):
        """Generate embeddings for a list of texts"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize texts
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**encoded)
            # Use mean pooling to get sentence embeddings
            embeddings = self.mean_pooling(outputs, encoded['attention_mask'])
            # Normalize embeddings
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.numpy()
    
    def mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to get sentence embeddings"""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# FAISS vector database implementation
class FaissVectorDB:
    def __init__(self, dimension, index_type="Flat"):
        self.dimension = dimension
        self.index_type = index_type
        
        if index_type == "Flat":
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        elif index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(dimension, 32)  # HNSW with 32 neighbors
            self.index.hnsw.efConstruction = 40
            self.index.hnsw.efSearch = 20
        elif index_type == "IVF":
            nlist = 100  # Number of clusters
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        self.texts = []
        self.ids = []
        self.is_trained = False
    
    def add(self, embeddings, texts, ids=None):
        """Add embeddings and corresponding texts to the index"""
        if ids is None:
            ids = list(range(len(self.ids), len(self.ids) + len(embeddings)))
        
        # Train IVF index if needed
        if self.index_type == "IVF" and not self.is_trained:
            self.index.train(embeddings)
            self.is_trained = True
        
        # Add to index
        self.index.add(embeddings.astype(np.float32))
        self.texts.extend(texts)
        self.ids.extend(ids)
    
    def search(self, query_embedding, k=5):
        """Search for the k most similar vectors"""
        if len(self.ids) == 0:
            return [], [], []
        
        # Ensure query embedding is normalized for cosine similarity
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        # Get results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # Valid index
                results.append({
                    'id': self.ids[idx],
                    'text': self.texts[idx],
                    'distance': float(distances[0][i]),
                    'similarity': float(distances[0][i])  # For IP index, distance = similarity
                })
        
        return results

# Demonstrate FAISS functionality
def demonstrate_faiss():
    # Sample documents
    documents = [
        "Machine learning algorithms can automatically improve with experience.",
        "Deep learning is a subset of machine learning that uses neural networks.",
        "Natural language processing enables computers to understand human language.",
        "Computer vision allows machines to interpret and understand visual information.",
        "Reinforcement learning involves agents learning through trial and error.",
        "Data science combines statistics, programming, and domain expertise.",
        "Artificial intelligence aims to create systems that can perform intelligent tasks.",
        "Neural networks are inspired by the human brain's structure and function."
    ]
    
    # Create embedder and vector database
    embedder = TextEmbedder()
    vector_db = FaissVectorDB(dimension=384, index_type="Flat")  # MiniLM has 384 dimensions
    
    # Generate embeddings and add to database
    print("Generating embeddings and building FAISS index...")
    embeddings = embedder.embed(documents)
    vector_db.add(embeddings, documents)
    
    print(f"Added {len(documents)} documents to FAISS index")
    print(f"Index type: {vector_db.index_type}")
    print(f"Vector dimension: {vector_db.dimension}")
    
    # Test search
    query = "What is deep learning?"
    query_embedding = embedder.embed(query)
    results = vector_db.search(query_embedding, k=3)
    
    print(f"\nQuery: {query}")
    print("Top 3 results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Similarity: {result['similarity']:.4f} - {result['text'][:100]}...")
    
    return vector_db, embedder

faiss_db, faiss_embedder = demonstrate_faiss()
```

### When and where we use FAISS

- **Large-scale similarity search**: When you need to search through millions or billions of vectors
- **Recommendation systems**: Finding similar items or users efficiently
- **Semantic search**: Retrieving documents based on meaning rather than keywords
- **Image retrieval**: Finding visually similar images
- **Real-time applications**: When low latency is crucial for user experience
- **Memory-constrained environments**: FAISS offers various trade-offs between accuracy and memory usage

```python
# When to use FAISS - Large-scale document retrieval example
def large_scale_retrieval_example():
    """Example of using FAISS for large-scale document retrieval"""
    
    # Generate synthetic large dataset
    num_documents = 10000
    documents = []
    
    topics = ["machine learning", "deep learning", "natural language processing", 
              "computer vision", "reinforcement learning", "data science"]
    
    for i in range(num_documents):
        topic = topics[i % len(topics)]
        document = f"This document is about {topic}. It contains information about {topic} techniques and applications. {topic} is an important field in artificial intelligence."
        documents.append(document)
    
    print(f"Generated {len(documents)} synthetic documents")
    
    # Create embedder and FAISS index
    embedder = TextEmbedder()
    vector_db = FaissVectorDB(dimension=384, index_type="HNSW")  # Use HNSW for large scale
    
    # Add documents in batches to manage memory
    batch_size = 1000
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
        
        embeddings = embedder.embed(batch_docs)
        vector_db.add(embeddings, batch_docs)
    
    print(f"Total documents in index: {len(vector_db.ids)}")
    
    # Test search performance
    import time
    query = "What are machine learning algorithms?"
    query_embedding = embedder.embed(query)
    
    start_time = time.time()
    results = vector_db.search(query_embedding, k=10)
    search_time = time.time() - start_time
    
    print(f"Search time: {search_time:.4f} seconds")
    print(f"Found {len(results)} results")
    print(f"Top result similarity: {results[0]['similarity']:.4f}")
    
    return vector_db

# Uncomment to run (takes time with large dataset)
# large_db = large_scale_retrieval_example()
```

### Real-world examples

- **Facebook**: Used for friend recommendation and content retrieval
- **Pinterest**: Powers visual search and recommendation systems
- **Spotify**: Finds similar songs and creates personalized playlists
- **E-commerce**: Product recommendation and visual search
- **Academic search**: Semantic search in research paper databases

### Problems and Advancements

**Problem**: Exact nearest neighbor search is computationally expensive for large datasets.
**Advancement**: Approximate nearest neighbor (ANN) algorithms like HNSW and IVF.

```python
# Compare different FAISS index types
def compare_faiss_indices():
    """Compare different FAISS index types for accuracy and speed"""
    
    # Generate test data
    np.random.seed(42)
    dimension = 128
    n_vectors = 10000
    n_queries = 100
    
    # Create random vectors
    vectors = np.random.random((n_vectors, dimension)).astype(np.float32)
    queries = np.random.random((n_queries, dimension)).astype(np.float32)
    
    # Normalize for cosine similarity
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    
    # Test different index types
    index_types = ["Flat", "HNSW", "IVF"]
    results = {}
    
    for index_type in index_types:
        print(f"\nTesting {index_type} index...")
        
        # Create index
        if index_type == "Flat":
            index = faiss.IndexFlatIP(dimension)
        elif index_type == "HNSW":
            index = faiss.IndexHNSWFlat(dimension, 32)
            index.hnsw.efConstruction = 40
            index.hnsw.efSearch = 20
        elif index_type == "IVF":
            nlist = 100
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(vectors)
        
        # Add vectors
        index.add(vectors)
        
        # Search
        import time
        start_time = time.time()
        distances, indices = index.search(queries, k=10)
        search_time = time.time() - start_time
        
        results[index_type] = {
            'search_time': search_time,
            'avg_time_per_query': search_time / n_queries,
            'index_size': index.ntotal
        }
        
        print(f"Search time: {search_time:.4f}s")
        print(f"Average time per query: {search_time/n_queries:.6f}s")
    
    return results

faiss_comparison = compare_faiss_indices()
```

## Chroma

### What is it?

Chroma is an open-source vector database designed specifically for AI applications, particularly for storing and querying embeddings generated by large language models. It provides a simple, developer-friendly interface with built-in support for metadata filtering, automatic embedding generation, and persistent storage. Chroma is designed to be lightweight, easy to integrate, and optimized for the types of queries common in AI applications.

### PyTorch Implementation

```python
import chromadb
from chromadb.utils import embedding_functions
import numpy as np
import uuid

# Chroma vector database implementation
class ChromaVectorDB:
    def __init__(self, collection_name="documents", embedding_function=None, persist_directory=None):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize Chroma client
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()
        
        # Set up embedding function
        if embedding_function is None:
            # Use Chroma's default embedding function
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        else:
            self.embedding_function = embedding_function
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
    
    def add_documents(self, documents, metadatas=None, ids=None):
        """Add documents to the collection"""
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        
        if metadatas is None:
            metadatas = [{} for _ in range(len(documents))]
        
        # Add to collection
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        return ids
    
    def search(self, query_text, n_results=5, where=None):
        """Search for similar documents"""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i] if results['metadatas'][0] else {},
                'distance': results['distances'][0][i]
            })
        
        return formatted_results
    
    def get_collection_info(self):
        """Get information about the collection"""
        return {
            'name': self.collection.name,
            'count': self.collection.count(),
            'metadata': self.collection.metadata
        }

# Demonstrate Chroma functionality
def demonstrate_chroma():
    # Sample documents with metadata
    documents = [
        "Machine learning algorithms can automatically improve with experience.",
        "Deep learning is a subset of machine learning that uses neural networks.",
        "Natural language processing enables computers to understand human language.",
        "Computer vision allows machines to interpret and understand visual information.",
        "Reinforcement learning involves agents learning through trial and error."
    ]
    
    metadatas = [
        {"category": "machine_learning", "source": "textbook"},
        {"category": "deep_learning", "source": "research_paper"},
        {"category": "nlp", "source": "tutorial"},
        {"category": "computer_vision", "source": "documentation"},
        {"category": "reinforcement_learning", "source": "blog"}
    ]
    
    # Create Chroma vector database
    chroma_db = ChromaVectorDB(
        collection_name="ai_documents",
        persist_directory="./chroma_db"
    )
    
    # Add documents
    ids = chroma_db.add_documents(documents, metadatas)
    print(f"Added {len(documents)} documents to Chroma collection")
    print(f"Collection info: {chroma_db.get_collection_info()}")
    
    # Test basic search
    query = "What is deep learning?"
    results = chroma_db.search(query, n_results=3)
    
    print(f"\nQuery: {query}")
    print("Top 3 results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Distance: {result['distance']:.4f} - {result['text'][:100]}...")
        print(f"   Metadata: {result['metadata']}")
    
    # Test filtered search
    print(f"\nFiltered search (category = 'machine_learning'):")
    filtered_results = chroma_db.search(
        "machine learning algorithms", 
        n_results=2, 
        where={"category": "machine_learning"}
    )
    
    for i, result in enumerate(filtered_results, 1):
        print(f"{i}. Distance: {result['distance']:.4f} - {result['text'][:100]}...")
    
    return chroma_db

chroma_db = demonstrate_chroma()
```

### When and where we use Chroma

- **RAG (Retrieval-Augmented Generation)**: Retrieving relevant context for LLMs
- **Document search applications**: Semantic search with metadata filtering
- **Prototype development**: Quick setup for AI applications
- **Small to medium datasets**: When you need persistent storage with simple querying
- **Applications with metadata**: When you need to filter results based on document attributes
- **Local development**: Easy to run locally without complex infrastructure

```python
# When to use Chroma - RAG application example
def rag_application_example():
    """Example of using Chroma for RAG applications"""
    
    # Create knowledge base documents
    knowledge_base = [
        "Python is a high-level programming language known for its readability and simplicity.",
        "PyTorch is an open-source machine learning library based on the Torch library.",
        "Transformers are deep learning models that use attention mechanisms for sequence processing.",
        "LangChain is a framework for developing applications powered by language models.",
        "Vector databases store embeddings and enable semantic similarity search.",
        "FAISS is a library for efficient similarity search developed by Facebook AI.",
        "Chroma is an open-source vector database designed for AI applications.",
        "RAG (Retrieval-Augmented Generation) combines retrieval and generation for better responses."
    ]
    
    # Create Chroma database
    rag_db = ChromaVectorDB(collection_name="rag_knowledge")
    
    # Add documents with source metadata
    metadatas = [{"source": "AI Knowledge Base"} for _ in knowledge_base]
    rag_db.add_documents(knowledge_base, metadatas)
    
    # Simulate RAG retrieval
    user_queries = [
        "What is PyTorch?",
        "How does RAG work?",
        "What are vector databases?"
    ]
    
    print("RAG Application Example:")
    print("=" * 50)
    
    for query in user_queries:
        print(f"\nUser Query: {query}")
        
        # Retrieve relevant context
        context_results = rag_db.search(query, n_results=2)
        context = "\n".join([result['text'] for result in context_results])
        
        print(f"Retrieved Context:\n{context}")
        print("-" * 30)
    
    return rag_db

rag_db = rag_application_example()
```

### Real-world examples

- **AI assistants**: Retrieving relevant information to answer user queries
- **Code documentation search**: Finding relevant code examples and documentation
- **Customer support**: Retrieving similar past issues and solutions
- **Research assistants**: Finding relevant academic papers and sources
- **Content recommendation**: Suggesting related articles or content

### Problems and Advancements

**Problem**: Chroma may not scale well to very large datasets compared to specialized solutions like FAISS.
**Advancement**: Integration with other vector databases and distributed computing frameworks.

```python
# Chroma vs FAISS comparison
def compare_chroma_faiss():
    """Compare Chroma and FAISS for different use cases"""
    
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing deals with human language understanding.",
        "Computer vision enables machines to see and interpret images.",
        "Reinforcement learning involves learning through rewards and punishments."
    ]
    
    # Test Chroma
    import time
    chroma_start = time.time()
    chroma_db = ChromaVectorDB(collection_name="comparison_test")
    chroma_db.add_documents(documents)
    chroma_add_time = time.time() - chroma_start
    
    chroma_search_start = time.time()
    chroma_results = chroma_db.search("What is machine learning?", n_results=2)
    chroma_search_time = time.time() - chroma_search_start
    
    # Test FAISS
    embedder = TextEmbedder()
    faiss_start = time.time()
    faiss_db = FaissVectorDB(dimension=384, index_type="Flat")
    embeddings = embedder.embed(documents)
    faiss_db.add(embeddings, documents)
    faiss_add_time = time.time() - faiss_start
    
    faiss_query_embedding = embedder.embed("What is machine learning?")
    faiss_search_start = time.time()
    faiss_results = faiss_db.search(faiss_query_embedding, k=2)
    faiss_search_time = time.time() - faiss_search_start
    
    print("Chroma vs FAISS Comparison:")
    print("=" * 40)
    print(f"Chroma - Add time: {chroma_add_time:.4f}s, Search time: {chroma_search_time:.4f}s")
    print(f"FAISS - Add time: {faiss_add_time:.4f}s, Search time: {faiss_search_time:.4f}s")
    print(f"Chroma results: {len(chroma_results)}")
    print(f"FAISS results: {len(faiss_results)}")
    
    return {
        'chroma': {'add_time': chroma_add_time, 'search_time': chroma_search_time},
        'faiss': {'add_time': faiss_add_time, 'search_time': faiss_search_time}
    }

comparison_results = compare_chroma_faiss()
```

## OpenAI Embeddings

### What is it?

OpenAI Embeddings are vector representations of text generated by OpenAI's embedding models, which convert text into high-dimensional numerical vectors that capture semantic meaning. These embeddings enable semantic search, clustering, and other similarity-based operations by representing the meaning of text in a way that similar concepts are close together in vector space. OpenAI's `text-embedding-ada-002` model is particularly popular for its balance of performance, cost, and quality.

### PyTorch Implementation

```python
import openai
import numpy as np
from typing import List, Union

# OpenAI Embedding implementation
class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str = "text-embedding-ada-002"):
        self.api_key = api_key
        self.model = model
        openai.api_key = api_key
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings using OpenAI API"""
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            response = openai.embeddings.create(
                input=texts,
                model=self.model
            )
            
            # Extract embeddings
            embeddings = []
            for item in response.data:
                embeddings.append(item.embedding)
            
            return np.array(embeddings)
        
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return np.array([])

# Integration with vector databases
class OpenAIVectorDB:
    def __init__(self, api_key: str, vector_db_type: str = "faiss", **kwargs):
        self.embedder = OpenAIEmbedder(api_key)
        self.vector_db_type = vector_db_type
        
        if vector_db_type == "faiss":
            dimension = 1536  # OpenAI ada-002 dimension
            self.vector_db = FaissVectorDB(dimension=dimension, **kwargs)
        elif vector_db_type == "chroma":
            # Chroma can use OpenAI embeddings directly
            import chromadb
            from chromadb.utils import embedding_functions
            
            self.client = chromadb.Client()
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name="text-embedding-ada-002"
            )
            self.collection = self.client.get_or_create_collection(
                name=kwargs.get("collection_name", "openai_docs"),
                embedding_function=self.embedding_function
            )
        else:
            raise ValueError(f"Unsupported vector DB type: {vector_db_type}")
    
    def add_documents(self, documents: List[str], metadatas: List[dict] = None, ids: List[str] = None):
        """Add documents to the vector database"""
        if self.vector_db_type == "faiss":
            embeddings = self.embedder.embed(documents)
            return self.vector_db.add(embeddings, documents, ids)
        elif self.vector_db_type == "chroma":
            if ids is None:
                import uuid
                ids = [str(uuid.uuid4()) for _ in range(len(documents))]
            if metadatas is None:
                metadatas = [{} for _ in range(len(documents))]
            
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            return ids
    
    def search(self, query: str, k: int = 5, **kwargs):
        """Search for similar documents"""
        if self.vector_db_type == "faiss":
            query_embedding = self.embedder.embed(query)
            return self.vector_db.search(query_embedding, k)
        elif self.vector_db_type == "chroma":
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                **kwargs
            )
            
            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'][0] else {},
                    'distance': results['distances'][0][i]
                })
            
            return formatted_results

# Note: OpenAI API key is required for this functionality
# For demonstration purposes, we'll show the structure without actual API calls
def demonstrate_openai_structure():
    """Demonstrate OpenAI embeddings structure (without actual API calls)"""
    
    print("OpenAI Embeddings Structure:")
    print("=" * 40)
    print("Model: text-embedding-ada-002")
    print("Dimensions: 1536")
    print("Input: Text (up to 8191 tokens)")
    print("Output: 1536-dimensional vector")
    print("Use case: Semantic search, clustering, classification")
    
    # Example usage pattern (requires API key)
    example_code = """
    # Initialize OpenAI embedder
    embedder = OpenAIEmbedder(api_key="your-api-key")
    
    # Generate embeddings
    embeddings = embedder.embed([
        "Machine learning is amazing",
        "Deep learning uses neural networks"
    ])
    
    # Use with FAISS
    faiss_db = OpenAIVectorDB(api_key="your-api-key", vector_db_type="faiss")
    faiss_db.add_documents(["Your documents here"])
    results = faiss_db.search("Your query here")
    """
    
    print("\nExample usage:")
    print(example_code)

demonstrate_openai_structure()
```

### When and where we use OpenAI Embeddings

- **Semantic search**: When you need high-quality embeddings without training your own models
- **Production applications**: When reliability and consistency are important
- **Multilingual applications**: OpenAI embeddings work well across multiple languages
- **Applications requiring high accuracy**: When embedding quality directly impacts application performance
- **Rapid prototyping**: When you want to quickly test embedding-based applications
- **Cost-effective solutions**: When the cost of API calls is acceptable for your use case

### Real-world examples

- **Enterprise search**: Internal document search with high accuracy requirements
- **Customer support systems**: Finding relevant support articles and solutions
- **Content recommendation**: High-quality content similarity for premium applications
- **Academic research**: Semantic search in research databases
- **Legal document analysis**: Finding relevant legal precedents and documents

### Problems and Advancements

**Problem**: API dependency and cost for large-scale applications.
**Advancement**: Hybrid approaches combining local and cloud embeddings, or fine-tuning open-source models.

## 🧪 Exercise: Document embedding

```python
def document_embedding_exercise():
    """Complete exercise: Document embedding with different approaches"""
    
    # Sample documents
    documents = [
        "Artificial intelligence is transforming the world through automation and intelligent systems.",
        "Machine learning algorithms learn patterns from data to make predictions and decisions.",
        "Deep learning uses neural networks with multiple layers to model complex patterns in data.",
        "Natural language processing enables computers to understand, interpret, and generate human language.",
        "Computer vision allows machines to extract meaningful information from digital images and videos.",
        "Reinforcement learning involves agents learning optimal behaviors through trial and error with rewards.",
        "Data science combines statistics, programming, and domain expertise to extract insights from data.",
        "Neural networks are inspired by the human brain and consist of interconnected processing units."
    ]
    
    print("Document Embedding Exercise")
    print("=" * 50)
    
    # 1. Hugging Face embeddings with FAISS
    print("\n1. Hugging Face embeddings with FAISS:")
    hf_embedder = TextEmbedder()
    faiss_db = FaissVectorDB(dimension=384, index_type="Flat")
    
    hf_embeddings = hf_embedder.embed(documents)
    faiss_db.add(hf_embeddings, documents)
    
    query = "What is machine learning?"
    hf_query_embedding = hf_embedder.embed(query)
    hf_results = faiss_db.search(hf_query_embedding, k=3)
    
    print(f"Query: {query}")
    for i, result in enumerate(hf_results, 1):
        print(f"  {i}. Similarity: {result['similarity']:.4f} - {result['text'][:80]}...")
    
    # 2. Hugging Face embeddings with Chroma
    print("\n2. Hugging Face embeddings with Chroma:")
    chroma_db = ChromaVectorDB(collection_name="exercise_docs")
    chroma_db.add_documents(documents)
    
    chroma_results = chroma_db.search(query, n_results=3)
    
    print(f"Query: {query}")
    for i, result in enumerate(chroma_results, 1):
        print(f"  {i}. Distance: {result['distance']:.4f} - {result['text'][:80]}...")
    
    # 3. Compare embedding dimensions and properties
    print("\n3. Embedding properties comparison:")
    print(f"Hugging Face MiniLM dimension: {hf_embeddings.shape[1]}")
    print(f"Number of documents embedded: {len(documents)}")
    print(f"Embedding data type: {hf_embeddings.dtype}")
    print(f"Embedding memory usage: {hf_embeddings.nbytes / 1024:.2f} KB")
    
    # 4. Test embedding quality with semantic similarity
    print("\n4. Semantic similarity test:")
    test_pairs = [
        ("Machine learning is powerful", "Deep learning uses neural networks"),
        ("Machine learning is powerful", "The weather is nice today"),
        ("Natural language processing", "Computer vision systems")
    ]
    
    for text1, text2 in test_pairs:
        emb1 = hf_embedder.embed(text1)
        emb2 = hf_embedder.embed(text2)
        similarity = np.dot(emb1, emb2.T)[0][0]
        print(f"'{text1}' vs '{text2}': {similarity:.4f}")
    
    return {
        'documents': documents,
        'faiss_db': faiss_db,
        'chroma_db': chroma_db,
        'embedder': hf_embedder
    }

exercise_results = document_embedding_exercise()
```

## 📌 Project: Semantic search system

```python
class SemanticSearchSystem:
    """Complete semantic search system using vector databases and embeddings"""
    
    def __init__(self, vector_db_type="chroma", embedding_model="huggingface", **kwargs):
        self.vector_db_type = vector_db_type
        self.embedding_model = embedding_model
        
        # Initialize embedder
        if embedding_model == "huggingface":
            self.embedder = TextEmbedder()
            embedding_dimension = 384
        elif embedding_model == "openai":
            if 'api_key' not in kwargs:
                raise ValueError("OpenAI API key required for OpenAI embeddings")
            self.embedder = OpenAIEmbedder(kwargs['api_key'])
            embedding_dimension = 1536
        else:
            raise ValueError(f"Unsupported embedding model: {embedding_model}")
        
        # Initialize vector database
        if vector_db_type == "faiss":
            index_type = kwargs.get("index_type", "Flat")
            self.vector_db = FaissVectorDB(dimension=embedding_dimension, index_type=index_type)
        elif vector_db_type == "chroma":
            collection_name = kwargs.get("collection_name", "semantic_search")
            persist_directory = kwargs.get("persist_directory", None)
            self.vector_db = ChromaVectorDB(
                collection_name=collection_name,
                persist_directory=persist_directory
            )
        else:
            raise ValueError(f"Unsupported vector DB type: {vector_db_type}")
        
        self.documents = []
        self.is_initialized = False
    
    def add_documents(self, documents, metadatas=None, chunk_size=None):
        """Add documents to the semantic search system"""
        if chunk_size:
            # Chunk large documents
            processed_docs = []
            processed_metadata = []
            
            for i, doc in enumerate(documents):
                if len(doc.split()) > chunk_size:
                    # Simple sentence-based chunking
                    sentences = doc.split('. ')
                    chunks = []
                    current_chunk = []
                    current_length = 0
                    
                    for sentence in sentences:
                        sentence_length = len(sentence.split())
                        if current_length + sentence_length > chunk_size:
                            if current_chunk:
                                chunks.append('. '.join(current_chunk) + '.')
                                current_chunk = [sentence]
                                current_length = sentence_length
                            else:
                                chunks.append(sentence + '.')
                        else:
                            current_chunk.append(sentence)
                            current_length += sentence_length
                    
                    if current_chunk:
                        chunks.append('. '.join(current_chunk) + '.')
                    
                    processed_docs.extend(chunks)
                    if metadatas:
                        chunk_metadata = [metadatas[i].copy() for _ in chunks]
                        # Add chunk info to metadata
                        for j, meta in enumerate(chunk_metadata):
                            meta['chunk_id'] = j
                            meta['original_doc_id'] = i
                        processed_metadata.extend(chunk_metadata)
                else:
                    processed_docs.append(doc)
                    if metadatas:
                        meta = metadatas[i].copy()
                        meta['chunk_id'] = 0
                        meta['original_doc_id'] = i
                        processed_metadata.append(meta)
            
            documents = processed_docs
            metadatas = processed_metadata if metadatas else None
        
        # Add to vector database
        if self.vector_db_type == "faiss":
            embeddings = self.embedder.embed(documents)
            self.vector_db.add(embeddings, documents)
        else:  # chroma
            self.vector_db.add_documents(documents, metadatas)
        
        self.documents.extend(documents)
        self.is_initialized = True
        
        return len(documents)
    
    def search(self, query, k=5, filters=None):
        """Perform semantic search"""
        if not self.is_initialized:
            raise ValueError("No documents added to the search system")
        
        if self.vector_db_type == "faiss":
            if filters:
                print("Warning: FAISS doesn't support metadata filtering")
            query_embedding = self.embedder.embed(query)
            results = self.vector_db.search(query_embedding, k=k)
        else:  # chroma
            results = self.vector_db.search(query, n_results=k, where=filters)
        
        return results
    
    def get_similar_documents(self, document_id, k=5):
        """Find documents similar to a given document"""
        if not self.is_initialized:
            raise ValueError("No documents added to the search system")
        
        if document_id >= len(self.documents):
            raise ValueError(f"Document ID {document_id} not found")
        
        query_document = self.documents[document_id]
        return self.search(query_document, k=k+1)[1:]  # Exclude the document itself
    
    def search_and_rerank(self, query, k=5, rerank_k=10):
        """Search and rerank results using cross-encoder"""
        # This is a simplified version - in practice, you'd use a cross-encoder model
        initial_results = self.search(query, k=rerank_k)
        
        # Simple reranking based on query-document similarity
        # In practice, use a cross-encoder like BERT for better reranking
        query_embedding = self.embedder.embed(query)
        
        reranked_results = []
        for result in initial_results:
            if self.vector_db_type == "faiss":
                doc_embedding = self.embedder.embed(result['text'])
                similarity = np.dot(query_embedding, doc_embedding.T)[0][0]
                result['rerank_score'] = float(similarity)
            else:  # chroma
                # Use distance as similarity score
                result['rerank_score'] = 1.0 - result['distance']
            reranked_results.append(result)
        
        # Sort by rerank score
        reranked_results.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        return reranked_results[:k]
    
    def get_collection_info(self):
        """Get information about the collection"""
        if self.vector_db_type == "faiss":
            return {
                'type': 'FAISS',
                'document_count': len(self.vector_db.ids),
                'embedding_model': self.embedding_model,
                'index_type': self.vector_db.index_type
            }
        else:  # chroma
            return self.vector_db.get_collection_info()

# Demonstrate the semantic search system
def demonstrate_semantic_search():
    """Demonstrate the complete semantic search system"""
    
    # Sample knowledge base
    knowledge_base = [
        "Artificial intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems.",
        "Machine learning is a subset of AI that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.",
        "Deep learning is a subset of machine learning that uses neural networks with three or more layers to simulate the behavior of the human brain.",
        "Natural language processing (NLP) is a branch of AI that helps computers understand, interpret, and manipulate human language.",
        "Computer vision is a field of AI that enables computers to derive meaningful information from digital images, videos, and other visual inputs.",
        "Reinforcement learning is an area of machine learning concerned with how intelligent agents ought to take actions in an environment to maximize cumulative reward.",
        "Data science is an interdisciplinary field that uses scientific methods, processes, algorithms, and systems to extract knowledge and insights from structured and unstructured data.",
        "Neural networks are computing systems vaguely inspired by the biological neural networks that constitute animal brains."
    ]
    
    # Create semantic search system
    search_system = SemanticSearchSystem(
        vector_db_type="chroma",
        embedding_model="huggingface",
        collection_name="ai_knowledge_base",
        persist_directory="./semantic_search_db"
    )
    
    # Add documents
    num_docs = search_system.add_documents(knowledge_base)
    print(f"Added {num_docs} documents to semantic search system")
    
    # Get system info
    print(f"System info: {search_system.get_collection_info()}")
    
    # Test basic search
    queries = [
        "What is machine learning?",
        "How does deep learning work?",
        "What is natural language processing?"
    ]
    
    print("\nSemantic Search Results:")
    print("=" * 60)
    
    for query in queries:
        print(f"\nQuery: {query}")
        results = search_system.search(query, k=2)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. Similarity: {result.get('similarity', 1-result.get('distance', 0)):.4f}")
            print(f"     Text: {result['text'][:120]}...")
            if 'metadata' in result:
                print(f"     Metadata: {result['metadata']}")
    
    # Test reranking
    print(f"\nReranked Results for: 'What is AI?'")
    reranked_results = search_system.search_and_rerank("What is AI?", k=2)
    for i, result in enumerate(reranked_results, 1):
        print(f"  {i}. Rerank Score: {result['rerank_score']:.4f} - {result['text'][:80]}...")
    
    # Test similar documents
    print(f"\nDocuments similar to document 0:")
    similar_docs = search_system.get_similar_documents(0, k=2)
    for i, result in enumerate(similar_docs, 1):
        print(f"  {i}. {result['text'][:80]}...")
    
    return search_system

# Run the semantic search system demonstration
semantic_search_system = demonstrate_semantic_search()

# Advanced: Web interface for semantic search (Streamlit-style pseudo code)
def create_semantic_search_app():
    """Create a Streamlit-like interface for semantic search"""
    
    app_code = '''
import streamlit as st
from semantic_search_system import SemanticSearchSystem
import os

# Initialize the search system
@st.cache_resource
def get_search_system():
    if os.path.exists("./semantic_search_db"):
        # Load existing database
        system = SemanticSearchSystem(
            vector_db_type="chroma",
            embedding_model="huggingface",
            collection_name="ai_knowledge_base",
            persist_directory="./semantic_search_db"
        )
    else:
        # Create new system
        system = SemanticSearchSystem(
            vector_db_type="chroma",
            embedding_model="huggingface",
            collection_name="ai_knowledge_base",
            persist_directory="./semantic_search_db"
        )
    return system

search_system = get_search_system()

st.title("🔍 Semantic Search System")
st.markdown("Search through your knowledge base using semantic similarity!")

# Document upload
st.header("Add Documents")
uploaded_docs = st.text_area("Enter documents (one per line):", height=200)
if st.button("Add Documents"):
    if uploaded_docs:
        docs = [doc.strip() for doc in uploaded_docs.split('\\n') if doc.strip()]
        num_added = search_system.add_documents(docs)
        st.success(f"Added {num_added} documents!")

# Search interface
st.header("Search")
query = st.text_input("Enter your search query:")
k = st.slider("Number of results:", 1, 10, 5)

if st.button("Search") and query:
    results = search_system.search(query, k=k)
    
    st.subheader(f"Results for: {query}")
    for i, result in enumerate(results, 1):
        similarity = result.get('similarity', 1 - result.get('distance', 0))
        st.markdown(f"**Result {i} (Similarity: {similarity:.4f})**")
        st.write(result['text'])
        st.markdown("---")

# System info
st.sidebar.header("System Info")
info = search_system.get_collection_info()
st.sidebar.json(info)
'''
    
    with open('semantic_search_app.py', 'w') as f:
        f.write(app_code)
    
    print("Semantic search Streamlit app created as 'semantic_search_app.py'")

create_semantic_search_app()
```