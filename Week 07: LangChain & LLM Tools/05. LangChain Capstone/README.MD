# Day 35 – LangChain Capstone  
## 📌 Project: RAG-based Chatbot Over Documents

---

## What is a RAG-based Chatbot?

**Retrieval-Augmented Generation (RAG)** is a powerful architecture that combines **information retrieval** with **generative language modeling**. Instead of relying solely on the static knowledge embedded in an LLM, RAG dynamically retrieves relevant external documents and uses them as context to generate more accurate, up-to-date, and domain-specific responses.

A **RAG-based chatbot over documents**:
- Accepts user queries
- Searches a **document corpus** (PDFs, text files, web pages, etc.)
- Retrieves **top-k relevant chunks**
- Feeds them to an LLM **with the user query**
- Generates **context-aware, grounded answers**
- Optionally includes **memory** for multi-turn conversations

This approach is critical for:
- Avoiding hallucinations
- Handling domain-specific or private data
- Providing source-backed answers
- Building AI agents that reason over real documents

---

## Essential Subtopics

### 1. Document Loading & Preprocessing

Documents must be parsed, cleaned, and split into manageable chunks.

```python
from langchain.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents_from_directory(dir_path: str, glob_pattern="**/*.pdf"):
    """Load all PDFs (or other formats) from a directory."""
    loader = DirectoryLoader(
        dir_path,
        glob=glob_pattern,
        loader_cls=PyPDFLoader  # or TextLoader for .txt
    )
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} documents")
    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into overlapping chunks for better context retention."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks")
    return chunks
```

---

### 2. Embedding & Vector Store Creation

Convert text chunks into dense vectors for semantic search.

```python
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def create_vectorstore(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Create FAISS vector store from document chunks."""
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("✅ Vector store created using FAISS + HuggingFace embeddings")
    return vectorstore, embeddings
```

> ✅ **Why FAISS?** Fast, in-memory, efficient for small-to-medium datasets  
> ✅ **Why HuggingFace?** No API key needed; open-source; runs locally

---

### 3. Retriever Setup

Configure how the system fetches relevant context.

```python
def create_retriever(vectorstore, k=4, score_threshold=0.4):
    """Create a similarity retriever with optional filtering."""
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": score_threshold}
    )
    return retriever
```

---

### 4. LLM Integration with Prompt Engineering

Use a prompt template that **forces citation** and **relevance**.

```python
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

def create_rag_prompt():
    template = """
    You are a helpful and precise assistant. Use ONLY the following retrieved context to answer the question.
    If the context doesn’t contain the answer, say "I don't know based on the provided documents."

    Context:
    {context}

    Question: {question}

    Answer:
    """
    return ChatPromptTemplate.from_template(template)

def create_llm(model="gpt-3.5-turbo", temperature=0.0):
    """Initialize LLM (can be swapped for open-source models like Llama 3 via Ollama)."""
    return ChatOpenAI(model=model, temperature=temperature)
```

---

### 5. RAG Chain Assembly

Combine retrieval, prompt, and LLM into a single pipeline.

```python
from langchain_core.runnables import RunnablePassthrough

def create_rag_chain(retriever, llm, prompt):
    """Build end-to-end RAG chain."""
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain
```

---

### 6. (Optional) Add Memory for Multi-Turn Chat

Enable conversation history using `ConversationalRetrievalChain`.

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

def create_conversational_rag_chain(llm, retriever):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True
    )
```

---

## When and Where We Use RAG-based Chatbots

| Scenario | Use Case | Why RAG? |
|--------|--------|--------|
| **Enterprise Knowledge Bases** | HR policies, internal docs | Avoid hallucination; ground answers in real data |
| **Legal & Compliance** | Contract analysis, regulation lookup | Precision + citation required |
| **Customer Support** | Product manuals, FAQs | Faster resolution with accurate info |
| **Research Assistance** | Scientific paper Q&A | Retrieve evidence from PDFs |
| **Education** | Textbook tutoring | Explain concepts with source context |

> 🔸 **Use RAG when**:  
> - Data is **private**, **dynamic**, or **domain-specific**  
> - You need **traceability** and **fact grounding**  
> - Fine-tuning LLMs is **impractical** or **costly**

---

## Real-World Applications and Examples

### Example 1: **Company Policy Assistant**
```python
# Load HR policy PDFs
docs = load_documents_from_directory("./docs/hr_policies/")
chunks = split_documents(docs)
vectorstore, _ = create_vectorstore(chunks)
retriever = create_retriever(vectorstore, k=3)

rag_chain = create_rag_chain(
    retriever,
    create_llm(),
    create_rag_prompt()
)

response = rag_chain.invoke("How many vacation days do employees get?")
print("🤖", response)
# Output: "Employees receive 15 paid vacation days per year as per Section 4.2 of the HR Policy Handbook."
```

### Example 2: **Scientific Paper Q&A**
```python
# Load research papers (PDFs)
papers = load_documents_from_directory("./papers/", glob_pattern="*.pdf")
chunks = split_documents(papers, chunk_size=500)  # smaller for technical content
vectorstore, _ = create_vectorstore(chunks)

# Use a more powerful model for reasoning
llm = create_llm(model="gpt-4-turbo", temperature=0.1)
retriever = create_retriever(vectorstore, k=5, score_threshold=0.3)

rag_chain = create_rag_chain(retriever, llm, create_rag_prompt())

answer = rag_chain.invoke("What dataset was used in the transformer paper?")
print("🔬", answer)
```

---

## Common Problems and How Advancements Solved Them

### ❌ Problem 1: **Poor Retrieval → Irrelevant Context → Garbage Output**

**Cause**: Chunking too large or too small; embeddings not semantic enough.

**✅ Advancement**:  
- **Parent-Child Chunking**: Store small chunks for retrieval, but retrieve full parent section for context.  
- **Re-Ranking**: Use cross-encoders (e.g., Cohere Rerank, `bge-reranker`) to re-score retrieved chunks.

**Enhanced Code**:
```python
# Parent-child splitting
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

def create_parent_child_retriever(documents):
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
    vectorstore = FAISS.from_texts([""], HuggingFaceEmbeddings())  # init empty
    store = InMemoryStore()
    
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    retriever.add_documents(documents)
    return retriever
```

---

### ❌ Problem 2: **LLM Ignores Context or Hallucinates**

**Cause**: Weak prompt; LLM overconfident.

**✅ Advancement**:  
- **Stronger prompt engineering** with explicit instructions  
- **Output parsing + validation**  
- **Citation requirement**

**Enhanced Prompt**:
```python
template = """
Answer the question using ONLY the provided context.
Cite the source by including [Document: filename, Page: X] if available.
If the answer is not in the context, say: "I cannot answer this based on the provided documents."

Context:
{context}

Question: {question}

Answer with citation:
"""
```

---

### ❌ Problem 3: **No Memory in Multi-Turn Conversations**

**Cause**: Stateless RAG chain.

**✅ Advancement**: Use **ConversationalRetrievalChain** or **custom stateful agents**.

**Working Code**:
```python
# Enable memory
chatbot = create_conversational_rag_chain(
    llm=create_llm(),
    retriever=create_retriever(vectorstore)
)

# Multi-turn interaction
response1 = chatbot({"question": "What is the refund policy?"})
response2 = chatbot({"question": "How long does it take to process?"})  # uses prior context
print("💬", response2['answer'])
```

---

### ❌ Problem 4: **Slow or Costly Inference**

**✅ Advancement**:  
- Swap OpenAI for **local LLMs** (Llama 3, Mistral) via **Ollama** or **vLLM**  
- Use **GPU-accelerated embeddings** (e.g., `sentence-transformers` with CUDA)

**Local LLM Integration**:
```python
from langchain_community.chat_models import ChatOllama

def create_local_llm(model="llama3"):
    return ChatOllama(model=model, temperature=0.0)

# Replace OpenAI with local model
llm = create_local_llm("mistral")  # or "llama3:8b"
```

---

## Full End-to-End Capstone: RAG Chatbot

```python
import os
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Load & split
loader = DirectoryLoader("./docs/", glob="**/*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# 2. Embed & store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

# 3. Build RAG chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
prompt = ChatPromptTemplate.from_template("""
Use only the following context to answer the question. If unsure, say "I don't know."

Context: {context}
Question: {question}
Answer:
""")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 4. Chat loop
print("🚀 RAG Chatbot Ready! Type 'quit' to exit.")
while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["quit", "exit"]:
        break
    response = rag_chain.invoke(user_input)
    print("🤖 Bot:", response)
```

---

## Bonus: Deploy with Streamlit (Optional Web UI)

```python
# streamlit_app.py
import streamlit as st
from your_rag_module import rag_chain  # from above

st.title("📚 Document Q&A Chatbot")
st.caption("Powered by RAG + LangChain")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.spinner("Thinking..."):
        response = rag_chain.invoke(prompt)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
```

Run with: `streamlit run streamlit_app.py`

---

Here is the **complete, production-ready RAG-based chatbot over documents** using LangChain, PyTorch-compatible embeddings, FAISS, and optional memory — structured exactly like your `README.MD` and ready to run locally.

---

### ✅ **Full Project Code: RAG-Based Chatbot Over Documents**

📁 **File structure:**
```
rag_chatbot/
├── docs/                     # ← Place your PDFs/TXTs here
├── rag_chatbot.py            # ← Main script
├── requirements.txt
└── .env                      # ← For OpenAI API key (optional)
```

---

### 📄 `requirements.txt`
```txt
langchain==0.1.16
langchain-community==0.0.34
langchain-core==0.1.40
faiss-cpu==1.7.4
sentence-transformers==2.2.2
torch==2.2.2
transformers==4.38.2
PyPDF2==3.0.1
python-dotenv
openai
streamlit  # optional for web UI
```

---

### 🔐 `.env` (Optional)
```env
OPENAI_API_KEY=your-api-key-here
```

---

### 🧠 `rag_chatbot.py`

```python
# rag_chatbot.py
import os
import warnings
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from typing import List, Optional

# Optional: Use OpenAI or fall back to local LLMs
try:
    from langchain.chat_models import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Load environment
load_dotenv()
warnings.filterwarnings("ignore")

class RAGChatbot:
    def __init__(
        self,
        docs_dir: str = "./docs",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k_retrieved: int = 4,
        use_memory: bool = True,
        use_openai: bool = False
    ):
        self.docs_dir = docs_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k_retrieved = k_retrieved
        self.use_memory = use_memory
        self.use_openai = use_openai and OPENAI_AVAILABLE

        # 1. Load & split documents
        self.documents = self._load_documents()
        self.chunks = self._split_documents()
        
        # 2. Create embeddings + vectorstore
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vectorstore = FAISS.from_documents(self.chunks, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": k_retrieved})
        
        # 3. Initialize memory and LLM
        if self.use_memory:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        self.llm = self._get_llm()
        
        # 4. Build chain
        if use_memory and self.use_openai:
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever,
                memory=self.memory,
                verbose=False
            )
        else:
            self.prompt = self._create_prompt()
            self.chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
            )

    def _load_documents(self) -> List:
        if not os.path.exists(self.docs_dir):
            os.makedirs(self.docs_dir)
            print(f"📁 Created '{self.docs_dir}' — please add PDF/TXT files!")
            return []
        
        loaders = []
        for ext, loader_cls in [(".pdf", PyPDFLoader), (".txt", TextLoader)]:
            try:
                loader = DirectoryLoader(self.docs_dir, glob=f"**/*{ext}", loader_cls=loader_cls)
                loaders.append(loader.load())
            except Exception as e:
                print(f"⚠️  No {ext} files or error: {e}")
        
        docs = [doc for sublist in loaders for doc in sublist] if loaders else []
        print(f"✅ Loaded {len(docs)} documents from '{self.docs_dir}'")
        return docs

    def _split_documents(self):
        if not self.documents:
            return []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(self.documents)
        print(f"✂️  Split into {len(chunks)} chunks")
        return chunks

    def _get_llm(self):
        if self.use_openai:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in .env")
            return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
        else:
            try:
                from langchain_community.chat_models import ChatOllama
                print("🧠 Using local LLM (Ollama: llama3) — make sure it's installed!")
                return ChatOllama(model="llama3", temperature=0.0)
            except ImportError:
                print("⚠️  Ollama not installed. Using mock response.")
                return MockLLM()

    def _create_prompt(self):
        template = """
        You are a precise assistant. Answer the question using ONLY the following context.
        If the answer isn't in the context, say: "I don't know based on the provided documents."

        Context:
        {context}

        Question: {question}

        Answer:
        """
        return ChatPromptTemplate.from_template(template)

    def ask(self, question: str) -> str:
        if not self.documents:
            return "❌ No documents loaded. Please add files to the 'docs/' folder."
        
        try:
            if self.use_memory and self.use_openai:
                response = self.chain({"question": question})
                return response["answer"]
            else:
                return self.chain.invoke(question)
        except Exception as e:
            return f"⚠️ Error: {str(e)}"

# Mock LLM fallback (for testing without Ollama/OpenAI)
class MockLLM:
    def invoke(self, prompt):
        return "🔍 This is a mock response. Install Ollama (`ollama run llama3`) or set OPENAI_API_KEY for real answers."

# CLI Chat Loop
if __name__ == "__main__":
    print("🦙 RAG Chatbot — Ask questions about your documents!\n")

    # Initialize chatbot
    bot = RAGChatbot(
        docs_dir="./docs",
        use_openai=False,      # Set True if you have OpenAI key
        use_memory=True
    )

    print("\n💡 Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("👋 Goodbye!")
            break
        response = bot.ask(user_input)
        print(f"🤖 Bot: {response}\n")
```

---

### 🌐 Optional: Web UI with Streamlit (`streamlit_app.py`)

```python
# streamlit_app.py
import streamlit as st
from rag_chatbot import RAGChatbot

st.set_page_config(page_title="📚 RAG Document Chatbot", layout="wide")
st.title("📚 RAG Document Chatbot")
st.caption("Ask questions about your PDFs & text files")

# Initialize chatbot (run once)
@st.cache_resource
def get_chatbot():
    return RAGChatbot(use_openai=False, use_memory=True)

bot = get_chatbot()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = bot.ask(prompt)
        st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
```

Run with:  
```bash
streamlit run streamlit_app.py
```

---

### 🧪 How to Use

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Add your documents** to the `docs/` folder (PDFs or TXTs).

3. **Run the chatbot**:
   ```bash
   python rag_chatbot.py
   ```

4. **(Optional)** For OpenAI:
   - Add your API key to `.env`
   - Set `use_openai=True`

5. **(Optional)** For local LLMs:
   - Install [Ollama](https://ollama.com/)
   - Run: `ollama run llama3`

---
