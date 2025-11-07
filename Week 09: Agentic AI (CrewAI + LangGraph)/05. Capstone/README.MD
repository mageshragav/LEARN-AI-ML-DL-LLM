Here is the **complete, runnable project code** for an **Agent System for Document Summarization & Q&A** using CrewAI, LangChain, and Ollama, structured exactly like your `README.MD` format.

---

### 🏗️ Project Structure

```
document_agents/
├── docs/                    # ← Place your PDFs/TXTs here
├── agent_system.py          # ← Main CrewAI agents
├── tools.py                 # ← Custom tools for agents
├── config.py                # ← Agent configurations
├── requirements.txt
└── README.md
```

---

### 📄 `requirements.txt`

```txt
crewai==0.28.8
langchain==0.1.16
langchain-community==0.0.34
faiss-cpu==1.7.4
sentence-transformers==2.2.2
PyPDF2==3.0.1
python-dotenv
ollama
pandas
numpy
scikit-learn
tqdm
```

> 💡 **Prerequisite**: Install [Ollama](https://ollama.com/) and run:  
> ```bash
> ollama pull llama3
> # or: ollama pull mistral, phi3, etc.
> ```

---

### 🧠 `config.py`

```python
# config.py
from langchain_community.chat_models import ChatOllama

# Initialize LLM
llm = ChatOllama(model="llama3", temperature=0.0)

# Agent configurations
AGENT_CONFIGS = {
    "researcher": {
        "role": "Senior Research Analyst",
        "goal": "Extract key information and summarize documents",
        "backstory": "You're a meticulous researcher who identifies the most important points in documents.",
    },
    "summarizer": {
        "role": "Content Summarizer",
        "goal": "Create concise, accurate summaries of research findings",
        "backstory": "You specialize in distilling complex information into clear, brief summaries.",
    },
    "qa_agent": {
        "role": "Document Q&A Specialist",
        "goal": "Answer user questions based on document content",
        "backstory": "You provide precise, citation-backed answers using retrieved document context.",
    }
}
```

---

### 🛠️ `tools.py`

```python
# tools.py
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
import PyPDF2
import os

class DocumentLoaderTool(BaseTool):
    """Tool to load and extract text from documents."""
    
    name = "document_loader"
    description = "Loads PDF or text files and extracts content"
    
    def _run(self, file_path: str) -> str:
        """Load document and return text."""
        try:
            if file_path.endswith('.pdf'):
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text()
                    return text
            elif file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                return "Unsupported file format"
        except Exception as e:
            return f"Error loading document: {str(e)}"

    async def _arun(self, file_path: str) -> str:
        """Async version of _run."""
        return self._run(file_path)

class DocumentSearchTool(BaseTool):
    """Tool to search within loaded document content."""
    
    name = "document_search"
    description = "Searches for specific terms within document text"
    
    def __init__(self, document_text: str):
        super().__init__()
        self.document_text = document_text
    
    def _run(self, query: str) -> str:
        """Search for query in document text."""
        query_lower = query.lower()
        sentences = self.document_text.split('.')
        results = []
        for sentence in sentences:
            if query_lower in sentence.lower():
                results.append(sentence.strip())
        return ". ".join(results[:5])  # Return top 5 matches
    
    async def _arun(self, query: str) -> str:
        """Async version of _run."""
        return self._run(query)

# Initialize tools
document_loader = DocumentLoaderTool()
```

---

### 🤖 `agent_system.py`

```python
# agent_system.py
import os
from crewai import Agent, Task, Crew, Process
from config import AGENT_CONFIGS, llm
from tools import document_loader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class DocumentAgentSystem:
    def __init__(self, docs_dir: str = "./docs"):
        self.docs_dir = docs_dir
        self.documents = self._load_documents()
        self.chunks = self._split_documents()
        self.vectorstore = self._create_vectorstore()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        
        # Initialize agents
        self.researcher = self._create_agent("researcher")
        self.summarizer = self._create_agent("summarizer")
        self.qa_agent = self._create_agent("qa_agent")
        
        # Create tasks
        self.research_task = self._create_research_task()
        self.summarize_task = self._create_summarize_task()
        self.qa_task = self._create_qa_task()
        
        # Create crew
        self.crew = Crew(
            agents=[self.researcher, self.summarizer, self.qa_agent],
            tasks=[self.research_task, self.summarize_task, self.qa_task],
            process=Process.sequential,
            memory=True,
            verbose=2
        )

    def _load_documents(self):
        """Load all documents from directory."""
        if not os.path.exists(self.docs_dir):
            os.makedirs(self.docs_dir)
            print(f"📁 Created '{self.docs_dir}' — add PDFs/TXTs!")
            return []
        
        docs = []
        for filename in os.listdir(self.docs_dir):
            filepath = os.path.join(self.docs_dir, filename)
            if filename.endswith(('.pdf', '.txt')):
                content = document_loader._run(filepath)
                docs.append({"source": filename, "content": content})
        print(f"✅ Loaded {len(docs)} documents")
        return docs

    def _split_documents(self):
        """Split documents into chunks."""
        if not self.documents:
            return []
        text = " ".join([doc["content"] for doc in self.documents])
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_text(text)
        print(f"✂️ Split into {len(chunks)} chunks")
        return chunks

    def _create_vectorstore(self):
        """Create FAISS vector store."""
        if not self.chunks:
            # Create empty vectorstore
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            return FAISS.from_texts([""], embeddings)
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return FAISS.from_texts(self.chunks, embeddings)

    def _create_agent(self, role: str):
        """Create an agent based on role."""
        config = AGENT_CONFIGS[role]
        return Agent(
            role=config["role"],
            goal=config["goal"],
            backstory=config["backstory"],
            verbose=True,
            allow_delegation=True,
            llm=llm
        )

    def _create_research_task(self):
        """Create research task."""
        return Task(
            description="Analyze documents to extract key information, facts, and insights.",
            expected_output="A comprehensive list of key points from the documents.",
            agent=self.researcher
        )

    def _create_summarize_task(self):
        """Create summarization task."""
        return Task(
            description="Create a concise summary of the research findings.",
            expected_output="A 200-300 word summary of the key points.",
            agent=self.summarizer,
            context=[self.research_task]
        )

    def _create_qa_task(self):
        """Create Q&A task."""
        return Task(
            description="Answer user questions based on document content. Use the retriever to find relevant context.",
            expected_output="A precise, citation-backed answer to the user's question.",
            agent=self.qa_agent
        )

    def run_research_and_summarize(self):
        """Run the research and summarization pipeline."""
        print("\n🔍 Starting Document Research & Summarization...")
        result = self.crew.kickoff()
        return result

    def answer_question(self, question: str):
        """Answer a specific question using the QA agent."""
        # Create a temporary QA crew for this question
        qa_crew = Crew(
            agents=[self.qa_agent],
            tasks=[
                Task(
                    description=f"Answer this question based on the documents: {question}",
                    expected_output="A precise, citation-backed answer.",
                    agent=self.qa_agent
                )
            ],
            verbose=1
        )
        result = qa_crew.kickoff()
        return result.raw

    def get_relevant_context(self, question: str):
        """Retrieve relevant document chunks for a question."""
        docs = self.retriever.invoke(question)
        return "\n\n".join([doc.page_content for doc in docs])

# CLI Interface
if __name__ == "__main__":
    system = DocumentAgentSystem()
    
    print("📚 Document Agent System Ready!")
    print("Commands: 'summarize', 'ask <question>', 'context <question>', 'quit'")
    
    while True:
        user_input = input("\n> ").strip()
        
        if user_input.lower() == "quit":
            break
        elif user_input.lower() == "summarize":
            result = system.run_research_and_summarize()
            print("\n📝 Summary:")
            print(result.raw)
        elif user_input.lower().startswith("ask "):
            question = user_input[4:]
            answer = system.answer_question(question)
            print(f"\n❓ Q: {question}")
            print(f"🤖 A: {answer}")
        elif user_input.lower().startswith("context "):
            question = user_input[8:]
            context = system.get_relevant_context(question)
            print(f"\n🔍 Retrieved Context:\n{context}")
        else:
            print("Invalid command. Try: 'summarize', 'ask <question>', 'context <question>', 'quit'")
```

---

### 🧪 How to Use

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Add your documents** to `./docs/` (PDF or TXT).

3. **Run the system**:
   ```bash
   python agent_system.py
   ```

4. **Use commands**:
   - `summarize` → Generate document summary
   - `ask What is the main topic?` → Q&A
   - `context How is X defined?` → Get relevant chunks
   - `quit` → Exit

---

### ✅ Features

- **CrewAI Agents**: Researcher, Summarizer, Q&A Specialist
- **Document Loading**: PDF + TXT support
- **Retrieval-Augmented Generation**: FAISS + HuggingFace embeddings
- **Ollama Integration**: Local LLMs (e.g., `llama3`)
- **CLI Interface**: Interactive command-line usage
- **Modular Design**: Easy to extend with new tools/agents

---

