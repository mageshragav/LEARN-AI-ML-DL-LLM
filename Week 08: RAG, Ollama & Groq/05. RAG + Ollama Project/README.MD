Here is a **complete, production-ready, end-to-end Hybrid QA System** using **RAG + Ollama**, including **document ingestion**, **retrieval**, **generation**, and **automatic evaluation**—fully structured for clarity and ready to run locally.

---

## 📁 Project Structure

```
rag-ollama-hybrid-qa/
├── docs/                     # Your PDF/TXT documents
├── data/
│   └── eval_questions.csv    # Evaluation questions + ground truth
├── rag_ollama_qa.py          # Main RAG + Ollama system
├── evaluate.py               # Automated evaluation
├── requirements.txt
└── README.md
```

---

## 📄 `requirements.txt`

```txt
langchain==0.1.16
langchain-community==0.0.34
langchain-core==0.1.40
faiss-cpu==1.7.4
sentence-transformers==2.2.2
PyPDF2==3.0.1
python-dotenv
ollama
pandas
numpy
scikit-learn
tqdm
matplotlib
```

> 💡 **Prerequisite**: Install [Ollama](https://ollama.com/) and run:  
> ```bash
> ollama pull llama3
> # or: ollama pull mistral, phi3, etc.
> ```

---

## 📄 `rag_ollama_qa.py`

```python
# rag_ollama_qa.py
import os
import warnings
from langchain.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatOllama

warnings.filterwarnings("ignore")

class RAGOllamaQA:
    def __init__(
        self,
        docs_dir: str = "./docs",
        model_name: str = "llama3",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k: int = 4
    ):
        self.docs_dir = docs_dir
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k = k

        # Step 1: Load & split documents
        self.documents = self._load_documents()
        self.chunks = self._split_documents()
        
        # Step 2: Embeddings + vector store
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        if self.chunks:
            self.vectorstore = FAISS.from_documents(self.chunks, self.embeddings)
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        else:
            self.retriever = None
        
        # Step 3: LLM
        self.llm = ChatOllama(model=model_name, temperature=0.0)
        
        # Step 4: RAG chain
        self.prompt = self._create_prompt()
        if self.retriever:
            self.rag_chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
        else:
            self.rag_chain = None

    def _load_documents(self):
        if not os.path.exists(self.docs_dir):
            os.makedirs(self.docs_dir)
            print(f"📁 Created '{self.docs_dir}' — add PDFs/TXTs!")
            return []
        
        loaders = []
        for ext, cls in [(".pdf", PyPDFLoader), (".txt", TextLoader)]:
            try:
                loader = DirectoryLoader(self.docs_dir, glob=f"**/*{ext}", loader_cls=cls)
                docs = loader.load()
                if docs:
                    print(f"✅ Loaded {len(docs)} {ext} files")
                    loaders.extend(docs)
            except Exception as e:
                print(f"⚠️ Error loading {ext}: {e}")
        return loaders

    def _split_documents(self):
        if not self.documents:
            return []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(self.documents)
        print(f"✂️ Split into {len(chunks)} chunks")
        return chunks

    def _create_prompt(self):
        template = """
        Answer the question using ONLY the provided context.
        If the answer is not in the context, say: "I don't know."

        Context:
        {context}

        Question: {question}

        Answer:
        """
        return ChatPromptTemplate.from_template(template)

    def ask(self, question: str) -> str:
        if not self.rag_chain:
            return "❌ No documents loaded. Add files to 'docs/' folder."
        return self.rag_chain.invoke(question)

    def get_retrieved_context(self, question: str) -> str:
        """For evaluation: return raw retrieved context"""
        if not self.retriever:
            return ""
        docs = self.retriever.invoke(question)
        return "\n\n".join([doc.page_content for doc in docs])
```

---

## 📄 `evaluate.py`

```python
# evaluate.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
import warnings
warnings.filterwarnings("ignore")

try:
    from bert_score import score as bert_score
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("⚠️ bert-score not installed. Install with: pip install bert-score")

from rag_ollama_qa import RAGOllamaQA

class HybridQAEvaluator:
    def __init__(self, rag_qa_system: RAGOllamaQA, eval_csv: str = "./data/eval_questions.csv"):
        self.rag_qa = rag_qa_system
        self.eval_df = pd.read_csv(eval_csv)
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    def text_similarity(self, pred: str, ref: str) -> float:
        if not pred.strip() or not ref.strip():
            return 0.0
        vectorizer = TfidfVectorizer().fit([pred, ref])
        tfidf = vectorizer.transform([pred, ref])
        return cosine_similarity(tfidf[0], tfidf[1])[0][0]

    def evaluate(self):
        results = []
        for _, row in self.eval_df.iterrows():
            question = row["question"]
            ground_truth = str(row["ground_truth"])

            # Get prediction and context
            prediction = self.rag_qa.ask(question)
            context = self.rag_qa.get_retrieved_context(question)

            # Metrics
            tfidf_sim = self.text_similarity(prediction, ground_truth)
            rouge_scores = self.scorer.score(ground_truth, prediction)
            rouge1 = rouge_scores['rouge1'].fmeasure
            rougeL = rouge_scores['rougeL'].fmeasure

            if BERT_AVAILABLE:
                P, R, F1 = bert_score([prediction], [ground_truth], lang="en", verbose=False)
                bert_f1 = F1.item()
            else:
                bert_f1 = None

            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "retrieved_context": context,
                "tfidf_similarity": tfidf_sim,
                "rouge1": rouge1,
                "rougeL": rougeL,
                "bert_f1": bert_f1
            })

        self.results_df = pd.DataFrame(results)
        return self.results_df

    def print_summary(self):
        print("\n📊 Evaluation Summary:")
        print(f"TF-IDF Similarity: {self.results_df['tfidf_similarity'].mean():.4f}")
        print(f"ROUGE-1 F1:        {self.results_df['rouge1'].mean():.4f}")
        print(f"ROUGE-L F1:        {self.results_df['rougeL'].mean():.4f}")
        if BERT_AVAILABLE:
            print(f"BERTScore F1:      {self.results_df['bert_f1'].mean():.4f}")

    def save_results(self, path: str = "evaluation_results.csv"):
        self.results_df.to_csv(path, index=False)
        print(f"✅ Results saved to {path}")

# Example usage
if __name__ == "__main__":
    # Create evaluation data if not exists
    eval_path = "./data/eval_questions.csv"
    os.makedirs("./data", exist_ok=True)
    if not os.path.exists(eval_path):
        sample_questions = pd.DataFrame({
            "question": ["What is the capital of France?", "Explain photosynthesis."],
            "ground_truth": ["Paris", "Photosynthesis is the process by which green plants convert light energy into chemical energy."]
        })
        sample_questions.to_csv(eval_path, index=False)
        print(f"📝 Created sample evaluation file at {eval_path}")

    # Initialize system
    qa_system = RAGOllamaQA(model_name="llama3")

    # Evaluate
    evaluator = HybridQAEvaluator(qa_system, eval_path)
    results = evaluator.evaluate()
    evaluator.print_summary()
    evaluator.save_results()
```

---

## 📄 Sample `data/eval_questions.csv`

```csv
question,ground_truth
What is the main topic of the document?,"The document discusses climate change impacts."
How many employees are in the company?,150
What is the refund policy duration?,30 days
```

> 💡 Replace with your own domain-specific questions and ground truth answers.

---

## ▶️ How to Run

### 1. **Install dependencies**
```bash
pip install -r requirements.txt
```

### 2. **Add your documents**
Place PDFs or TXT files in `./docs/`

### 3. **Run the QA system (interactive)**
```python
from rag_ollama_qa import RAGOllamaQA

bot = RAGOllamaQA(model_name="llama3")
print(bot.ask("Summarize the key points in the documents."))
```

### 4. **Run evaluation**
```bash
python evaluate.py
```

---

## ✅ Features

- **Local & Private**: No data leaves your machine
- **Hybrid Architecture**: Retrieval + Generation
- **Multiple Evaluation Metrics**: TF-IDF, ROUGE, BERTScore
- **Flexible LLM**: Swap `llama3` ↔ `mistral` ↔ `phi3`
- **Extensible**: Add your own metrics or models

---
