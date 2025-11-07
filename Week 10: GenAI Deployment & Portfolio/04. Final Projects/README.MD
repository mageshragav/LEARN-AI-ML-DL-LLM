Excellent! Let's tackle **Day 49 – Final Projects**, where we’ll build two powerful AI applications:

1. **Invoice Q&A Bot** – Extract information from invoices and answer user questions.
2. **Resume Screener with LangChain** – Screen resumes and match them with job descriptions.

For each project, I'll provide:
- 🧠 Concept explanation
- 🛠️ Essential subtopics and code
- 🌍 Real-world applications
- ⚠️ Common problems and advancements
- 📌 Full project structure and code samples

---

## 📌 Project 1: Invoice Q&A Bot

### 🧠 What is an Invoice Q&A Bot?

An **Invoice Q&A Bot** uses **Natural Language Processing (NLP)** and **document understanding** to extract structured data (e.g., invoice number, date, total amount) from unstructured or semi-structured invoice documents (PDFs, images, etc.) and allows users to ask questions about the content.

### 🛠️ Essential Subtopics

#### 1. **Document Parsing**
Use libraries like `PyMuPDF`, `pdfplumber`, or `PyPDF2` to extract text from PDF invoices.

#### 2. **Entity Recognition**
Use NLP models (e.g., spaCy, Hugging Face Transformers) to identify key fields like:
- Invoice number
- Date
- Total amount
- Vendor name

#### 3. **Question Answering (QA)**
Use a pre-trained QA model (e.g., from Hugging Face) to answer user questions based on invoice content.

---

### 🧪 Code Example: Invoice Q&A Bot

#### Step 1: Extract Text from Invoice

```python
import pdfplumber

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

invoice_text = extract_text_from_pdf("invoice.pdf")
```

#### Step 2: Load QA Model

```python
from transformers import pipeline

qa_pipeline = pipeline("question-answering")

def ask_question(question, context):
    result = qa_pipeline(question=question, context=context)
    return result['answer']

user_question = "What is the total amount?"
answer = ask_question(user_question, invoice_text)
print("Answer:", answer)
```

---

### 🌍 Real-World Applications

- **Finance automation**: Automatically extract and verify invoice data.
- **Customer support bots**: Allow users to ask questions about their invoices.
- **Audit and compliance**: Quickly retrieve specific invoice data for audits.

---

### ⚠️ Common Problems and Advancements

#### Problem 1: **Poor OCR Quality**
- **Issue**: Scanned invoices often have poor text extraction.
- **Solution**: Use advanced OCR like **Tesseract with preprocessing** or cloud-based OCR like **Google Vision API**.

#### Problem 2: **Context Length Limits**
- **Issue**: Large invoices exceed token limits of QA models.
- **Solution**: Use **sliding window** or **document chunking** strategies.

#### Problem 3: **Lack of Structure**
- **Issue**: Invoices vary in layout and language.
- **Solution**: Use **LayoutLM** or **Donut** models that understand document structure.

---

---

## 📌 Project 2: Resume Screener with LangChain

### 🧠 What is a Resume Screener with LangChain?

A **Resume Screener with LangChain** is an AI-powered system that:
- Parses resumes (PDFs, DOCX)
- Matches resume content against job descriptions
- Uses **LangChain** to orchestrate multiple LLM calls for scoring, summarization, and reasoning

### 🛠️ Essential Subtopics

#### 1. **Resume Parsing**
Extract text from resumes using `PyMuPDF`, `python-docx`, or `OCR`.

#### 2. **Embedding and Similarity Search**
Convert resume and job description text into embeddings and compute similarity (e.g., cosine similarity).

#### 3. **LangChain Chains**
Use LangChain to create a pipeline that:
- Summarizes the resume
- Matches it to the job description
- Scores the match
- Generates reasoning

---

### 🧪 Code Example: Resume Screener with LangChain

#### Step 1: Parse Resume

```python
import fitz  # PyMuPDF

def extract_resume_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

resume_text = extract_resume_text("resume.pdf")
```

#### Step 2: Load Job Description

```python
job_description = """
We are looking for a software engineer with experience in Python, FastAPI, and cloud deployment.
Knowledge of NLP and LLMs is a plus.
"""
```

#### Step 3: Use LangChain for Matching

```python
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = OpenAI(temperature=0)

prompt = PromptTemplate(
    input_variables=["resume", "job"],
    template="""
    Compare the following resume and job description. Score the match from 0 to 100 and explain your reasoning.

    Resume:
    {resume}

    Job Description:
    {job}
    """
)

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(resume=resume_text, job=job_description)
print(result)
```

---

### 🌍 Real-World Applications

- **HR automation**: Automatically screen hundreds of resumes.
- **Job portals**: Offer real-time resume-job matching.
- **Recruitment agencies**: Speed up shortlisting and improve accuracy.

---

### ⚠️ Common Problems and Advancements

#### Problem 1: **Unstructured Resume Formats**
- **Issue**: Resumes come in many formats (PDF, DOCX, images).
- **Solution**: Use multi-format parsers and OCR tools.

#### Problem 2: **Bias in Matching**
- **Issue**: Models may favor certain keywords or demographics.
- **Solution**: Use **bias mitigation techniques** and **explainable AI**.

#### Problem 3: **Expensive LLM Calls**
- **Issue**: Using GPT-4 for every resume can be costly.
- **Solution**: Use **embedding-based filtering** first, then LLM reasoning only on top matches.

---

## 📌 Project Structure (Example)

```
/final-projects
│
├── invoice_qa_bot/
│   ├── main.py
│   ├── invoice.pdf
│   └── requirements.txt
│
└── resume_screener/
    ├── app.py
    ├── resume.pdf
    ├── job_description.txt
    └── requirements.txt
```

---

## 🚀 Next Steps

Would you like to:
- Build a **Gradio or Streamlit UI** for either project?
- Deploy these apps using **FastAPI + Docker**?
- Use **vector databases (e.g., Chroma, Pinecone)** to improve resume matching?
- Integrate with **cloud APIs (e.g., OpenAI, Google Document AI)**?

Let me know how you'd like to proceed!

Great! Let's build **both projects from scratch** with **full integration setup**, including:

- ✅ Complete code for **Invoice Q&A Bot**
- ✅ Complete code for **Resume Screener with LangChain**
- ✅ Streamlit UIs for both
- ✅ FastAPI backend (optional)
- ✅ Requirements and setup instructions
- ✅ Docker integration (optional)

---

## 📌 Project 1: Invoice Q&A Bot (Full Code)

### 📁 Project Structure

```
invoice_qa_bot/
├── app.py                 # Streamlit UI
├── invoice_qa_engine.py   # QA logic
├── requirements.txt
└── sample_invoice.pdf     # Test invoice
```

---

### 🔧 `invoice_qa_engine.py`

```python
import pdfplumber
from transformers import pipeline

# Load QA pipeline
qa_pipeline = pipeline("question-answering")

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

def answer_question(question, context):
    result = qa_pipeline(question=question, context=context)
    return result['answer']
```

---

### 🎨 `app.py` (Streamlit UI)

```python
import streamlit as st
from invoice_qa_engine import extract_text_from_pdf, answer_question

st.set_page_config(page_title="Invoice Q&A Bot", layout="centered")
st.title("📄 Invoice Q&A Bot")

uploaded_file = st.file_uploader("Upload an invoice (PDF)", type=["pdf"])

if uploaded_file:
    with open("sample_invoice.pdf", "wb") as f:
        f.write(uploaded_file.read())

    invoice_text = extract_text_from_pdf("sample_invoice.pdf")
    st.success("Invoice uploaded and processed!")

    st.subheader("Ask a Question:")
    user_question = st.text_input("e.g. What is the total amount?")

    if st.button("Get Answer"):
        if user_question.strip() == "":
            st.warning("Please enter a question.")
        else:
            answer = answer_question(user_question, invoice_text)
            st.success(f"Answer: {answer}")
```

---

### 📦 `requirements.txt`

```
streamlit
pdfplumber
torch
transformers
sentencepiece
```

---

---

## 📌 Project 2: Resume Screener with LangChain (Full Code)

### 📁 Project Structure

```
resume_screener/
├── app.py                 # Streamlit UI
├── matcher.py             # Matching logic
├── requirements.txt
├── sample_resume.pdf
└── sample_job.txt
```

---

### 🔧 `matcher.py`

```python
import fitz  # PyMuPDF
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def extract_resume_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def load_job_description(txt_path):
    with open(txt_path, "r") as file:
        return file.read()

def match_resume_to_job(resume_text, job_description):
    llm = OpenAI(temperature=0)
    prompt = PromptTemplate(
        input_variables=["resume", "job"],
        template="""
        You are an expert recruiter. Compare the resume and job description below.
        Give a match score from 0 to 100 and explain your reasoning.

        Resume:
        {resume}

        Job Description:
        {job}
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(resume=resume_text, job=job_description)
```

---

### 🎨 `app.py` (Streamlit UI)

```python
import streamlit as st
from matcher import extract_resume_text, load_job_description, match_resume_to_job

st.set_page_config(page_title="Resume Screener", layout="wide")
st.title("📄 Resume Screener with LangChain")

st.sidebar.header("Upload Files")
resume_file = st.sidebar.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_file = st.sidebar.file_uploader("Upload Job Description (TXT)", type=["txt"])

if resume_file and job_file:
    with open("sample_resume.pdf", "wb") as f:
        f.write(resume_file.read())

    with open("sample_job.txt", "wb") as f:
        f.write(job_file.read())

    resume_text = extract_resume_text("sample_resume.pdf")
    job_description = load_job_description("sample_job.txt")

    st.subheader("Resume Preview")
    st.text_area("Content", resume_text, height=300)

    st.subheader("Job Description")
    st.text_area("Content", job_description, height=200)

    if st.button("Match Resume to Job"):
        result = match_resume_to_job(resume_text, job_description)
        st.subheader("Matching Result")
        st.success(result)
```

---

### 📦 `requirements.txt`

```
streamlit
pymupdf
langchain
openai
tiktoken
```

---

## 🐳 Docker Setup (Optional)

### `Dockerfile`

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run

```bash
docker build -t invoice-qa-bot .
docker run -p 8501:8501 invoice-qa-bot
```

---

## 🌐 Deployment Options

### 1. **Local Testing**
```bash
streamlit run app.py
```

### 2. **Heroku**
- Create `Procfile`:
  ```
  web: streamlit run app.py --server.port $PORT
  ```
- Push to Heroku as shown in previous session.

### 3. **Hugging Face Spaces**
- Upload code as a Space repo.
- It will auto-deploy with Streamlit.

---

## 🚀 Optional Enhancements

- Use **ChromaDB or FAISS** for semantic resume search.
- Add **user authentication** (e.g., Firebase, Auth0).
- Use **FastAPI** to expose endpoints and build a React frontend.
- Integrate **Google Document AI** or **AWS Textract** for better OCR.

---

Would you like help with any of these enhancements or a walk-through of deployment on a specific platform?

Great! Let's enhance the **Resume Screener with LangChain** project by integrating **ChromaDB**, a vector database that allows us to store and query resume embeddings efficiently. This will enable **semantic search** across resumes and improve matching quality.

---

## 🧠 Why Use ChromaDB?

- **Embedding Storage**: Store resume embeddings for fast retrieval.
- **Semantic Search**: Find resumes similar to a job description using vector similarity.
- **Scalability**: Efficiently handle hundreds or thousands of resumes.
- **Integration**: Works natively with LangChain and OpenAI embeddings.

---

## 📁 Updated Project Structure

```
resume_screener_chroma/
├── app.py                 # Streamlit UI
├── matcher.py             # Matching logic with Chroma
├── store_resumes.py       # Script to index resumes into Chroma
├── requirements.txt
├── sample_resume.pdf
└── sample_job.txt
```

---

## 🔧 `store_resumes.py` (Index Resumes into ChromaDB)

```python
import fitz  # PyMuPDF
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def index_resumes_to_chroma(resume_paths, persist_directory="chroma_db"):
    embeddings = OpenAIEmbeddings()
    documents = []

    for path in resume_paths:
        text = extract_text_from_pdf(path)
        documents.append(text)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.create_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectorstore.persist()
    print("Resumes indexed successfully!")

if __name__ == "__main__":
    resume_files = ["sample_resume.pdf"]  # Add more resumes as needed
    index_resumes_to_chroma(resume_files)
```

---

## 🔧 `matcher.py` (Matching Logic with ChromaDB)

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def match_resume_to_job(job_description, persist_directory="chroma_db"):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    # Retrieve top resumes similar to job description
    results = vectorstore.similarity_search(job_description, k=3)
    top_resumes = [doc.page_content for doc in results]

    llm = OpenAI(temperature=0)
    prompt = PromptTemplate(
        input_variables=["job", "resumes"],
        template="""
        You are an expert recruiter. Given the job description and resumes below,
        rank the resumes and provide a score (0-100) and reasoning for each.

        Job Description:
        {job}

        Resumes:
        {resumes}
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(job=job_description, resumes="\n\n".join(top_resumes))
```

---

## 🎨 `app.py` (Streamlit UI with Chroma)

```python
import streamlit as st
from matcher import match_resume_to_job
from store_resumes import index_resumes_to_chroma
import os

st.set_page_config(page_title="Resume Screener with ChromaDB", layout="wide")
st.title("📄 Resume Screener with ChromaDB")

st.sidebar.header("Controls")
if st.sidebar.button("Re-index Resumes"):
    resume_files = ["sample_resume.pdf"]  # Add more as needed
    index_resumes_to_chroma(resume_files)
    st.sidebar.success("Resumes re-indexed!")

job_description = st.text_area("Paste Job Description Here", height=200)

if st.button("Match Resumes to Job"):
    if not job_description.strip():
        st.warning("Please enter a job description.")
    else:
        with st.spinner("Matching..."):
            result = match_resume_to_job(job_description)
            st.subheader("Matching Results")
            st.success(result)
```

---

## 📦 `requirements.txt`

```
streamlit
pymupdf
langchain
openai
tiktoken
chromadb
pypdf
```

---

## 🚀 How to Run

### 1. **Index Resumes into ChromaDB**

```bash
python store_resumes.py
```

This will create a `chroma_db` directory with indexed resume embeddings.

### 2. **Run Streamlit App**

```bash
streamlit run app.py
```

### 3. **Usage Flow**
- Upload or place resumes in the project folder.
- Click “Re-index Resumes” to update ChromaDB.
- Paste a job description and click “Match Resumes to Job”.

---

## 🌐 Deployment Notes

- **Local**: ChromaDB persists locally by default.
- **Cloud (Heroku/Render)**: Use cloud storage or vector DBs like Pinecone or Weaviate for persistence.
- **Hugging Face Spaces**: Works well, but ChromaDB must be re-indexed on each restart unless connected to cloud storage.

---

## ⚙️ Optional Enhancements

- **Use Pinecone/Weaviate**: Replace ChromaDB with cloud vector DBs for persistence across sessions.
- **Add Metadata**: Store resume filenames, upload dates, etc., alongside embeddings.
- **UI Improvements**: Add file upload widget for resumes and job descriptions.
- **User Accounts**: Use Firebase/Auth0 to allow users to manage their own resume sets.

---

Would you like help setting up **Pinecone** or **Weaviate** instead of ChromaDB for persistent cloud storage? Or want to build a **FastAPI backend** for this resume matcher?

