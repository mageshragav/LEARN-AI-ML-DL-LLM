

# Day 25 – NLP Capstone: Resume Parser using NER and Classifier

## What is a Resume Parser and Why is it Important for NLP?

A Resume Parser is a software tool that automatically analyzes and extracts structured information from unstructured resume documents (like PDFs or DOCX). It converts free-form text into a structured format, such as JSON or XML, identifying key pieces of information like the candidate's name, contact details, skills, work experience, education, and more.

This project is a capstone for NLP because it's not a single, simple task. It's a complex pipeline that combines several advanced NLP techniques:

1.  **Document Processing:** Handling different file formats (PDF, DOCX) and extracting raw text.
2.  **Named Entity Recognition (NER):** Identifying and categorizing key entities like `PERSON` (name), `EMAIL`, `PHONE_NUMBER`, `SKILL`, `DEGREE`, `COMPANY`, `LOCATION`.
3.  **Text Classification:** Categorizing the entire resume into a predefined job domain (e.g., `Software Engineering`, `Data Science`, `Marketing`, `Project Management`).
4.  **Information Extraction:** Going beyond simple entities to understand relationships, such as linking a `COMPANY` to a `JOB_TITLE` and a `DATE_RANGE` within the work experience section.

Building a resume parser is important for NLP because:

-   **It's a Real-World, High-Impact Application:** Companies use resume parsers to automate the screening of thousands of job applications, saving countless hours of manual work for HR departments.
-   **It's a Multi-Faceted Problem:** It forces you to think about how to combine different NLP models into a single, coherent system.
-   **It Deals with Noisy, Unstructured Data:** Resumes have no standard format. They use different layouts, bullet points, fonts, and jargon, making them a perfect testbed for robust NLP techniques.
-   **It Demonstrates the Value of Transfer Learning:** We will use pre-trained Transformer models (like BERT) and fine-tune them, which is the standard and most effective approach in modern NLP.

## Essential Subtopics

### 1. Named Entity Recognition (NER) for Resumes

**What is NER in the context of Resumes?**
Named Entity Recognition is the subtask of NLP that seeks to locate and classify named entities mentioned in unstructured text into pre-defined categories. For a resume parser, we're not just interested in standard entities like `PERSON` or `LOCATION`, but also in domain-specific ones like `SKILL`, `JOB_TITLE`, `COMPANY`, `DEGREE`, and `CERTIFICATION`.

The goal is to turn this:
`"John Doe, a Senior Software Engineer at Google, has a Master's degree in Computer Science from MIT and is skilled in Python and TensorFlow."`

Into this structured data:
```json
{
  "PERSON": ["John Doe"],
  "JOB_TITLE": ["Senior Software Engineer"],
  "COMPANY": ["Google"],
  "DEGREE": ["Master's degree"],
  "FIELD_OF_STUDY": ["Computer Science"],
  "UNIVERSITY": ["MIT"],
  "SKILL": ["Python", "TensorFlow"]
}
```

**Implementation (Fine-tuning a Transformer for NER with `transformers`):**

For this, we'll need a labeled dataset where each word in a resume is tagged with an entity type (e.g., `B-SKILL`, `I-SKILL`, `O`). For this example, we'll simulate the process, as creating a full labeled dataset is a massive undertaking. We will use a pre-trained model and show how you would fine-tune it.

```python
# This is a conceptual example. In a real project, you would load your own labeled dataset.
import torch
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
import numpy as np

# --- 1. Define Labels and Load a Pre-trained Model ---
# Define the NER labels specific to resumes
label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-SKILL", "I-SKILL", "B-JOB", "I-JOB", "B-DEGREE", "I-DEGREE"]
label_map = {label: i for i, label in enumerate(label_list)}
num_labels = len(label_list)

model_name = "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForTokenClassification.from_pretrained(model_name, num_labels=num_labels)

# --- 2. Prepare Data (Conceptual) ---
# In a real scenario, you would load your dataset here.
# Let's assume we have a list of examples:
# resume_data = [
#     {"text": "John Doe worked at Google as a Data Scientist.", "tags": ["B-PER", "I-PER", "O", "B-ORG", "O", "B-JOB", "I-JOB", "O"]},
#     # ... more examples
# ]

# The key steps are:
# 1. Tokenize the text.
# 2. Align the labels with the tokens (since BERT uses WordPiece, one word can be split into multiple tokens).
# 3. Create attention masks to ignore padding tokens.

# --- 3. Fine-tuning the Model (Conceptual) ---
# The Hugging Face `Trainer` API makes this straightforward.
# training_args = TrainingArguments(
#     output_dir='./results',
#     num_train_epochs=3,
#     per_device_train_batch_size=16,
#     save_steps=10_000,
#     save_total_limit=2,
# )
#
# # trainer = Trainer(
# #     model=model,
# #     args=training_args,
# #     train_dataset=your_train_dataset,  # A PyTorch Dataset
# #     eval_dataset=your_eval_dataset    # A PyTorch Dataset
# # )
# #
# # trainer.train()
# #
# # # Save the fine-tuned model
# # model.save_pretrained("resume_ner_model")
# # tokenizer.save_pretrained("resume_ner_model")

# --- 4. Using a Pre-trained NER Model for Demonstration ---
# For this example, let's use a general-purpose NER model from Hugging Face
# to show the mechanics. A resume-specific fine-tuned model would perform much better.
from transformers import pipeline

# Use a pipeline for easy inference
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", tokenizer="dbmdz/bert-large-cased-finetuned-conll03-english")

resume_text = """
John Doe
123 Main St, Anytown, USA
john.doe@email.com | (123) 456-7890 | linkedin.com/in/johndoe

PROFESSIONAL SUMMARY
Senior Software Engineer with 8+ years of experience in full-stack development, cloud computing, and machine learning. Proficient in Python, Java, and AWS.

WORK EXPERIENCE
Senior Software Engineer, Google, Mountain View, CA
June 2018 - Present
- Led a team of 5 engineers to develop a scalable microservices architecture.
- Implemented machine learning models for recommendation systems using TensorFlow.

Software Developer, Microsoft, Redmond, WA
May 2015 - May 2018
- Developed and maintained features for the Azure cloud platform.
- Worked primarily with C# and .NET framework.

EDUCATION
Master of Science in Computer Science, Stanford University
2013 - 2015
Bachelor of Science in Engineering, MIT
2009 - 2013
"""

print("--- Extracting Entities with a Pre-trained Model ---")
entities = ner_pipeline(resume_text)

# Group subwords back into full words/tokens
current_word = ""
current_entity = ""
for entity in entities:
    word, label, score = entity['word'], entity['entity'], entity['score']
    if word.startswith("##"):
        current_word += word[2:]
    else:
        if current_word:
            print(f"Entity: {current_word.strip()}, Label: {current_entity}")
        current_word = word
        current_entity = label
# Print the last entity
if current_word:
    print(f"Entity: {current_word.strip()}, Label: {current_entity}")

print("\nNote: A general model identifies 'John', 'Doe', 'Google', 'Mountain View', etc.")
print("A fine-tuned model would also identify 'Python', 'Java', 'AWS', 'Stanford University', 'MIT', etc.")
```

**When and where we use NER in this project:**
-   **As the core information extraction module.** It's the first and most critical step in parsing the resume.
-   **To populate a structured database.** The extracted entities (skills, companies, degrees) become fields in a candidate profile in an Applicant Tracking System (ATS).
-   **To power search and filtering.** Recruiters can filter for candidates who have specific skills (`SKILL: "Python"`) or have worked at certain companies (`COMPANY: "Google"`).

**Real-world examples:**
-   **LinkedIn Resume Parser:** When you upload a resume to LinkedIn, it uses a sophisticated parser to auto-fill your profile fields like "Current Position," "Past Companies," and "Education."
-   **Applicant Tracking Systems (ATS):** Tools like Workday, Greenhouse, and Lever all have built-in resume parsers to screen incoming applications, rank candidates based on keyword matches (skills), and identify top talent.
-   **HR Tech Startups:** Companies like Hiretual, SeekOut, and Eightfold AI build advanced AI-powered sourcing tools that parse millions of public profiles and resumes to build searchable talent databases.

**Problems and Advancements:**
-   **Problem:** Standard NER models trained on news articles (like CoNLL-2003) fail to identify domain-specific entities crucial for resumes, such as `SKILL` (e.g., "TensorFlow," "Agile Methodology"), `CERTIFICATION` (e.g., "PMP," "AWS Certified Developer"), or `JOB_TITLE` (e.g., "Chief Technology Officer").
  -   **Advancement:** **Fine-tuning Transformer models on domain-specific data.** By taking a powerful pre-trained model like BERT or RoBERTa and fine-tuning it on a dataset of resumes manually labeled with resume-specific entities, the model learns to recognize these critical categories with high accuracy. This is the single most important advancement for this task.
-   **Problem:** Resumes are complex documents with structure (sections, bullet points) and layout. A pure text-based NER model might struggle with context, for example, distinguishing "Java" the programming language from "Java" the island.
  -   **Advancement:** **Layout-Aware Models and Multi-Modal Learning.** Advanced parsers don't just look at the text; they also analyze the document's layout. They use Optical Character Recognition (OCR) engines that preserve positional information (bounding boxes) for each word. Models can then use spatial features (e.g., "Is this word in the 'Skills' section?", "Is it a bullet point?") to improve entity classification. Models like LayoutLM and Donut are designed for this purpose, combining text, layout, and even image information.

### 2. Text Classification for Resume Categorization

**What is Text Classification for Resumes?**
While NER extracts specific pieces of information, text classification aims to assign the entire resume document to one or more predefined categories. In our project, the goal is to read a resume and classify it into a job domain, such as `Software Engineering`, `Data Science`, `Product Management`, `Marketing`, or `Sales`.

This is a multi-class text classification problem. The model learns the typical vocabulary, skills, and experience patterns associated with each job category. For example, a resume classified as `Data Science` will likely contain keywords like "Python," "R," "Machine Learning," "SQL," and "Statistics," while a `Marketing` resume might contain "SEO," "Google Analytics," "Campaign Management," and "Content Strategy."

**Implementation (Fine-tuning a Transformer for Text Classification):**

Again, we'll need a labeled dataset of resumes where each resume is tagged with a job category. We'll simulate the fine-tuning process.

```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import pandas as pd

# --- 1. Define Labels and Load a Pre-trained Model ---
# Define the job categories
categories = ["Software Engineering", "Data Science", "Product Management", "Marketing", "Other"]
label_map = {cat: i for i, cat in enumerate(categories)}
num_labels = len(categories)

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# --- 2. Prepare Data (Conceptual Dataset Class) ---
# In a real project, you would load your data (e.g., from a CSV file)
# data = pd.read_csv("labeled_resumes.csv") # Columns: 'resume_text', 'category'

class ResumeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item_idx):
        text = str(self.texts[item_idx])
        label = self.labels[item_idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# # --- 3. Fine-tuning the Model (Conceptual) ---
# # Assume we have train_df and val_df pandas DataFrames
# train_dataset = ResumeDataset(train_df['resume_text'].to_numpy(), train_df['category'].map(label_map).to_numpy(), tokenizer)
# val_dataset = ResumeDataset(val_df['resume_text'].to_numpy(), val_df['category'].map(label_map).to_numpy(), tokenizer)
#
# training_args = TrainingArguments(
#     output_dir='./results_classifier',
#     num_train_epochs=4,
#     per_device_train_batch_size=8,
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     logging_dir='./logs',
# )
#
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset
# )
#
# trainer.train()
#
# # Save the fine-tuned model
# model.save_pretrained("resume_classifier_model")
# tokenizer.save_pretrained("resume_classifier_model")


# --- 4. Using the Fine-tuned Model for Inference ---
# For demonstration, let's simulate loading a fine-tuned model and using it.
# (In a real app, you would load the actual saved model)
# model_classifier = BertForSequenceClassification.from_pretrained("resume_classifier_model")
# tokenizer_classifier = BertTokenizer.from_pretrained("resume_classifier_model")
model_classifier = model # Using the instantiated model for demo
tokenizer_classifier = tokenizer

def classify_resume(text, model, tokenizer, label_map):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model(**encoding)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1).item()
    
    # Reverse the label map to get the category name
    reverse_label_map = {v: k for k, v in label_map.items()}
    return reverse_label_map[predictions]

# Example resumes for classification
resume_se = """
Experienced software engineer with a strong background in object-oriented programming and software design.
Proficient in Java, Spring Boot, Microservices, and cloud platforms like AWS and Azure.
Led the development of a high-traffic e-commerce platform serving millions of users.
"""

resume_ds = """
Data Scientist with a passion for extracting insights from complex datasets.
Expertise in Python, R, SQL, and machine learning frameworks like Scikit-learn, TensorFlow, and PyTorch.
Experience in building predictive models, A/B testing, and data visualization.
"""

resume_pm = """
Strategic Product Manager with a track record of launching successful products in the B2B SaaS space.
Skilled in market research, user story creation, roadmap planning, and cross-functional team leadership.
Certified Scrum Product Owner (CSPO) with strong analytical and communication skills.
"""

print("\n--- Classifying Resumes (Simulated) ---")
# In a real scenario, these would be much more accurate
print(f"Resume 1 classified as: {classify_resume(resume_se, model_classifier, tokenizer_classifier, label_map)}")
print(f"Resume 2 classified as: {classify_resume(resume_ds, model_classifier, tokenizer_classifier, label_map)}")
print(f"Resume 3 classified as: {classify_resume(resume_pm, model_classifier, tokenizer_classifier, label_map)}")
```

**When and where we use text classification in this project:**
-   **As the primary categorization module.** It provides a high-level summary of the candidate's profile.
-   **To route resumes to the right recruiters or hiring managers.** A resume classified as `Software Engineering` can be automatically sent to the engineering recruitment team.
-   **To build talent pools.** Companies can create searchable databases of candidates categorized by their profession, making it easy to find talent for specific roles.
-   **For analytics and reporting.** To understand the composition of the applicant pool (e.g., "What percentage of applicants are for Data Science roles?").

**Real-world examples:**
-   **Job Boards:** When you upload a resume to sites like Indeed or Monster, they automatically suggest job categories you might be interested in based on a classification of your profile.
-   **Internal HR Systems:** Large corporations use classification to tag incoming applications, making it easier for HR business partners to find relevant candidates when a new role opens up.
-   **Marketplace Platforms:** Freelance platforms like Upwork or Toptal use classification to categorize freelancer profiles, allowing clients to easily find and hire for specific skills.

**Problems and Advancements:**
-   **Problem:** Resumes are long documents, often exceeding the maximum token limit (e.g., 512 for BERT) of standard Transformer models. Truncating them can lead to loss of critical information.
  -   **Advancement:** **Long-Document Transformers and Hierarchical Models.** To handle long text, several strategies exist:
      1.  **Longformer, BigBird:** These are Transformer architectures modified to handle sequences of thousands of tokens, making them suitable for long documents.
      2.  **Hierarchical Classification:** The document is split into smaller chunks (e.g., by section). Each chunk is encoded by a Transformer, and then the chunk representations are aggregated (e.g., averaged, or fed into another model) to produce a final document-level classification.
-   **Problem:** A resume can belong to multiple categories (e.g., a "Machine Learning Engineer" has skills in both `Software Engineering` and `Data Science`). Standard single-label classification is insufficient.
  -   **Advancement:** **Multi-Label Classification.** This is a variation of the classification task where the model can predict multiple labels for a single instance. The final layer of the network uses a sigmoid activation function for each label instead of a softmax, and the loss function is changed to Binary Cross-Entropy. This allows the model to independently predict the presence of each job category.
    ```python
    # For multi-label, the final layer and loss change
    # model.classifier = nn.Linear(config.hidden_size, num_labels) # num_labels is now the number of possible categories
    # loss_fct = nn.BCEWithLogitsLoss() # Instead of CrossEntropyLoss
    ```

## 📌 Project: Resume Parser using NER and Classifier

Now, let's combine these concepts into a single, end-to-end application. This project will be a Python script that takes a resume text file, processes it with our (simulated) NER and classification models, and outputs a structured JSON object.

**Implementation (`resume_parser_app.py`):**

This script will simulate the full pipeline. In a real-world scenario, you would replace the simulated model loading with your actual fine-tuned models.

```python
import json
import re
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# --- 1. Load Models (Simulated) ---
# In a real application, you would load your fine-tuned models here.
# For this example, we use pre-trained models and rule-based extraction for demonstration.

print("Loading models...")

# NER Model (using a general-purpose one for demo)
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True)

# Classifier Model (simulated)
# We'll use a simple keyword-based approach for demonstration
# In a real app, you would load your fine-tuned BERT classifier.
# model_classifier = AutoModelForSequenceClassification.from_pretrained("your_resume_classifier_model")
# tokenizer_classifier = AutoTokenizer.from_pretrained("your_resume_classifier_model")

def simple_keyword_classifier(text):
    """A simple rule-based classifier for demonstration."""
    text_lower = text.lower()
    if any(keyword in text_lower for keyword in ["python", "java", "c++", "software engineer", "developer", "api", "github"]):
        return "Software Engineering"
    elif any(keyword in text_lower for keyword in ["data scientist", "machine learning", "tensorflow", "pytorch", "sql", "r language"]):
        return "Data Science"
    elif any(keyword in text_lower for keyword in ["product manager", "roadmap", "user stories", "agile", "scrum", "market research"]):
        return "Product Management"
    elif any(keyword in text_lower for keyword in ["marketing", "seo", "sem", "campaign", "brand", "google analytics"]):
        return "Marketing"
    else:
        return "Other"

print("Models loaded (simulated).")

# --- 2. Define Helper Functions ---
def extract_email_phone(text):
    """Uses regex to find email and phone numbers."""
    email = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    phone = re.findall(r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
    return email[0] if email else None, phone[0] if phone else None

def extract_skills(text, skill_keywords):
    """Extracts skills based on a predefined list."""
    text_lower = text.lower()
    found_skills = []
    for skill in skill_keywords:
        if skill.lower() in text_lower:
            found_skills.append(skill)
    return list(set(found_skills)) # Return unique skills

# --- 3. The Main Parsing Function ---
def parse_resume(resume_text):
    """
    Parses a single resume text to extract structured information.
    """
    parsed_data = {
        "contact_info": {},
        "entities": {},
        "classification": None,
        "raw_text": resume_text
    }

    # 1. Extract Contact Info (Rule-based)
    email, phone = extract_email_phone(resume_text)
    if email: parsed_data["contact_info"]["email"] = email
    if phone: parsed_data["contact_info"]["phone"] = phone

    # 2. Extract Entities using NER Model
    ner_results = ner_pipeline(resume_text)
    for entity in ner_results:
        entity_group = entity['entity_group']
        word = entity['word']
        if entity_group not in parsed_data["entities"]:
            parsed_data["entities"][entity_group] = []
        parsed_data["entities"][entity_group].append(word)
    
    # 3. Classify Resume
    # In a real app: classification = classify_resume(resume_text, model_classifier, tokenizer_classifier)
    classification = simple_keyword_classifier(resume_text)
    parsed_data["classification"] = classification

    # 4. Extract Skills (Hybrid approach: NER + Keywords)
    # The general NER model won't find skills, so we use a keyword list
    # A fine-tuned NER model would do this automatically.
    common_skills = ["Python", "Java", "C++", "JavaScript", "SQL", "TensorFlow", "PyTorch", "Scikit-learn",
                     "AWS", "Azure", "Docker", "Kubernetes", "Agile", "Scrum", "Project Management",
                     "Marketing", "SEO", "Data Analysis", "Machine Learning"]
    skills = extract_skills(resume_text, common_skills)
    if skills:
        parsed_data["entities"]["SKILL"] = skills # Add to our entities dict

    return parsed_data

# --- 4. Main Execution Block ---
if __name__ == "__main__":
    # Sample resume text
    sample_resume = """
    Jane Smith
    Data Scientist | jane.smith@email.com | (555) 123-4567 | linkedin.com/in/janesmith

    PROFILE
    Results-oriented Data Scientist with 5+ years of experience in leveraging machine learning and data analysis to drive business solutions. Proficient in Python, R, SQL, and TensorFlow.

    SKILLS
    Programming Languages: Python, R, SQL
    Machine Learning: Scikit-learn, TensorFlow, PyTorch, XGBoost
    Data Visualization: Tableau, Matplotlib, Seaborn
    Cloud Platforms: AWS, Azure

    PROFESSIONAL EXPERIENCE
    Senior Data Analyst, Innovate Corp, New York, NY
    Jan 2020 - Present
    - Developed predictive models to forecast customer churn, reducing churn by 15%.
    - Analyzed large datasets using SQL and Python to identify key business insights.
    - Collaborated with product and engineering teams to integrate models into production.

    Data Scientist, Data Solutions Inc., Boston, MA
    Jun 2017 - Dec 2019
    - Built machine learning pipelines for real-time fraud detection.
    - Conducted A/B testing to evaluate the effectiveness of new features.
    - Presented findings to stakeholders through clear and concise visualizations.

    EDUCATION
    Master of Science in Data Science, Stanford University
    2015 - 2017
    Bachelor of Science in Statistics, UC Berkeley
    2011 - 2015
    """

    print("\n--- Parsing Sample Resume ---")
    parsed_json = parse_resume(sample_resume)

    # Pretty print the JSON output
    print(json.dumps(parsed_json, indent=2))

    # You can save this to a file
    with open("parsed_resume.json", "w") as f:
        json.dump(parsed_json, f, indent=2)
    
    print("\nParsed resume saved to parsed_resume.json")
```

**How to Run and Extend the Project:**

1.  **Save the Code:** Save the code above as `resume_parser_app.py`.
2.  **Install Dependencies:**
    ```bash
    pip install torch transformers pandas regex
    ```
3.  **Run the Script:**
    ```bash
    python resume_parser_app.py
    ```

You will see the structured JSON output printed to your console and saved in `parsed_resume.json`.

**Next Steps for a Real-World Application:**

1.  **Data Collection and Labeling:** This is the most critical and labor-intensive step.
    *   Collect hundreds or thousands of resumes in PDF/DOCX format.
    *   Use a library like `PyPDF2` or `python-docx` to extract the raw text.
    *   Create a labeling tool (or use one like Label Studio) to manually tag entities (`SKILL`, `JOB_TITLE`, etc.) and assign job categories to each resume.

2.  **Model Fine-Tuning:**
    *   **For NER:** Use your labeled NER data to fine-tune a `BertForTokenClassification` model as shown in the NER section.
    *   **For Classification:** Use your labeled classification data to fine-tune a `BertForSequenceClassification` model as shown in the Classification section.

3.  **Build a Web Interface (Streamlit):**
    *   Create a new file `app/streamlit_resume_parser.py`.
    *   Add a file uploader widget (`st.file_uploader`) to accept PDF or DOCX files.
    *   Add text extraction logic inside the app.
    *   Load your *actual* fine-tuned models using `@st.cache_resource`.
    *   Call your `parse_resume` function (updated to use your real models).
    *   Display the structured JSON output in a clean, formatted way, perhaps using `st.json`.

4.  **Deployment:**
    *   Deploy your Streamlit app to Streamlit Community Cloud or another platform, just as we did in the CNN project.
