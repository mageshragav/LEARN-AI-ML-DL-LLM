# LLM App: News Summarizer using HuggingFace + Streamlit

## 📌 Project: News Summarizer

This project demonstrates building a complete LLM application that summarizes news articles using Hugging Face transformers and Streamlit for the web interface.

### What is it?

A news summarizer is an application that takes a long news article as input and generates a concise, coherent summary that captures the key points. This leverages Large Language Models (LLMs) from Hugging Face to understand the content and produce human-readable summaries, making it easier for users to quickly grasp the essence of lengthy articles.

### PyTorch Implementation

```python
# First, let's install required packages
# Note: This would typically be in a requirements.txt file or installation script
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    pipeline,
    BartForConditionalGeneration,
    T5ForConditionalGeneration
)
import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import time

# Define the news summarizer class
class NewsSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        """
        Initialize the news summarizer with a pre-trained model
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load the appropriate model based on type
        if "t5" in model_name.lower():
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        else:
            self.model = BartForConditionalGeneration.from_pretrained(model_name)
        
        # Create a summarization pipeline
        self.summarizer = pipeline(
            "summarization",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        print(f"Loaded model: {model_name}")
    
    def preprocess_text(self, text, max_length=1024):
        """
        Preprocess the input text to ensure it's in the right format
        """
        # Clean the text
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        text = text.strip()
        
        # Truncate if too long (models have token limits)
        if len(text) > max_length:
            text = text[:max_length]
        
        return text
    
    def summarize(self, text, max_length=150, min_length=50, do_sample=False):
        """
        Generate a summary for the input text
        """
        # Preprocess the text
        processed_text = self.preprocess_text(text)
        
        # Generate summary using the pipeline
        summary = self.summarizer(
            processed_text,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample,
            truncation=True
        )
        
        return summary[0]['summary_text']
    
    def batch_summarize(self, texts, max_length=150, min_length=50):
        """
        Summarize multiple texts at once
        """
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        summaries = self.summarizer(
            processed_texts,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            truncation=True
        )
        
        return [summary['summary_text'] for summary in summaries]

# Example of how to use the summarizer
def demonstrate_summarizer():
    """
    Demonstrate the summarizer functionality
    """
    # Sample news article
    sample_article = """
    The global economy is showing signs of recovery as inflation rates begin to stabilize in many countries. 
    Central banks around the world are carefully monitoring the situation and adjusting their monetary policies accordingly. 
    The International Monetary Fund has revised its growth forecasts upward for several major economies, 
    citing improved consumer spending and business investment. However, challenges remain in the labor market, 
    where unemployment rates are still elevated in some regions. Supply chain disruptions continue to impact 
    global trade, though there are signs that these issues are gradually resolving. Technology companies 
    reported strong earnings this quarter, driven by increased demand for cloud services and artificial intelligence applications. 
    The stock markets have responded positively to these developments, with major indices reaching new highs. 
    Investors remain cautious about potential future risks, including geopolitical tensions and climate-related events. 
    Governments are implementing new policies to support economic growth and job creation, while also addressing 
    long-term sustainability concerns. The outlook for the next quarter appears optimistic, though uncertainties remain.
    """
    
    print("Sample News Article:")
    print(sample_article[:200] + "...")
    print(f"\nOriginal length: {len(sample_article)} characters")
    
    # Initialize summarizer
    summarizer = NewsSummarizer("facebook/bart-large-cnn")
    
    # Generate summary
    summary = summarizer.summarize(sample_article)
    
    print(f"\nGenerated Summary:")
    print(summary)
    print(f"\nSummary length: {len(summary)} characters")
    print(f"Reduction: {((len(sample_article) - len(summary)) / len(sample_article)) * 100:.1f}%")

# Run demonstration
demonstrate_summarizer()
```

### When and where we use news summarization

- **News aggregation platforms**: Providing quick summaries of multiple articles
- **Social media monitoring**: Summarizing trending news topics
- **Corporate newsrooms**: Quickly understanding market-moving news
- **Educational institutions**: Summarizing research papers or news for students
- **Personal productivity**: Helping users quickly scan through daily news
- **Content curation**: Creating digestible content for newsletters or apps

### Real-world examples

- **Reuters**: Uses AI to generate earnings report summaries
- **Associated Press**: Employs automated summarization for sports and financial news
- **Google News**: Provides article snippets and summaries
- **Financial services**: Summarizing market reports and earnings calls
- **Research platforms**: Condensing academic papers and reports

### Problems and Advancements

**Problem**: Maintaining factual accuracy while reducing text length.
**Advancement**: Better pre-trained models and fine-tuning techniques.

**Problem**: Handling domain-specific terminology and context.
**Advancement**: Domain-adapted models and multi-document summarization.

## Streamlit App Implementation

```python
# Create the Streamlit app
streamlit_app_code = '''
import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import requests
from bs4 import BeautifulSoup
import re
import time

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .summary-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .metric-container {
        background-color: #e6f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "original_text" not in st.session_state:
    st.session_state.original_text = ""
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

# App title
st.markdown("<h1 class='main-header'>📰 News Summarizer</h1>", unsafe_allow_html=True)
st.markdown("### Transform long articles into concise summaries using state-of-the-art AI")

# Model selection
st.sidebar.header("Model Configuration")
model_option = st.sidebar.selectbox(
    "Choose a summarization model:",
    [
        "facebook/bart-large-cnn",
        "t5-small",
        "t5-base",
        "google/pegasus-xsum"
    ],
    index=0
)

# Summary parameters
st.sidebar.header("Summary Settings")
max_length = st.sidebar.slider("Max summary length", 50, 500, 150)
min_length = st.sidebar.slider("Min summary length", 10, 150, 50)

# Initialize summarizer
@st.cache_resource
def load_summarizer(model_name):
    """Load the summarization model"""
    try:
        summarizer = pipeline(
            "summarization",
            model=model_name,
            tokenizer=model_name,
            device=0 if torch.cuda.is_available() else -1
        )
        return summarizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to extract text from URL
def extract_text_from_url(url):
    """Extract text content from a URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = " ".join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        st.error(f"Error extracting text from URL: {e}")
        return None

# Function to preprocess text
def preprocess_text(text, max_length=2048):
    """Preprocess the input text"""
    # Clean the text
    text = re.sub(r"\\s+", " ", text)  # Replace multiple spaces with single space
    text = text.strip()
    
    # Truncate if too long (models have token limits)
    if len(text) > max_length:
        text = text[:max_length]
    
    return text

# Main app layout
tab1, tab2 = st.tabs(["📝 Text Input", "🔗 URL Input"])

with tab1:
    st.subheader("Enter News Article")
    input_text = st.text_area(
        "Paste your news article here:",
        height=300,
        placeholder="Enter the full text of the news article you want to summarize..."
    )

with tab2:
    st.subheader("Enter Article URL")
    url = st.text_input(
        "Or provide a URL to a news article:",
        placeholder="https://example.com/news-article"
    )
    
    if url:
        if st.button("Extract Article from URL"):
            with st.spinner("Extracting article content..."):
                extracted_text = extract_text_from_url(url)
                if extracted_text:
                    input_text = extracted_text
                    st.success("Article extracted successfully!")
                else:
                    st.error("Failed to extract text from the URL")

# Summarization button
if st.button("Generate Summary", type="primary", use_container_width=True):
    if input_text.strip():
        # Preprocess input
        processed_text = preprocess_text(input_text)
        
        # Store original text in session state
        st.session_state.original_text = processed_text
        
        # Load model
        with st.spinner(f"Loading model: {model_option}..."):
            summarizer = load_summarizer(model_option)
            if summarizer is None:
                st.stop()
        
        # Generate summary
        with st.spinner("Generating summary..."):
            start_time = time.time()
            summary_result = summarizer(
                processed_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True
            )
            end_time = time.time()
            
            summary = summary_result[0]["summary_text"]
            st.session_state.summary = summary
        
        st.success(f"Summary generated in {end_time - start_time:.2f} seconds!")
    else:
        st.warning("Please enter some text or provide a URL")

# Display results if available
if st.session_state.summary:
    st.header("Generated Summary")
    st.markdown(f"<div class='summary-box'>{st.session_state.summary}</div>", unsafe_allow_html=True)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class='metric-container'>
            <h4>Original Length</h4>
            <p>{len(st.session_state.original_text):,} characters</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-container'>
            <h4>Summary Length</h4>
            <p>{len(st.session_state.summary):,} characters</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        reduction = ((len(st.session_state.original_text) - len(st.session_state.summary)) / len(st.session_state.original_text)) * 100
        st.markdown(f"""
        <div class='metric-container'>
            <h4>Reduction</h4>
            <p>{reduction:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Show original text
    with st.expander("View Original Text"):
        st.text_area("Original Article", st.session_state.original_text, height=300)
    
    # Download summary
    st.download_button(
        label="Download Summary",
        data=st.session_state.summary,
        file_name="news_summary.txt",
        mime="text/plain"
    )

# App information
with st.sidebar:
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This app uses state-of-the-art transformer models to generate concise summaries of news articles.
    
    **Features:**
    - Multiple model options
    - URL extraction
    - Customizable summary length
    - Performance metrics
    """)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Powered by Hugging Face Transformers and Streamlit</p>", unsafe_allow_html=True)
'''

# Save the Streamlit app to a file
with open('news_summarizer_app.py', 'w') as f:
    f.write(streamlit_app_code)

print("Streamlit app created as 'news_summarizer_app.py'")
print("To run the app:")
print("1. Install required packages: pip install streamlit transformers torch requests beautifulsoup4")
print("2. Run the app: streamlit run news_summarizer_app.py")
```

## Advanced Features Implementation

```python
# Advanced summarization features
class AdvancedNewsSummarizer(NewsSummarizer):
    def __init__(self, model_name="facebook/bart-large-cnn"):
        super().__init__(model_name)
        
        # Additional models for different summarization approaches
        self.lead3_summarizer = self._create_lead3_summarizer()
    
    def _create_lead3_summarizer(self):
        """Simple extractive summarizer that takes first 3 sentences"""
        def lead3_summarize(text):
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            # Take first 3 sentences or all if less than 3
            selected = sentences[:3] if len(sentences) >= 3 else sentences
            return ' '.join(selected) + '.'
        return lead3_summarize
    
    def multi_model_summarize(self, text, models=["facebook/bart-large-cnn", "google/pegasus-xsum"]):
        """Generate summaries using multiple models"""
        summaries = {}
        
        for model_name in models:
            try:
                temp_summarizer = NewsSummarizer(model_name)
                summary = temp_summarizer.summarize(text)
                summaries[model_name] = summary
            except Exception as e:
                print(f"Error with model {model_name}: {e}")
        
        return summaries
    
    def extract_key_points(self, text):
        """Extract key points from the text"""
        # This is a simplified version - in practice, you'd use more sophisticated NLP
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        # Take sentences that are likely to be important (contain key terms)
        key_terms = ['important', 'significant', 'major', 'key', 'crucial', 'critical', 'notable', 'primary']
        key_points = []
        
        for sentence in sentences[:10]:  # Check first 10 sentences
            if any(term in sentence.lower() for term in key_terms):
                key_points.append(sentence.strip())
        
        return key_points
    
    def summarize_with_insights(self, text):
        """Generate summary with additional insights"""
        # Generate main summary
        main_summary = self.summarize(text)
        
        # Extract key points
        key_points = self.extract_key_points(text)
        
        # Calculate statistics
        word_count = len(text.split())
        sentence_count = len(re.split(r'[.!?]+', text))
        
        return {
            'summary': main_summary,
            'key_points': key_points,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'summary_length': len(main_summary),
            'reduction_percentage': ((word_count - len(main_summary.split())) / word_count) * 100
        }

# Example usage of advanced features
def demonstrate_advanced_features():
    sample_article = """
    The global economy is showing signs of recovery as inflation rates begin to stabilize in many countries. 
    Central banks around the world are carefully monitoring the situation and adjusting their monetary policies accordingly. 
    The International Monetary Fund has revised its growth forecasts upward for several major economies, 
    citing improved consumer spending and business investment. However, challenges remain in the labor market, 
    where unemployment rates are still elevated in some regions. Supply chain disruptions continue to impact 
    global trade, though there are signs that these issues are gradually resolving. Technology companies 
    reported strong earnings this quarter, driven by increased demand for cloud services and artificial intelligence applications. 
    The stock markets have responded positively to these developments, with major indices reaching new highs. 
    Investors remain cautious about potential future risks, including geopolitical tensions and climate-related events. 
    Governments are implementing new policies to support economic growth and job creation, while also addressing 
    long-term sustainability concerns. The outlook for the next quarter appears optimistic, though uncertainties remain.
    """
    
    print("Demonstrating Advanced Features:")
    print("="*50)
    
    # Initialize advanced summarizer
    advanced_summarizer = AdvancedNewsSummarizer()
    
    # Generate summary with insights
    insights = advanced_summarizer.summarize_with_insights(sample_article)
    
    print(f"Summary: {insights['summary']}")
    print(f"\nKey Points: {insights['key_points']}")
    print(f"\nStatistics:")
    print(f"  Original words: {insights['word_count']}")
    print(f"  Original sentences: {insights['sentence_count']}")
    print(f"  Summary length: {insights['summary_length']} characters")
    print(f"  Reduction: {insights['reduction_percentage']:.1f}%")

demonstrate_advanced_features()
```

## Deployment Considerations

```python
# Deployment configuration example
deployment_config = {
    "model_loading": {
        "cache_dir": "./model_cache",
        "use_gpu": True,
        "batch_size": 1
    },
    "api_settings": {
        "max_input_length": 1024,
        "max_output_length": 256,
        "timeout": 30
    },
    "streamlit_config": {
        "server_port": 8501,
        "server_headless": True,
        "browser_gather_usage_stats": False
    }
}

print("Deployment Configuration:")
print(deployment_config)
```

