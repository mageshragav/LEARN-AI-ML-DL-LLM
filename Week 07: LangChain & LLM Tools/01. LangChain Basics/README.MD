# Day 31 – LangChain Basics

## What is LangChain?

LangChain is an open-source framework designed to simplify the development of applications powered by large language models (LLMs). It provides a structured way to chain together LLM calls, integrate with external data sources, and manage memory and context across interactions.

LangChain addresses common challenges in building LLM applications by providing modular components that can be combined in various ways. It acts as a "middleware" layer between your application and the LLM, offering abstractions for prompt management, chaining operations, and maintaining conversational context.

## Essential Subtopics

### 1. Prompt Templates

Prompt templates in LangChain are structured formats for creating prompts with placeholders for dynamic content. They allow you to define reusable prompt structures that can be filled with specific values at runtime.

**Key Concepts:**
- Template formatting with variables
- Parameter injection
- Reusable prompt structures
- Integration with different LLM providers

**Coding Example - Using PromptTemplate:**

```python
from langchain.prompts import PromptTemplate

# Create a simple prompt template
template = """
Question: {question}
Answer: Let's think step by step.
"""

prompt = PromptTemplate(
    input_variables=["question"],
    template=template,
)

# Format the prompt with a specific question
formatted_prompt = prompt.format(question="What is the capital of France?")
print(formatted_prompt)

# Output:
"""
Question: What is the capital of France?
Answer: Let's think step by step.
"""
```

**Coding Example - Using FewShotPromptTemplate:**

```python
from langchain.prompts import FewShotPromptTemplate, PromptTemplate

# Create example prompts
example_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="Question: {question}\n{answer}",
)

# Create examples
examples = [
    {
        "question": "Who was the first person to walk on the moon?",
        "answer": "Neil Armstrong was the first person to walk on the moon in 1969."
    },
    {
        "question": "What is the largest planet in our solar system?",
        "answer": "Jupiter is the largest planet in our solar system."
    }
]

# Create a few-shot prompt template
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"],
)

# Format the prompt
formatted_prompt = few_shot_prompt.format(input="What is the smallest country in the world?")
print(formatted_prompt)
```

### 2. Chains

Chains in LangChain are sequences of operations that can include LLM calls, data processing, and other actions. They allow you to create complex workflows by combining simple components in a structured way.

**Key Concepts:**
- Sequential operations
- LLMChain for basic LLM interactions
- SimpleSequentialChain for chaining multiple operations
- TransformChain for data processing

**Coding Example - Using LLMChain:**

```python
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

# Initialize the LLM
llm = OpenAI(temperature=0.7)

# Create a prompt template
template = """
What is a good name for a company that makes {product}?
"""

prompt = PromptTemplate(
    input_variables=["product"],
    template=template,
)

# Create an LLM chain
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain
result = chain.run("eco-friendly water bottles")
print(result)
```

**Coding Example - Using SimpleSequentialChain:**

```python
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain, SimpleSequentialChain

# Initialize the LLM
llm = OpenAI(temperature=0.7)

# First chain: Generate a company name
name_template = """
What is a good name for a company that makes {product}?
"""
name_prompt = PromptTemplate(
    input_variables=["product"],
    template=name_template,
)
name_chain = LLMChain(llm=llm, prompt=name_prompt)

# Second chain: Generate a tagline based on the company name
tagline_template = """
Generate a catchy tagline for a company called {company_name}.
"""
tagline_prompt = PromptTemplate(
    input_variables=["company_name"],
    template=tagline_template,
)
tagline_chain = LLMChain(llm=llm, prompt=tagline_prompt)

# Create a sequential chain
overall_chain = SimpleSequentialChain(
    chains=[name_chain, tagline_chain],
    verbose=True
)

# Run the chain
result = overall_chain.run("eco-friendly water bottles")
print(result)
```

### 3. Memory

Memory in LangChain refers to the ability to maintain context across multiple interactions with an LLM. It allows applications to remember previous parts of a conversation, which is essential for building coherent conversational agents.

**Key Concepts:**
- ConversationBufferMemory for storing full conversation history
- ConversationSummaryMemory for summarizing conversations
- ConversationKGMemory for storing knowledge graph representations
- Memory integration with chains

**Coding Example - Using ConversationBufferMemory:**

```python
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# Initialize the LLM
llm = OpenAI(temperature=0.7)

# Create a prompt template that includes conversation history
template = """
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{chat_history}

Human: {human_input}
AI:
"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"],
    template=template,
)

# Create a memory buffer
memory = ConversationBufferMemory(memory_key="chat_history")

# Create an LLM chain with memory
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

# First interaction
chain.run("Hi, my name is John!")

# Second interaction (the AI remembers the name)
chain.run("What's my name?")
```

**Coding Example - Using ConversationSummaryMemory:**

```python
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryMemory

# Initialize the LLM
llm = OpenAI(temperature=0.7)

# Create a prompt template
template = """
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation summary:
{chat_history}

Human: {human_input}
AI:
"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"],
    template=template,
)

# Create a summary memory
memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history")

# Create an LLM chain with summary memory
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

# Multiple interactions
chain.run("Hi, I'm planning a trip to Japan. Can you recommend some places to visit?")
chain.run("What's the best time of year to visit Japan?")
chain.run("I'm interested in traditional Japanese culture. Any specific recommendations?")
```

## When and Where We Use LangChain

### When to Use LangChain:

1. **When building complex LLM applications**: LangChain provides structure for applications that require multiple LLM calls or complex workflows.

2. **When you need to maintain conversation context**: Memory components make it easier to build conversational agents that remember previous interactions.

3. **When integrating with external data sources**: LangChain offers tools for connecting LLMs with databases, APIs, and other data sources.

4. **When you want to experiment with different prompt strategies**: Prompt templates make it easy to test and refine different prompting approaches.

### Where We Use LangChain:

1. **Conversational AI**: Building chatbots and virtual assistants that maintain context across multiple turns.

2. **Document Analysis**: Creating systems that can answer questions about documents or summarize content.

3. **Automated Workflows**: Developing applications that use LLMs as part of larger automated processes.

4. **Personalized Content Generation**: Creating systems that generate content tailored to user preferences and history.

**Coding Example - Document Q&A with LangChain:**

```python
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Load a document
loader = TextLoader("path/to/your/document.txt")

# Create an index from the document
index = VectorstoreIndexCreator().from_loaders([loader])

# Create a question-answering chain
llm = OpenAI()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=index.vectorstore.as_retriever(),
)

# Ask questions about the document
question = "What is the main topic of the document?"
answer = qa_chain.run(question)
print(f"Question: {question}")
print(f"Answer: {answer}")
```

## Real-World Applications and Examples

### 1. Customer Support Chatbot

**Application**: Building a customer support chatbot that can answer questions about products and services while maintaining conversation context.

**Coding Example - Customer support chatbot with memory:**

```python
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentType, initialize_agent, load_tools

# Initialize the LLM
llm = OpenAI(temperature=0.7)

# Create a prompt template for customer support
template = """
You are a helpful customer support assistant for an e-commerce company called ShopEasy. 
You help customers with questions about products, orders, shipping, and returns.
If you don't know the answer to a question, you say that you don't know and offer to transfer the customer to a human agent.

Current conversation:
{chat_history}

Customer: {customer_input}
Assistant:
"""

prompt = PromptTemplate(
    input_variables=["chat_history", "customer_input"],
    template=template,
)

# Create a memory buffer
memory = ConversationBufferMemory(memory_key="chat_history")

# Create an LLM chain with memory
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

# Simulate a customer conversation
print("Customer Support Chatbot")
print("Type 'quit' to end the conversation\n")

while True:
    user_input = input("Customer: ")
    if user_input.lower() == 'quit':
        print("Assistant: Thank you for contacting ShopEasy support. Have a great day!")
        break
    
    response = chain.run(customer_input=user_input)
    print(f"Assistant: {response}")
```

### 2. Document Summarization Pipeline

**Application**: Creating a pipeline that can load documents, extract key information, and generate summaries.

**Coding Example - Document summarization with chains:**

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain, SimpleSequentialChain

# Initialize the LLM
llm = OpenAI(temperature=0.7)

# Load a PDF document
loader = PyPDFLoader("path/to/document.pdf")
documents = loader.load()

# Split the document into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# First chain: Extract key points from each chunk
extract_template = """
Extract the key points from the following text:

Text: {text}

Key points:
"""
extract_prompt = PromptTemplate(
    input_variables=["text"],
    template=extract_template,
)
extract_chain = LLMChain(llm=llm, prompt=extract_prompt)

# Second chain: Summarize the key points
summarize_template = """
Summarize the following key points into a concise summary:

Key points: {key_points}

Summary:
"""
summarize_prompt = PromptTemplate(
    input_variables=["key_points"],
    template=summarize_template,
)
summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt)

# Process each chunk and combine the results
all_key_points = []
for text in texts[:5]:  # Process first 5 chunks for demo
    key_points = extract_chain.run(text.page_content)
    all_key_points.append(key_points)

# Combine all key points
combined_key_points = "\n".join(all_key_points)

# Generate the final summary
summary = summarize_chain.run(key_points=combined_key_points)
print("Document Summary:")
print(summary)
```

### 3. Personalized Content Recommendation

**Application**: Building a system that recommends content based on user preferences and interaction history.

**Coding Example - Content recommendation with memory:**

```python
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryMemory

# Initialize the LLM
llm = OpenAI(temperature=0.7)

# Create a prompt template for content recommendation
template = """
You are a personalized content recommendation assistant. You recommend articles, videos, and other content 
based on the user's interests and interaction history. You provide a brief description of why you think 
each recommendation would be interesting to the user.

User profile summary:
{chat_history}

Current request: {user_request}

Recommendations:
"""

prompt = PromptTemplate(
    input_variables=["chat_history", "user_request"],
    template=template,
)

# Create a summary memory to track user preferences
memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history")

# Create an LLM chain with memory
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

# Simulate user interactions
print("Content Recommendation Assistant")
print("Type 'quit' to end the session\n")

# Initial user preferences
chain.run("I'm interested in technology and artificial intelligence. I like in-depth articles and tutorials.")
chain.run("I also enjoy content about space exploration and astronomy.")

# Get recommendations
while True:
    user_input = input("What kind of content are you looking for today? ")
    if user_input.lower() == 'quit':
        print("Assistant: Goodbye! Come back anytime for more personalized recommendations.")
        break
    
    response = chain.run(user_request=user_input)
    print(f"Assistant: {response}")
```

## Common Problems and How Advancements Have Solved Them

### 1. Problem: Managing Context in Conversations

**Issue**: Maintaining context across multiple turns in a conversation is challenging, especially with token limitations in LLMs.

**Solution**: LangChain's memory components provide different strategies for managing conversation context, from storing full history to creating summaries.

**Problem Code (Basic conversation without context management):**

```python
from langchain.llms import OpenAI

# Initialize the LLM
llm = OpenAI(temperature=0.7)

# First interaction
response1 = llm("Hi, my name is Sarah. I'm planning a trip to Italy.")
print(f"AI: {response1}")

# Second interaction (no context about previous conversation)
response2 = llm("Can you recommend places to visit in Italy?")
print(f"AI: {response2}")  # The AI doesn't remember the user's name or previous context
```

**Enhancement Code (Using ConversationBufferMemory for context):**

```python
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# Initialize the LLM
llm = OpenAI(temperature=0.7)

# Create a prompt template that includes conversation history
template = """
The following is a conversation between a human and an AI assistant.

Current conversation:
{chat_history}

Human: {human_input}
AI:
"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"],
    template=template,
)

# Create a memory buffer
memory = ConversationBufferMemory(memory_key="chat_history")

# Create an LLM chain with memory
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

# First interaction
response1 = chain.run("Hi, my name is Sarah. I'm planning a trip to Italy.")
print(f"AI: {response1}")

# Second interaction (with context)
response2 = chain.run("Can you recommend places to visit in Italy?")
print(f"AI: {response2}")  # The AI remembers the user's name and previous context
```

### 2. Problem: Complex Workflows with Multiple LLM Calls

**Issue**: Building applications that require multiple LLM calls in a specific sequence can be complex and error-prone.

**Solution**: LangChain's chaining capabilities allow you to create structured workflows that combine multiple operations.

**Problem Code (Manual sequence of LLM calls):**

```python
from langchain.llms import OpenAI

# Initialize the LLM
llm = OpenAI(temperature=0.7)

# Manual sequence of operations
# Step 1: Generate a blog post topic
topic_prompt = "Generate a blog post topic about artificial intelligence."
topic = llm(topic_prompt)
print(f"Topic: {topic}")

# Step 2: Generate an outline for the topic
outline_prompt = f"Generate an outline for a blog post with the topic: {topic}"
outline = llm(outline_prompt)
print(f"Outline: {outline}")

# Step 3: Write the blog post based on the outline
blog_prompt = f"Write a blog post based on this outline: {outline}"
blog_post = llm(blog_prompt)
print(f"Blog Post: {blog_post}")
```

**Enhancement Code (Using SimpleSequentialChain for structured workflow):**

```python
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain, SimpleSequentialChain

# Initialize the LLM
llm = OpenAI(temperature=0.7)

# First chain: Generate a blog post topic
topic_template = "Generate a blog post topic about artificial intelligence."
topic_prompt = PromptTemplate(
    input_variables=[],
    template=topic_template,
)
topic_chain = LLMChain(llm=llm, prompt=topic_prompt)

# Second chain: Generate an outline for the topic
outline_template = "Generate an outline for a blog post with the topic: {topic}"
outline_prompt = PromptTemplate(
    input_variables=["topic"],
    template=outline_template,
)
outline_chain = LLMChain(llm=llm, prompt=outline_prompt)

# Third chain: Write the blog post based on the outline
blog_template = "Write a blog post based on this outline: {outline}"
blog_prompt = PromptTemplate(
    input_variables=["outline"],
    template=blog_template,
)
blog_chain = LLMChain(llm=llm, prompt=blog_prompt)

# Create a sequential chain
overall_chain = SimpleSequentialChain(
    chains=[topic_chain, outline_chain, blog_chain],
    verbose=True
)

# Run the chain
result = overall_chain.run({})
print(f"Final Blog Post: {result}")
```

### 3. Problem: Integrating External Data with LLMs

**Issue**: Incorporating external data sources into LLM applications can be challenging, especially when dealing with large datasets.

**Solution**: LangChain provides tools for loading, processing, and indexing external data, making it easier to integrate with LLMs.

**Problem Code (Manual data integration):**

```python
from langchain.llms import OpenAI
import pandas as pd

# Initialize the LLM
llm = OpenAI(temperature=0.7)

# Load data from a CSV file
df = pd.read_csv("products.csv")

# Manually format data for the LLM
product_info = ""
for _, row in df.iterrows():
    product_info += f"Product: {row['name']}, Price: ${row['price']}, Category: {row['category']}\n"

# Create a prompt with the data
prompt = f"""
Based on the following product information, recommend the best product for someone who wants a high-quality laptop under $1000.

{product_info}

Recommendation:
"""

# Get the recommendation
recommendation = llm(prompt)
print(recommendation)
```

**Enhancement Code (Using LangChain's data loading and indexing):**

```python
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Load data from a CSV file
loader = CSVLoader(file_path="products.csv")

# Create an index from the data
index = VectorstoreIndexCreator().from_loaders([loader])

# Create a question-answering chain
llm = OpenAI()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=index.vectorstore.as_retriever(),
)

# Ask a question about the data
question = "What is the best laptop under $1000?"
answer = qa_chain.run(question)
print(f"Question: {question}")
print(f"Answer: {answer}")
```

## 🧪 Exercise: Basic LLM App

Let's create a basic LLM application using LangChain that can answer questions and maintain conversation context:

```python
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import os

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

class BasicLLMApp:
    def __init__(self):
        # Initialize the LLM
        self.llm = OpenAI(temperature=0.7)
        
        # Create a prompt template
        template = """
        You are a helpful AI assistant. You answer questions accurately and provide detailed explanations.
        If you don't know the answer to a question, you say so honestly.
        
        Current conversation:
        {chat_history}
        
        Human: {human_input}
        AI:
        """
        
        self.prompt = PromptTemplate(
            input_variables=["chat_history", "human_input"],
            template=template,
        )
        
        # Create a memory buffer
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        
        # Create an LLM chain with memory
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory,
            verbose=True
        )
    
    def run(self):
        print("Basic LLM App")
        print("Type 'quit' to exit, 'clear' to clear conversation history\n")
        
        while True:
            user_input = input("You: ")
            
            if user_input.lower() == 'quit':
                print("AI: Goodbye!")
                break
            
            if user_input.lower() == 'clear':
                self.memory.clear()
                print("AI: Conversation history cleared.")
                continue
            
            response = self.chain.run(human_input=user_input)
            print(f"AI: {response}")

# Create and run the app
if __name__ == "__main__":
    app = BasicLLMApp()
    app.run()
```

## 📌 Project: Q&A Bot with LangChain

Now, let's build a more sophisticated Q&A bot that can answer questions based on a knowledge base:

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
import os
import glob

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

class QABot:
    def __init__(self, knowledge_base_path=None):
        # Initialize the LLM
        self.llm = OpenAI(temperature=0.7)
        
        # Initialize memory for conversation context
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        
        # Create a prompt template for general conversation
        self.conversation_prompt = PromptTemplate(
            input_variables=["chat_history", "human_input"],
            template="""
            You are a helpful AI assistant with access to a knowledge base. 
            You can answer questions based on the knowledge base and maintain conversation context.
            
            Current conversation:
            {chat_history}
            
            Human: {human_input}
            AI:
            """
        )
        
        # Create a conversation chain
        self.conversation_chain = LLMChain(
            llm=self.llm,
            prompt=self.conversation_prompt,
            memory=self.memory,
            verbose=False
        )
        
        # Initialize the knowledge base
        self.knowledge_base = None
        self.qa_chain = None
        
        if knowledge_base_path:
            self.load_knowledge_base(knowledge_base_path)
    
    def load_knowledge_base(self, path):
        """Load documents from a path and create a knowledge base"""
        print("Loading knowledge base...")
        
        # Load documents
        documents = []
        if os.path.isdir(path):
            # Load all text files in the directory
            for file_path in glob.glob(os.path.join(path, "*.txt")):
                loader = TextLoader(file_path)
                documents.extend(loader.load())
        else:
            # Load a single file
            loader = TextLoader(path)
            documents = loader.load()
        
        if not documents:
            print("No documents found.")
            return
        
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings()
        self.knowledge_base = FAISS.from_documents(texts, embeddings)
        
        # Create a QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.knowledge_base.as_retriever(),
            verbose=False
        )
        
        print(f"Knowledge base loaded with {len(texts)} text chunks.")
    
    def ask_question(self, question):
        """Ask a question based on the knowledge base"""
        if not self.qa_chain:
            return "No knowledge base loaded. Please load a knowledge base first."
        
        try:
            answer = self.qa_chain.run(question)
            return answer
        except Exception as e:
            return f"Error answering question: {str(e)}"
    
    def chat(self, message):
        """Have a general conversation"""
        try:
            response = self.conversation_chain.run(human_input=message)
            return response
        except Exception as e:
            return f"Error in conversation: {str(e)}"
    
    def clear_memory(self):
        """Clear the conversation memory"""
        self.memory.clear()
        return "Conversation memory cleared."
    
    def run(self):
        """Run the interactive Q&A bot"""
        print("Q&A Bot with LangChain")
        print("Commands:")
        print("  'ask <question>' - Ask a question based on the knowledge base")
        print("  'chat <message>' - Have a general conversation")
        print("  'load <path>' - Load a knowledge base from a file or directory")
        print("  'clear' - Clear conversation history")
        print("  'quit' - Exit the bot\n")
        
        while True:
            user_input = input("You: ")
            
            if user_input.lower() == 'quit':
                print("Bot: Goodbye!")
                break
            
            if user_input.lower() == 'clear':
                response = self.clear_memory()
                print(f"Bot: {response}")
                continue
            
            if user_input.lower().startswith('load '):
                path = user_input[5:].strip()
                self.load_knowledge_base(path)
                continue
            
            if user_input.lower().startswith('ask '):
                question = user_input[4:].strip()
                response = self.ask_question(question)
                print(f"Bot: {response}")
                continue
            
            if user_input.lower().startswith('chat '):
                message = user_input[5:].strip()
                response = self.chat(message)
                print(f"Bot: {response}")
                continue
            
            # Default to conversation if no command is specified
            response = self.chat(user_input)
            print(f"Bot: {response}")

# Create and run the Q&A bot
if __name__ == "__main__":
    # Create a sample knowledge base file for demonstration
    sample_text = """
    Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.
    
    Machine Learning is a subset of AI that focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy. Machine learning is an important component of the growing field of data science.
    
    Deep Learning is a subset of machine learning based on artificial neural networks. Deep learning models are trained using large sets of labeled data and neural network architectures that learn features directly from the data without the need for manual feature extraction.
    
    Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret and manipulate human language. NLP draws from many disciplines including computer science and computational linguistics in its pursuit to fill the gap between human communication and computer understanding.
    """
    
    with open("sample_knowledge.txt", "w") as f:
        f.write(sample_text)
    
    # Create and run the bot
    bot = QABot()
    bot.load_knowledge_base("sample_knowledge.txt")
    bot.run()
```

This Q&A bot includes:
1. Knowledge base loading from text files
2. Question answering based on the knowledge base
3. General conversation capabilities with memory
4. Command-based interface for different operations
5. Error handling and user feedback

