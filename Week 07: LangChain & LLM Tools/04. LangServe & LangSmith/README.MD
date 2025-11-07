# Day 34 – LangServe & LangSmith

## What are LangServe & LangSmith?

**LangServe** is a framework for serving LangChain applications as REST APIs. It provides tools to package, deploy, and scale LangChain chains, making it easier to productionize LLM-powered applications. LangServe handles the complexities of API creation, request handling, and response formatting, allowing developers to focus on building the core logic of their applications.

**LangSmith** is a platform for observability, tracing, and debugging LangChain applications. It provides tools to monitor the performance of LLM applications, trace the execution of chains, identify bottlenecks, and debug issues. LangSmith helps developers understand how their applications are behaving in production and optimize them for better performance and reliability.

Together, LangServe and LangSmith form a powerful combination for deploying and monitoring production-grade LLM applications.

## Essential Subtopics

### 1. LangServe: Serving LangChain Applications

LangServe simplifies the process of deploying LangChain chains as web services. It provides a standardized way to expose chains through HTTP endpoints, handling input validation, serialization, and other web service concerns.

**Key Concepts:**
- API endpoint creation
- Input/output schemas
- Request handling
- Deployment options

**Coding Example - Creating a simple LangServe app:**

```python
from fastapi import FastAPI
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langserve import add_routes

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

# Create a FastAPI app
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server for LangChain chains",
)

# Add the chain to the app
add_routes(
    app,
    chain,
    path="/company-name",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
```

**Coding Example - Running and testing the LangServe app:**

```python
import requests
import json

# Define the API endpoint
url = "http://localhost:8000/company-name/invoke"

# Define the input data
data = {
    "input": {"product": "eco-friendly water bottles"}
}

# Make a POST request to the API
response = requests.post(url, json=data)

# Print the response
if response.status_code == 200:
    result = response.json()
    print("Generated company name:")
    print(result["output"])
else:
    print(f"Error: {response.status_code}")
    print(response.text)
```

### 2. LangSmith: Observability and Tracing

LangSmith provides tools for monitoring and debugging LangChain applications. It allows you to trace the execution of chains, track performance metrics, and identify issues in your applications.

**Key Concepts:**
- Execution tracing
- Performance monitoring
- Error tracking
- Debugging tools

**Coding Example - Setting up LangSmith tracing:**

```python
import os
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langsmith import Client

# Set up LangSmith environment variables
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-api-key"
os.environ["LANGCHAIN_PROJECT"] = "my-langchain-project"

# Initialize LangSmith client
client = Client()

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

# Create an LLM chain with tracing enabled
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain with tracing
result = chain.run(product="eco-friendly water bottles")
print(result)
```

**Coding Example - Retrieving and analyzing traces:**

```python
from langsmith import Client
import pandas as pd

# Initialize LangSmith client
client = Client()

# Get runs from your project
runs = client.list_runs(project_name="my-langchain-project")

# Convert to a DataFrame for analysis
data = []
for run in runs:
    data.append({
        "run_id": run.id,
        "name": run.name,
        "start_time": run.start_time,
        "end_time": run.end_time,
        "execution_time": (run.end_time - run.start_time).total_seconds(),
        "status": run.status,
        "error": run.error if run.error else None
    })

df = pd.DataFrame(data)

# Display basic statistics
print(f"Total runs: {len(df)}")
print(f"Average execution time: {df['execution_time'].mean():.2f} seconds")
print(f"Success rate: {(df['status'] == 'success').mean() * 100:.1f}%")

# Display the slowest runs
print("\nSlowest runs:")
print(df.nlargest(5, 'execution_time')[['name', 'execution_time', 'status']])
```

### 3. Advanced LangServe Features

LangServe provides several advanced features for building robust production applications, including authentication, rate limiting, and custom middleware.

**Key Concepts:**
- Authentication and authorization
- Rate limiting
- Custom middleware
- Input validation

**Coding Example - Adding authentication to LangServe:**

```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langserve import add_routes

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

# Create a FastAPI app
app = FastAPI(
    title="Authenticated LangChain Server",
    version="1.0",
    description="An authenticated API server for LangChain chains",
)

# Set up API key authentication
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Valid API keys (in a real app, these would be stored securely)
VALID_API_KEYS = ["your-api-key-1", "your-api-key-2"]

async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )
    return api_key

# Add the chain to the app with authentication
add_routes(
    app,
    chain,
    path="/company-name",
    dependencies=[Depends(get_api_key)],
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
```

**Coding Example - Adding rate limiting to LangServe:**

```python
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware import Middleware
from fastapi.middleware.base import BaseHTTPMiddleware
from time import time
from collections import defaultdict
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langserve import add_routes

# Rate limiting middleware
class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, rate_limit=10, time_window=60):
        super().__init__(app)
        self.rate_limit = rate_limit
        self.time_window = time_window
        self.requests = defaultdict(list)
    
    async def dispatch(self, request: Request, call_next):
        # Get client IP
        client_ip = request.client.host
        
        # Get current time
        current_time = time()
        
        # Clean old requests
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if current_time - req_time < self.time_window
        ]
        
        # Check rate limit
        if len(self.requests[client_ip]) >= self.rate_limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
            )
        
        # Record the request
        self.requests[client_ip].append(current_time)
        
        # Process the request
        response = await call_next(request)
        return response

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

# Create a FastAPI app with middleware
app = FastAPI(
    title="Rate Limited LangChain Server",
    version="1.0",
    description="A rate-limited API server for LangChain chains",
    middleware=[Middleware(RateLimitMiddleware, rate_limit=5, time_window=60)],
)

# Add the chain to the app
add_routes(
    app,
    chain,
    path="/company-name",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
```

### 4. LangSmith Advanced Features

LangSmith provides advanced features for deep analysis and optimization of LangChain applications, including custom metrics, dataset evaluation, and comparison tools.

**Key Concepts:**
- Custom metrics
- Dataset evaluation
- A/B testing
- Performance optimization

**Coding Example - Creating custom metrics in LangSmith:**

```python
from langsmith import Client, Run, EvaluationResult
from typing import Dict, Any

# Initialize LangSmith client
client = Client()

# Define a custom metric function
def response_length_metric(run: Run, example: Dict[str, Any]) -> EvaluationResult:
    """Evaluate the length of the response."""
    response = run.outputs.get("output", "")
    length = len(response.split())
    
    # Score based on desired length (e.g., 50-100 words is ideal)
    if 50 <= length <= 100:
        score = 1.0
    elif 30 <= length < 50 or 100 < length <= 150:
        score = 0.7
    else:
        score = 0.3
    
    return EvaluationResult(
        key="response_length",
        score=score,
        comment=f"Response length: {length} words"
    )

# Define another custom metric
def sentiment_metric(run: Run, example: Dict[str, Any]) -> EvaluationResult:
    """Evaluate the sentiment of the response (simulated)."""
    response = run.outputs.get("output", "")
    
    # In a real implementation, you would use a sentiment analysis model
    # For this example, we'll use a simple heuristic
    positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic"]
    negative_words = ["bad", "terrible", "awful", "horrible", "poor", "worst"]
    
    positive_count = sum(1 for word in positive_words if word in response.lower())
    negative_count = sum(1 for word in negative_words if word in response.lower())
    
    if positive_count > negative_count:
        score = 1.0
        sentiment = "positive"
    elif negative_count > positive_count:
        score = 0.0
        sentiment = "negative"
    else:
        score = 0.5
        sentiment = "neutral"
    
    return EvaluationResult(
        key="sentiment",
        score=score,
        comment=f"Sentiment: {sentiment}"
    )

# Register the custom metrics
client.create_evaluation_function(response_length_metric)
client.create_evaluation_function(sentiment_metric)

# Create a dataset for evaluation
dataset = client.create_dataset(
    name="company-name-evaluation",
    description="Dataset for evaluating company name generation",
)

# Add examples to the dataset
examples = [
    {"input": {"product": "eco-friendly water bottles"}},
    {"input": {"product": "smart home devices"}},
    {"input": {"product": "vegan snacks"}},
]

for example in examples:
    client.create_example(
        inputs=example["input"],
        dataset_id=dataset.id,
    )

# Run evaluation
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

# Run the evaluation
evaluation_results = client.evaluate(
    chain,
    dataset_id=dataset.id,
    evaluators=[response_length_metric, sentiment_metric],
)

# Print the results
for result in evaluation_results:
    print(f"Example: {result.example.inputs}")
    print(f"Output: {result.run.outputs}")
    print(f"Evaluation Scores: {result.evaluation_results}")
    print("-" * 50)
```

## When and Where We Use LangServe & LangSmith

### When to Use LangServe:

1. **When deploying LangChain applications to production**: LangServe provides the tools needed to expose chains as web services.

2. **When you need to scale your LLM applications**: LangServe makes it easier to handle increased traffic and load.

3. **When you need to integrate with other systems**: LangServe's REST API interface allows for easy integration with web applications, mobile apps, and other services.

4. **When you need to secure your LLM applications**: LangServe supports authentication, authorization, and other security features.

### When to Use LangSmith:

1. **When monitoring production LLM applications**: LangSmith provides observability tools to track performance and usage.

2. **When debugging complex chains**: LangSmith's tracing capabilities help identify issues in chain execution.

3. **When optimizing LLM applications**: LangSmith's metrics and evaluation tools help identify areas for improvement.

4. **When conducting A/B testing**: LangSmith supports comparing different versions of chains to determine which performs better.

### Where We Use LangServe & LangSmith:

1. **Production LLM APIs**: LangServe for serving the APIs, LangSmith for monitoring their performance.

2. **Customer Support Chatbots**: LangServe for deploying the chatbot service, LangSmith for tracking conversation quality and identifying issues.

3. **Content Generation Systems**: LangServe for exposing content generation capabilities, LangSmith for monitoring output quality and consistency.

4. **Enterprise LLM Applications**: LangServe for secure deployment, LangSmith for compliance monitoring and performance tracking.

**Coding Example - End-to-end deployment and monitoring:**

```python
# langserve_app.py
from fastapi import FastAPI
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langserve import add_routes
import os

# Set environment variables
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-api-key"
os.environ["LANGCHAIN_PROJECT"] = "customer-support-chatbot"

# Initialize the LLM
llm = OpenAI(temperature=0.7)

# Create a prompt template for customer support
template = """
You are a helpful customer support assistant for an e-commerce company called ShopEasy. 
You help customers with questions about products, orders, shipping, and returns.
If you don't know the answer to a question, you say that you don't know and offer to transfer the customer to a human agent.

Customer: {customer_input}
Assistant:
"""
prompt = PromptTemplate(
    input_variables=["customer_input"],
    template=template,
)

# Create an LLM chain
chain = LLMChain(llm=llm, prompt=prompt)

# Create a FastAPI app
app = FastAPI(
    title="ShopEasy Customer Support",
    version="1.0",
    description="Customer support chatbot for ShopEasy",
)

# Add the chain to the app
add_routes(
    app,
    chain,
    path="/chat",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
```

```python
# monitor_app.py
from langsmith import Client
import pandas as pd
import matplotlib.pyplot as plt
import time

# Initialize LangSmith client
client = Client()

def monitor_app():
    """Monitor the LangServe app using LangSmith"""
    print("Monitoring ShopEasy Customer Support App")
    print("-" * 50)
    
    while True:
        # Get recent runs
        runs = client.list_runs(
            project_name="customer-support-chatbot",
            execution_order="desc",
            limit=10
        )
        
        if not runs:
            print("No recent runs found.")
            time.sleep(30)
            continue
        
        # Calculate metrics
        total_runs = len(runs)
        successful_runs = sum(1 for run in runs if run.status == "success")
        avg_execution_time = sum(
            (run.end_time - run.start_time).total_seconds() 
            for run in runs if run.end_time and run.start_time
        ) / total_runs
        
        # Display metrics
        print(f"\nTotal runs: {total_runs}")
        print(f"Success rate: {successful_runs/total_runs*100:.1f}%")
        print(f"Avg execution time: {avg_execution_time:.2f} seconds")
        
        # Display recent errors
        errors = [run for run in runs if run.status == "error"]
        if errors:
            print("\nRecent errors:")
            for error in errors[:3]:
                print(f"- {error.error[:100]}...")
        
        # Wait before next check
        time.sleep(30)

if __name__ == "__main__":
    monitor_app()
```

## Real-World Applications and Examples

### 1. Production Chatbot Service

**Application**: Deploying a customer support chatbot as a production service with monitoring and debugging capabilities.

**Coding Example - Production chatbot with LangServe and LangSmith:**

```python
# chatbot_service.py
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory
from langserve import add_routes
from langsmith import Client
import os
import uuid
from typing import Dict, Any

# Set environment variables
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-api-key"
os.environ["LANGCHAIN_PROJECT"] = "production-chatbot"

# Initialize LangSmith client
langsmith_client = Client()

# Initialize the LLM
llm = OpenAI(temperature=0.7)

# Create a dictionary to store conversation memories
conversation_memories: Dict[str, ConversationBufferMemory] = {}

# Create a prompt template for customer support
template = """
You are a helpful customer support assistant for an e-commerce company called ShopEasy. 
You help customers with questions about products, orders, shipping, and returns.
If you don't know the answer to a question, you say that you don't know and offer to transfer the customer to a human agent.

Current conversation:
{history}

Customer: {input}
Assistant:
"""
prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=template,
)

# Create a FastAPI app
app = FastAPI(
    title="ShopEasy Customer Support",
    version="1.0",
    description="Production customer support chatbot for ShopEasy",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoint to start a new conversation
@app.post("/conversation/new")
async def new_conversation():
    """Start a new conversation and return a conversation ID."""
    conversation_id = str(uuid.uuid4())
    conversation_memories[conversation_id] = ConversationBufferMemory()
    return {"conversation_id": conversation_id}

# Endpoint to continue a conversation
@app.post("/conversation/{conversation_id}/chat")
async def chat(conversation_id: str, input_data: Dict[str, str]):
    """Continue a conversation with the chatbot."""
    if conversation_id not in conversation_memories:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    
    # Get the memory for this conversation
    memory = conversation_memories[conversation_id]
    
    # Create a chain with the memory
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
    )
    
    # Get the user input
    user_input = input_data.get("input", "")
    
    # Generate a response
    response = chain.run(input=user_input)
    
    # Log the interaction to LangSmith
    langsmith_client.create_run(
        name="chatbot_interaction",
        inputs={"input": user_input, "conversation_id": conversation_id},
        outputs={"output": response},
        project_name=os.environ["LANGCHAIN_PROJECT"],
    )
    
    return {"response": response}

# Endpoint to get conversation history
@app.get("/conversation/{conversation_id}/history")
async def get_history(conversation_id: str):
    """Get the conversation history."""
    if conversation_id not in conversation_memories:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    
    memory = conversation_memories[conversation_id]
    return {"history": memory.buffer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
```

### 2. Content Generation API

**Application**: Creating a content generation API with monitoring for quality and performance.

**Coding Example - Content generation API with monitoring:**

```python
# content_generation_api.py
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langserve import add_routes
from langsmith import Client, Run, EvaluationResult
from typing import Dict, Any
import os
import re

# Set environment variables
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-api-key"
os.environ["LANGCHAIN_PROJECT"] = "content-generation-api"

# Initialize LangSmith client
langsmith_client = Client()

# Initialize the LLM
llm = OpenAI(temperature=0.7)

# Create a prompt template for blog post generation
blog_template = """
Write a blog post about {topic} with the following requirements:
- Target audience: {audience}
- Tone: {tone}
- Length: Approximately {word_count} words
- Include a catchy title
- Structure with an introduction, body, and conclusion

Blog Post:
"""
blog_prompt = PromptTemplate(
    input_variables=["topic", "audience", "tone", "word_count"],
    template=blog_template,
)

# Create a prompt template for social media post generation
social_template = """
Create a {platform} post about {topic} with the following requirements:
- Target audience: {audience}
- Tone: {tone}
- Include relevant hashtags
- Maximum length: {max_length} characters

Social Media Post:
"""
social_prompt = PromptTemplate(
    input_variables=["platform", "topic", "audience", "tone", "max_length"],
    template=social_template,
)

# Create evaluation functions
def word_count_metric(run: Run, example: Dict[str, Any]) -> EvaluationResult:
    """Evaluate if the generated content meets the word count requirement."""
    target_word_count = example.get("word_count", 500)
    content = run.outputs.get("output", "")
    actual_word_count = len(content.split())
    
    # Calculate score based on how close to the target
    if actual_word_count == 0:
        score = 0.0
    else:
        # Score is 1.0 if within 10% of target, decreases linearly beyond that
        ratio = actual_word_count / target_word_count
        if 0.9 <= ratio <= 1.1:
            score = 1.0
        else:
            score = max(0.0, 1.0 - abs(ratio - 1.0))
    
    return EvaluationResult(
        key="word_count",
        score=score,
        comment=f"Target: {target_word_count}, Actual: {actual_word_count}"
    )

def length_metric(run: Run, example: Dict[str, Any]) -> EvaluationResult:
    """Evaluate if the generated content meets the character length requirement."""
    max_length = example.get("max_length", 280)
    content = run.outputs.get("output", "")
    actual_length = len(content)
    
    # Score is 1.0 if under or equal to max length, 0.0 otherwise
    score = 1.0 if actual_length <= max_length else 0.0
    
    return EvaluationResult(
        key="length",
        score=score,
        comment=f"Max: {max_length}, Actual: {actual_length}"
    )

# Register evaluation functions
langsmith_client.create_evaluation_function(word_count_metric)
langsmith_client.create_evaluation_function(length_metric)

# Create chains
blog_chain = LLMChain(llm=llm, prompt=blog_prompt)
social_chain = LLMChain(llm=llm, prompt=social_prompt)

# Create a FastAPI app
app = FastAPI(
    title="Content Generation API",
    version="1.0",
    description="API for generating various types of content",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add the chains to the app
add_routes(
    app,
    blog_chain,
    path="/generate-blog",
)

add_routes(
    app,
    social_chain,
    path="/generate-social-post",
)

# Custom endpoint for generating content with evaluation
@app.post("/generate-with-evaluation")
async def generate_with_evaluation(input_data: Dict[str, Any]):
    """Generate content and evaluate it."""
    content_type = input_data.get("type", "blog")
    
    if content_type == "blog":
        # Generate blog post
        result = blog_chain.run(
            topic=input_data.get("topic", "artificial intelligence"),
            audience=input_data.get("audience", "general"),
            tone=input_data.get("tone", "informative"),
            word_count=input_data.get("word_count", 500),
        )
        
        # Evaluate the result
        evaluation = word_count_metric(
            Run(outputs={"output": result}),
            {"word_count": input_data.get("word_count", 500)}
        )
        
    elif content_type == "social":
        # Generate social media post
        result = social_chain.run(
            platform=input_data.get("platform", "Twitter"),
            topic=input_data.get("topic", "artificial intelligence"),
            audience=input_data.get("audience", "general"),
            tone=input_data.get("tone", "informative"),
            max_length=input_data.get("max_length", 280),
        )
        
        # Evaluate the result
        evaluation = length_metric(
            Run(outputs={"output": result}),
            {"max_length": input_data.get("max_length", 280)}
        )
        
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid content type. Must be 'blog' or 'social'.",
        )
    
    # Log to LangSmith
    langsmith_client.create_run(
        name=f"generate_{content_type}_with_evaluation",
        inputs=input_data,
        outputs={"output": result, "evaluation": evaluation.dict()},
        project_name=os.environ["LANGCHAIN_PROJECT"],
    )
    
    return {
        "content": result,
        "evaluation": {
            "metric": evaluation.key,
            "score": evaluation.score,
            "comment": evaluation.comment
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
```

### 3. Document Processing Pipeline

**Application**: Creating a document processing pipeline with monitoring for performance and accuracy.

**Coding Example - Document processing pipeline with tracing:**

```python
# document_processing_pipeline.py
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langsmith import Client
from typing import Dict, Any
import os
import tempfile
import uuid
import time

# Set environment variables
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-api-key"
os.environ["LANGCHAIN_PROJECT"] = "document-processing-pipeline"

# Initialize LangSmith client
langsmith_client = Client()

# Initialize the LLM
llm = OpenAI(temperature=0.7)

# Create a dictionary to store document indexes
document_indexes: Dict[str, Any] = {}

# Create a FastAPI app
app = FastAPI(
    title="Document Processing Pipeline",
    version="1.0",
    description="API for processing documents and answering questions about them",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoint to upload and process a document
@app.post("/document/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document."""
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp_file:
        tmp_file.write(await file.read())
        tmp_file_path = tmp_file.name
    
    try:
        # Load the document
        if file.filename.endswith('.pdf'):
            loader = PyPDFLoader(tmp_file_path)
        elif file.filename.endswith('.txt'):
            loader = TextLoader(tmp_file_path)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported file type. Only PDF and TXT files are supported.",
            )
        
        documents = loader.load()
        
        # Split the documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(texts, embeddings)
        
        # Create a document ID
        document_id = str(uuid.uuid4())
        
        # Store the vector store
        document_indexes[document_id] = {
            "vector_store": vector_store,
            "filename": file.filename,
            "upload_time": time.time(),
            "chunk_count": len(texts),
        }
        
        # Log to LangSmith
        langsmith_client.create_run(
            name="document_upload",
            inputs={"filename": file.filename, "size": len(texts)},
            outputs={"document_id": document_id},
            project_name=os.environ["LANGCHAIN_PROJECT"],
        )
        
        return {
            "document_id": document_id,
            "filename": file.filename,
            "chunk_count": len(texts),
        }
    
    finally:
        # Clean up the temporary file
        os.unlink(tmp_file_path)

# Endpoint to ask a question about a document
@app.post("/document/{document_id}/question")
async def ask_question(document_id: str, question_data: Dict[str, str]):
    """Ask a question about a processed document."""
    if document_id not in document_indexes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )
    
    # Get the vector store
    vector_store = document_indexes[document_id]["vector_store"]
    
    # Create a QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
    )
    
    # Get the question
    question = question_data.get("question", "")
    
    # Generate an answer
    start_time = time.time()
    answer = qa_chain.run(question)
    end_time = time.time()
    
    # Log to LangSmith
    langsmith_client.create_run(
        name="document_question",
        inputs={"document_id": document_id, "question": question},
        outputs={"answer": answer},
        project_name=os.environ["LANGCHAIN_PROJECT"],
        start_time=start_time,
        end_time=end_time,
    )
    
    return {
        "answer": answer,
        "processing_time": end_time - start_time,
    }

# Endpoint to list uploaded documents
@app.get("/documents")
async def list_documents():
    """List all uploaded documents."""
    return [
        {
            "document_id": doc_id,
            "filename": doc_info["filename"],
            "upload_time": doc_info["upload_time"],
            "chunk_count": doc_info["chunk_count"],
        }
        for doc_id, doc_info in document_indexes.items()
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
```

## Common Problems and How Advancements Have Solved Them

### 1. Problem: Deploying LangChain Applications to Production

**Issue**: Deploying LangChain applications to production can be challenging due to the need for API creation, request handling, and other web service concerns.

**Solution**: LangServe provides a standardized way to deploy LangChain chains as REST APIs, handling the complexities of web service creation.

**Problem Code (Manual API creation):**

```python
from flask import Flask, request, jsonify
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
import os

# Set environment variables
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

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

# Create a Flask app
app = Flask(__name__)

# Define an endpoint
@app.route('/generate-name', methods=['POST'])
def generate_name():
    data = request.json
    product = data.get('product', '')
    
    if not product:
        return jsonify({'error': 'Product is required'}), 400
    
    try:
        result = chain.run(product=product)
        return jsonify({'name': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

**Enhancement Code (Using LangServe):**

```python
from fastapi import FastAPI
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langserve import add_routes
import os

# Set environment variables
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

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

# Create a FastAPI app
app = FastAPI(
    title="Company Name Generator",
    version="1.0",
    description="API for generating company names",
)

# Add the chain to the app
add_routes(
    app,
    chain,
    path="/generate-name",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
```

### 2. Problem: Monitoring and Debugging LLM Applications

**Issue**: Monitoring and debugging LLM applications in production is difficult due to the complexity of chain execution and the lack of visibility into internal processes.

**Solution**: LangSmith provides observability tools for tracing chain execution, monitoring performance, and debugging issues.

**Problem Code (Basic chain without monitoring):**

```python
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain, SimpleSequentialChain
import os

# Set environment variables
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

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

**Enhancement Code (With LangSmith tracing):**

```python
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain, SimpleSequentialChain
from langsmith import Client
import os

# Set environment variables
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-api-key"
os.environ["LANGCHAIN_PROJECT"] = "company-name-generator"

# Initialize LangSmith client
client = Client()

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

# Run the chain with tracing
result = overall_chain.run("eco-friendly water bottles")
print(result)

# Get the runs from LangSmith
runs = client.list_runs(project_name="company-name-generator")

# Analyze the runs
for run in runs:
    print(f"Run ID: {run.id}")
    print(f"Name: {run.name}")
    print(f"Execution time: {(run.end_time - run.start_time).total_seconds():.2f} seconds")
    print(f"Status: {run.status}")
    print("-" * 50)
```

### 3. Problem: Evaluating LLM Application Performance

**Issue**: Evaluating the performance and quality of LLM applications is challenging due to the subjective nature of language generation and the lack of standardized metrics.

**Solution**: LangSmith provides tools for defining custom evaluation metrics, running evaluations on datasets, and comparing different versions of chains.

**Problem Code (Basic evaluation without metrics):**

```python
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
import os

# Set environment variables
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Initialize the LLM
llm = OpenAI(temperature=0.7)

# Create a prompt template
template = """
Summarize the following text in {max_sentences} sentences:

Text: {text}

Summary:
"""
prompt = PromptTemplate(
    input_variables=["text", "max_sentences"],
    template=template,
)

# Create an LLM chain
chain = LLMChain(llm=llm, prompt=prompt)

# Test data
test_cases = [
    {
        "text": "The James Webb Space Telescope is a space telescope designed primarily to conduct infrared astronomy. As the largest optical telescope in space, its high resolution and sensitivity allow it to view objects too old, distant, or faint for the Hubble Space Telescope.",
        "max_sentences": 2,
    },
    {
        "text": "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.",
        "max_sentences": 1,
    },
]

# Run the chain on test cases
for i, test_case in enumerate(test_cases):
    result = chain.run(
        text=test_case["text"],
        max_sentences=test_case["max_sentences"],
    )
    print(f"Test Case {i+1}:")
    print(f"Expected sentences: {test_case['max_sentences']}")
    print(f"Summary: {result}")
    print("-" * 50)
```

**Enhancement Code (With LangSmith evaluation):**

```python
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langsmith import Client, Run, EvaluationResult
from typing import Dict, Any
import os

# Set environment variables
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-api-key"
os.environ["LANGCHAIN_PROJECT"] = "summarization-evaluation"

# Initialize LangSmith client
client = Client()

# Initialize the LLM
llm = OpenAI(temperature=0.7)

# Create a prompt template
template = """
Summarize the following text in {max_sentences} sentences:

Text: {text}

Summary:
"""
prompt = PromptTemplate(
    input_variables=["text", "max_sentences"],
    template=template,
)

# Create an LLM chain
chain = LLMChain(llm=llm, prompt=prompt)

# Define evaluation functions
def sentence_count_metric(run: Run, example: Dict[str, Any]) -> EvaluationResult:
    """Evaluate if the summary has the correct number of sentences."""
    expected_sentences = example.get("max_sentences", 3)
    summary = run.outputs.get("output", "")
    
    # Count sentences (simple heuristic)
    actual_sentences = len([s.strip() for s in summary.split('.') if s.strip()])
    
    # Calculate score based on how close to the expected count
    if actual_sentences == expected_sentences:
        score = 1.0
    else:
        score = max(0.0, 1.0 - abs(actual_sentences - expected_sentences) / expected_sentences)
    
    return EvaluationResult(
        key="sentence_count",
        score=score,
        comment=f"Expected: {expected_sentences}, Actual: {actual_sentences}"
    )

def relevance_metric(run: Run, example: Dict[str, Any]) -> EvaluationResult:
    """Evaluate the relevance of the summary (simulated)."""
    text = example.get("text", "")
    summary = run.outputs.get("output", "")
    
    # In a real implementation, you would use a more sophisticated method
    # For this example, we'll use a simple keyword overlap heuristic
    text_words = set(text.lower().split())
    summary_words = set(summary.lower().split())
    
    if not summary_words:
        return EvaluationResult(
            key="relevance",
            score=0.0,
            comment="Empty summary"
        )
    
    overlap = len(text_words.intersection(summary_words))
    relevance = overlap / len(summary_words)
    
    return EvaluationResult(
        key="relevance",
        score=relevance,
        comment=f"Keyword overlap: {overlap}/{len(summary_words)}"
    )

# Register evaluation functions
client.create_evaluation_function(sentence_count_metric)
client.create_evaluation_function(relevance_metric)

# Create a dataset for evaluation
dataset = client.create_dataset(
    name="summarization-test-cases",
    description="Test cases for summarization evaluation",
)

# Add examples to the dataset
test_cases = [
    {
        "text": "The James Webb Space Telescope is a space telescope designed primarily to conduct infrared astronomy. As the largest optical telescope in space, its high resolution and sensitivity allow it to view objects too old, distant, or faint for the Hubble Space Telescope.",
        "max_sentences": 2,
    },
    {
        "text": "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.",
        "max_sentences": 1,
    },
]

for test_case in test_cases:
    client.create_example(
        inputs=test_case,
        dataset_id=dataset.id,
    )

# Run the evaluation
evaluation_results = client.evaluate(
    chain,
    dataset_id=dataset.id,
    evaluators=[sentence_count_metric, relevance_metric],
)

# Print the results
for result in evaluation_results:
    print(f"Example: {result.example.inputs}")
    print(f"Output: {result.run.outputs}")
    print(f"Evaluation Scores: {result.evaluation_results}")
    print("-" * 50)
```

## 🧪 Exercise: Deploy LangChain App

Let's deploy a simple LangChain app using LangServe:

```python
# deploy_app.py
from fastapi import FastAPI
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langserve import add_routes
import os

# Set environment variables
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Initialize the LLM
llm = OpenAI(temperature=0.7)

# Create a prompt template for a travel assistant
template = """
You are a helpful travel assistant. Provide recommendations for {destination} based on the following preferences:
- Budget: {budget}
- Duration: {duration}
- Interests: {interests}

Travel Recommendations:
"""
prompt = PromptTemplate(
    input_variables=["destination", "budget", "duration", "interests"],
    template=template,
)

# Create an LLM chain
chain = LLMChain(llm=llm, prompt=prompt)

# Create a FastAPI app
app = FastAPI(
    title="Travel Assistant API",
    version="1.0",
    description="API for travel recommendations",
)

# Add the chain to the app
add_routes(
    app,
    chain,
    path="/travel-recommendations",
)

# Add a health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
```

Now, let's create a client to test the deployed app:

```python
# test_deployed_app.py
import requests
import json

# Define the API endpoints
base_url = "http://localhost:8000"
health_url = f"{base_url}/health"
recommendations_url = f"{base_url}/travel-recommendations/invoke"

# Test the health check endpoint
print("Testing health check endpoint...")
response = requests.get(health_url)
if response.status_code == 200:
    print("Health check passed:", response.json())
else:
    print("Health check failed:", response.status_code)

# Test the travel recommendations endpoint
print("\nTesting travel recommendations endpoint...")
data = {
    "input": {
        "destination": "Japan",
        "budget": "medium",
        "duration": "2 weeks",
        "interests": "culture, food, nature"
    }
}

response = requests.post(recommendations_url, json=data)
if response.status_code == 200:
    result = response.json()
    print("Travel recommendations:")
    print(result["output"])
else:
    print("Request failed:", response.status_code)
    print(response.text)
```

## 📌 Project: Chain Tracing & Debug Interface

Let's build a comprehensive chain tracing and debug interface using LangSmith:

```python
# tracing_interface.py
from langsmith import Client
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import os

class ChainTracingInterface:
    def __init__(self, project_name=None):
        """Initialize the tracing interface."""
        # Set environment variables
        os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-api-key"
        
        # Initialize LangSmith client
        self.client = Client()
        
        # Set the project name
        self.project_name = project_name
        
        # Cache for storing runs
        self.runs_cache = []
        self.last_refresh_time = None
    
    def set_project(self, project_name):
        """Set the project to trace."""
        self.project_name = project_name
        self.runs_cache = []
        self.last_refresh_time = None
    
    def refresh_runs(self, force=False):
        """Refresh the runs cache."""
        if not self.project_name:
            print("No project set. Use set_project() to set a project.")
            return
        
        # Only refresh if forced or if it's been more than 5 minutes
        current_time = datetime.now()
        if not force and self.last_refresh_time and (current_time - self.last_refresh_time).total_seconds() < 300:
            return
        
        print(f"Refreshing runs for project: {self.project_name}")
        
        # Get runs from the last 24 hours
        runs = self.client.list_runs(
            project_name=self.project_name,
            start_time=datetime.now() - timedelta(days=1),
            execution_order="desc",
        )
        
        # Update the cache
        self.runs_cache = list(runs)
        self.last_refresh_time = current_time
        
        print(f"Retrieved {len(self.runs_cache)} runs")
    
    def get_run_statistics(self):
        """Get statistics about the runs."""
        if not self.runs_cache:
            self.refresh_runs()
        
        if not self.runs_cache:
            return None
        
        # Calculate statistics
        total_runs = len(self.runs_cache)
        successful_runs = sum(1 for run in self.runs_cache if run.status == "success")
        error_runs = sum(1 for run in self.runs_cache if run.status == "error")
        
        execution_times = []
        for run in self.runs_cache:
            if run.end_time and run.start_time:
                execution_times.append((run.end_time - run.start_time).total_seconds())
        
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        max_execution_time = max(execution_times) if execution_times else 0
        min_execution_time = min(execution_times) if execution_times else 0
        
        return {
            "total_runs": total_runs,
            "successful_runs": successful_runs,
            "error_runs": error_runs,
            "success_rate": (successful_runs / total_runs * 100) if total_runs > 0 else 0,
            "avg_execution_time": avg_execution_time,
            "max_execution_time": max_execution_time,
            "min_execution_time": min_execution_time,
        }
    
    def display_statistics(self):
        """Display run statistics."""
        stats = self.get_run_statistics()
        
        if not stats:
            print("No statistics available.")
            return
        
        print("\nRun Statistics")
        print("-" * 50)
        print(f"Total runs: {stats['total_runs']}")
        print(f"Successful runs: {stats['successful_runs']}")
        print(f"Error runs: {stats['error_runs']}")
        print(f"Success rate: {stats['success_rate']:.1f}%")
        print(f"Average execution time: {stats['avg_execution_time']:.2f} seconds")
        print(f"Max execution time: {stats['max_execution_time']:.2f} seconds")
        print(f"Min execution time: {stats['min_execution_time']:.2f} seconds")
    
    def get_recent_errors(self, limit=10):
        """Get recent error runs."""
        if not self.runs_cache:
            self.refresh_runs()
        
        errors = [run for run in self.runs_cache if run.status == "error"]
        return errors[:limit]
    
    def display_recent_errors(self, limit=10):
        """Display recent errors."""
        errors = self.get_recent_errors(limit)
        
        if not errors:
            print("No recent errors found.")
            return
        
        print(f"\nRecent Errors (last {min(limit, len(errors))})")
        print("-" * 50)
        
        for error in errors:
            print(f"Run ID: {error.id}")
            print(f"Name: {error.name}")
            print(f"Error: {error.error[:100]}...")
            print(f"Time: {error.start_time}")
            print("-" * 30)
    
    def get_slowest_runs(self, limit=10):
        """Get the slowest runs."""
        if not self.runs_cache:
            self.refresh_runs()
        
        # Filter runs with execution times
        runs_with_times = []
        for run in self.runs_cache:
            if run.end_time and run.start_time:
                execution_time = (run.end_time - run.start_time).total_seconds()
                runs_with_times.append((run, execution_time))
        
        # Sort by execution time (descending)
        runs_with_times.sort(key=lambda x: x[1], reverse=True)
        
        return [run for run, _ in runs_with_times[:limit]]
    
    def display_slowest_runs(self, limit=10):
        """Display the slowest runs."""
        slowest_runs = self.get_slowest_runs(limit)
        
        if not slowest_runs:
            print("No runs with execution times found.")
            return
        
        print(f"\nSlowest Runs (last {min(limit, len(slowest_runs))})")
        print("-" * 50)
        
        for run in slowest_runs:
            execution_time = (run.end_time - run.start_time).total_seconds()
            print(f"Run ID: {run.id}")
            print(f"Name: {run.name}")
            print(f"Execution time: {execution_time:.2f} seconds")
            print(f"Time: {run.start_time}")
            print("-" * 30)
    
    def plot_execution_times(self):
        """Plot execution times over time."""
        if not self.runs_cache:
            self.refresh_runs()
        
        # Filter runs with execution times
        runs_with_times = []
        for run in self.runs_cache:
            if run.end_time and run.start_time:
                execution_time = (run.end_time - run.start_time).total_seconds()
                runs_with_times.append((run.start_time, execution_time))
        
        if not runs_with_times:
            print("No runs with execution times found.")
            return
        
        # Sort by start time
        runs_with_times.sort(key=lambda x: x[0])
        
        # Extract times and execution times
        times = [rt[0] for rt in runs_with_times]
        execution_times = [rt[1] for rt in runs_with_times]
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        plt.plot(times, execution_times, 'o-')
        plt.title('Execution Times Over Time')
        plt.xlabel('Time')
        plt.ylabel('Execution Time (seconds)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_status_distribution(self):
        """Plot the distribution of run statuses."""
        if not self.runs_cache:
            self.refresh_runs()
        
        # Count statuses
        status_counts = {}
        for run in self.runs_cache:
            status = run.status
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Create the plot
        plt.figure(figsize=(8, 6))
        plt.pie(status_counts.values(), labels=status_counts.keys(), autopct='%1.1f%%')
        plt.title('Run Status Distribution')
        plt.tight_layout()
        plt.show()
    
    def get_run_details(self, run_id):
        """Get details for a specific run."""
        try:
            run = self.client.read_run(run_id)
            return run
        except Exception as e:
            print(f"Error getting run details: {str(e)}")
            return None
    
    def display_run_details(self, run_id):
        """Display details for a specific run."""
        run = self.get_run_details(run_id)
        
        if not run:
            return
        
        print(f"\nRun Details")
        print("-" * 50)
        print(f"Run ID: {run.id}")
        print(f"Name: {run.name}")
        print(f"Status: {run.status}")
        print(f"Start Time: {run.start_time}")
        print(f"End Time: {run.end_time}")
        
        if run.end_time and run.start_time:
            execution_time = (run.end_time - run.start_time).total_seconds()
            print(f"Execution Time: {execution_time:.2f} seconds")
        
        print(f"Inputs: {run.inputs}")
        print(f"Outputs: {run.outputs}")
        
        if run.error:
            print(f"Error: {run.error}")
    
    def run_interactive_interface(self):
        """Run an interactive interface for exploring runs."""
        print("Chain Tracing & Debug Interface")
        print("-" * 50)
        
        if not self.project_name:
            project_name = input("Enter project name: ")
            self.set_project(project_name)
        
        self.refresh_runs()
        
        while True:
            print("\nOptions:")
            print("1. Display statistics")
            print("2. Display recent errors")
            print("3. Display slowest runs")
            print("4. Plot execution times")
            print("5. Plot status distribution")
            print("6. Get run details")
            print("7. Refresh runs")
            print("8. Change project")
            print("9. Exit")
            
            choice = input("\nEnter your choice: ")
            
            if choice == "1":
                self.display_statistics()
            elif choice == "2":
                limit = input("Enter number of errors to display (default 10): ") or "10"
                self.display_recent_errors(int(limit))
            elif choice == "3":
                limit = input("Enter number of slowest runs to display (default 10): ") or "10"
                self.display_slowest_runs(int(limit))
            elif choice == "4":
                self.plot_execution_times()
            elif choice == "5":
                self.plot_status_distribution()
            elif choice == "6":
                run_id = input("Enter run ID: ")
                self.display_run_details(run_id)
            elif choice == "7":
                self.refresh_runs(force=True)
            elif choice == "8":
                project_name = input("Enter new project name: ")
                self.set_project(project_name)
                self.refresh_runs()
            elif choice == "9":
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please try again.")

# Create and run the interface
if __name__ == "__main__":
    interface = ChainTracingInterface()
    interface.run_interactive_interface()
```

This tracing and debug interface provides:
1. Statistics about chain runs (success rate, execution times, etc.)
2. Recent error tracking
3. Identification of slowest runs
4. Visualization of execution times over time
5. Visualization of status distribution
6. Detailed view of individual runs
7. Interactive interface for exploring runs

