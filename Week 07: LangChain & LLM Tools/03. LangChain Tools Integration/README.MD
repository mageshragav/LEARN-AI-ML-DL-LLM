# Day 33 – LangChain Tools Integration

## What is LangChain Tools Integration?

LangChain Tools Integration is a powerful feature that allows AI agents to interact with external systems, APIs, databases, and other computational resources. It enables agents to perform actions beyond generating text, such as calling APIs, running code, accessing databases, performing calculations, and retrieving information from external sources.

This integration is crucial for building practical AI applications that can solve real-world problems by leveraging external tools and data sources. Instead of being limited to pattern matching and text generation, agents can take actions, retrieve fresh information, and perform complex tasks.

## Essential Subtopics

### 1. Tool Calling Fundamentals

Tool calling is the mechanism by which an LLM can request to execute specific functions or tools. The LLM determines when to use a tool, which tool to use, and what parameters to pass to the tool.

**PyTorch Implementation (Conceptual - using LangChain with PyTorch for custom tool logic):**

```python
import torch
import torch.nn as nn
import numpy as np
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Type
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
import requests
import pandas as pd

class CalculatorTool(BaseTool):
    """A tool for performing mathematical calculations using PyTorch"""
    
    name = "calculator"
    description = "Useful for performing mathematical calculations"
    
    def _run(
        self,
        expression: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the calculator."""
        try:
            # Example: Use PyTorch for tensor operations
            # For complex calculations, we can leverage PyTorch's tensor operations
            if "torch" in expression:
                # This is a simplified example - in practice, you'd need to safely evaluate
                # expressions to prevent code injection
                pass
            
            # For basic calculations, use eval safely
            allowed_chars = set('0123456789+-*/().% ')
            if all(c in allowed_chars for c in expression):
                result = eval(expression)
                return f"Result: {result}"
            else:
                return "Error: Invalid characters in expression"
        except Exception as e:
            return f"Error: {str(e)}"

    async def _arun(
        self,
        expression: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the calculator asynchronously."""
        return self._run(expression, run_manager=run_manager)

class PyTorchTensorTool(BaseTool):
    """A tool for performing PyTorch tensor operations"""
    
    name = "pytorch_tensor_operations"
    description = "Useful for performing PyTorch tensor operations and calculations"
    
    def _run(
        self,
        operation: str,
        tensor_data: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Perform PyTorch tensor operations."""
        try:
            # Convert string data to tensor
            tensor_list = [float(x) for x in tensor_data.split(',')]
            tensor = torch.tensor(tensor_list)
            
            if operation == "mean":
                result = torch.mean(tensor).item()
            elif operation == "sum":
                result = torch.sum(tensor).item()
            elif operation == "std":
                result = torch.std(tensor).item()
            elif operation == "max":
                result = torch.max(tensor).item()
            elif operation == "min":
                result = torch.min(tensor).item()
            else:
                return f"Unknown operation: {operation}"
            
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"

    async def _arun(
        self,
        operation: str,
        tensor_data: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tensor operations asynchronously."""
        return self._run(operation, tensor_data, run_manager=run_manager)

# Example of creating a custom tool with PyTorch integration
class PyTorchModelInferenceTool(BaseTool):
    """A tool for running PyTorch model inference"""
    
    name = "pytorch_model_inference"
    description = "Useful for running inference on a pre-trained PyTorch model"
    
    def __init__(self):
        super().__init__()
        # Initialize a simple PyTorch model for demonstration
        self.model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        # Load pre-trained weights if available
        # self.model.load_state_dict(torch.load("model_weights.pth"))
        self.model.eval()
    
    def _run(
        self,
        input_data: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Run PyTorch model inference."""
        try:
            # Convert input string to tensor
            input_list = [float(x) for x in input_data.split(',')]
            if len(input_list) != 10:
                return f"Error: Expected 10 features, got {len(input_list)}"
            
            input_tensor = torch.tensor(input_list, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                output = self.model(input_tensor)
                prediction = output.item()
            
            return f"Model prediction: {prediction}"
        except Exception as e:
            return f"Error during inference: {str(e)}"

    async def _arun(
        self,
        input_data: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Run PyTorch model inference asynchronously."""
        return self._run(input_data, run_manager=run_manager)

# Initialize tools
calculator_tool = CalculatorTool()
tensor_tool = PyTorchTensorTool()
model_inference_tool = PyTorchModelInferenceTool()

print("Tools initialized:")
print(f"- {calculator_tool.name}: {calculator_tool.description}")
print(f"- {tensor_tool.name}: {tensor_tool.description}")
print(f"- {model_inference_tool.name}: {model_inference_tool.description}")
```

### 2. Retrieval Chains

Retrieval chains combine the power of document retrieval with language models to provide contextually relevant responses based on external knowledge sources.

**PyTorch Implementation:**

```python
import torch
import torch.nn.functional as F
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
import numpy as np
from typing import List

class PyTorchRetrievalChain:
    """A retrieval chain that uses PyTorch for similarity calculations"""
    
    def __init__(self, documents: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.documents = documents
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        
        # Create document embeddings using PyTorch
        self.doc_embeddings = self._create_document_embeddings()
        
        # Create FAISS index
        self.vectorstore = FAISS.from_texts(documents, self.embeddings)
    
    def _create_document_embeddings(self):
        """Create embeddings for all documents using PyTorch-compatible embeddings"""
        embeddings_list = []
        for doc in self.documents:
            embedding = self.embeddings.embed_query(doc)
            embeddings_list.append(embedding)
        
        # Convert to PyTorch tensor
        return torch.tensor(embeddings_list, dtype=torch.float32)
    
    def retrieve_similar_documents(self, query: str, k: int = 3):
        """Retrieve similar documents using PyTorch-based similarity"""
        # Get query embedding
        query_embedding = torch.tensor(self.embeddings.embed_query(query), dtype=torch.float32).unsqueeze(0)
        
        # Calculate cosine similarity using PyTorch
        similarities = F.cosine_similarity(query_embedding, self.doc_embeddings)
        
        # Get top-k most similar documents
        top_k_indices = torch.topk(similarities, k).indices
        top_k_similarities = torch.topk(similarities, k).values
        
        results = []
        for idx, sim in zip(top_k_indices, top_k_similarities):
            results.append({
                'document': self.documents[idx.item()],
                'similarity': sim.item()
            })
        
        return results
    
    def run_query(self, query: str):
        """Run a query against the retrieval chain"""
        # Use the FAISS vectorstore for retrieval
        docs = self.vectorstore.similarity_search(query, k=3)
        
        # Combine documents for context
        context = "\n".join([doc.page_content for doc in docs])
        
        return {
            'query': query,
            'context': context,
            'documents': docs
        }

# Example usage
sample_documents = [
    "The stock market is a complex system that involves buying and selling shares of companies.",
    "Machine learning is a subset of artificial intelligence that focuses on algorithms that learn from data.",
    "Climate change refers to long-term shifts in global temperatures and weather patterns.",
    "Deep learning uses neural networks with multiple layers to model complex patterns in data.",
    "Natural language processing enables computers to understand and generate human language."
]

retrieval_chain = PyTorchRetrievalChain(sample_documents)
print("Retrieval chain initialized with sample documents")

# Test retrieval
query = "What is machine learning?"
results = retrieval_chain.retrieve_similar_documents(query)
print(f"\nQuery: {query}")
print("Top similar documents:")
for i, result in enumerate(results):
    print(f"{i+1}. Similarity: {result['similarity']:.3f}")
    print(f"   Document: {result['document'][:100]}...")
```

### 3. Tool Integration with LLMs

Combining tools with language models to create intelligent agents that can reason and act.

**PyTorch Implementation:**

```python
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.agents.agent_toolkits import create_react_agent
from langchain.tools import Tool
import json

class ToolBasedAgent:
    """An agent that uses tools to answer questions"""
    
    def __init__(self, tools: List[BaseTool], llm_model: str = "gpt-3.5-turbo"):
        self.tools = tools
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        
        # Initialize the agent with tools
        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def run_query(self, query: str):
        """Run a query through the tool-based agent"""
        try:
            result = self.agent.run(query)
            return result
        except Exception as e:
            return f"Error: {str(e)}"

# Create a more complex tool for data analysis
class PyTorchDataAnalysisTool(BaseTool):
    """A tool for performing data analysis with PyTorch"""
    
    name = "pytorch_data_analysis"
    description = "Useful for performing data analysis tasks like correlation, regression, etc."
    
    def _run(
        self,
        data_str: str,
        operation: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Perform data analysis using PyTorch."""
        try:
            # Parse data string (expected format: "x1,y1;x2,y2;...")
            data_pairs = data_str.split(';')
            x_values = []
            y_values = []
            
            for pair in data_pairs:
                x, y = pair.split(',')
                x_values.append(float(x))
                y_values.append(float(y))
            
            x_tensor = torch.tensor(x_values, dtype=torch.float32)
            y_tensor = torch.tensor(y_values, dtype=torch.float32)
            
            if operation == "correlation":
                # Calculate correlation coefficient
                x_mean = torch.mean(x_tensor)
                y_mean = torch.mean(y_tensor)
                numerator = torch.sum((x_tensor - x_mean) * (y_tensor - y_mean))
                denominator = torch.sqrt(torch.sum((x_tensor - x_mean)**2) * torch.sum((y_tensor - y_mean)**2))
                correlation = numerator / denominator
                return f"Correlation coefficient: {correlation.item():.4f}"
            
            elif operation == "mean_difference":
                return f"Mean difference (x - y): {(torch.mean(x_tensor) - torch.mean(y_tensor)).item():.4f}"
            
            else:
                return f"Unknown operation: {operation}"
                
        except Exception as e:
            return f"Error: {str(e)}"

    async def _arun(
        self,
        data_str: str,
        operation: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Run data analysis asynchronously."""
        return self._run(data_str, operation, run_manager=run_manager)

# Initialize tools for the agent
tools = [
    calculator_tool,
    tensor_tool,
    model_inference_tool,
    PyTorchDataAnalysisTool()
]

# Create the agent
agent = ToolBasedAgent(tools)
print("Tool-based agent initialized with multiple PyTorch-integrated tools")
```

## When and Where We Use LangChain Tools Integration

LangChain Tools Integration is used in the following scenarios:

1. **Research and Analysis**: When agents need to access current information, research papers, or databases to provide accurate answers.

2. **Data Processing**: When agents need to perform calculations, analyze datasets, or run statistical operations.

3. **API Integration**: When agents need to interact with external services like weather APIs, financial data providers, or social media platforms.

4. **File Processing**: When agents need to read, write, or process files of various formats.

5. **Complex Problem Solving**: When problems require multiple steps involving different tools and data sources.

6. **Real-time Information**: When agents need access to live data like stock prices, news, or weather.

**Code Example:**

```python
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import SerpAPIWrapper

# Example: Using tools for research
def create_research_agent():
    # Web search tool
    search_tool = DuckDuckGoSearchRun()
    
    # Calculator tool (from our implementation above)
    calc_tool = calculator_tool
    
    # Combine tools
    tools = [
        Tool(
            name="web_search",
            func=search_tool.run,
            description="Useful for searching the web for current information"
        ),
        calc_tool
    ]
    
    research_agent = ToolBasedAgent(tools)
    return research_agent

# Example usage
research_agent = create_research_agent()
print("Research agent created with web search capabilities")
```

## Real-World Applications and Examples

### 1. Financial Analysis Assistant

```python
# Example: Financial analysis using multiple tools
class FinancialAnalysisTool(BaseTool):
    """A tool for financial calculations and analysis"""
    
    name = "financial_analyzer"
    description = "Useful for financial calculations like compound interest, ROI, etc."
    
    def _run(
        self,
        principal: float,
        rate: float,
        time: float,
        operation: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Perform financial calculations."""
        try:
            if operation == "compound_interest":
                amount = principal * (1 + rate/100) ** time
                return f"Compound interest result: {amount:.2f}"
            elif operation == "simple_interest":
                interest = (principal * rate * time) / 100
                return f"Simple interest: {interest:.2f}"
            else:
                return f"Unknown operation: {operation}"
        except Exception as e:
            return f"Error: {str(e)}"

    async def _arun(
        self,
        principal: float,
        rate: float,
        time: float,
        operation: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Run financial analysis asynchronously."""
        return self._run(principal, rate, time, operation, run_manager=run_manager)

# Create financial analysis agent
financial_tools = [
    FinancialAnalysisTool(),
    calculator_tool
]

financial_agent = ToolBasedAgent(financial_tools)
print("Financial analysis agent created")
```

### 2. Scientific Calculator Agent

```python
import math

class ScientificCalculatorTool(BaseTool):
    """A tool for scientific calculations"""
    
    name = "scientific_calculator"
    description = "Useful for scientific calculations including trigonometry, logarithms, etc."
    
    def _run(
        self,
        expression: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Perform scientific calculations."""
        try:
            # Safely evaluate scientific expressions
            allowed_names = {
                k: v for k, v in math.__dict__.items() if not k.startswith("__")
            }
            allowed_names.update({"abs": abs, "round": round})
            
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return f"Scientific calculation result: {result}"
        except Exception as e:
            return f"Error in scientific calculation: {str(e)}"

    async def _arun(
        self,
        expression: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Run scientific calculation asynchronously."""
        return self._run(expression, run_manager=run_manager)

# Example usage
scientific_tools = [
    ScientificCalculatorTool(),
    PyTorchTensorTool()
]

scientific_agent = ToolBasedAgent(scientific_tools)
print("Scientific calculator agent created")
```

## Common Problems and How Advancements Solved Them

### Problem 1: Tool Selection and Coordination

**Issue**: Early tool integration systems struggled with determining which tool to use and when to use it, leading to inefficient or incorrect responses.

**Advancement**: Modern agents use sophisticated reasoning mechanisms to determine tool usage, including:
- Chain-of-thought prompting for better reasoning
- Tool calling APIs that provide structured output
- Multi-step planning capabilities

**Problem Code:**
```python
# Problem: Simple tool selection without proper reasoning
def simple_tool_selector(query, tools):
    """Simple tool selection without reasoning - PROBLEMATIC"""
    if "calculate" in query.lower():
        return tools[0]  # Always use calculator
    elif "search" in query.lower():
        return tools[1]  # Always use search
    else:
        return None  # No tool selected
```

**Enhanced Code:**
```python
class AdvancedToolSelector:
    """Advanced tool selection with reasoning"""
    
    def __init__(self, tools):
        self.tools = {tool.name: tool for tool in tools}
        self.tool_descriptions = {tool.name: tool.description for tool in tools}
    
    def select_and_run_tools(self, query):
        """Select and run appropriate tools based on query analysis"""
        # Analyze query to determine required tools
        required_tools = self._analyze_query(query)
        
        results = []
        for tool_name in required_tools:
            if tool_name in self.tools:
                tool = self.tools[tool_name]
                # Extract parameters from query
                params = self._extract_parameters(query, tool)
                try:
                    result = tool._run(**params) if params else tool._run(query)
                    results.append({
                        'tool': tool_name,
                        'result': result,
                        'query': query
                    })
                except Exception as e:
                    results.append({
                        'tool': tool_name,
                        'error': f"Error: {str(e)}",
                        'query': query
                    })
        
        return results
    
    def _analyze_query(self, query):
        """Analyze query to determine which tools are needed"""
        # This would use more sophisticated NLP in practice
        tools_needed = []
        
        if any(word in query.lower() for word in ['calculate', 'compute', 'math', 'add', 'subtract', 'multiply', 'divide']):
            tools_needed.append('calculator')
        
        if any(word in query.lower() for word in ['mean', 'average', 'sum', 'std', 'standard deviation']):
            tools_needed.append('pytorch_tensor_operations')
        
        if any(word in query.lower() for word in ['predict', 'model', 'inference']):
            tools_needed.append('pytorch_model_inference')
        
        return tools_needed or list(self.tools.keys())[:1]  # Default to first tool if none found
    
    def _extract_parameters(self, query, tool):
        """Extract parameters from query for tool execution"""
        # This would use more sophisticated parameter extraction in practice
        if tool.name == 'calculator':
            # Extract mathematical expression
            import re
            numbers = re.findall(r'\d+\.?\d*', query)
            if len(numbers) >= 1:
                return {'expression': query}
        elif tool.name == 'pytorch_tensor_operations':
            # Extract operation and data
            if 'mean' in query.lower():
                return {'operation': 'mean', 'tensor_data': '1,2,3,4,5'}
            elif 'sum' in query.lower():
                return {'operation': 'sum', 'tensor_data': '1,2,3,4,5'}
        
        return {}

# Initialize advanced tool selector
advanced_selector = AdvancedToolSelector(tools)
print("Advanced tool selector initialized")
```

### Problem 2: Context Management and Memory

**Issue**: Tools often operated in isolation without maintaining context from previous interactions, leading to fragmented responses.

**Advancement**: Modern systems incorporate:
- Memory mechanisms to maintain conversation context
- State management for multi-step processes
- Context-aware tool execution

**Problem Code:**
```python
# Problem: Tools don't maintain context
class ContextlessToolAgent:
    """Tool agent without context management - PROBLEMATIC"""
    
    def __init__(self, tools):
        self.tools = tools
    
    def run(self, query):
        """Run query without considering context"""
        for tool in self.tools:
            if self._should_use_tool(query, tool):
                return tool._run(query)
        return "No suitable tool found"
    
    def _should_use_tool(self, query, tool):
        return True  # Always try to use tools without context
```

**Enhanced Code:**
```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

class ContextAwareToolAgent:
    """Tool agent with context management and memory"""
    
    def __init__(self, tools, llm):
        self.tools = {tool.name: tool for tool in tools}
        self.llm = llm
        self.memory = ConversationBufferMemory()
        self.conversation_history = []
    
    def run(self, query):
        """Run query with context awareness"""
        # Add current query to history
        self.conversation_history.append({"role": "user", "content": query})
        
        # Analyze if tools are needed based on context
        tool_usage = self._determine_tool_usage(query)
        
        if tool_usage:
            # Execute tools with context
            tool_results = []
            for tool_name, params in tool_usage.items():
                if tool_name in self.tools:
                    tool = self.tools[tool_name]
                    try:
                        result = tool._run(**params) if params else tool._run(query)
                        tool_results.append({
                            'tool': tool_name,
                            'result': result,
                            'query': query
                        })
                    except Exception as e:
                        tool_results.append({
                            'tool': tool_name,
                            'error': f"Error: {str(e)}",
                            'query': query
                        })
            
            # Combine tool results with conversation
            combined_result = self._combine_results(query, tool_results)
            
            # Add to memory
            self.conversation_history.append({"role": "assistant", "content": combined_result})
            
            return combined_result
        else:
            # Use LLM directly for general conversation
            context = self._get_context()
            response = f"Context: {context}\nQuery: {query}\nResponse: [LLM response would go here]"
            self.conversation_history.append({"role": "assistant", "content": response})
            return response
    
    def _determine_tool_usage(self, query):
        """Determine which tools to use based on query and context"""
        # Analyze query in context of conversation history
        context = self._get_context()
        
        tool_usage = {}
        
        if any(word in query.lower() for word in ['calculate', 'compute', 'math']):
            tool_usage['calculator'] = {'expression': self._extract_math_expression(query)}
        
        if any(word in query.lower() for word in ['mean', 'average', 'sum']):
            tool_usage['pytorch_tensor_operations'] = {
                'operation': 'mean' if 'mean' in query.lower() else 'sum',
                'tensor_data': '1,2,3,4,5'  # This would come from context or user input
            }
        
        return tool_usage
    
    def _extract_math_expression(self, query):
        """Extract mathematical expression from query"""
        # Simplified extraction - in practice, this would be more sophisticated
        import re
        expression = re.sub(r'[^\d+\-*/().\s]', '', query)
        return expression.strip() or query
    
    def _get_context(self):
        """Get recent conversation context"""
        recent_messages = self.conversation_history[-5:]  # Last 5 messages
        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_messages])
        return context
    
    def _combine_results(self, query, tool_results):
        """Combine tool results into coherent response"""
        if len(tool_results) == 1:
            return tool_results[0]['result']
        else:
            combined = "Tool Results:\n"
            for result in tool_results:
                combined += f"- {result['tool']}: {result['result']}\n"
            return combined

# Example usage
context_agent = ContextAwareToolAgent(tools, None)  # Using None for LLM in this example
print("Context-aware tool agent initialized")
```

### Problem 3: Error Handling and Robustness

**Issue**: Tools often failed catastrophically when encountering unexpected inputs or external service failures, breaking the entire conversation flow.

**Advancement**: Modern systems include:
- Comprehensive error handling and fallback mechanisms
- Graceful degradation when tools fail
- Input validation and sanitization
- Retry mechanisms for external services

**Problem Code:**
```python
# Problem: No error handling
def fragile_tool_caller(tool, query):
    """Tool caller without error handling - FRAGILE"""
    return tool._run(query)  # Will crash if tool fails
```

**Enhanced Code:**
```python
import time
import logging
from typing import Optional, Dict, Any

class RobustToolCaller:
    """Robust tool calling with error handling and retry mechanisms"""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logging.getLogger(__name__)
    
    def call_tool_with_retry(self, tool: BaseTool, params: Dict[str, Any]) -> str:
        """Call tool with retry mechanism and error handling"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Attempt {attempt + 1} for tool {tool.name}")
                
                # Validate parameters
                validated_params = self._validate_parameters(tool, params)
                
                # Call the tool
                result = tool._run(**validated_params)
                
                self.logger.info(f"Tool {tool.name} succeeded on attempt {attempt + 1}")
                return result
                
            except Exception as e:
                last_error = e
                self.logger.warning(f"Tool {tool.name} failed on attempt {attempt + 1}: {str(e)}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    self.logger.error(f"All {self.max_retries} attempts failed for tool {tool.name}")
        
        # All retries failed, return error message
        error_msg = f"Tool {tool.name} failed after {self.max_retries} attempts. Last error: {str(last_error)}"
        self.logger.error(error_msg)
        return f"Error: {error_msg}"
    
    def _validate_parameters(self, tool: BaseTool, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize tool parameters"""
        # This would implement validation based on tool requirements
        # For now, just return the parameters
        return params
    
    def call_multiple_tools(self, tools: List[tuple]) -> Dict[str, str]:
        """Call multiple tools safely"""
        results = {}
        
        for tool, params in tools:
            try:
                result = self.call_tool_with_retry(tool, params)
                results[tool.name] = result
            except Exception as e:
                results[tool.name] = f"Error: {str(e)}"
        
        return results

# Initialize robust tool caller
robust_caller = RobustToolCaller(max_retries=3, retry_delay=1.0)
print("Robust tool caller initialized with error handling")
```

## 🧪 Exercise: Tool-based Chatbot

```python
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
import asyncio

class ToolBasedChatbot:
    """A chatbot that uses tools to answer questions"""
    
    def __init__(self, tools: List[BaseTool]):
        self.tools = tools
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        # Create the agent
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use the provided tools to answer questions."),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        agent = create_tool_calling_agent(self.llm, tools, prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    def chat(self, message: str, history: list = None):
        """Chat with the tool-based chatbot"""
        if history is None:
            history = []
        
        result = self.agent_executor.invoke({
            "input": message,
            "chat_history": history
        })
        
        return result["output"]

# Create and test the tool-based chatbot
chatbot_tools = [
    calculator_tool,
    tensor_tool,
    PyTorchDataAnalysisTool()
]

chatbot = ToolBasedChatbot(chatbot_tools)
print("Tool-based chatbot initialized")

# Example interactions
print("\n=== Tool-Based Chatbot Demo ===")
try:
    response1 = chatbot.chat("Calculate 25 * 4 + 10")
    print(f"Response: {response1}")
    
    response2 = chatbot.chat("Find the mean of 1, 2, 3, 4, 5 using PyTorch")
    print(f"Response: {response2}")
    
    response3 = chatbot.chat("Analyze the correlation between these values: 1,2; 2,4; 3,6; 4,8")
    print(f"Response: {response3}")
    
except Exception as e:
    print(f"Error in chatbot demo: {e}")
    print("Note: This requires a valid OpenAI API key to run fully")
```

## 📌 Project: PDF Chatbot with Memory

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import os

class PDFChatbotWithMemory:
    """A chatbot that can answer questions about PDF documents with memory"""
    
    def __init__(self, pdf_path: str, model_name: str = "gpt-3.5-turbo"):
        self.pdf_path = pdf_path
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Load and process PDF
        self._load_pdf()
        
        # Initialize LLM and memory
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create conversational retrieval chain
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(),
            memory=self.memory,
            verbose=True
        )
    
    def _load_pdf(self):
        """Load and process PDF document"""
        # Load PDF
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        texts = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vectorstore = FAISS.from_documents(texts, self.embeddings)
        print(f"PDF loaded and indexed. Found {len(texts)} text chunks.")
    
    def ask(self, question: str) -> str:
        """Ask a question about the PDF content"""
        try:
            response = self.conversation_chain({"question": question})
            return response["answer"]
        except Exception as e:
            return f"Error processing question: {str(e)}"
    
    def get_conversation_history(self):
        """Get the conversation history"""
        return self.memory.buffer

# Example usage (note: you would need an actual PDF file)
"""
# To use this, uncomment and provide a PDF file path
pdf_chatbot = PDFChatbotWithMemory("path/to/your/document.pdf")

# Example questions
questions = [
    "What is the main topic of this document?",
    "Can you summarize the key points?",
    "What are the conclusions mentioned in the document?"
]

for question in questions:
    answer = pdf_chatbot.ask(question)
    print(f"Q: {question}")
    print(f"A: {answer}\n")

print("Conversation History:")
print(pdf_chatbot.get_conversation_history())
"""

print("PDF Chatbot with Memory structure implemented")
print("Note: To run fully, you would need to provide an actual PDF file path")

# Summary of LangChain Tools Integration
print("\n=== LangChain Tools Integration Summary ===")
print("1. Tool Calling: Mechanism for LLMs to use external functions")
print("2. Retrieval Chains: Combining document retrieval with LLMs")
print("3. Context Management: Maintaining conversation history")
print("4. Error Handling: Robust tool execution with fallbacks")
print("5. Real-world Applications: Research, analysis, automation")
print("6. Advanced Features: Multi-tool coordination, memory integration")
```

This comprehensive implementation demonstrates LangChain Tools Integration with PyTorch integration, covering all essential aspects from basic tool creation to advanced features like memory management and error handling, with practical applications in research, analysis, and document processing.