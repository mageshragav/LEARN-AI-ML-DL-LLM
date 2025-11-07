# Day 43 – LangGraph Workflows

## Graph-based agent flows

### What is it?

Graph-based agent flows represent AI agent workflows as directed graphs where nodes are stateful functions (like tools, LLM calls, or decision points) and edges represent the flow of information or control between these functions. Unlike linear execution, graph-based flows allow for complex, conditional, and iterative behaviors, enabling agents to handle multi-step tasks, incorporate feedback loops, and make dynamic decisions based on intermediate results. This approach provides a more natural and flexible way to model complex AI workflows compared to traditional sequential programming.

### PyTorch Implementation

```python
import asyncio
from typing import Dict, List, Any, Optional, Tuple
import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from enum import Enum
import time
import random

# State management for the graph
@dataclass
class AgentState:
    """State object that flows through the graph"""
    input_query: str = ""
    context: str = ""
    retrieved_docs: List[str] = field(default_factory=list)
    generated_response: str = ""
    search_results: List[str] = field(default_factory=list)
    intermediate_steps: List[Dict] = field(default_factory=list)
    final_answer: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class NodeType(Enum):
    """Types of nodes in the graph"""
    LLM = "llm"
    TOOL = "tool"
    CONDITIONAL = "conditional"
    END = "end"
    START = "start"

@dataclass
class GraphNode:
    """Represents a node in the agent workflow graph"""
    name: str
    node_type: NodeType
    function: callable
    dependencies: List[str] = field(default_factory=list)
    
    def execute(self, state: AgentState) -> AgentState:
        """Execute the node function with the current state"""
        try:
            return self.function(state)
        except Exception as e:
            state.error = str(e)
            return state

class LangGraph:
    """Graph-based agent workflow engine"""
    
    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.graph = nx.DiGraph()
        self.edges: List[Tuple[str, str]] = []
    
    def add_node(self, name: str, node_type: NodeType, function: callable, dependencies: List[str] = None):
        """Add a node to the workflow graph"""
        node = GraphNode(name, node_type, function, dependencies or [])
        self.nodes[name] = node
        self.graph.add_node(name)
        
        # Add dependencies as edges
        if dependencies:
            for dep in dependencies:
                self.add_edge(dep, name)
    
    def add_edge(self, from_node: str, to_node: str):
        """Add an edge between nodes"""
        if from_node in self.nodes and to_node in self.nodes:
            self.graph.add_edge(from_node, to_node)
            self.edges.append((from_node, to_node))
    
    def get_execution_order(self) -> List[str]:
        """Get the topological order of node execution"""
        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            # If there are cycles, use a different approach
            return self._handle_cycles()
    
    def _handle_cycles(self) -> List[str]:
        """Handle cyclic dependencies in the graph"""
        # For simplicity, return a basic execution order
        return list(self.nodes.keys())
    
    def execute(self, initial_state: AgentState) -> AgentState:
        """Execute the graph workflow"""
        execution_order = self.get_execution_order()
        state = initial_state
        
        print(f"Execution order: {execution_order}")
        
        for node_name in execution_order:
            if node_name in self.nodes:
                print(f"Executing node: {node_name}")
                node = self.nodes[node_name]
                state = node.execute(state)
                
                # If there's an error, stop execution
                if state.error:
                    print(f"Error in node {node_name}: {state.error}")
                    break
        
        return state

# Example node functions for the graph
def retrieve_documents(state: AgentState) -> AgentState:
    """Node function to retrieve relevant documents"""
    print(f"Retrieving documents for query: {state.input_query[:50]}...")
    
    # Simulate document retrieval
    mock_docs = [
        f"Document about {state.input_query} - This is relevant information.",
        f"Another document related to {state.input_query} - Contains useful details.",
        f"Third document for {state.input_query} - Provides additional context."
    ]
    
    state.retrieved_docs = mock_docs
    state.context = " ".join(mock_docs)
    state.intermediate_steps.append({
        "step": "document_retrieval",
        "query": state.input_query,
        "num_docs": len(mock_docs)
    })
    
    return state

def generate_response(state: AgentState) -> AgentState:
    """Node function to generate response using LLM"""
    print("Generating response based on context...")
    
    # Simulate LLM response generation
    response = f"Based on the provided context, here's my response to your query about '{state.input_query}'. "
    response += f"I found {len(state.retrieved_docs)} relevant documents. "
    response += "The key points are that this is a well-documented topic with multiple resources available."
    
    state.generated_response = response
    state.final_answer = response
    state.intermediate_steps.append({
        "step": "response_generation",
        "response_length": len(response)
    })
    
    return state

def validate_response(state: AgentState) -> AgentState:
    """Node function to validate the generated response"""
    print("Validating response quality...")
    
    # Simple validation logic
    if len(state.generated_response) < 20:
        state.error = "Response too short"
        return state
    
    # Add validation metadata
    state.metadata["response_quality"] = "good"
    state.metadata["validation_passed"] = True
    state.intermediate_steps.append({
        "step": "response_validation",
        "quality": "good"
    })
    
    return state

def conditional_routing(state: AgentState) -> AgentState:
    """Node function for conditional routing based on state"""
    print("Performing conditional routing...")
    
    # Example: If the query is about Python, route differently
    if "python" in state.input_query.lower():
        state.metadata["routing_decision"] = "python_specialized"
        print("Routing to Python-specific processing...")
    else:
        state.metadata["routing_decision"] = "general"
    
    state.intermediate_steps.append({
        "step": "conditional_routing",
        "decision": state.metadata["routing_decision"]
    })
    
    return state

# Demonstrate graph-based workflow
def demonstrate_graph_workflow():
    """Demonstrate the graph-based agent workflow"""
    
    print("Creating Graph-based Agent Workflow")
    print("=" * 50)
    
    # Create the graph
    workflow = LangGraph()
    
    # Add nodes to the workflow
    workflow.add_node("start", NodeType.START, lambda state: state)
    workflow.add_node("conditional_routing", NodeType.CONDITIONAL, conditional_routing, ["start"])
    workflow.add_node("retrieve_documents", NodeType.TOOL, retrieve_documents, ["start"])
    workflow.add_node("generate_response", NodeType.LLM, generate_response, ["retrieve_documents"])
    workflow.add_node("validate_response", NodeType.TOOL, validate_response, ["generate_response"])
    
    # Create initial state
    initial_state = AgentState(
        input_query="What is machine learning?",
        metadata={"workflow_id": "demo_001"}
    )
    
    # Execute the workflow
    final_state = workflow.execute(initial_state)
    
    print(f"\nFinal state:")
    print(f"Input: {final_state.input_query}")
    print(f"Response: {final_state.final_answer[:100]}...")
    print(f"Retrieved docs: {len(final_state.retrieved_docs)}")
    print(f"Intermediate steps: {len(final_state.intermediate_steps)}")
    print(f"Metadata: {final_state.metadata}")
    
    # Visualize the graph
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(workflow.graph)
    nx.draw(workflow.graph, pos, with_labels=True, node_color='lightblue', 
            node_size=3000, font_size=10, font_weight='bold', arrows=True)
    plt.title("Agent Workflow Graph")
    plt.show()
    
    return workflow, final_state

workflow, final_state = demonstrate_graph_workflow()
```

### When and where we use graph-based agent flows

- **Complex decision-making**: When agents need to make multiple conditional decisions based on intermediate results
- **Multi-step reasoning**: For tasks requiring multiple reasoning steps with feedback loops
- **Tool orchestration**: When agents need to use various tools in a specific sequence or conditionally
- **Conversational agents**: For handling complex dialog flows with memory and context
- **Research assistants**: For conducting multi-step research with iterative refinement
- **Workflow automation**: For automating complex business processes with multiple decision points

### Real-world examples

- **Customer support agents**: Routing queries to appropriate specialists based on content analysis
- **Legal research assistants**: Following complex research paths based on case law and precedents
- **Medical diagnosis systems**: Following decision trees based on symptoms and test results
- **Financial advisors**: Making investment recommendations based on multiple factors and conditions
- **Content moderation**: Following different review paths based on content type and risk level

### Problems and Advancements

**Problem**: Managing state and ensuring consistency across complex graph executions.
**Advancement**: State management systems with checkpointing and recovery mechanisms.

**Problem**: Debugging and monitoring complex graph-based workflows.
**Advancement**: Visualization tools and comprehensive logging systems for graph execution.

## 🧪 Exercise: Build agent workflow

```python
def build_agent_workflow_exercise():
    """Complete exercise: Build a graph-based agent workflow"""
    
    print("Agent Workflow Exercise")
    print("=" * 40)
    
    # Define the workflow components
    class ResearchWorkflow:
        def __init__(self):
            self.graph = LangGraph()
            self.setup_workflow()
        
        def setup_workflow(self):
            """Setup the research workflow graph"""
            # Node functions
            def query_analysis(state: AgentState) -> AgentState:
                print("Analyzing query...")
                state.metadata["query_complexity"] = "medium"
                state.metadata["query_type"] = "informational"
                state.intermediate_steps.append({"step": "query_analysis"})
                return state
            
            def topic_research(state: AgentState) -> AgentState:
                print("Researching topic...")
                # Simulate research
                research_results = [
                    f"Research finding 1 about {state.input_query}",
                    f"Research finding 2 about {state.input_query}", 
                    f"Research finding 3 about {state.input_query}"
                ]
                state.search_results = research_results
                state.intermediate_steps.append({
                    "step": "topic_research",
                    "findings_count": len(research_results)
                })
                return state
            
            def source_evaluation(state: AgentState) -> AgentState:
                print("Evaluating sources...")
                # Simulate source evaluation
                state.metadata["source_quality"] = "high"
                state.metadata["reliability_score"] = 0.8
                state.intermediate_steps.append({
                    "step": "source_evaluation",
                    "quality": "high"
                })
                return state
            
            def synthesis(state: AgentState) -> AgentState:
                print("Synthesizing information...")
                synthesis = f"Based on research about '{state.input_query}', here's a comprehensive synthesis. "
                synthesis += f"We found {len(state.search_results)} key findings from reliable sources. "
                synthesis += "The main points indicate that this is a well-researched topic with strong evidence."
                
                state.generated_response = synthesis
                state.final_answer = synthesis
                state.intermediate_steps.append({
                    "step": "synthesis",
                    "response_length": len(synthesis)
                })
                return state
            
            def quality_check(state: AgentState) -> AgentState:
                print("Performing quality check...")
                # Simple quality check
                if len(state.generated_response) > 50 and state.metadata.get("reliability_score", 0) > 0.7:
                    state.metadata["quality_check_passed"] = True
                else:
                    state.metadata["quality_check_passed"] = False
                
                state.intermediate_steps.append({
                    "step": "quality_check",
                    "passed": state.metadata["quality_check_passed"]
                })
                return state
            
            # Build the graph
            self.graph.add_node("query_analysis", NodeType.TOOL, query_analysis)
            self.graph.add_node("topic_research", NodeType.TOOL, topic_research, ["query_analysis"])
            self.graph.add_node("source_evaluation", NodeType.TOOL, source_evaluation, ["topic_research"])
            self.graph.add_node("synthesis", NodeType.LLM, synthesis, ["source_evaluation"])
            self.graph.add_node("quality_check", NodeType.TOOL, quality_check, ["synthesis"])
        
        def execute_workflow(self, query: str) -> AgentState:
            """Execute the research workflow"""
            initial_state = AgentState(input_query=query)
            return self.graph.execute(initial_state)
    
    # Create and test the workflow
    research_agent = ResearchWorkflow()
    
    # Test queries
    test_queries = [
        "What are the benefits of machine learning?",
        "How does deep learning work?",
        "What is natural language processing?"
    ]
    
    results = []
    for query in test_queries:
        print(f"\nProcessing query: {query}")
        result = research_agent.execute_workflow(query)
        results.append(result)
        
        print(f"Final answer: {result.final_answer[:100]}...")
        print(f"Metadata: {result.metadata}")
        print(f"Steps taken: {len(result.intermediate_steps)}")
    
    # Analyze workflow performance
    print(f"\nWorkflow Analysis:")
    print(f"Total queries processed: {len(results)}")
    print(f"Average response length: {np.mean([len(r.final_answer) for r in results]):.2f}")
    print(f"Quality check pass rate: {sum(1 for r in results if r.metadata.get('quality_check_passed', False)) / len(results) * 100:.1f}%")
    
    return research_agent, results

# Run the exercise
research_agent, exercise_results = build_agent_workflow_exercise()
```

## 📌 Project: Conversational retrieval agent

```python
class ConversationalRetrievalAgent:
    """Complete conversational retrieval agent using graph-based workflows"""
    
    def __init__(self, knowledge_base=None):
        self.knowledge_base = knowledge_base or self._create_sample_knowledge_base()
        self.conversation_history = []
        self.graph = self._build_conversation_graph()
        
    def _create_sample_knowledge_base(self):
        """Create a sample knowledge base for the agent"""
        return {
            "machine_learning": [
                "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
                "It uses algorithms to parse data, learn from that data, and make informed decisions based on what it has learned.",
                "Common applications include image recognition, natural language processing, and recommendation systems."
            ],
            "deep_learning": [
                "Deep learning is a subset of machine learning that uses neural networks with three or more layers.",
                "These neural networks attempt to simulate the behavior of the human brain, though they are much simpler in their operation.",
                "It's particularly effective for image and speech recognition tasks."
            ],
            "nlp": [
                "Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language.",
                "The ultimate objective of NLP is to read, decipher, understand, and make sense of human languages in a manner that is valuable.",
                "Applications include chatbots, translation services, and sentiment analysis."
            ],
            "computer_vision": [
                "Computer vision is a field of artificial intelligence that trains computers to interpret and understand the visual world.",
                "Using digital images from cameras and videos and deep learning models, machines can accurately identify and classify objects.",
                "Applications include autonomous vehicles, medical image analysis, and facial recognition."
            ]
        }
    
    def _build_conversation_graph(self):
        """Build the conversation workflow graph"""
        graph = LangGraph()
        
        def context_enrichment(state: AgentState) -> AgentState:
            """Enrich context with conversation history"""
            print("Enriching context with conversation history...")
            
            # Add recent conversation history to context
            recent_history = self.conversation_history[-3:]  # Last 3 exchanges
            history_context = " ".join([f"Q: {ex['query']} A: {ex['response']}" for ex in recent_history])
            
            state.context = f"Previous context: {history_context}\nCurrent query: {state.input_query}"
            state.intermediate_steps.append({"step": "context_enrichment", "history_length": len(recent_history)})
            
            return state
        
        def query_understanding(state: AgentState) -> AgentState:
            """Understand the user query and identify intent"""
            print("Understanding query...")
            
            # Simple intent classification
            query_lower = state.input_query.lower()
            if any(word in query_lower for word in ["what is", "define", "explain"]):
                intent = "definition"
            elif any(word in query_lower for word in ["how", "process", "steps"]):
                intent = "process"
            elif any(word in query_lower for word in ["benefit", "advantage", "good"]):
                intent = "benefit"
            else:
                intent = "general"
            
            state.metadata["query_intent"] = intent
            state.intermediate_steps.append({"step": "query_understanding", "intent": intent})
            
            return state
        
        def information_retrieval(state: AgentState) -> AgentState:
            """Retrieve relevant information from knowledge base"""
            print("Retrieving information...")
            
            # Simple keyword-based retrieval
            query_lower = state.input_query.lower()
            retrieved_docs = []
            
            for topic, docs in self.knowledge_base.items():
                if topic.replace("_", " ") in query_lower or topic in query_lower:
                    retrieved_docs.extend(docs)
            
            # If no specific topic found, return general knowledge
            if not retrieved_docs:
                all_docs = []
                for docs in self.knowledge_base.values():
                    all_docs.extend(docs)
                retrieved_docs = all_docs[:3]  # Take first 3 documents
            
            state.retrieved_docs = retrieved_docs
            state.context = state.context + " " + " ".join(retrieved_docs)
            state.intermediate_steps.append({
                "step": "information_retrieval", 
                "docs_retrieved": len(retrieved_docs)
            })
            
            return state
        
        def response_generation(state: AgentState) -> AgentState:
            """Generate response based on context and intent"""
            print("Generating response...")
            
            intent = state.metadata.get("query_intent", "general")
            context = state.context
            query = state.input_query
            
            # Generate response based on intent
            if intent == "definition":
                response = f"Regarding your question about '{query}', here's what I know: "
                response += " ".join(state.retrieved_docs[:2])  # Use first 2 retrieved docs
            elif intent == "process":
                response = f"To understand the process of '{query}', consider this: "
                response += " ".join(state.retrieved_docs[:2])
            elif intent == "benefit":
                response = f"The benefits of '{query}' include: "
                response += " ".join(state.retrieved_docs[:2])
            else:
                response = f"About '{query}': "
                response += " ".join(state.retrieved_docs[:3])
            
            # Add follow-up suggestion
            response += f" Would you like to know more about {query.split()[-1]}?"
            
            state.generated_response = response
            state.final_answer = response
            state.intermediate_steps.append({
                "step": "response_generation",
                "response_length": len(response)
            })
            
            return state
        
        def conversation_tracking(state: AgentState) -> AgentState:
            """Track conversation for context management"""
            print("Tracking conversation...")
            
            # Add to conversation history
            self.conversation_history.append({
                "query": state.input_query,
                "response": state.final_answer,
                "timestamp": time.time()
            })
            
            # Keep only recent history (last 10 exchanges)
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            state.intermediate_steps.append({
                "step": "conversation_tracking",
                "history_length": len(self.conversation_history)
            })
            
            return state
        
        def response_validation(state: AgentState) -> AgentState:
            """Validate response quality"""
            print("Validating response...")
            
            response = state.final_answer
            query = state.input_query
            
            # Simple validation checks
            validation_score = 0
            
            # Check if response contains query terms
            query_words = set(query.lower().split())
            response_words = set(response.lower().split())
            term_overlap = len(query_words.intersection(response_words))
            validation_score += term_overlap * 0.1
            
            # Check response length
            if len(response) > 50:
                validation_score += 0.3
            else:
                validation_score += 0.1
            
            # Check for follow-up suggestions
            if "more" in response.lower() or "know" in response.lower():
                validation_score += 0.2
            
            state.metadata["response_validation_score"] = min(validation_score, 1.0)
            state.intermediate_steps.append({
                "step": "response_validation",
                "validation_score": validation_score
            })
            
            return state
        
        # Build the graph
        graph.add_node("context_enrichment", NodeType.TOOL, context_enrichment)
        graph.add_node("query_understanding", NodeType.TOOL, query_understanding, ["context_enrichment"])
        graph.add_node("information_retrieval", NodeType.TOOL, information_retrieval, ["query_understanding"])
        graph.add_node("response_generation", NodeType.LLM, response_generation, ["information_retrieval"])
        graph.add_node("conversation_tracking", NodeType.TOOL, conversation_tracking, ["response_generation"])
        graph.add_node("response_validation", NodeType.TOOL, response_validation, ["conversation_tracking"])
        
        return graph
    
    def chat(self, user_input: str) -> str:
        """Process user input and return agent response"""
        # Create initial state
        initial_state = AgentState(
            input_query=user_input,
            metadata={"timestamp": time.time()}
        )
        
        # Execute the workflow
        final_state = self.graph.execute(initial_state)
        
        # Return the response
        return final_state.final_answer
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []
    
    def get_conversation_history(self):
        """Get conversation history"""
        return self.conversation_history
    
    def get_workflow_graph(self):
        """Get the workflow graph for visualization"""
        return self.graph

# Demonstrate the conversational retrieval agent
def demonstrate_conversational_agent():
    """Demonstrate the complete conversational retrieval agent"""
    
    print("Conversational Retrieval Agent Demo")
    print("=" * 50)
    
    # Create the agent
    agent = ConversationalRetrievalAgent()
    
    # Simulate a conversation
    conversation_flow = [
        "What is machine learning?",
        "How does it work?",
        "What are its benefits?",
        "Tell me about deep learning",
        "How is it different from machine learning?"
    ]
    
    print("Starting conversation demo...")
    print("-" * 30)
    
    for i, user_input in enumerate(conversation_flow, 1):
        print(f"Round {i}:")
        print(f"User: {user_input}")
        
        response = agent.chat(user_input)
        print(f"Agent: {response}")
        
        print(f"Conversation history length: {len(agent.get_conversation_history())}")
        print("-" * 30)
    
    # Show conversation history
    print("\nFull Conversation History:")
    for i, exchange in enumerate(agent.get_conversation_history(), 1):
        print(f"{i}. Q: {exchange['query'][:50]}...")
        print(f"   A: {exchange['response'][:100]}...")
    
    # Test follow-up question that should use context
    print(f"\nTesting context-aware follow-up:")
    follow_up = "Can you elaborate on that?"
    response = agent.chat(follow_up)
    print(f"User: {follow_up}")
    print(f"Agent: {response}")
    
    # Show graph structure
    print(f"\nWorkflow Graph Nodes: {list(agent.get_workflow_graph().nodes.keys())}")
    
    return agent

# Create and run the conversational agent
conversational_agent = demonstrate_conversational_agent()

# Advanced: Multi-modal conversation agent
class MultiModalConversationalAgent(ConversationalRetrievalAgent):
    """Extended conversational agent with multi-modal capabilities"""
    
    def __init__(self, knowledge_base=None):
        super().__init__(knowledge_base)
        self.image_context = {}
        self.file_attachments = []
    
    def _build_conversation_graph(self):
        """Build enhanced conversation workflow graph with multi-modal support"""
        graph = LangGraph()
        
        def multimodal_input_processing(state: AgentState) -> AgentState:
            """Process multi-modal inputs including text and metadata"""
            print("Processing multi-modal input...")
            
            # Check for image context in metadata
            if state.metadata.get("image_attached"):
                state.context += " [Image attached: User has provided an image for analysis]"
                state.metadata["input_modality"] = "text_with_image"
            else:
                state.metadata["input_modality"] = "text_only"
            
            # Check for file attachments
            if state.metadata.get("file_attached"):
                state.context += " [File attached: User has provided a document]"
                state.metadata["input_modality"] = "text_with_file"
            
            state.intermediate_steps.append({
                "step": "multimodal_input_processing",
                "modality": state.metadata["input_modality"]
            })
            
            return state
        
        # Use the same nodes as parent class but add multi-modal processing
        # ... (include all previous nodes here)
        
        # Add multi-modal processing node at the beginning
        graph.add_node("multimodal_input_processing", NodeType.TOOL, multimodal_input_processing)
        
        # Connect to existing workflow
        graph.add_node("context_enrichment", NodeType.TOOL, self._get_context_enrichment_func())
        graph.add_edge("multimodal_input_processing", "context_enrichment")
        
        # Add remaining nodes (simplified for this example)
        # ... (connect remaining nodes as in parent class)
        
        return graph
    
    def _get_context_enrichment_func(self):
        """Get context enrichment function for multi-modal"""
        def context_enrichment(state: AgentState) -> AgentState:
            # Enhanced context enrichment
            return super()._get_context_enrichment_func()(state)
        return context_enrichment
    
    def chat_with_context(self, user_input: str, image_attached: bool = False, file_attached: bool = False):
        """Chat with additional context about attached media"""
        initial_state = AgentState(
            input_query=user_input,
            metadata={
                "timestamp": time.time(),
                "image_attached": image_attached,
                "file_attached": file_attached
            }
        )
        
        final_state = self.graph.execute(initial_state)
        return final_state.final_answer

# Performance comparison and optimization
def performance_comparison():
    """Compare performance of different workflow approaches"""
    
    print("\nPerformance Comparison")
    print("=" * 30)
    
    # Sequential approach (traditional)
    def sequential_approach(query):
        start_time = time.time()
        
        # Simulate sequential processing
        time.sleep(0.1)  # Query analysis
        time.sleep(0.1)  # Information retrieval
        time.sleep(0.1)  # Response generation
        time.sleep(0.05) # Validation
        
        processing_time = time.time() - start_time
        return f"Sequential response for: {query}", processing_time
    
    # Graph-based approach
    def graph_approach(query):
        start_time = time.time()
        
        # Create simple graph-based workflow
        agent = ConversationalRetrievalAgent()
        response = agent.chat(query)
        
        processing_time = time.time() - start_time
        return response, processing_time
    
    # Test both approaches
    test_queries = ["What is AI?", "How does ML work?", "Explain NLP?"]
    
    print("Sequential Approach:")
    seq_times = []
    for query in test_queries:
        _, seq_time = sequential_approach(query)
        seq_times.append(seq_time)
        print(f"  Query: {query[:15]}... Time: {seq_time:.3f}s")
    
    print(f"  Average: {np.mean(seq_times):.3f}s")
    
    print("\nGraph-based Approach:")
    graph_times = []
    for query in test_queries:
        _, graph_time = graph_approach(query)
        graph_times.append(graph_time)
        print(f"  Query: {query[:15]}... Time: {graph_time:.3f}s")
    
    print(f"  Average: {np.mean(graph_times):.3f}s")
    
    print(f"\nImprovement: {((np.mean(seq_times) - np.mean(graph_times)) / np.mean(seq_times) * 100):.1f}% faster")

performance_comparison()
```