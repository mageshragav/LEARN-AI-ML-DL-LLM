# Day 41 – Agents & Planning

## What are Agents & Planning in LLM Systems?

### Agents
In the context of LLM systems, agents are autonomous programs that use language models to perform tasks by making decisions, taking actions, and using tools to achieve specific goals. Unlike simple chains that follow a predetermined path, agents can dynamically decide their next steps based on the current context and available tools.

### Planning
Planning is the process by which agents determine the sequence of actions needed to achieve a goal. It involves breaking down complex tasks into smaller, manageable steps and deciding the order in which to execute them. Effective planning is crucial for agents to handle complex, multi-step problems.

## Essential Subtopics

### 1. ReAct (Reasoning and Acting)

**What it is:**
ReAct is a paradigm that combines reasoning and acting in language models. It enables models to generate both reasoning traces and task-specific actions in an interleaved manner, allowing them to dynamically adjust their approach based on intermediate results.

**Explanation:**
ReAct agents work by:
1. Reasoning about the task at hand
2. Deciding on an action to take
3. Executing the action and observing the result
4. Using the result to inform the next reasoning step
5. Repeating this process until the task is complete

This approach allows agents to handle complex tasks that require multiple steps and external information retrieval.

**Coding Example:**
```python
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chat_models import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

class ReActAgent:
    """
    A ReAct (Reasoning and Acting) agent implementation.
    """
    def __init__(self, api_key, model_name="llama3-8b-8192"):
        """
        Initialize the ReAct agent.
        
        Args:
            api_key: Groq API key
            model_name: Name of the model to use
        """
        # Initialize the model
        self.llm = ChatGroq(model=model_name, groq_api_key=api_key)
        
        # Load tools
        self.tools = load_tools(["serpapi", "llm-math"], llm=self.llm)
        
        # Initialize the agent
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
    
    def run(self, task):
        """
        Run the agent on a task.
        
        Args:
            task: The task to perform
            
        Returns:
            Result of the task
        """
        return self.agent.run(task)

# Example usage
# react_agent = ReActAgent("your-groq-api-key")

# Simple task
# result = react_agent.run("What is the current population of France and what is 5% of that number?")
# print(result)

# More complex task
# complex_result = react_agent.run("Find the CEO of the company that created the iPhone and calculate how old they would be in dog years (1 dog year = 7 human years).")
# print(complex_result)
```

### 2. AutoGPT

**What it is:**
AutoGPT is an experimental open-source application that demonstrates the capabilities of GPT-4 (and other LLMs) to autonomously perform tasks by breaking them down into smaller sub-tasks, executing them, and iterating until the overall goal is achieved. It represents one of the earliest examples of autonomous AI agents.

**Explanation:**
AutoGPT works by:
1. Receiving a high-level goal from the user
2. Breaking down the goal into smaller tasks
3. Prioritizing and executing tasks
4. Reflecting on the results and adjusting the plan
5. Continuing this process until the goal is achieved

Unlike simple agents, AutoGPT has long-term memory and can work towards goals over extended periods, making decisions about what to do next without human intervention.

**Coding Example:**
```python
import time
import json
from typing import List, Dict, Any, Callable
from langchain.chat_models import ChatGroq
from langchain.schema import HumanMessage, SystemMessage, AIMessage

class AutoGPTAgent:
    """
    A simplified AutoGPT-style agent implementation.
    """
    def __init__(self, api_key, model_name="llama3-8b-8192"):
        """
        Initialize the AutoGPT agent.
        
        Args:
            api_key: Groq API key
            model_name: Name of the model to use
        """
        self.llm = ChatGroq(model=model_name, groq_api_key=api_key)
        self.memory = []  # Stores conversation history
        self.task_queue = []  # Stores tasks to be executed
        self.completed_tasks = []  # Stores completed tasks
        self.tools = {}  # Available tools
        
        # Register default tools
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register default tools."""
        # Search tool
        def search(query):
            """Mock search function."""
            return f"Search results for '{query}': This is a mock search result."
        
        # Calculate tool
        def calculate(expression):
            """Mock calculate function."""
            try:
                return f"Calculation result: {eval(expression)}"
            except:
                return "Error: Invalid expression"
        
        # Memory tool
        def store_memory(key, value):
            """Store a value in memory."""
            self.memory.append({"key": key, "value": value, "timestamp": time.time()})
            return f"Stored '{key}' in memory."
        
        def retrieve_memory(key):
            """Retrieve a value from memory."""
            for item in reversed(self.memory):
                if item["key"] == key:
                    return f"Retrieved '{key}': {item['value']}"
            return f"No memory found for key '{key}'."
        
        # Register tools
        self.tools["search"] = search
        self.tools["calculate"] = calculate
        self.tools["store_memory"] = store_memory
        self.tools["retrieve_memory"] = retrieve_memory
    
    def register_tool(self, name: str, func: Callable):
        """
        Register a custom tool.
        
        Args:
            name: Name of the tool
            func: Function to register
        """
        self.tools[name] = func
    
    def plan(self, goal: str) -> List[str]:
        """
        Create a plan to achieve a goal.
        
        Args:
            goal: The goal to achieve
            
        Returns:
            List of tasks
        """
        system_prompt = """You are an AI planning assistant. Your task is to break down a goal into a series of smaller, achievable tasks.
        
        Return your response as a JSON array of strings, where each string is a task.
        
        Example:
        Goal: "Research the history of artificial intelligence"
        Response: ["Search for the origins of artificial intelligence", "Find key milestones in AI development", "Identify important figures in AI history"]
        """
        
        user_prompt = f"Goal: {goal}"
        
        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        try:
            # Parse JSON response
            tasks = json.loads(response.content)
            return tasks if isinstance(tasks, list) else [tasks]
        except:
            # Fallback if JSON parsing fails
            return [f"Research: {goal}"]
    
    def execute_task(self, task: str) -> str:
        """
        Execute a single task.
        
        Args:
            task: The task to execute
            
        Returns:
            Result of the task
        """
        system_prompt = """You are an AI task execution assistant. Your task is to determine what action to take based on the given task.
        
        Available tools:
        - search: Search for information online
        - calculate: Perform mathematical calculations
        - store_memory: Store information in memory
        - retrieve_memory: Retrieve information from memory
        
        Return your response as a JSON object with two fields:
        - "tool": The name of the tool to use
        - "input": The input to provide to the tool
        
        If no tool is needed, return a JSON object with a single field:
        - "response": Your response to the task
        """
        
        user_prompt = f"Task: {task}"
        
        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        try:
            # Parse JSON response
            action = json.loads(response.content)
            
            if "tool" in action and "input" in action:
                # Execute tool
                tool_name = action["tool"]
                tool_input = action["input"]
                
                if tool_name in self.tools:
                    result = self.tools[tool_name](tool_input)
                    return f"Executed {tool_name} with input '{tool_input}': {result}"
                else:
                    return f"Error: Unknown tool '{tool_name}'"
            elif "response" in action:
                return action["response"]
            else:
                return "Error: Invalid action format"
        except:
            # Fallback if JSON parsing fails
            return f"Executed task: {task}"
    
    def run(self, goal: str, max_iterations: int = 10) -> Dict[str, Any]:
        """
        Run the AutoGPT agent on a goal.
        
        Args:
            goal: The goal to achieve
            max_iterations: Maximum number of iterations
            
        Returns:
            Result of the execution
        """
        print(f"Starting AutoGPT with goal: {goal}")
        
        # Create plan
        print("Creating plan...")
        self.task_queue = self.plan(goal)
        print(f"Plan created with {len(self.task_queue)} tasks")
        
        # Execute tasks
        iteration = 0
        while self.task_queue and iteration < max_iterations:
            iteration += 1
            print(f"\nIteration {iteration}/{max_iterations}")
            
            # Get next task
            task = self.task_queue.pop(0)
            print(f"Executing task: {task}")
            
            # Execute task
            result = self.execute_task(task)
            print(f"Result: {result}")
            
            # Store completed task
            self.completed_tasks.append({
                "task": task,
                "result": result,
                "iteration": iteration
            })
            
            # Reflect and adjust plan if needed
            if iteration < max_iterations and self.task_queue:
                print("Reflecting on progress...")
                reflection = self._reflect(goal, self.completed_tasks)
                print(f"Reflection: {reflection}")
                
                # Adjust plan based on reflection
                if "new_tasks" in reflection:
                    new_tasks = reflection["new_tasks"]
                    if new_tasks:
                        print(f"Adding new tasks: {new_tasks}")
                        self.task_queue.extend(new_tasks)
        
        # Final summary
        print("\nExecution complete")
        summary = self._summarize(goal, self.completed_tasks)
        print(f"Summary: {summary}")
        
        return {
            "goal": goal,
            "completed_tasks": self.completed_tasks,
            "summary": summary,
            "iterations": iteration
        }
    
    def _reflect(self, goal: str, completed_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Reflect on progress and adjust plan.
        
        Args:
            goal: The overall goal
            completed_tasks: List of completed tasks
            
        Returns:
            Reflection result
        """
        system_prompt = """You are an AI reflection assistant. Your task is to reflect on the progress made towards a goal and determine if any adjustments are needed.
        
        Return your response as a JSON object with the following fields:
        - "progress": A description of the progress made so far
        - "adjustments_needed": Whether adjustments are needed (true/false)
        - "new_tasks": A list of new tasks to add (empty list if no adjustments needed)
        """
        
        # Format completed tasks
        tasks_text = "\n".join([
            f"- {task['task']}: {task['result']}"
            for task in completed_tasks
        ])
        
        user_prompt = f"""
        Goal: {goal}
        
        Completed tasks:
        {tasks_text}
        """
        
        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        try:
            # Parse JSON response
            reflection = json.loads(response.content)
            return reflection
        except:
            # Fallback if JSON parsing fails
            return {
                "progress": "Progress has been made",
                "adjustments_needed": False,
                "new_tasks": []
            }
    
    def _summarize(self, goal: str, completed_tasks: List[Dict[str, Any]]) -> str:
        """
        Summarize the execution results.
        
        Args:
            goal: The overall goal
            completed_tasks: List of completed tasks
            
        Returns:
            Summary
        """
        system_prompt = """You are an AI summarization assistant. Your task is to summarize the results of executing tasks towards a goal.
        
        Provide a concise summary of what was accomplished and how it relates to the goal.
        """
        
        # Format completed tasks
        tasks_text = "\n".join([
            f"- {task['task']}: {task['result']}"
            for task in completed_tasks
        ])
        
        user_prompt = f"""
        Goal: {goal}
        
        Completed tasks:
        {tasks_text}
        
        Provide a summary of what was accomplished.
        """
        
        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        return response.content

# Example usage
# autogpt = AutoGPTAgent("your-groq-api-key")

# Register a custom tool
# def get_weather(location):
#     """Get weather information for a location."""
#     return f"The weather in {location} is sunny with a temperature of 72°F."

# autogpt.register_tool("get_weather", get_weather)

# Run the agent
# result = autogpt.run("Research the current weather in San Francisco and suggest appropriate clothing.")
# print(f"Final result: {result['summary']}")
```

### 3. CrewAI

**What it is:**
CrewAI is a framework for orchestrating role-playing, autonomous AI agents. It enables the creation of multi-agent systems where different agents with specific roles, goals, and capabilities collaborate to solve complex tasks.

**Explanation:**
CrewAI works by:
1. Defining different agent roles with specific capabilities
2. Assigning tasks to agents based on their roles
3. Allowing agents to collaborate and communicate
4. Coordinating the execution of tasks to achieve a common goal

This approach is particularly useful for complex problems that require diverse expertise and coordination between different specialized agents.

**Coding Example:**
```python
from crewai import Agent, Task, Crew, Process
from langchain.chat_models import ChatGroq
from langchain.tools import Tool
from typing import List, Dict, Any

class CrewAISystem:
    """
    A CrewAI multi-agent system implementation.
    """
    def __init__(self, api_key, model_name="llama3-8b-8192"):
        """
        Initialize the CrewAI system.
        
        Args:
            api_key: Groq API key
            model_name: Name of the model to use
        """
        self.llm = ChatGroq(model=model_name, groq_api_key=api_key)
        self.agents = {}
        self.tasks = {}
        self.crew = None
    
    def create_agent(self, name: str, role: str, goal: str, backstory: str, tools: List[Tool] = None):
        """
        Create an agent with a specific role.
        
        Args:
            name: Name of the agent
            role: Role of the agent
            goal: Goal of the agent
            backstory: Backstory of the agent
            tools: List of tools available to the agent
        """
        agent = Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            tools=tools or [],
            llm=self.llm,
            verbose=True
        )
        
        self.agents[name] = agent
        return agent
    
    def create_task(self, name: str, description: str, agent: str, expected_output: str = ""):
        """
        Create a task for an agent.
        
        Args:
            name: Name of the task
            description: Description of the task
            agent: Name of the agent to assign the task to
            expected_output: Expected output of the task
        """
        if agent not in self.agents:
            raise ValueError(f"Agent '{agent}' not found")
        
        task = Task(
            description=description,
            agent=self.agents[agent],
            expected_output=expected_output
        )
        
        self.tasks[name] = task
        return task
    
    def create_crew(self, process: Process = Process.sequential):
        """
        Create a crew from the defined agents and tasks.
        
        Args:
            process: Process to use (sequential or hierarchical)
        """
        agents_list = list(self.agents.values())
        tasks_list = list(self.tasks.values())
        
        self.crew = Crew(
            agents=agents_list,
            tasks=tasks_list,
            process=process,
            verbose=True
        )
    
    def run(self):
        """
        Run the crew.
        
        Returns:
            Result of the crew execution
        """
        if not self.crew:
            raise ValueError("Crew not created. Call create_crew() first.")
        
        return self.crew.kickoff()

# Example usage
def create_crewai_example():
    """Create an example CrewAI system."""
    # Initialize the system
    crew_system = CrewAISystem("your-groq-api-key")
    
    # Define tools
    def search_tool(query):
        """Mock search tool."""
        return f"Search results for '{query}': This is a mock search result."
    
    def analyze_tool(text):
        """Mock analysis tool."""
        return f"Analysis of '{text}': This is a mock analysis."
    
    search = Tool(
        name="search",
        func=search_tool,
        description="Search for information online"
    )
    
    analyze = Tool(
        name="analyze",
        func=analyze_tool,
        description="Analyze text or data"
    )
    
    # Create agents
    researcher = crew_system.create_agent(
        name="Researcher",
        role="Senior Research Analyst",
        goal="Uncover cutting-edge developments in AI and data science",
        backstory="""You work at a leading tech think tank.
        Your expertise lies in identifying emerging trends.
        You have a knack for dissecting complex data and presenting actionable insights.""",
        tools=[search]
    )
    
    writer = crew_system.create_agent(
        name="Writer",
        role="Tech Content Strategist",
        goal="Craft compelling content on tech advancements",
        backstory="""You are a renowned Content Strategist, known for your insightful and engaging articles.
        You transform complex concepts into compelling narratives.""",
        tools=[analyze]
    )
    
    # Create tasks
    task1 = crew_system.create_task(
        name="research_task",
        description="Conduct a comprehensive analysis of the latest advancements in AI in 2024.",
        agent="Researcher",
        expected_output="Full analysis report in bullet points"
    )
    
    task2 = crew_system.create_task(
        name="write_task",
        description="Using the insights provided, develop an engaging blog post that highlights the most significant AI advancements.",
        agent="Writer",
        expected_output="Full blog post of at least 4 paragraphs"
    )
    
    # Create and run crew
    crew_system.create_crew()
    result = crew_system.run()
    
    return result

# Run the example
# crew_result = create_crewai_example()
# print(f"CrewAI result: {crew_result}")
```

## When and Where We Use Agents & Planning

### When to Use Agents:
- When dealing with complex, multi-step problems
- When tasks require dynamic decision-making
- When external information retrieval is needed
- When tasks need to be adapted based on intermediate results
- When different types of expertise are required

### When to Use Planning:
- When breaking down complex goals into manageable tasks
- When coordinating multiple agents or tools
- When tasks need to be executed in a specific order
- When adapting to changing requirements or conditions
- When optimizing for efficiency or resource usage

### Where to Use Agents:
- Research and analysis tasks
- Content creation and curation
- Customer service and support
- Data analysis and visualization
- Project management and coordination

### Where to Use Planning:
- Workflow automation systems
- Multi-agent research systems
- Complex problem-solving applications
- Adaptive learning systems
- Resource allocation and optimization

**Coding Example:**
```python
class MultiAgentSystem:
    """
    A multi-agent system that combines ReAct, AutoGPT, and CrewAI approaches.
    """
    def __init__(self, api_key):
        """
        Initialize the multi-agent system.
        
        Args:
            api_key: Groq API key
        """
        self.react_agent = ReActAgent(api_key)
        self.autogpt_agent = AutoGPTAgent(api_key)
        self.crewai_system = CrewAISystem(api_key)
        
        # Define which approach to use for different types of tasks
        self.task_router = {
            "simple": self._handle_simple_task,
            "complex": self._handle_complex_task,
            "collaborative": self._handle_collaborative_task
        }
    
    def _handle_simple_task(self, task):
        """
        Handle a simple task using ReAct.
        
        Args:
            task: The task to handle
            
        Returns:
            Result of the task
        """
        return self.react_agent.run(task)
    
    def _handle_complex_task(self, task):
        """
        Handle a complex task using AutoGPT.
        
        Args:
            task: The task to handle
            
        Returns:
            Result of the task
        """
        return self.autogpt_agent.run(task)
    
    def _handle_collaborative_task(self, task):
        """
        Handle a collaborative task using CrewAI.
        
        Args:
            task: The task to handle
            
        Returns:
            Result of the task
        """
        # This would require setting up the CrewAI system with appropriate agents and tasks
        # For this example, we'll just return a placeholder
        return f"Collaborative task '{task}' would be handled by CrewAI"
    
    def classify_task(self, task):
        """
        Classify a task to determine the appropriate approach.
        
        Args:
            task: The task to classify
            
        Returns:
            Classification of the task
        """
        system_prompt = """You are a task classification assistant. Your task is to classify the given task into one of the following categories:
        
        - simple: A straightforward task that can be completed with a few steps
        - complex: A multi-step task that requires planning and adaptation
        - collaborative: A task that would benefit from multiple agents with different expertise
        
        Return only the category name.
        """
        
        response = self.react_agent.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Task: {task}")
        ])
        
        classification = response.content.lower().strip()
        
        # Ensure the classification is valid
        if classification not in self.task_router:
            classification = "simple"  # Default to simple
        
        return classification
    
    def run(self, task):
        """
        Run the appropriate agent for the task.
        
        Args:
            task: The task to run
            
        Returns:
            Result of the task
        """
        # Classify the task
        classification = self.classify_task(task)
        print(f"Task classified as: {classification}")
        
        # Route to the appropriate handler
        handler = self.task_router[classification]
        return handler(task)

# Example usage
# multi_agent = MultiAgentSystem("your-groq-api-key")

# Simple task
# simple_result = multi_agent.run("What is the capital of France?")
# print(f"Simple task result: {simple_result}")

# Complex task
# complex_result = multi_agent.run("Research the history of artificial intelligence and identify the key milestones.")
# print(f"Complex task result: {complex_result}")

# Collaborative task
# collaborative_result = multi_agent.run("Create a comprehensive report on renewable energy sources, including research, analysis, and recommendations.")
# print(f"Collaborative task result: {collaborative_result}")
```

## Real-World Applications and Examples

### 1. Research Assistant

**Application:**
A research assistant that can gather information from multiple sources, analyze it, and synthesize findings into a comprehensive report.

**Coding Example:**
```python
class ResearchAssistant:
    """
    A research assistant using agents and planning.
    """
    def __init__(self, api_key):
        """
        Initialize the research assistant.
        
        Args:
            api_key: Groq API key
        """
        self.llm = ChatGroq(model="llama3-8b-8192", groq_api_key=api_key)
        self.autogpt = AutoGPTAgent(api_key)
        
        # Register research-specific tools
        self._register_research_tools()
    
    def _register_research_tools(self):
        """Register research-specific tools."""
        # Search tool
        def search_academic(query):
            """Mock academic search."""
            return f"Academic search results for '{query}': Found 5 relevant papers."
        
        # Summarize tool
        def summarize_text(text):
            """Mock text summarization."""
            return f"Summary of text: This is a summary of the provided text."
        
        # Extract key points tool
        def extract_key_points(text):
            """Mock key point extraction."""
            return f"Key points: 1. Point 1, 2. Point 2, 3. Point 3"
        
        # Register tools
        self.autogpt.register_tool("search_academic", search_academic)
        self.autogpt.register_tool("summarize", summarize_text)
        self.autogpt.register_tool("extract_key_points", extract_key_points)
    
    def research_topic(self, topic):
        """
        Research a topic and generate a report.
        
        Args:
            topic: The topic to research
            
        Returns:
            Research report
        """
        goal = f"Research the topic '{topic}' and generate a comprehensive report including key findings, important sources, and future directions."
        
        result = self.autogpt.run(goal)
        return result["summary"]
    
    def answer_research_question(self, question):
        """
        Answer a research question.
        
        Args:
            question: The research question
            
        Returns:
            Answer to the question
        """
        system_prompt = """You are a research assistant. Your task is to answer the given research question based on your knowledge.
        
        Provide a comprehensive answer that includes:
        1. A direct answer to the question
        2. Key points and evidence supporting the answer
        3. Relevant context and background information
        4. Sources or references (if applicable)
        """
        
        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Research question: {question}")
        ])
        
        return response.content

# Example usage
# research_assistant = ResearchAssistant("your-groq-api-key")

# Research a topic
# report = research_assistant.research_topic("The impact of artificial intelligence on healthcare")
# print(f"Research report: {report}")

# Answer a research question
# answer = research_assistant.answer_research_question("How does machine learning improve medical diagnosis?")
# print(f"Research answer: {answer}")
```

### 2. Project Management Assistant

**Application:**
A project management assistant that can help plan, track, and manage projects by breaking them down into tasks, assigning resources, and monitoring progress.

**Coding Example:**
```python
class ProjectManagementAssistant:
    """
    A project management assistant using agents and planning.
    """
    def __init__(self, api_key):
        """
        Initialize the project management assistant.
        
        Args:
            api_key: Groq API key
        """
        self.llm = ChatGroq(model="llama3-8b-8192", groq_api_key=api_key)
        self.autogpt = AutoGPTAgent(api_key)
        
        # Project data
        self.projects = {}
        
        # Register project management tools
        self._register_pm_tools()
    
    def _register_pm_tools(self):
        """Register project management tools."""
        # Create task tool
        def create_task(task_name, description, priority="medium"):
            """Mock task creation."""
            return f"Task '{task_name}' created with priority '{priority}'."
        
        # Assign resource tool
        def assign_resource(task_name, resource_name):
            """Mock resource assignment."""
            return f"Resource '{resource_name}' assigned to task '{task_name}'."
        
        # Update progress tool
        def update_progress(task_name, progress_percent):
            """Mock progress update."""
            return f"Task '{task_name}' progress updated to {progress_percent}%."
        
        # Register tools
        self.autogpt.register_tool("create_task", create_task)
        self.autogpt.register_tool("assign_resource", assign_resource)
        self.autogpt.register_tool("update_progress", update_progress)
    
    def create_project(self, project_name, description):
        """
        Create a new project.
        
        Args:
            project_name: Name of the project
            description: Description of the project
            
        Returns:
            Project information
        """
        self.projects[project_name] = {
            "name": project_name,
            "description": description,
            "tasks": [],
            "status": "planning"
        }
        
        return f"Project '{project_name}' created successfully."
    
    def plan_project(self, project_name):
        """
        Plan a project by breaking it down into tasks.
        
        Args:
            project_name: Name of the project
            
        Returns:
            Project plan
        """
        if project_name not in self.projects:
            return f"Project '{project_name}' not found."
        
        project = self.projects[project_name]
        goal = f"Create a detailed project plan for '{project_name}' with the following description: {project['description']}. The plan should include all necessary tasks, dependencies, and resource assignments."
        
        result = self.autogpt.run(goal)
        
        # Update project status
        project["status"] = "planned"
        project["plan"] = result["summary"]
        
        return result["summary"]
    
    def execute_project(self, project_name):
        """
        Execute a project by carrying out its plan.
        
        Args:
            project_name: Name of the project
            
        Returns:
            Execution results
        """
        if project_name not in self.projects:
            return f"Project '{project_name}' not found."
        
        project = self.projects[project_name]
        
        if project["status"] != "planned":
            return f"Project '{project_name}' is not ready for execution. Current status: {project['status']}"
        
        goal = f"Execute the project plan for '{project_name}'. Carry out all tasks in the plan, track progress, and report on the final outcome."
        
        result = self.autogpt.run(goal)
        
        # Update project status
        project["status"] = "completed"
        project["execution_results"] = result["summary"]
        
        return result["summary"]
    
    def get_project_status(self, project_name):
        """
        Get the status of a project.
        
        Args:
            project_name: Name of the project
            
        Returns:
            Project status
        """
        if project_name not in self.projects:
            return f"Project '{project_name}' not found."
        
        project = self.projects[project_name]
        
        status = f"Project: {project['name']}\n"
        status += f"Description: {project['description']}\n"
        status += f"Status: {project['status']}\n"
        
        if "plan" in project:
            status += f"Plan: {project['plan']}\n"
        
        if "execution_results" in project:
            status += f"Execution Results: {project['execution_results']}\n"
        
        return status

# Example usage
# pm_assistant = ProjectManagementAssistant("your-groq-api-key")

# Create a project
# create_result = pm_assistant.create_project("Website Redesign", "Redesign the company website with a modern look and improved user experience.")
# print(create_result)

# Plan the project
# plan_result = pm_assistant.plan_project("Website Redesign")
# print(f"Project plan: {plan_result}")

# Execute the project
# execution_result = pm_assistant.execute_project("Website Redesign")
# print(f"Execution result: {execution_result}")

# Get project status
# status = pm_assistant.get_project_status("Website Redesign")
# print(f"Project status: {status}")
```

## Common Problems and How Advancements Have Solved Them

### Problem 1: Limited Autonomy in LLM Applications

**Problem:**
Early LLM applications were limited in their ability to act autonomously. They could respond to prompts but couldn't make decisions, take actions, or adapt their approach based on intermediate results.

**Solution:**
Agent frameworks like ReAct, AutoGPT, and CrewAI provide LLMs with the ability to:
1. Reason about tasks and decide on actions
2. Execute actions using available tools
3. Observe the results of their actions
4. Adapt their approach based on observations
5. Work towards goals over extended periods

**Problem Code:**
```python
# Old approach - simple prompt-response
def simple_llm_response(prompt, api_key):
    """
    Get a simple response from an LLM.
    
    Args:
        prompt: Input prompt
        api_key: API key
        
    Returns:
        Response from the LLM
    """
    llm = ChatGroq(model="llama3-8b-8192", groq_api_key=api_key)
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

# This approach is limited to direct responses
# api_key = "your-groq-api-key"
# response = simple_llm_response("What is the capital of France?", api_key)
# print(response)
```

**Enhancement Code:**
```python
# New approach - autonomous agent
def autonomous_agent_response(goal, api_key):
    """
    Get a response from an autonomous agent.
    
    Args:
        goal: Goal for the agent to achieve
        api_key: API key
        
    Returns:
        Result of the agent's work
    """
    agent = AutoGPTAgent(api_key)
    result = agent.run(goal)
    return result["summary"]

# This approach can handle complex, multi-step tasks
# api_key = "your-groq-api-key"
# result = autonomous_agent_response("Research the capital of France and find its current population.", api_key)
# print(result)
```

### Problem 2: Lack of Coordination Between Multiple LLMs

**Problem:**
Complex tasks often require different types of expertise, but coordinating multiple LLMs with different specializations was challenging.

**Solution:**
Multi-agent frameworks like CrewAI enable:
1. Definition of agents with specific roles and capabilities
2. Assignment of tasks to appropriate agents
3. Coordination and collaboration between agents
4. Integration of results from multiple agents

**Problem Code:**
```python
# Old approach - single LLM for all tasks
def single_llm_for_all_tasks(tasks, api_key):
    """
    Use a single LLM for all tasks.
    
    Args:
        tasks: List of tasks to perform
        api_key: API key
        
    Returns:
        Results of the tasks
    """
    llm = ChatGroq(model="llama3-8b-8192", groq_api_key=api_key)
    results = []
    
    for task in tasks:
        response = llm.invoke([HumanMessage(content=task)])
        results.append(response.content)
    
    return results

# This approach uses the same LLM for all tasks, regardless of specialization
# api_key = "your-groq-api-key"
# tasks = ["Research AI trends", "Write a blog post about AI", "Analyze the impact of AI on jobs"]
# results = single_llm_for_all_tasks(tasks, api_key)
# for i, result in enumerate(results):
#     print(f"Task {i+1}: {result}")
```

**Enhancement Code:**
```python
# New approach - specialized agents for different tasks
def multi_agent_for_different_tasks(goal, api_key):
    """
    Use multiple specialized agents for different aspects of a goal.
    
    Args:
        goal: Overall goal to achieve
        api_key: API key
        
    Returns:
        Result of the agents' collaboration
    """
    # Create CrewAI system
    crew_system = CrewAISystem(api_key)
    
    # Create specialized agents
    researcher = crew_system.create_agent(
        name="Researcher",
        role="Research Analyst",
        goal="Conduct thorough research on topics",
        backstory="You are an expert researcher with access to vast information sources."
    )
    
    writer = crew_system.create_agent(
        name="Writer",
        role="Content Writer",
        goal="Create engaging and informative content",
        backstory="You are a skilled writer who can transform research into compelling narratives."
    )
    
    analyst = crew_system.create_agent(
        name="Analyst",
        role="Data Analyst",
        goal="Analyze information and extract insights",
        backstory="You are an analytical thinker who can identify patterns and draw conclusions."
    )
    
    # Create tasks for each agent
    research_task = crew_system.create_task(
        name="research",
        description=f"Research the topic: {goal}",
        agent="Researcher",
        expected_output="Comprehensive research findings"
    )
    
    analysis_task = crew_system.create_task(
        name="analysis",
        description="Analyze the research findings and extract key insights",
        agent="Analyst",
        expected_output="Key insights and patterns"
    )
    
    writing_task = crew_system.create_task(
        name="writing",
        description="Create a comprehensive report based on the research and analysis",
        agent="Writer",
        expected_output="Well-written report"
    )
    
    # Create and run crew
    crew_system.create_crew()
    result = crew_system.run()
    
    return result

# This approach uses specialized agents for different aspects of the goal
# api_key = "your-groq-api-key"
# goal = "The impact of artificial intelligence on healthcare"
# result = multi_agent_for_different_tasks(goal, api_key)
# print(result)
```

## 🧪 Exercise: Plan-Execute Loop

Let's create an exercise to implement a plan-execute loop using agents.

```python
def plan_execute_loop_exercise(api_key):
    """
    Exercise: Implement a plan-execute loop.
    
    Args:
        api_key: Groq API key
    """
    print("Starting Plan-Execute Loop Exercise")
    
    # Initialize the AutoGPT agent
    agent = AutoGPTAgent(api_key)
    
    # Register a custom tool for this exercise
    def execute_step(step_description):
        """
        Execute a step in the plan.
        
        Args:
            step_description: Description of the step to execute
            
        Returns:
            Result of executing the step
        """
        print(f"Executing step: {step_description}")
        # In a real implementation, this would actually execute the step
        return f"Step executed: {step_description}"
    
    agent.register_tool("execute_step", execute_step)
    
    # Define a goal that requires planning and execution
    goal = "Plan and execute a simple website redesign project. The project should include research, design, and implementation phases."
    
    # Run the agent
    result = agent.run(goal, max_iterations=5)
    
    # Print the results
    print("\nPlan-Execute Loop Results:")
    print(f"Goal: {goal}")
    print(f"Summary: {result['summary']}")
    print(f"Iterations: {result['iterations']}")
    print("\nCompleted Tasks:")
    for i, task in enumerate(result['completed_tasks'], 1):
        print(f"{i}. {task['task']}: {task['result']}")

# Example usage
# plan_execute_loop_exercise("your-groq-api-key")
```

## 📌 Project: Multi-agent Q&A Bot

Now, let's create a comprehensive project - a multi-agent Q&A bot that uses different agents for different types of questions.

```python
import os
import time
import json
import argparse
from typing import Dict, List, Any, Optional
from langchain.chat_models import ChatGroq
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

class MultiAgentQABot:
    """
    A multi-agent Q&A bot with specialized agents for different types of questions.
    """
    def __init__(self, api_key, config_path=None):
        """
        Initialize the multi-agent Q&A bot.
        
        Args:
            api_key: Groq API key
            config_path: Path to configuration file (optional)
        """
        # Load configuration
        self.config = self._load_config(config_path) if config_path else {}
        
        # Initialize the LLM
        self.llm = ChatGroq(model="llama3-8b-8192", groq_api_key=api_key)
        
        # Initialize memory
        self.memory = ConversationBufferMemory()
        
        # Initialize agents
        self.agents = {}
        self._initialize_agents()
        
        # Initialize conversation history
        self.conversation_history = []
    
    def _load_config(self, config_path):
        """
        Load configuration from a file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Configuration dictionary
        """
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _initialize_agents(self):
        """Initialize the specialized agents."""
        # General knowledge agent
        self.agents["general"] = {
            "name": "General Knowledge Agent",
            "description": "Answers general knowledge questions",
            "system_prompt": """You are a helpful assistant with broad general knowledge. 
            Answer the user's question accurately and concisely. 
            If you don't know the answer, say so honestly."""
        }
        
        # Technical agent
        self.agents["technical"] = {
            "name": "Technical Agent",
            "description": "Answers technical and scientific questions",
            "system_prompt": """You are a technical expert with deep knowledge of science, technology, and mathematics.
            Provide detailed, accurate answers to technical questions.
            Include relevant formulas, concepts, and explanations where appropriate."""
        }
        
        # Creative agent
        self.agents["creative"] = {
            "name": "Creative Agent",
            "description": "Handles creative and open-ended questions",
            "system_prompt": """You are a creative thinker with a flair for imaginative and original responses.
            Answer creative questions with engaging, thoughtful, and innovative responses.
            Feel free to explore different perspectives and possibilities."""
        }
        
        # Factual agent
        self.agents["factual"] = {
            "name": "Factual Agent",
            "description": "Provides factual information with sources",
            "system_prompt": """You are a factual researcher who provides accurate information with sources.
            Answer questions with factual information, and always include sources or references where possible.
            If you cannot verify a fact, say so."""
        }
    
    def classify_question(self, question):
        """
        Classify a question to determine the appropriate agent.
        
        Args:
            question: The question to classify
            
        Returns:
            Classification of the question
        """
        system_prompt = """You are a question classification assistant. Your task is to classify the given question into one of the following categories:
        
        - general: General knowledge questions that don't require specialized expertise
        - technical: Questions about science, technology, mathematics, or other technical subjects
        - creative: Questions that require imagination, originality, or creative thinking
        - factual: Questions that require specific factual information with sources
        
        Return only the category name.
        """
        
        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Question: {question}")
        ])
        
        classification = response.content.lower().strip()
        
        # Ensure the classification is valid
        if classification not in self.agents:
            classification = "general"  # Default to general
        
        return classification
    
    def route_to_agent(self, question, classification):
        """
        Route a question to the appropriate agent.
        
        Args:
            question: The question to route
            classification: Classification of the question
            
        Returns:
            Response from the agent
        """
        agent = self.agents[classification]
        
        # Create the prompt
        system_message = SystemMessage(content=agent["system_prompt"])
        human_message = HumanMessage(content=question)
        
        # Get response
        response = self.llm.invoke([system_message, human_message])
        
        return {
            "response": response.content,
            "agent": agent["name"],
            "classification": classification
        }
    
    def answer_question(self, question):
        """
        Answer a question using the appropriate agent.
        
        Args:
            question: The question to answer
            
        Returns:
            Answer to the question
        """
        # Classify the question
        classification = self.classify_question(question)
        
        # Route to the appropriate agent
        result = self.route_to_agent(question, classification)
        
        # Update memory and conversation history
        self.memory.chat_memory.add_user_message(question)
        self.memory.chat_memory.add_ai_message(result["response"])
        
        self.conversation_history.append({
            "timestamp": time.time(),
            "question": question,
            "answer": result["response"],
            "agent": result["agent"],
            "classification": classification
        })
        
        return result
    
    def chat_loop(self):
        """
        Run an interactive chat loop.
        """
        print("Multi-Agent Q&A Bot")
        print("Type 'exit' to end the conversation.")
        print("Type 'agents' to see available agents.")
        print("Type 'history' to see conversation history.")
        print("-" * 50)
        
        while True:
            # Get user input
            user_input = input("You: ")
            
            # Check for exit command
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            
            # Check for agents command
            if user_input.lower() == 'agents':
                self._print_agents()
                continue
            
            # Check for history command
            if user_input.lower() == 'history':
                self._print_history()
                continue
            
            # Answer question
            result = self.answer_question(user_input)
            
            # Print response
            print(f"{result['agent']}: {result['response']}")
            print(f"[Classification: {result['classification']}]")
            print("-" * 50)
    
    def _print_agents(self):
        """Print information about available agents."""
        print("Available Agents:")
        for key, agent in self.agents.items():
            print(f"- {key}: {agent['name']} - {agent['description']}")
        print("-" * 50)
    
    def _print_history(self):
        """Print the conversation history."""
        print("Conversation History:")
        for i, entry in enumerate(self.conversation_history, 1):
            timestamp = time.strftime("%H:%M:%S", time.localtime(entry["timestamp"]))
            print(f"{i}. [{timestamp}] {entry['question']}")
            print(f"   Answered by {entry['agent']} ({entry['classification']})")
        print("-" * 50)
    
    def save_conversation(self, file_path):
        """
        Save the conversation history to a file.
        
        Args:
            file_path: Path to save the conversation
        """
        with open(file_path, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)
        
        print(f"Conversation saved to {file_path}")
    
    def load_conversation(self, file_path):
        """
        Load a conversation history from a file.
        
        Args:
            file_path: Path to load the conversation from
        """
        with open(file_path, 'r') as f:
            loaded_history = json.load(f)
        
        self.conversation_history = loaded_history
        
        # Rebuild memory
        self.memory.clear()
        for entry in self.conversation_history:
            self.memory.chat_memory.add_user_message(entry["question"])
            self.memory.chat_memory.add_ai_message(entry["answer"])
        
        print(f"Conversation loaded from {file_path}")

class EnhancedMultiAgentQABot(MultiAgentQABot):
    """
    An enhanced version of the multi-agent Q&A bot with additional features.
    """
    def __init__(self, api_key, config_path=None):
        """
        Initialize the enhanced multi-agent Q&A bot.
        
        Args:
            api_key: Groq API key
            config_path: Path to configuration file (optional)
        """
        super().__init__(api_key, config_path)
        
        # Initialize tools
        self.tools = {}
        self._initialize_tools()
    
    def _initialize_tools(self):
        """Initialize tools for the agents."""
        # Search tool
        def search(query):
            """Mock search function."""
            return f"Search results for '{query}': This is a mock search result."
        
        # Calculate tool
        def calculate(expression):
            """Mock calculate function."""
            try:
                return f"Calculation result: {eval(expression)}"
            except:
                return "Error: Invalid expression"
        
        # Get weather tool
        def get_weather(location):
            """Mock weather function."""
            return f"The weather in {location} is sunny with a temperature of 72°F."
        
        # Register tools
        self.tools["search"] = search
        self.tools["calculate"] = calculate
        self.tools["get_weather"] = get_weather
        
        # Add tool-using agents
        self.agents["research"] = {
            "name": "Research Agent",
            "description": "Conducts research using search tools",
            "system_prompt": """You are a research assistant with access to search tools.
            When asked a question, use the search tool to find relevant information.
            Provide comprehensive answers based on your research."""
        }
        
        self.agents["calculation"] = {
            "name": "Calculation Agent",
            "description": "Performs mathematical calculations",
            "system_prompt": """You are a mathematical assistant with access to calculation tools.
            When asked a mathematical question, use the calculate tool to perform the computation.
            Provide clear, step-by-step explanations of your calculations."""
        }
        
        self.agents["weather"] = {
            "name": "Weather Agent",
            "description": "Provides weather information",
            "system_prompt": """You are a weather assistant with access to weather tools.
            When asked about weather, use the get_weather tool to find current conditions.
            Provide helpful weather information and advice."""
        }
    
    def route_to_agent(self, question, classification):
        """
        Route a question to the appropriate agent, with tool support.
        
        Args:
            question: The question to route
            classification: Classification of the question
            
        Returns:
            Response from the agent
        """
        agent = self.agents[classification]
        
        # For tool-using agents, we need a different approach
        if classification in ["research", "calculation", "weather"]:
            return self._route_to_tool_agent(question, classification, agent)
        else:
            # Use the parent method for non-tool agents
            return super()._route_to_agent(question, classification)
    
    def _route_to_tool_agent(self, question, classification, agent):
        """
        Route a question to a tool-using agent.
        
        Args:
            question: The question to route
            classification: Classification of the question
            agent: Agent information
            
        Returns:
            Response from the agent
        """
        # Create a system prompt that includes tool information
        system_prompt = f"""{agent['system_prompt']}
        
        Available tools:
        - search: Search for information online
        - calculate: Perform mathematical calculations
        - get_weather: Get weather information for a location
        
        To use a tool, include the tool name and input in your response in this format:
        TOOL: tool_name, input
        """
        
        # Get initial response
        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=question)
        ])
        
        # Check if the agent wants to use a tool
        if response.content.startswith("TOOL:"):
            # Extract tool name and input
            tool_parts = response.content[5:].split(", ", 1)
            tool_name = tool_parts[0]
            tool_input = tool_parts[1] if len(tool_parts) > 1 else ""
            
            # Execute the tool
            if tool_name in self.tools:
                tool_result = self.tools[tool_name](tool_input)
                
                # Get a new response with the tool result
                follow_up_prompt = f"""You used the {tool_name} tool with input "{tool_input}" and got this result:
                
                {tool_result}
                
                Now, provide a complete answer to the original question: {question}"""
                
                final_response = self.llm.invoke([
                    SystemMessage(content=agent["system_prompt"]),
                    HumanMessage(content=follow_up_prompt)
                ])
                
                return {
                    "response": final_response.content,
                    "agent": agent["name"],
                    "classification": classification,
                    "tool_used": tool_name
                }
            else:
                return {
                    "response": f"Error: Unknown tool '{tool_name}'",
                    "agent": agent["name"],
                    "classification": classification
                }
        else:
            return {
                "response": response.content,
                "agent": agent["name"],
                "classification": classification
            }

def create_sample_config():
    """Create a sample configuration file."""
    config = {
        "model_name": "llama3-8b-8192",
        "max_history": 100,
        "auto_save": True,
        "save_interval": 10
    }
    
    with open("qa_bot_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("Sample configuration created: qa_bot_config.json")

def main():
    """Main function for the multi-agent Q&A bot project."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Multi-Agent Q&A Bot")
    parser.add_argument("--api-key", type=str, required=True, help="Groq API key")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--create-config", action="store_true", help="Create a sample configuration file")
    parser.add_argument("--enhanced", action="store_true", help="Use the enhanced version with tools")
    parser.add_argument("--load", type=str, help="Path to load conversation from")
    
    args = parser.parse_args()
    
    # Create sample config if requested
    if args.create_config:
        create_sample_config()
        return
    
    # Initialize the bot
    if args.enhanced:
        print("Initializing Enhanced Multi-Agent Q&A Bot...")
        bot = EnhancedMultiAgentQABot(args.api_key, args.config)
    else:
        print("Initializing Multi-Agent Q&A Bot...")
        bot = MultiAgentQABot(args.api_key, args.config)
    
    # Load conversation if specified
    if args.load:
        bot.load_conversation(args.load)
    
    # Start chat loop
    bot.chat_loop()

if __name__ == "__main__":
    main()
```
