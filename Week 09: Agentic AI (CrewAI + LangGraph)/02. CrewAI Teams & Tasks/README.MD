# Day 42 – CrewAI Teams & Tasks  
## 📌 Project: SEO Content Bot  

---

### What is CrewAI?

**CrewAI** is an open-source framework for building autonomous AI agent teams that collaborate to complete complex tasks. Unlike single-agent systems, CrewAI enables:

- **Role-based agents**: Each agent has a defined role, goal, and backstory
- **Task delegation**: Tasks are assigned based on agent expertise
- **Team memory**: Shared context across agents
- **Sequential or parallel execution**
- **Autonomous planning and tool use**

It’s ideal for workflows like research → writing → editing → publishing.

---

### Essential Subtopics

#### 1. Role Assignment

Each agent in CrewAI has:
- `role`: Job title (e.g., "Senior Researcher")
- `goal`: Objective (e.g., "Find latest trends in AI")
- `backstory`: Expertise justification
- `tools`: Available functions (optional)

```python
from crewai import Agent

researcher = Agent(
    role="Senior Research Analyst",
    goal="Uncover cutting-edge developments in AI and data science",
    backstory=(
        "You're a seasoned researcher with a knack for finding "
        "the most relevant information from diverse sources."
    ),
    verbose=True,
    allow_delegation=True
)

writer = Agent(
    role="Technical Content Writer",
    goal="Transform complex tech insights into engaging blog posts",
    backstory=(
        "You're a skilled writer who turns technical jargon "
        "into compelling, easy-to-understand content."
    ),
    verbose=True,
    allow_delegation=False
)
```

> ✅ **Best Practice**: Use `verbose=True` for debugging, `allow_delegation` to enable inter-agent task passing.

---

#### 2. Task Creation

Tasks define what each agent does:
- `description`: Clear instructions
- `expected_output`: Desired format
- `agent`: Who performs it
- `async_execution`: Run in parallel?

```python
from crewai import Task

research_task = Task(
    description="Research the latest AI trends in 2025",
    expected_output="A detailed list of at least 5 key AI trends with sources",
    agent=researcher
)

write_task = Task(
    description="Write a blog post about the top AI trends",
    expected_output="A well-structured, SEO-optimized blog post of 800+ words",
    agent=writer,
    async_execution=False
)
```

> 🔁 `async_execution=True` runs tasks in parallel; `False` ensures sequential flow.

---

#### 3. Team Memory (Context Sharing)

CrewAI automatically passes output from one task to the next. No manual state management needed.

```python
# The writer will receive the researcher's output as context
write_task.context = [research_task]
```

This enables **context-aware generation** without extra code.

---

#### 4. Crew Assembly

The `Crew` object ties everything together:
- Agents
- Tasks
- Process type (`sequential`, `hierarchical`)

```python
from crewai import Crew
from langchain_community.llms import Ollama

# Optional: Use local LLM
ollama_llm = Ollama(model="llama3")

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process="sequential",  # or "hierarchical"
    memory=True,           # Enable team memory
    cache=True,
    llm=ollama_llm         # Optional: override default LLM
)
```

> 💡 Set `memory=True` to persist context across runs.

---

#### 5. Execution & Output

Run the crew and get results:

```python
result = crew.kickoff()
print("📄 Generated Content:")
print(result.raw)  # Full output
print("\n📊 Token Usage:", result.token_usage)
```

Use `result.pydantic` if you defined structured outputs.

---

### When and Where We Use CrewAI

| Scenario | Why CrewAI? |
|--------|------------|
| **Content Creation** | Research + write + edit pipelines |
| **Market Analysis** | Gather data → analyze → report |
| **Code Generation** | Design → implement → test → document |
| **Customer Support** | Triage → resolve → escalate |
| **Academic Research** | Survey papers, literature reviews |

> ✅ Use when tasks require **specialization**, **sequencing**, and **context sharing**.

---

### Real-World Applications and Examples

#### Example 1: Market Research Team

```python
analyst = Agent(role="Market Analyst", ...)
strategist = Agent(role="Growth Strategist", ...)

analyze_task = Task(description="Analyze Q2 sales data", agent=analyst)
plan_task = Task(description="Create growth strategy", agent=strategist, context=[analyze_task])

crew = Crew(agents=[analyst, strategist], tasks=[analyze_task, plan_task])
result = crew.kickoff()
```

#### Example 2: Documentation Generator

```python
code_reader = Agent(role="Code Reviewer", goal="Extract functionality from code")
doc_writer = Agent(role="Technical Writer", goal="Generate user docs")

read_task = Task("Understand main.py", agent=code_reader)
write_doc_task = Task("Write README.md", agent=doc_writer, context=[read_task])

crew = Crew(agents=[code_reader, doc_writer], tasks=[read_task, write_doc_task])
```

---

### Common Problems and How Advancements Solved Them

---

#### ❌ Problem 1: Agents Work in Isolation (No Context)

**Issue**: Early frameworks had no built-in way to pass outputs between agents.

**Advancement**: CrewAI uses `task.context` and automatic dependency resolution.

**Problem Code**:
```python
# Manual, fragile context passing
research_output = researcher.execute(research_prompt)
writer_prompt = f"Use this: {research_output} to write..."
final = writer.execute(writer_prompt)
```

**✅ Enhanced Code**:
```python
# CrewAI handles context automatically
write_task.context = [research_task]  # ← Clean, declarative
result = crew.kickoff()               # ← Outputs chained
```

---

#### ❌ Problem 2: No Shared Memory Across Runs

**Issue**: Teams couldn’t remember past interactions.

**Advancement**: CrewAI supports **persistent memory** via vector stores.

**Enhanced Code**:
```python
from crewai.memory.storage.local_context_storage import LocalContextStorage

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    memory=True,
    memory_backend=LocalContextStorage()  # Saves to disk
)
```

Now the crew “remembers” previous sessions.

---

#### ❌ Problem 3: Poor Role Clarity → Chaotic Output

**Issue**: Generic roles lead to inconsistent behavior.

**Advancement**: Explicit `backstory` and `goal` create behavioral boundaries.

**Weak Prompt**:
```python
Agent(role="Writer", goal="Write something")
```

**✅ Strong Prompt**:
```python
Agent(
    role="SEO Content Writer",
    goal="Write engaging, keyword-rich articles that rank on Google",
    backstory="You've written 500+ blog posts ranking on page 1 of Google..."
)
```

---

#### ❌ Problem 4: Sequential Bottlenecks

**Issue**: Waiting for one agent blocks others.

**Advancement**: `async_execution=True` for parallelism.

**Parallel Tasks**:
```python
task1 = Task("Research Python libraries", agent=researcher, async_execution=True)
task2 = Task("Research JavaScript frameworks", agent=researcher, async_execution=True)

# Both run concurrently
crew = Crew(agents=[researcher], tasks=[task1, task2])
```

---

## 🧪 Exercise: Researcher + Writer Agents

```python
# researcher_writer_exercise.py
from crewai import Agent, Task, Crew

# Define agents
researcher = Agent(
    role="Research Specialist",
    goal="Find accurate and up-to-date information",
    backstory="You're a meticulous researcher with access to the web.",
    verbose=True
)

writer = Agent(
    role="Content Creator",
    goal="Write clear, engaging summaries",
    backstory="You turn complex findings into simple, compelling stories.",
    verbose=True
)

# Define tasks
research_task = Task(
    description="What are the benefits of meditation for mental health?",
    expected_output="A bullet-point list of 5 evidence-based benefits",
    agent=researcher
)

write_task = Task(
    description="Write a 200-word social media post about meditation benefits",
    expected_output="A friendly, engaging post with hashtags",
    agent=writer,
    context=[research_task]
)

# Assemble crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process="sequential"
)

# Run
result = crew.kickoff()
print(result.raw)
```

---

## 📌 Project: SEO Content Bot

```python
# seo_content_bot.py
from crewai import Agent, Task, Crew, Process
from langchain.tools import DuckDuckGoSearchRun
import os

# Optional: Set API keys
os.environ["OPENAI_API_KEY"] = "your-key-here"

# Tools
search_tool = DuckDuckGoSearchRun()

# Agents
researcher = Agent(
    role="Senior SEO Researcher",
    goal="Find trending topics and high-ranking keywords in AI",
    backstory="You know how to find what people are searching for and what ranks on Google.",
    verbose=True,
    tools=[search_tool],
    allow_delegation=True
)

writer = Agent(
    role="SEO Content Writer",
    goal="Write blog posts that rank on Google",
    backstory=(
        "You're a professional blogger who writes SEO-optimized, "
        "reader-friendly content that ranks on page 1."
    ),
    verbose=True,
    allow_delegation=False
)

editor = Agent(
    role="Chief Editor",
    goal="Ensure content is polished, factual, and on-brand",
    backstory="You have 20 years of experience editing tech publications.",
    verbose=True,
    allow_delegation=True
)

# Tasks
research_task = Task(
    description=(
        "Identify the top 3 trending AI topics in 2025. "
        "Find related long-tail keywords with high search volume."
    ),
    expected_output="A list of 3 topics and 5 keywords each, with sources",
    agent=researcher,
    tools=[search_tool]
)

write_task = Task(
    description="Write a 1000-word SEO blog post about the #1 topic",
    expected_output=(
        "A complete blog post with H1, H2s, meta description, "
        "and keyword placement. Include a call-to-action."
    ),
    agent=writer,
    context=[research_task]
)

edit_task = Task(
    description="Edit the blog post for clarity, tone, and SEO",
    expected_output="Final polished blog post ready for publishing",
    agent=editor,
    context=[write_task]
)

# Create crew
seo_crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, write_task, edit_task],
    process=Process.sequential,
    memory=True,
    verbose=2
)

# Execute
print("🚀 Starting SEO Content Bot...")
result = seo_crew.kickoff()

# Save output
with open("output/seo_blog_post.md", "w") as f:
    f.write("# Final SEO Blog Post\n\n")
    f.write(result.raw)

print("✅ Content generated and saved to output/seo_blog_post.md")
```

---

### Bonus: Add Human Feedback Loop

```python
# Allow human input before finalizing
def approve_content():
    print("\n📝 Final Content:\n")
    print(result.raw)
    return input("\nApprove? (y/n): ").lower() == 'y'

if approve_content():
    publish(result.raw)  # Your publishing function
else:
    print("🔄 Revising content...")
```

---

### Summary: CrewAI Best Practices

| Feature | Recommendation |
|-------|----------------|
| **Roles** | Be specific: “Senior Data Scientist” > “Analyst” |
| **Goals** | Action-oriented: “Generate X” not “Be good at Y” |
| **Backstory** | Justify expertise to guide tone |
| **Tasks** | Define `expected_output` clearly |
| **Memory** | Use `memory=True` for multi-session workflows |
| **Tools** | Assign only necessary tools per agent |
| **Process** | `sequential` for pipelines, `hierarchical` for manager-led teams |

---