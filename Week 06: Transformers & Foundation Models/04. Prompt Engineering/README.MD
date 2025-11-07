# Day 29 – Prompt Engineering

## What is Prompt Engineering?

Prompt engineering is the practice of designing and optimizing input prompts to effectively guide large language models (LLMs) to produce desired outputs. It's a critical skill for working with LLMs because the quality, structure, and content of the prompt significantly influence the model's response.

Prompt engineering involves understanding how LLMs interpret different types of instructions, examples, and formatting, and leveraging this knowledge to craft prompts that elicit accurate, relevant, and useful responses.

## Essential Subtopics

### 1. Few-shot Prompting

Few-shot prompting is a technique where you provide a small number of examples in the prompt to guide the model's understanding of the task. This approach leverages the in-context learning capabilities of LLMs, allowing them to learn patterns from the examples without explicit training.

**Key Concepts:**
- In-context learning
- Example selection and ordering
- Task demonstration through examples
- Balancing between zero-shot and few-shot approaches

**Coding Example - Few-shot prompting for sentiment analysis:**

```python
from transformers import pipeline

# Initialize a text generation pipeline
generator = pipeline("text-generation", model="gpt2")

# Few-shot prompt for sentiment analysis
few_shot_prompt = """
Classify the sentiment of the following reviews as Positive, Negative, or Neutral.

Example 1:
Review: "This product is amazing! I love it."
Sentiment: Positive

Example 2:
Review: "The service was terrible and the food was cold."
Sentiment: Negative

Example 3:
Review: "The movie was okay, not great but not bad either."
Sentiment: Neutral

Now, classify this review:
Review: "The battery life on this phone is incredible, it lasts all day."
Sentiment:
"""

# Generate response
result = generator(few_shot_prompt, max_length=200, num_return_sequences=1)
print(result[0]['generated_text'])
```

### 2. Chain-of-Thought (CoT) Prompting

Chain-of-Thought prompting is a technique that encourages the model to break down complex problems into intermediate steps before arriving at the final answer. This approach is particularly effective for reasoning tasks, as it mimics human problem-solving processes.

**Key Concepts:**
- Step-by-step reasoning
- Intermediate step generation
- Complex problem decomposition
- Improving accuracy on multi-step problems

**Coding Example - CoT prompting for math problems:**

```python
from transformers import pipeline

# Initialize a text generation pipeline
generator = pipeline("text-generation", model="gpt2")

# Chain-of-Thought prompt for a math problem
cot_prompt = """
Solve the following math problem by showing your step-by-step reasoning.

Problem: John has 5 apples. He gives 2 apples to Mary and then buys 3 more apples. How many apples does John have now?

Solution:
Step 1: John starts with 5 apples.
Step 2: He gives 2 apples to Mary, so he has 5 - 2 = 3 apples left.
Step 3: He buys 3 more apples, so he has 3 + 3 = 6 apples.
Final Answer: John has 6 apples.

Now, solve this problem:
Problem: Sarah has 12 books. She gives 3 books to her friend and then buys 5 more books. How many books does Sarah have now?

Solution:
"""

# Generate response
result = generator(cot_prompt, max_length=200, num_return_sequences=1)
print(result[0]['generated_text'])
```

### 3. Persona Prompting

Persona prompting involves instructing the model to adopt a specific persona or character when generating responses. This technique is useful for tailoring the model's output style, tone, and perspective to match particular use cases or user preferences.

**Key Concepts:**
- Role specification
- Style and tone adaptation
- Perspective taking
- Contextual framing

**Coding Example - Persona prompting for different writing styles:**

```python
from transformers import pipeline

# Initialize a text generation pipeline
generator = pipeline("text-generation", model="gpt2")

# Persona prompt for a specific writing style
persona_prompt = """
You are a professional travel blogger with a witty and engaging writing style. Write a short paragraph about visiting Paris.

Persona: Professional travel blogger, witty and engaging style
Topic: Visiting Paris
Response:
"""

# Generate response
result = generator(persona_prompt, max_length=150, num_return_sequences=1)
print(result[0]['generated_text'])

# Now try a different persona
different_persona_prompt = """
You are a formal historian specializing in European architecture. Write a short paragraph about visiting Paris.

Persona: Formal historian specializing in European architecture
Topic: Visiting Paris
Response:
"""

# Generate response
result = generator(different_persona_prompt, max_length=150, num_return_sequences=1)
print(result[0]['generated_text'])
```

### 4. Advanced Prompting Techniques

Beyond the core techniques, there are several advanced prompting strategies that can further improve model performance:

**Key Concepts:**
- Instruction following
- Output formatting
- Constraint specification
- Prompt chaining

**Coding Example - Advanced prompting with output formatting:**

```python
from transformers import pipeline

# Initialize a text generation pipeline
generator = pipeline("text-generation", model="gpt2")

# Advanced prompt with specific output format
advanced_prompt = """
Extract the key information from the following text and format it as JSON.

Text: "Apple Inc. announced its latest iPhone, the iPhone 14, at their headquarters in Cupertino, California on September 7, 2022. The new device features a 6.1-inch display and starts at $799."

Instructions:
1. Extract the company name, product name, announcement location, announcement date, display size, and starting price.
2. Format the extracted information as a JSON object with appropriate keys.
3. Do not include any other text in your response.

Response:
"""

# Generate response
result = generator(advanced_prompt, max_length=200, num_return_sequences=1)
print(result[0]['generated_text'])
```

## When and Where We Use Prompt Engineering

### When to Use Prompt Engineering:

1. **When working with zero-shot or few-shot learning**: Prompt engineering is essential when you want the model to perform tasks it wasn't explicitly trained on.

2. **When you need to control output format**: For applications requiring specific response formats (JSON, XML, etc.), prompt engineering helps structure the output.

3. **When fine-tuning isn't feasible**: When computational resources or labeled data are limited, prompt engineering offers a way to adapt model behavior without retraining.

4. **When you need to improve model reasoning**: For complex tasks requiring multi-step reasoning, techniques like Chain-of-Thought can significantly improve performance.

### Where We Use Prompt Engineering:

1. **Content Generation**: Crafting prompts for blog posts, marketing copy, or creative writing with specific styles and tones.

2. **Data Extraction and Processing**: Designing prompts to extract structured information from unstructured text.

3. **Question Answering Systems**: Formulating questions and context to elicit accurate answers from knowledge bases.

4. **Code Generation**: Creating prompts that generate code snippets or entire functions based on natural language descriptions.

**Coding Example - Prompt engineering for code generation:**

```python
from transformers import pipeline

# Initialize a text generation pipeline
generator = pipeline("text-generation", model="gpt2")

# Well-engineered prompt for code generation
code_prompt = """
Generate a Python function that calculates the factorial of a number using recursion.

Requirements:
1. The function should be named "factorial"
2. It should take one parameter "n"
3. It should return the factorial of n
4. Include error handling for negative numbers
5. Add a docstring explaining the function

```python
def factorial(n):
    \"\"\"
    Calculate the factorial of a number using recursion.
    
    Args:
        n (int): The number to calculate the factorial for
        
    Returns:
        int: The factorial of n
        
    Raises:
        ValueError: If n is negative
    \"\"\"
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    elif n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)
```

Now, generate a Python function that checks if a string is a palindrome:

```python
"""

# Generate response
result = generator(code_prompt, max_length=300, num_return_sequences=1)
print(result[0]['generated_text'])
```

## Real-World Applications and Examples

### 1. Content Creation for Marketing

**Application**: Generating marketing copy with specific brand voice and style guidelines.

**Coding Example - Persona-based marketing copy generation:**

```python
from transformers import pipeline

# Initialize a text generation pipeline
generator = pipeline("text-generation", model="gpt2")

# Persona prompt for marketing copy
marketing_prompt = """
You are a marketing specialist for a luxury watch brand. Write a product description for a new watch model called "Chronomaster Elite".

Brand Persona: Luxury, sophisticated, timeless
Product: Chronomaster Elite watch
Key Features: Swiss movement, sapphire crystal, water-resistant to 100m, limited edition
Target Audience: Affluent professionals aged 35-55
Tone: Elegant, exclusive, refined

Product Description:
"""

# Generate response
result = generator(marketing_prompt, max_length=200, num_return_sequences=1)
print(result[0]['generated_text'])
```

### 2. Automated Customer Support

**Application**: Creating a customer support bot that can answer common questions and provide troubleshooting steps.

**Coding Example - Few-shot prompting for customer support:**

```python
from transformers import pipeline

# Initialize a text generation pipeline
generator = pipeline("text-generation", model="gpt2")

# Few-shot prompt for customer support
support_prompt = """
As a customer support representative, provide helpful and polite responses to customer inquiries.

Example 1:
Customer: "My order hasn't arrived yet. It was supposed to be delivered yesterday."
Response: "I'm sorry to hear that your order hasn't arrived as expected. I'd be happy to help you track it down. Could you please provide your order number so I can check the status for you?"

Example 2:
Customer: "I received the wrong item in my package."
Response: "I apologize for the mix-up with your order. We'll make this right for you. Could you please share your order number and a photo of the item you received? We'll arrange for the correct item to be sent to you right away."

Now, respond to this customer inquiry:
Customer: "The product I received is damaged. What should I do?"
Response:
"""

# Generate response
result = generator(support_prompt, max_length=150, num_return_sequences=1)
print(result[0]['generated_text'])
```

### 3. Educational Content Generation

**Application**: Creating educational materials with explanations tailored to different learning levels and styles.

**Coding Example - Chain-of-Thought prompting for educational content:**

```python
from transformers import pipeline

# Initialize a text generation pipeline
generator = pipeline("text-generation", model="gpt2")

# CoT prompt for explaining a scientific concept
education_prompt = """
Explain the process of photosynthesis in a way that a middle school student can understand. Break down the explanation into simple steps.

Step 1: Start with what plants need for photosynthesis.
Step 2: Explain how plants capture sunlight.
Step 3: Describe how plants use water and carbon dioxide.
Step 4: Explain what plants produce as a result.
Step 5: Summarize the importance of photosynthesis.

Explanation:
"""

# Generate response
result = generator(education_prompt, max_length=300, num_return_sequences=1)
print(result[0]['generated_text'])
```

## Common Problems and How Advancements Have Solved Them

### 1. Problem: Inconsistent Output Quality

**Issue**: LLMs can produce inconsistent outputs for the same prompt, with variations in quality, format, and relevance.

**Solution**: Techniques like temperature control, top-k/nucleus sampling, and prompt standardization help improve consistency.

**Problem Code (Inconsistent generation):**

```python
from transformers import pipeline

# Initialize a text generation pipeline
generator = pipeline("text-generation", model="gpt2")

# Basic prompt without control parameters
prompt = "List three benefits of regular exercise:"

# Generate multiple responses to show inconsistency
for i in range(3):
    result = generator(prompt, max_length=100)
    print(f"Response {i+1}: {result[0]['generated_text']}\n")
```

**Enhancement Code (Controlled generation for consistency):**

```python
from transformers import pipeline

# Initialize a text generation pipeline
generator = pipeline("text-generation", model="gpt2")

# Improved prompt with clear instructions
prompt = """
List three benefits of regular exercise in a numbered format. Each benefit should be a complete sentence.

Benefits of regular exercise:
1.
"""

# Generate with controlled parameters for consistency
for i in range(3):
    result = generator(
        prompt,
        max_length=100,
        temperature=0.7,  # Lower temperature for more focused output
        top_p=0.9,       # Nucleus sampling
        do_sample=True,
        num_return_sequences=1
    )
    print(f"Response {i+1}: {result[0]['generated_text']}\n")
```

### 2. Problem: Difficulty with Complex Reasoning

**Issue**: LLMs often struggle with complex reasoning tasks, making errors in multi-step problems or logical deductions.

**Solution**: Chain-of-Thought prompting and other reasoning techniques break down complex problems into manageable steps.

**Problem Code (Direct question without reasoning steps):**

```python
from transformers import pipeline

# Initialize a text generation pipeline
generator = pipeline("text-generation", model="gpt2")

# Complex reasoning problem without CoT
reasoning_prompt = """
A bakery sells 3 types of cookies: chocolate chip at $1 each, oatmeal at $1.50 each, and peanut butter at $1.25 each. 
If a customer buys 2 chocolate chip, 3 oatmeal, and 1 peanut butter cookie, and pays with a $10 bill, how much change do they get?
"""

# Generate response
result = generator(reasoning_prompt, max_length=150)
print(result[0]['generated_text'])
```

**Enhancement Code (Chain-of-Thought for complex reasoning):**

```python
from transformers import pipeline

# Initialize a text generation pipeline
generator = pipeline("text-generation", model="gpt2")

# Complex reasoning problem with CoT
cot_reasoning_prompt = """
Solve this math problem by showing your step-by-step reasoning:

Problem: A bakery sells 3 types of cookies: chocolate chip at $1 each, oatmeal at $1.50 each, and peanut butter at $1.25 each. 
If a customer buys 2 chocolate chip, 3 oatmeal, and 1 peanut butter cookie, and pays with a $10 bill, how much change do they get?

Solution:
Step 1: Calculate the cost of chocolate chip cookies: 2 cookies × $1 each = $2
Step 2: Calculate the cost of oatmeal cookies: 3 cookies × $1.50 each = $4.50
Step 3: Calculate the cost of peanut butter cookies: 1 cookie × $1.25 each = $1.25
Step 4: Calculate the total cost: $2 + $4.50 + $1.25 = $7.75
Step 5: Calculate the change: $10 - $7.75 = $2.25

Final Answer: The customer gets $2.25 in change.

Now, solve this problem:
Problem: A store sells apples at $0.75 each, oranges at $0.60 each, and bananas at $0.40 each. 
If a customer buys 4 apples, 2 oranges, and 3 bananas, and pays with a $10 bill, how much change do they get?

Solution:
"""

# Generate response
result = generator(cot_reasoning_prompt, max_length=250)
print(result[0]['generated_text'])
```

### 3. Problem: Lack of Domain-Specific Knowledge

**Issue**: General-purpose LLMs may lack specialized knowledge for specific domains, leading to inaccurate or incomplete responses.

**Solution**: Combining prompt engineering with retrieval-augmented generation (RAG) to incorporate domain-specific knowledge.

**Problem Code (General model without domain knowledge):**

```python
from transformers import pipeline

# Initialize a text generation pipeline
generator = pipeline("text-generation", model="gpt2")

# Question requiring specialized medical knowledge
medical_prompt = """
What are the recommended treatments for stage 2 hypertension according to the latest guidelines?
"""

# Generate response
result = generator(medical_prompt, max_length=200)
print(result[0]['generated_text'])
```

**Enhancement Code (RAG approach with domain knowledge):**

```python
from transformers import pipeline

# Initialize a text generation pipeline
generator = pipeline("text-generation", model="gpt2")

# Domain-specific knowledge base (in practice, this would be retrieved from a database)
medical_knowledge = """
According to the 2023 American Heart Association guidelines:
- Stage 2 hypertension is defined as blood pressure ≥140/90 mmHg
- First-line treatments include ACE inhibitors, ARBs, calcium channel blockers, or thiazide diuretics
- Lifestyle modifications are recommended alongside medication
- Combination therapy may be needed for blood pressure >20/10 mmHg above target
"""

# Question with provided context
rag_medical_prompt = f"""
Based on the following medical guidelines, answer the question:

Medical Guidelines:
{medical_knowledge}

Question: What are the recommended treatments for stage 2 hypertension according to the latest guidelines?

Answer:
"""

# Generate response
result = generator(rag_medical_prompt, max_length=250)
print(result[0]['generated_text'])
```

## 🧪 Exercise: Prompt Crafting

Let's practice crafting effective prompts for different tasks:

```python
from transformers import pipeline
import json

# Initialize a text generation pipeline
generator = pipeline("text-generation", model="gpt2")

def evaluate_prompt(prompt, expected_aspects=None):
    """Generate a response and evaluate if it meets expected aspects"""
    print(f"Prompt:\n{prompt}\n")
    
    # Generate response
    result = generator(prompt, max_length=200, num_return_sequences=1)
    response = result[0]['generated_text']
    print(f"Response:\n{response}\n")
    
    # Evaluate if expected aspects are present
    if expected_aspects:
        evaluation = {}
        for aspect in expected_aspects:
            evaluation[aspect] = aspect.lower() in response.lower()
        
        print("Evaluation:")
        for aspect, present in evaluation.items():
            status = "✓" if present else "✗"
            print(f"{status} {aspect}: {'Present' if present else 'Missing'}")
    
    print("-" * 50)
    return response

# Exercise 1: Basic few-shot prompting
print("Exercise 1: Basic few-shot prompting")
few_shot_prompt = """
Classify the following emails as "Urgent", "Normal", or "Spam".

Example 1:
Email: "URGENT: Your account will be suspended in 24 hours"
Classification: Urgent

Example 2:
Email: "Meeting reminder: Team standup tomorrow at 10 AM"
Classification: Normal

Example 3:
Email: "Congratulations! You've won a $1,000,000 prize!"
Classification: Spam

Now, classify this email:
Email: "Action required: Please update your payment information"
Classification:
"""
evaluate_prompt(few_shot_prompt, expected_aspects=["Urgent", "Normal", "Spam"])

# Exercise 2: Chain-of-Thought prompting
print("Exercise 2: Chain-of-Thought prompting")
cot_prompt = """
Solve this word problem by showing your step-by-step reasoning:

Problem: A train leaves Chicago at 2 PM traveling at 60 mph. Another train leaves New York at 3 PM traveling at 50 mph. 
If the distance between Chicago and New York is 800 miles, at what time will the trains meet?

Solution:
Step 1: 
"""
evaluate_prompt(cot_prompt, expected_aspects=["train", "meet", "time"])

# Exercise 3: Persona prompting
print("Exercise 3: Persona prompting")
persona_prompt = """
You are a professional chef with a passion for Italian cuisine. Write a short paragraph describing how to make the perfect spaghetti carbonara.

Persona: Professional chef, Italian cuisine expert
Topic: Making spaghetti carbonara
Response:
"""
evaluate_prompt(persona_prompt, expected_aspects=["spaghetti", "carbonara", "eggs", "cheese"])

# Exercise 4: Output formatting
print("Exercise 4: Output formatting")
formatting_prompt = """
Extract the key information from the following restaurant review and format it as JSON.

Review: "I had an amazing dinner at Bella Italia last night. The pasta was perfectly cooked and the service was exceptional. 
I ordered the fettuccine alfredo for $18 and it was worth every penny. The atmosphere was cozy and romantic, perfect for a date night. 
I would definitely recommend this restaurant to anyone looking for authentic Italian cuisine."

Instructions:
1. Extract the restaurant name, dish ordered, price, atmosphere, and recommendation.
2. Format the extracted information as a JSON object.
3. Do not include any other text in your response.

Response:
"""
evaluate_prompt(formatting_prompt, expected_aspects=["restaurant", "dish", "price", "atmosphere"])
```

## 📌 Project: Prompt Testing Dashboard

Let's build a simple prompt testing dashboard that allows you to experiment with different prompts and evaluate their effectiveness:

```python
import ipywidgets as widgets
from IPython.display import display, HTML
from transformers import pipeline
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np

class PromptTestingDashboard:
    def __init__(self):
        # Initialize the model
        self.generator = pipeline("text-generation", model="gpt2")
        
        # Create UI elements
        self.create_ui()
        
        # Initialize results storage
        self.results = []
    
    def create_ui(self):
        # Create prompt input
        self.prompt_input = widgets.Textarea(
            value='',
            placeholder='Enter your prompt here...',
            description='Prompt:',
            layout=widgets.Layout(width='100%', height='150px')
        )
        
        # Create model parameters
        self.max_length = widgets.IntSlider(
            value=100,
            min=10,
            max=500,
            step=10,
            description='Max Length:',
            continuous_update=False
        )
        
        self.temperature = widgets.FloatSlider(
            value=0.7,
            min=0.1,
            max=2.0,
            step=0.1,
            description='Temperature:',
            continuous_update=False
        )
        
        self.top_p = widgets.FloatSlider(
            value=0.9,
            min=0.1,
            max=1.0,
            step=0.05,
            description='Top P:',
            continuous_update=False
        )
        
        # Create generate button
        self.generate_button = widgets.Button(
            description='Generate Response',
            button_style='primary',
            layout=widgets.Layout(width='200px')
        )
        self.generate_button.on_click(self.on_generate_click)
        
        # Create save result button
        self.save_button = widgets.Button(
            description='Save Result',
            button_style='success',
            layout=widgets.Layout(width='200px')
        )
        self.save_button.on_click(self.on_save_click)
        
        # Create clear results button
        self.clear_button = widgets.Button(
            description='Clear Results',
            button_style='warning',
            layout=widgets.Layout(width='200px')
        )
        self.clear_button.on_click(self.on_clear_click)
        
        # Create output area
        self.output_area = widgets.Output()
        
        # Create results table
        self.results_table = widgets.Output()
        
        # Create evaluation metrics
        self.evaluation_metrics = widgets.Output()
        
        # Arrange UI elements
        prompt_params = widgets.VBox([
            self.prompt_input,
            widgets.HBox([self.max_length, self.temperature, self.top_p]),
            widgets.HBox([self.generate_button, self.save_button, self.clear_button])
        ])
        
        self.dashboard = widgets.Tab([
            widgets.VBox([prompt_params, self.output_area]),
            widgets.VBox([self.results_table]),
            widgets.VBox([self.evaluation_metrics])
        ])
        self.dashboard.set_title(0, 'Prompt Testing')
        self.dashboard.set_title(1, 'Results History')
        self.dashboard.set_title(2, 'Evaluation Metrics')
    
    def on_generate_click(self, b):
        with self.output_area:
            self.output_area.clear_output()
            
            # Get prompt and parameters
            prompt = self.prompt_input.value
            max_length = self.max_length.value
            temperature = self.temperature.value
            top_p = self.top_p.value
            
            if not prompt:
                print("Please enter a prompt.")
                return
            
            print("Generating response...")
            start_time = time.time()
            
            # Generate response
            try:
                result = self.generator(
                    prompt,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    num_return_sequences=1
                )
                
                response = result[0]['generated_text']
                generation_time = time.time() - start_time
                
                # Display results
                print(f"Generation time: {generation_time:.2f} seconds")
                print("\nGenerated Response:")
                print("-" * 50)
                print(response)
                
                # Store current result
                self.current_result = {
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'prompt': prompt,
                    'response': response,
                    'max_length': max_length,
                    'temperature': temperature,
                    'top_p': top_p,
                    'generation_time': generation_time
                }
                
            except Exception as e:
                print(f"Error generating response: {str(e)}")
    
    def on_save_click(self, b):
        if hasattr(self, 'current_result'):
            self.results.append(self.current_result)
            self.update_results_table()
            self.update_evaluation_metrics()
            with self.output_area:
                print("Result saved successfully!")
        else:
            with self.output_area:
                print("No result to save. Please generate a response first.")
    
    def on_clear_click(self, b):
        self.results = []
        self.update_results_table()
        self.update_evaluation_metrics()
        with self.results_table:
            self.results_table.clear_output()
            print("Results cleared.")
    
    def update_results_table(self):
        with self.results_table:
            self.results_table.clear_output()
            
            if not self.results:
                print("No results saved yet.")
                return
            
            # Create DataFrame from results
            df = pd.DataFrame(self.results)
            
            # Display table
            display(HTML(df[['timestamp', 'max_length', 'temperature', 'top_p', 'generation_time']].to_html(index=False)))
            
            # Show prompt and response for the most recent result
            if self.results:
                latest = self.results[-1]
                print("\nMost Recent Result:")
                print(f"Prompt: {latest['prompt']}")
                print(f"Response: {latest['response']}")
    
    def update_evaluation_metrics(self):
        with self.evaluation_metrics:
            self.evaluation_metrics.clear_output()
            
            if not self.results:
                print("No results to evaluate.")
                return
            
            # Calculate metrics
            generation_times = [r['generation_time'] for r in self.results]
            response_lengths = [len(r['response'].split()) for r in self.results]
            temperatures = [r['temperature'] for r in self.results]
            
            # Create plots
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            
            # Generation time over time
            axs[0, 0].plot(range(len(generation_times)), generation_times, 'o-')
            axs[0, 0].set_title('Generation Time Over Experiments')
            axs[0, 0].set_xlabel('Experiment Number')
            axs[0, 0].set_ylabel('Time (seconds)')
            
            # Response length over time
            axs[0, 1].plot(range(len(response_lengths)), response_lengths, 'o-')
            axs[0, 1].set_title('Response Length Over Experiments')
            axs[0, 1].set_xlabel('Experiment Number')
            axs[0, 1].set_ylabel('Word Count')
            
            # Temperature vs. Generation Time
            axs[1, 0].scatter(temperatures, generation_times)
            axs[1, 0].set_title('Temperature vs. Generation Time')
            axs[1, 0].set_xlabel('Temperature')
            axs[1, 0].set_ylabel('Time (seconds)')
            
            # Temperature vs. Response Length
            axs[1, 1].scatter(temperatures, response_lengths)
            axs[1, 1].set_title('Temperature vs. Response Length')
            axs[1, 1].set_xlabel('Temperature')
            axs[1, 1].set_ylabel('Word Count')
            
            plt.tight_layout()
            plt.show()
            
            # Print summary statistics
            print("\nSummary Statistics:")
            print(f"Average Generation Time: {np.mean(generation_times):.2f} seconds")
            print(f"Average Response Length: {np.mean(response_lengths):.2f} words")
            print(f"Temperature Range: {min(temperatures):.1f} - {max(temperatures):.1f}")
    
    def display(self):
        display(self.dashboard)

# Create and display the dashboard
dashboard = PromptTestingDashboard()
dashboard.display()
```

This prompt testing dashboard allows you to:
1. Experiment with different prompts and model parameters
2. Generate and evaluate responses
3. Save and compare results across multiple experiments
4. Visualize metrics like generation time and response length
5. Analyze how different parameters affect output quality

