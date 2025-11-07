# Day 48 – LLMOps Best Practices

## Prompt Evaluation

### What is it?

Prompt evaluation is the systematic process of measuring, analyzing, and comparing the quality and effectiveness of prompts used with Large Language Models (LLMs). It ensures that prompts consistently produce desired outputs across different models, tasks, and contexts. Effective prompt evaluation involves both quantitative metrics (accuracy, relevance, coherence) and qualitative assessments (fluency, tone, safety), enabling data-driven decisions for prompt optimization and versioning.

### PyTorch Implementation

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import re
import time
from sklearn.metrics.pairwise import cosine_similarity

# Text embedding model for semantic evaluation
class PromptEvaluator:
    def __init__(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModel.from_pretrained(embedding_model)
        self.model.eval()
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a text"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pooling
            embeddings = self.mean_pooling(outputs, inputs['attention_mask'])
            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.numpy()
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        return cosine_similarity(emb1, emb2)[0][0]
    
    def evaluate_prompt(self, prompt: str, expected_output: str, actual_output: str) -> Dict[str, float]:
        """Evaluate a prompt based on multiple metrics"""
        metrics = {}
        
        # Semantic similarity
        metrics['semantic_similarity'] = self.semantic_similarity(expected_output, actual_output)
        
        # Length ratio
        expected_len = len(expected_output.split())
        actual_len = len(actual_output.split())
        metrics['length_ratio'] = min(actual_len / expected_len, expected_len / actual_len) if expected_len > 0 and actual_len > 0 else 0
        
        # Keyword coverage
        expected_keywords = set(expected_output.lower().split())
        actual_keywords = set(actual_output.lower().split())
        if expected_keywords:
            metrics['keyword_coverage'] = len(expected_keywords.intersection(actual_keywords)) / len(expected_keywords)
        else:
            metrics['keyword_coverage'] = 0
        
        # Fluency (simple proxy)
        metrics['fluency_score'] = min(1.0, len(actual_output) / 10.0) if len(actual_output) > 0 else 0
        
        return metrics

# Demonstrate prompt evaluation
def demonstrate_prompt_evaluation():
    """Demonstrate prompt evaluation functionality"""
    
    print("Prompt Evaluation Demo")
    print("=" * 40)
    
    # Initialize evaluator
    evaluator = PromptEvaluator()
    
    # Sample prompts and outputs
    test_cases = [
        {
            'prompt': "Explain what machine learning is in simple terms.",
            'expected': "Machine learning is a type of artificial intelligence that allows computers to learn from data without being explicitly programmed.",
            'actual': "Machine learning enables computers to learn patterns from data and make predictions without explicit programming."
        },
        {
            'prompt': "Summarize the benefits of exercise.",
            'expected': "Exercise improves physical health, mental wellbeing, and overall quality of life.",
            'actual': "Regular exercise enhances physical fitness, reduces stress, and boosts mood and energy levels."
        },
        {
            'prompt': "What is the capital of France?",
            'expected': "The capital of France is Paris.",
            'actual': "Paris is the capital city of France."
        }
    ]
    
    results = []
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Prompt: {case['prompt']}")
        print(f"Expected: {case['expected']}")
        print(f"Actual: {case['actual']}")
        
        metrics = evaluator.evaluate_prompt(case['prompt'], case['expected'], case['actual'])
        results.append(metrics)
        
        print("Metrics:")
        for metric, score in metrics.items():
            print(f"  {metric}: {score:.4f}")
    
    return results

evaluation_results = demonstrate_prompt_evaluation()
```

### When and where we use prompt evaluation

- **Prompt engineering**: Iteratively improving prompts based on performance metrics
- **A/B testing**: Comparing different prompt versions in production
- **Model migration**: Ensuring consistent performance when switching LLMs
- **Quality assurance**: Validating prompt performance before deployment
- **Continuous monitoring**: Tracking prompt performance over time
- **Regulatory compliance**: Ensuring prompts meet safety and fairness standards

### Real-world examples

- **Customer support**: Evaluating prompts that generate responses to user queries
- **Content generation**: Assessing prompts for blog posts, product descriptions, and marketing copy
- **Code generation**: Measuring prompt effectiveness for generating functional code
- **Medical applications**: Ensuring prompts produce accurate and safe medical information
- **Financial services**: Validating prompts for risk assessment and investment advice

### Problems and Advancements

**Problem**: Subjective evaluation criteria and lack of standardized metrics.
**Advancement**: Automated evaluation frameworks with multiple dimensions and human-in-the-loop validation.

**Problem**: Context-dependent prompt performance that varies across domains.
**Advancement**: Domain-specific evaluation benchmarks and adaptive scoring systems.

## Feedback Loops

### What is it?

Feedback loops in LLMOps are mechanisms that capture user interactions, model outputs, and performance metrics to continuously improve prompt effectiveness and model behavior. These loops collect data from various sources (user ratings, click-through rates, error reports) and feed it back into the prompt optimization pipeline, enabling iterative refinement and adaptation to changing requirements or user preferences.

### PyTorch Implementation

```python
import json
from datetime import datetime, timedelta
from collections import defaultdict, deque
import sqlite3

class FeedbackLoop:
    def __init__(self, db_path="prompt_feedback.db"):
        self.db_path = db_path
        self.init_database()
        self.feedback_buffer = deque(maxlen=1000)  # In-memory buffer for recent feedback
        
    def init_database(self):
        """Initialize the feedback database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prompt_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_id TEXT,
                prompt_version TEXT,
                user_id TEXT,
                input_text TEXT,
                model_output TEXT,
                user_rating REAL,
                user_feedback TEXT,
                metadata TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prompt_performance (
                prompt_id TEXT,
                prompt_version TEXT,
                metric_name TEXT,
                metric_value REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def record_feedback(self, 
                       prompt_id: str,
                       prompt_version: str,
                       user_id: str,
                       input_text: str,
                       model_output: str,
                       user_rating: float = None,
                       user_feedback: str = None,
                       metadata: Dict = None):
        """Record user feedback for a prompt"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO prompt_feedback 
            (prompt_id, prompt_version, user_id, input_text, model_output, user_rating, user_feedback, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (prompt_id, prompt_version, user_id, input_text, model_output, user_rating, user_feedback, json.dumps(metadata or {})))
        
        conn.commit()
        conn.close()
        
        # Add to in-memory buffer
        feedback_record = {
            'prompt_id': prompt_id,
            'prompt_version': prompt_version,
            'user_id': user_id,
            'input_text': input_text,
            'model_output': model_output,
            'user_rating': user_rating,
            'user_feedback': user_feedback,
            'metadata': metadata,
            'timestamp': datetime.now()
        }
        self.feedback_buffer.append(feedback_record)
    
    def get_feedback_stats(self, prompt_id: str, hours: int = 24) -> Dict:
        """Get feedback statistics for a prompt"""
        since_time = datetime.now() - timedelta(hours=hours)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get rating statistics
        cursor.execute('''
            SELECT AVG(user_rating), COUNT(user_rating), MIN(user_rating), MAX(user_rating)
            FROM prompt_feedback 
            WHERE prompt_id = ? AND timestamp > ?
        ''', (prompt_id, since_time))
        
        avg_rating, rating_count, min_rating, max_rating = cursor.fetchone()
        
        # Get feedback count
        cursor.execute('''
            SELECT COUNT(*) FROM prompt_feedback 
            WHERE prompt_id = ? AND timestamp > ?
        ''', (prompt_id, since_time))
        
        total_feedback = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'prompt_id': prompt_id,
            'hours': hours,
            'avg_rating': avg_rating or 0,
            'rating_count': rating_count or 0,
            'min_rating': min_rating or 0,
            'max_rating': max_rating or 0,
            'total_feedback': total_feedback,
            'feedback_rate': rating_count / total_feedback if total_feedback > 0 else 0
        }
    
    def analyze_feedback_trends(self, prompt_id: str, days: int = 7) -> Dict:
        """Analyze feedback trends over time"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get daily feedback statistics
        cursor.execute('''
            SELECT 
                DATE(timestamp) as date,
                AVG(user_rating) as avg_rating,
                COUNT(*) as feedback_count
            FROM prompt_feedback 
            WHERE prompt_id = ? AND timestamp > DATE('now', '-{} days')
            GROUP BY DATE(timestamp)
            ORDER BY date
        '''.format(days), (prompt_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return {'trend_data': [], 'trend_summary': 'No feedback data available'}
        
        trend_data = []
        for date, avg_rating, count in rows:
            trend_data.append({
                'date': date,
                'avg_rating': avg_rating,
                'feedback_count': count
            })
        
        # Calculate trend summary
        ratings = [d['avg_rating'] for d in trend_data if d['avg_rating'] is not None]
        if len(ratings) > 1:
            trend = 'improving' if ratings[-1] > ratings[0] else 'declining' if ratings[-1] < ratings[0] else 'stable'
        else:
            trend = 'insufficient data'
        
        return {
            'trend_data': trend_data,
            'trend_summary': trend
        }
    
    def trigger_prompt_update(self, prompt_id: str, threshold: float = 3.5) -> bool:
        """Determine if a prompt needs updating based on feedback"""
        stats = self.get_feedback_stats(prompt_id, hours=48)
        
        if stats['rating_count'] >= 10 and stats['avg_rating'] < threshold:
            return True
        return False

# Demonstrate feedback loop functionality
def demonstrate_feedback_loop():
    """Demonstrate feedback loop functionality"""
    
    print("Feedback Loop Demo")
    print("=" * 30)
    
    # Initialize feedback loop
    feedback_system = FeedbackLoop()
    
    # Simulate user feedback
    sample_feedback = [
        {
            'prompt_id': 'summary_v1',
            'prompt_version': '1.0',
            'user_id': 'user123',
            'input_text': 'The quick brown fox jumps over the lazy dog.',
            'model_output': 'A fox jumps over a dog.',
            'user_rating': 4.2,
            'user_feedback': 'Good summary but could be more detailed.'
        },
        {
            'prompt_id': 'summary_v1',
            'prompt_version': '1.0',
            'user_id': 'user456',
            'input_text': 'Artificial intelligence is transforming the world.',
            'model_output': 'AI is changing everything.',
            'user_rating': 2.8,
            'user_feedback': 'Too brief and lacks substance.'
        },
        {
            'prompt_id': 'summary_v1',
            'prompt_version': '1.0',
            'user_id': 'user789',
            'input_text': 'Machine learning algorithms learn from data.',
            'model_output': 'ML algorithms learn patterns from data to make predictions.',
            'user_rating': 4.8,
            'user_feedback': 'Excellent summary!'
        }
    ]
    
    # Record feedback
    for feedback in sample_feedback:
        feedback_system.record_feedback(**feedback)
        print(f"Recorded feedback from {feedback['user_id']}")
    
    # Get feedback statistics
    stats = feedback_system.get_feedback_stats('summary_v1', hours=24)
    print(f"\nFeedback Statistics for 'summary_v1':")
    print(f"Average Rating: {stats['avg_rating']:.2f}")
    print(f"Total Feedback: {stats['total_feedback']}")
    print(f"Rating Rate: {stats['feedback_rate']:.2%}")
    
    # Check if prompt needs updating
    needs_update = feedback_system.trigger_prompt_update('summary_v1', threshold=4.0)
    print(f"\nPrompt needs update: {needs_update}")
    
    # Analyze trends
    trends = feedback_system.analyze_feedback_trends('summary_v1', days=7)
    print(f"Trend analysis: {trends['trend_summary']}")
    
    return feedback_system

feedback_loop = demonstrate_feedback_loop()
```

### When and where we use feedback loops

- **Active learning**: Prioritizing uncertain predictions for human review
- **Personalization**: Adapting prompts based on individual user preferences
- **Error correction**: Learning from user corrections and model mistakes
- **Performance monitoring**: Detecting drift in model performance over time
- **A/B testing**: Collecting user preferences between different prompt versions
- **Content moderation**: Improving safety and appropriateness of responses

### Real-world examples

- **Chatbots**: Learning from user thumbs up/down feedback to improve responses
- **Recommendation systems**: Adapting prompts based on click-through and engagement rates
- **Search engines**: Refining query understanding based on user click patterns
- **E-commerce**: Optimizing product description prompts based on conversion rates
- **Healthcare**: Improving medical response prompts based on clinician feedback

### Problems and Advancements

**Problem**: Feedback bias and low response rates from users.
**Advancement**: Implicit feedback collection and active learning strategies.

**Problem**: Delayed feedback making real-time adaptation challenging.
**Advancement**: Streaming feedback processing and online learning algorithms.

## 🧪 Exercise: Prompt comparison

```python
def prompt_comparison_exercise():
    """Complete exercise: Compare different prompts using evaluation metrics"""
    
    print("Prompt Comparison Exercise")
    print("=" * 50)
    
    # Initialize evaluator
    evaluator = PromptEvaluator()
    feedback_system = FeedbackLoop()
    
    # Define prompt variants for the same task
    prompt_variants = {
        'basic': "Summarize this text: {text}",
        'detailed': "Provide a concise summary of the following text, highlighting the key points and main ideas: {text}",
        'instruction': "Your task is to create a summary of the given text. Focus on the most important information and keep it brief: {text}",
        'structured': "Summarize the text below in exactly 3 sentences:\n\n{text}",
        'role_based': "You are an expert editor. Create a professional summary of this text: {text}"
    }
    
    # Test texts
    test_texts = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to parse data, learn from that data, and make informed decisions based on what it has learned.",
        "Climate change is affecting weather patterns globally, leading to more frequent and severe natural disasters. Rising temperatures are causing sea levels to rise and disrupting ecosystems worldwide.",
        "Blockchain technology provides a decentralized and secure way to record transactions. It uses cryptographic techniques to ensure data integrity and transparency across a distributed network."
    ]
    
    # Expected summaries (ideal outputs)
    expected_summaries = [
        "Machine learning enables systems to learn from data without explicit programming using algorithms to make informed decisions.",
        "Climate change is causing more severe weather events and rising sea levels by disrupting global weather patterns and ecosystems.",
        "Blockchain provides secure, decentralized transaction recording using cryptography for data integrity and transparency."
    ]
    
    # Mock LLM responses for each prompt variant
    def mock_llm_response(prompt_template: str, text: str) -> str:
        """Simulate LLM response based on prompt quality"""
        prompt = prompt_template.format(text=text)
        
        # Simple simulation based on prompt type
        if 'basic' in prompt_template.lower():
            # Basic prompt produces shorter, less detailed responses
            return text.split('.')[0] + '.'
        elif 'detailed' in prompt_template.lower():
            # Detailed prompt produces comprehensive responses
            sentences = text.split('.')
            return ' '.join([s.strip() + '.' for s in sentences[:2] if s.strip()])
        elif 'instruction' in prompt_template.lower():
            # Instruction prompt produces task-focused responses
            return f"Summary: {text.split('.')[0]}."
        elif 'structured' in prompt_template.lower():
            # Structured prompt produces exactly 3 sentences
            sentences = text.split('.')
            result_sentences = []
            for i in range(min(3, len(sentences))):
                if sentences[i].strip():
                    result_sentences.append(sentences[i].strip() + '.')
            return ' '.join(result_sentences)
        elif 'role_based' in prompt_template.lower():
            # Role-based prompt produces professional, polished responses
            return f"Professional Summary: {text.split('.')[0]}. This represents a key advancement in the field."
        else:
            return text.split('.')[0] + '.'
    
    # Evaluate all prompt variants
    comparison_results = {}
    
    for prompt_name, prompt_template in prompt_variants.items():
        print(f"\nEvaluating prompt: {prompt_name}")
        variant_results = []
        
        for i, (text, expected) in enumerate(zip(test_texts, expected_summaries)):
            actual = mock_llm_response(prompt_template, text)
            
            # Evaluate
            metrics = evaluator.evaluate_prompt(prompt_template, expected, actual)
            
            # Record feedback (simulate user rating based on quality)
            user_rating = min(5.0, max(1.0, metrics['semantic_similarity'] * 5))
            
            feedback_system.record_feedback(
                prompt_id=prompt_name,
                prompt_version='1.0',
                user_id=f'test_user_{i}',
                input_text=text,
                model_output=actual,
                user_rating=user_rating,
                user_feedback=f'Quality score: {user_rating:.1f}',
                metadata={'test_case': i, 'metrics': metrics}
            )
            
            variant_results.append({
                'text_index': i,
                'actual_output': actual,
                'metrics': metrics,
                'user_rating': user_rating
            })
            
            print(f"  Test {i+1}: Semantic similarity = {metrics['semantic_similarity']:.4f}, Rating = {user_rating:.1f}")
        
        comparison_results[prompt_name] = variant_results
    
    # Generate comparison report
    print(f"\n" + "="*60)
    print("PROMPT COMPARISON REPORT")
    print("="*60)
    
    # Calculate average metrics for each prompt
    prompt_averages = {}
    for prompt_name, results in comparison_results.items():
        avg_semantic = np.mean([r['metrics']['semantic_similarity'] for r in results])
        avg_rating = np.mean([r['user_rating'] for r in results])
        avg_length_ratio = np.mean([r['metrics']['length_ratio'] for r in results])
        avg_keyword_coverage = np.mean([r['metrics']['keyword_coverage'] for r in results])
        
        prompt_averages[prompt_name] = {
            'avg_semantic_similarity': avg_semantic,
            'avg_user_rating': avg_rating,
            'avg_length_ratio': avg_length_ratio,
            'avg_keyword_coverage': avg_keyword_coverage
        }
        
        print(f"\n{prompt_name.upper()}:")
        print(f"  Avg Semantic Similarity: {avg_semantic:.4f}")
        print(f"  Avg User Rating: {avg_rating:.2f}")
        print(f"  Avg Length Ratio: {avg_length_ratio:.4f}")
        print(f"  Avg Keyword Coverage: {avg_keyword_coverage:.4f}")
    
    # Find best prompt
    best_prompt = max(prompt_averages.keys(), key=lambda x: prompt_averages[x]['avg_user_rating'])
    print(f"\n🏆 BEST PROMPT: {best_prompt}")
    print(f"   Average User Rating: {prompt_averages[best_prompt]['avg_user_rating']:.2f}")
    
    # Create visualization data
    df_data = []
    for prompt_name, metrics in prompt_averages.items():
        df_data.append({
            'Prompt': prompt_name,
            'Semantic Similarity': metrics['avg_semantic_similarity'],
            'User Rating': metrics['avg_user_rating'],
            'Length Ratio': metrics['avg_length_ratio'],
            'Keyword Coverage': metrics['avg_keyword_coverage']
        })
    
    results_df = pd.DataFrame(df_data)
    print(f"\nComparison DataFrame:")
    print(results_df.round(4))
    
    return comparison_results, prompt_averages, results_df

# Run the prompt comparison exercise
comparison_results, prompt_averages, results_df = prompt_comparison_exercise()
```

## 📌 Project: LLM prompt registry

```python
import yaml
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
import sqlite3

class PromptRegistry:
    """Complete LLM prompt registry system with versioning and evaluation"""
    
    def __init__(self, registry_path: str = "prompt_registry", db_path: str = "prompt_registry.db"):
        self.registry_path = Path(registry_path)
        self.db_path = db_path
        self.registry_path.mkdir(exist_ok=True)
        
        # Initialize database
        self.init_database()
        
        # Initialize evaluator and feedback system
        self.evaluator = PromptEvaluator()
        self.feedback_system = FeedbackLoop(db_path=db_path)
    
    def init_database(self):
        """Initialize the prompt registry database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_id TEXT UNIQUE,
                prompt_text TEXT,
                description TEXT,
                tags TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                version TEXT,
                metadata TEXT,
                hash TEXT UNIQUE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prompt_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_id TEXT,
                evaluation_metrics TEXT,
                evaluation_score REAL,
                evaluator TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prompt_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_id TEXT,
                version TEXT,
                prompt_text TEXT,
                changes TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def generate_prompt_hash(self, prompt_text: str) -> str:
        """Generate a hash for prompt text to detect duplicates"""
        return hashlib.md5(prompt_text.encode('utf-8')).hexdigest()
    
    def register_prompt(self, 
                       prompt_id: str,
                       prompt_text: str,
                       description: str = "",
                       tags: List[str] = None,
                       metadata: Dict = None) -> str:
        """Register a new prompt in the registry"""
        prompt_hash = self.generate_prompt_hash(prompt_text)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO prompts 
                (prompt_id, prompt_text, description, tags, version, metadata, hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                prompt_id,
                prompt_text,
                description,
                ','.join(tags or []),
                '1.0',
                json.dumps(metadata or {}),
                prompt_hash
            ))
            
            conn.commit()
            print(f"✅ Prompt '{prompt_id}' registered successfully!")
            
            # Save to YAML file
            self._save_prompt_to_yaml(prompt_id, prompt_text, description, tags, metadata, '1.0')
            
            return prompt_id
            
        except sqlite3.IntegrityError as e:
            if 'hash' in str(e):
                print(f"⚠️  Prompt already exists (duplicate hash): {prompt_hash}")
            elif 'prompt_id' in str(e):
                print(f"⚠️  Prompt ID already exists: {prompt_id}")
            else:
                print(f"❌ Error registering prompt: {e}")
            return None
        
        finally:
            conn.close()
    
    def _save_prompt_to_yaml(self, prompt_id: str, prompt_text: str, description: str, 
                           tags: List[str], metadata: Dict, version: str):
        """Save prompt to YAML file"""
        prompt_data = {
            'prompt_id': prompt_id,
            'version': version,
            'prompt_text': prompt_text,
            'description': description,
            'tags': tags or [],
            'metadata': metadata or {},
            'created_at': datetime.now().isoformat()
        }
        
        yaml_file = self.registry_path / f"{prompt_id}_v{version}.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(prompt_data, f, default_flow_style=False, indent=2)
    
    def get_prompt(self, prompt_id: str, version: str = None) -> Optional[Dict]:
        """Retrieve a prompt from the registry"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if version:
            cursor.execute('''
                SELECT prompt_text, description, tags, metadata, version 
                FROM prompts 
                WHERE prompt_id = ? AND version = ?
            ''', (prompt_id, version))
        else:
            cursor.execute('''
                SELECT prompt_text, description, tags, metadata, version 
                FROM prompts 
                WHERE prompt_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            ''', (prompt_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'prompt_id': prompt_id,
                'prompt_text': result[0],
                'description': result[1],
                'tags': result[2].split(',') if result[2] else [],
                'metadata': json.loads(result[3]) if result[3] else {},
                'version': result[4]
            }
        return None
    
    def list_prompts(self, tag: str = None) -> List[Dict]:
        """List all prompts in the registry"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if tag:
            cursor.execute('''
                SELECT prompt_id, description, tags, version, created_at 
                FROM prompts 
                WHERE tags LIKE ?
                ORDER BY created_at DESC
            ''', (f'%{tag}%',))
        else:
            cursor.execute('''
                SELECT prompt_id, description, tags, version, created_at 
                FROM prompts 
                ORDER BY created_at DESC
            ''')
        
        results = cursor.fetchall()
        conn.close()
        
        prompts = []
        for row in results:
            prompts.append({
                'prompt_id': row[0],
                'description': row[1],
                'tags': row[2].split(',') if row[2] else [],
                'version': row[3],
                'created_at': row[4]
            })
        
        return prompts
    
    def evaluate_prompt(self, prompt_id: str, test_cases: List[Dict]) -> Dict:
        """Evaluate a prompt against test cases"""
        prompt_data = self.get_prompt(prompt_id)
        if not prompt_data:
            raise ValueError(f"Prompt '{prompt_id}' not found in registry")
        
        prompt_text = prompt_data['prompt_text']
        evaluation_results = []
        
        for test_case in test_cases:
            # Simulate LLM response
            actual_output = self._simulate_llm_response(prompt_text, test_case.get('input', ''))
            
            # Evaluate
            metrics = self.evaluator.evaluate_prompt(
                prompt_text,
                test_case.get('expected_output', ''),
                actual_output
            )
            
            evaluation_results.append({
                'test_case': test_case,
                'actual_output': actual_output,
                'metrics': metrics
            })
        
        # Calculate overall score
        avg_semantic_similarity = np.mean([r['metrics']['semantic_similarity'] for r in evaluation_results])
        avg_keyword_coverage = np.mean([r['metrics']['keyword_coverage'] for r in evaluation_results])
        
        overall_score = (avg_semantic_similarity + avg_keyword_coverage) / 2
        
        # Store evaluation in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO prompt_evaluations 
            (prompt_id, evaluation_metrics, evaluation_score, evaluator)
            VALUES (?, ?, ?, ?)
        ''', (
            prompt_id,
            json.dumps(evaluation_results),
            overall_score,
            'automated_evaluator'
        ))
        
        conn.commit()
        conn.close()
        
        return {
            'prompt_id': prompt_id,
            'overall_score': overall_score,
            'detailed_results': evaluation_results,
            'timestamp': datetime.now().isoformat()
        }
    
    def _simulate_llm_response(self, prompt_template: str, input_text: str) -> str:
        """Simulate LLM response for evaluation"""
        # Simple simulation - in real use, this would call an actual LLM
        if '{text}' in prompt_template:
            full_prompt = prompt_template.format(text=input_text)
        else:
            full_prompt = f"{prompt_template}\n\n{input_text}"
        
        # Generate simple response based on prompt content
        if 'summarize' in prompt_template.lower():
            return f"Summary of: {input_text[:50]}..."
        elif 'explain' in prompt_template.lower():
            return f"Explanation: {input_text}"
        else:
            return f"Response to: {input_text[:30]}..."
    
    def version_prompt(self, prompt_id: str, new_prompt_text: str, changes: str = "") -> str:
        """Create a new version of an existing prompt"""
        current_prompt = self.get_prompt(prompt_id)
        if not current_prompt:
            raise ValueError(f"Prompt '{prompt_id}' not found")
        
        # Get current version and increment
        current_version = current_prompt['version']
        version_parts = current_version.split('.')
        new_version = f"{version_parts[0]}.{int(version_parts[1]) + 1}"
        
        # Register new version
        new_prompt_id = f"{prompt_id}"
        metadata = current_prompt['metadata']
        metadata['previous_version'] = current_version
        metadata['changes'] = changes
        
        success = self.register_prompt(
            prompt_id=new_prompt_id,
            prompt_text=new_prompt_text,
            description=current_prompt['description'],
            tags=current_prompt['tags'],
            metadata=metadata
        )
        
        if success:
            # Record version in versions table
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO prompt_versions 
                (prompt_id, version, prompt_text, changes)
                VALUES (?, ?, ?, ?)
            ''', (prompt_id, new_version, new_prompt_text, changes))
            
            conn.commit()
            conn.close()
            
            print(f"✅ New version {new_version} created for prompt '{prompt_id}'")
            return new_version
        
        return None
    
    def export_registry(self, output_path: str = "prompt_registry_export.json") -> str:
        """Export the entire registry to JSON"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM prompts')
        prompts = cursor.fetchall()
        
        cursor.execute('SELECT * FROM prompt_evaluations')
        evaluations = cursor.fetchall()
        
        conn.close()
        
        export_data = {
            'prompts': [],
            'evaluations': [],
            'export_timestamp': datetime.now().isoformat()
        }
        
        # Convert prompts to dict
        for prompt in prompts:
            export_data['prompts'].append({
                'id': prompt[0],
                'prompt_id': prompt[1],
                'prompt_text': prompt[2],
                'description': prompt[3],
                'tags': prompt[4],
                'created_at': prompt[5],
                'updated_at': prompt[6],
                'version': prompt[7],
                'metadata': prompt[8],
                'hash': prompt[9]
            })
        
        # Convert evaluations to dict  
        for eval_ in evaluations:
            export_data['evaluations'].append({
                'id': eval_[0],
                'prompt_id': eval_[1],
                'evaluation_metrics': eval_[2],
                'evaluation_score': eval_[3],
                'evaluator': eval_[4],
                'timestamp': eval_[5]
            })
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✅ Registry exported to {output_path}")
        return output_path

# Demonstrate the prompt registry project
def demonstrate_prompt_registry():
    """Demonstrate the complete prompt registry system"""
    
    print("LLM Prompt Registry Project")
    print("=" * 50)
    
    # Initialize registry
    registry = PromptRegistry()
    
    # Register sample prompts
    sample_prompts = [
        {
            'prompt_id': 'text_summarization_basic',
            'prompt_text': 'Summarize this text: {text}',
            'description': 'Basic text summarization prompt',
            'tags': ['summarization', 'basic'],
            'metadata': {'task_type': 'summarization', 'difficulty': 'easy'}
        },
        {
            'prompt_id': 'text_summarization_advanced',
            'prompt_text': 'Provide a concise summary of the following text, highlighting the key points and main ideas: {text}',
            'description': 'Advanced text summarization prompt with detailed instructions',
            'tags': ['summarization', 'advanced'],
            'metadata': {'task_type': 'summarization', 'difficulty': 'medium'}
        },
        {
            'prompt_id': 'question_answering',
            'prompt_text': 'Answer the following question based on the provided context:\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:',
            'description': 'Question answering with context',
            'tags': ['qa', 'context-based'],
            'metadata': {'task_type': 'question_answering', 'requires_context': True}
        },
        {
            'prompt_id': 'code_generation',
            'prompt_text': 'Write a Python function that {task}. Include proper documentation and error handling.',
            'description': 'Code generation prompt for Python functions',
            'tags': ['code', 'python', 'generation'],
            'metadata': {'task_type': 'code_generation', 'language': 'python'}
        }
    ]
    
    print("Registering sample prompts...")
    for prompt_data in sample_prompts:
        registry.register_prompt(**prompt_data)
    
    # List all prompts
    print(f"\n📋 All registered prompts:")
    all_prompts = registry.list_prompts()
    for prompt in all_prompts:
        print(f"  • {prompt['prompt_id']} (v{prompt['version']}) - {', '.join(prompt['tags'])}")
    
    # List prompts by tag
    print(f"\n🏷️  Prompts with 'summarization' tag:")
    summarization_prompts = registry.list_prompts(tag='summarization')
    for prompt in summarization_prompts:
        print(f"  • {prompt['prompt_id']} - {prompt['description']}")
    
    # Retrieve a specific prompt
    print(f"\n🔍 Retrieving prompt 'text_summarization_basic':")
    retrieved_prompt = registry.get_prompt('text_summarization_basic')
    if retrieved_prompt:
        print(f"Prompt text: {retrieved_prompt['prompt_text']}")
        print(f"Version: {retrieved_prompt['version']}")
        print(f"Tags: {', '.join(retrieved_prompt['tags'])}")
    
    # Create test cases for evaluation
    test_cases = [
        {
            'input': 'Machine learning is a subset of artificial intelligence.',
            'expected_output': 'ML is a subset of AI.'
        },
        {
            'input': 'Deep learning uses neural networks with multiple layers.',
            'expected_output': 'DL uses multi-layer neural networks.'
        }
    ]
    
    # Evaluate a prompt
    print(f"\n📊 Evaluating prompt 'text_summarization_basic':")
    try:
        evaluation_result = registry.evaluate_prompt('text_summarization_basic', test_cases)
        print(f"Overall score: {evaluation_result['overall_score']:.4f}")
        print(f"Test cases evaluated: {len(evaluation_result['detailed_results'])}")
    except ValueError as e:
        print(f"Error: {e}")
    
    # Create a new version
    print(f"\n🆕 Creating new version of 'text_summarization_basic':")
    new_version = registry.version_prompt(
        'text_summarization_basic',
        'Create a brief summary of this text, focusing on the main ideas: {text}',
        'Improved instructions for better summary quality'
    )
    
    if new_version:
        print(f"New version created: {new_version}")
    
    # Export registry
    print(f"\n📤 Exporting registry...")
    export_file = registry.export_registry()
    print(f"Registry exported to: {export_file}")
    
    # Show registry directory contents
    print(f"\n📁 Registry files created:")
    for file_path in registry.registry_path.glob("*.yaml"):
        print(f"  • {file_path.name}")
    
    return registry

# Run the prompt registry demonstration
prompt_registry = demonstrate_prompt_registry()

# Advanced: Web interface for prompt registry (Streamlit-style)
def create_prompt_registry_app():
    """Create a Streamlit-like interface for the prompt registry"""
    
    app_code = '''
import streamlit as st
import os
import json
from prompt_registry import PromptRegistry

# Initialize the prompt registry
@st.cache_resource
def get_registry():
    return PromptRegistry()

registry = get_registry()

st.set_page_config(
    page_title="LLM Prompt Registry",
    page_icon="📝",
    layout="wide"
)

st.title("📝 LLM Prompt Registry")
st.markdown("Manage, evaluate, and version your LLM prompts")

# Sidebar navigation
page = st.sidebar.selectbox("Navigation", ["Register Prompt", "Browse Prompts", "Evaluate Prompt", "Version Management"])

if page == "Register Prompt":
    st.header("Register New Prompt")
    
    with st.form("register_prompt"):
        prompt_id = st.text_input("Prompt ID", help="Unique identifier for the prompt")
        prompt_text = st.text_area("Prompt Text", height=200, help="The actual prompt text with placeholders like {text}")
        description = st.text_input("Description", help="Brief description of what the prompt does")
        tags = st.text_input("Tags", help="Comma-separated tags for categorization")
        metadata_json = st.text_area("Metadata (JSON)", value="{}", help="Additional metadata in JSON format")
        
        submitted = st.form_submit_button("Register Prompt")
        
        if submitted:
            try:
                metadata = json.loads(metadata_json) if metadata_json else {}
                tag_list = [tag.strip() for tag in tags.split(",")] if tags else []
                
                result = registry.register_prompt(
                    prompt_id=prompt_id,
                    prompt_text=prompt_text,
                    description=description,
                    tags=tag_list,
                    metadata=metadata
                )
                
                if result:
                    st.success(f"✅ Prompt '{prompt_id}' registered successfully!")
                else:
                    st.error("❌ Failed to register prompt. Check if ID already exists.")
            except json.JSONDecodeError:
                st.error("❌ Invalid JSON in metadata field")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

elif page == "Browse Prompts":
    st.header("Browse Prompts")
    
    # Filter by tag
    tag_filter = st.text_input("Filter by tag (optional)")
    
    if tag_filter:
        prompts = registry.list_prompts(tag=tag_filter)
        st.info(f"Showing prompts with tag '{tag_filter}'")
    else:
        prompts = registry.list_prompts()
        st.info(f"Showing all prompts ({len(prompts)} total)")
    
    for prompt in prompts:
        with st.expander(f"{prompt['prompt_id']} (v{prompt['version']})"):
            st.write(f"**Description:** {prompt['description']}")
            st.write(f"**Tags:** {', '.join(prompt['tags'])}")
            st.write(f"**Created:** {prompt['created_at']}")
            
            # Get full prompt details
            full_prompt = registry.get_prompt(prompt['prompt_id'])
            if full_prompt:
                st.text_area("Prompt Text", full_prompt['prompt_text'], height=100, disabled=True)

elif page == "Evaluate Prompt":
    st.header("Evaluate Prompt")
    
    # Select prompt to evaluate
    all_prompts = registry.list_prompts()
    prompt_options = [p['prompt_id'] for p in all_prompts]
    
    if prompt_options:
        selected_prompt = st.selectbox("Select Prompt to Evaluate", prompt_options)
        
        st.subheader("Test Cases")
        st.markdown("Add test cases for evaluation (JSON format)")
        
        test_cases_json = st.text_area(
            "Test Cases (JSON)", 
            value='[{"input": "Sample text", "expected_output": "Expected summary"}]',
            height=200
        )
        
        if st.button("Evaluate Prompt"):
            try:
                test_cases = json.loads(test_cases_json)
                result = registry.evaluate_prompt(selected_prompt, test_cases)
                
                st.success(f"✅ Evaluation completed! Score: {result['overall_score']:.4f}")
                
                # Show detailed results
                st.subheader("Detailed Results")
                for i, detail in enumerate(result['detailed_results']):
                    with st.expander(f"Test Case {i+1}"):
                        st.write(f"**Input:** {detail['test_case'].get('input', 'N/A')}")
                        st.write(f"**Expected:** {detail['test_case'].get('expected_output', 'N/A')}")
                        st.write(f"**Actual:** {detail['actual_output']}")
                        st.write(f"**Semantic Similarity:** {detail['metrics']['semantic_similarity']:.4f}")
                        st.write(f"**Keyword Coverage:** {detail['metrics']['keyword_coverage']:.4f}")
                        
            except json.JSONDecodeError:
                st.error("❌ Invalid JSON format")
            except Exception as e:
                st.error(f"❌ Evaluation error: {str(e)}")
    else:
        st.info("No prompts available. Register some prompts first!")

elif page == "Version Management":
    st.header("Version Management")
    
    all_prompts = registry.list_prompts()
    prompt_options = [p['prompt_id'] for p in all_prompts]
    
    if prompt_options:
        selected_prompt = st.selectbox("Select Prompt to Version", prompt_options)
        
        # Show current version
        current = registry.get_prompt(selected_prompt)
        if current:
            st.info(f"Current version: {current['version']}")
            
            with st.form("new_version"):
                new_prompt_text = st.text_area("New Prompt Text", value=current['prompt_text'], height=200)
                changes_description = st.text_input("Changes Description", help="Describe what changed in this version")
                
                submitted = st.form_submit_button("Create New Version")
                
                if submitted:
                    new_version = registry.version_prompt(
                        selected_prompt, 
                        new_prompt_text, 
                        changes_description
                    )
                    if new_version:
                        st.success(f"✅ New version {new_version} created!")
                    else:
                        st.error("❌ Failed to create new version")
    else:
        st.info("No prompts available. Register some prompts first!")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ for LLMOps")
'''
    
    with open('prompt_registry_app.py', 'w') as f:
        f.write(app_code)
    
    print("Prompt registry Streamlit app created as 'prompt_registry_app.py'")

create_prompt_registry_app()
```