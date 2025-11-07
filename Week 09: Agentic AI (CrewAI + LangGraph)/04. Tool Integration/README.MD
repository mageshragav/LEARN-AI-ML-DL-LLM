# Day 44 – Tool Integration

## Web browsing

### What is it?

Web browsing integration in AI agents refers to the capability of language models to interact with the internet, retrieve information from websites, and process web content in real-time. This enables agents to access up-to-date information, perform research tasks, and interact with online resources beyond their static training data. Modern web browsing tools can parse HTML, handle JavaScript, extract relevant content, and even interact with web forms and APIs.

### PyTorch Implementation

```python
import requests
from bs4 import BeautifulSoup
import asyncio
import aiohttp
import time
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import re

class WebBrowsingAgent:
    """AI agent with web browsing capabilities"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Initialize text processing model
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model.eval()
    
    def extract_text_from_url(self, url: str) -> Dict:
        """Extract and process text content from a URL"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract main content
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'content|main|article'))
            
            if main_content:
                text = main_content.get_text()
            else:
                text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_text = " ".join(chunk for chunk in chunks if chunk)
            
            # Extract title
            title = soup.find('title')
            title = title.get_text().strip() if title else "No title found"
            
            # Extract meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            meta_desc = meta_desc.get('content', '') if meta_desc else ''
            
            return {
                'url': url,
                'title': title,
                'meta_description': meta_desc,
                'content': clean_text,
                'word_count': len(clean_text.split()),
                'success': True
            }
            
        except Exception as e:
            return {
                'url': url,
                'error': str(e),
                'success': False
            }
    
    def search_and_extract(self, query: str, max_results: int = 5) -> List[Dict]:
        """Simulate search and extract relevant URLs (in practice, use a search API)"""
        # This is a simplified version - in practice, use Google Search API, SerpAPI, etc.
        # For demonstration, we'll create mock results
        mock_results = [
            {
                'url': f'https://example.com/article-{i}',
                'title': f'Mock Article {i} for {query}',
                'snippet': f'This is a mock snippet for query: {query}'
            }
            for i in range(1, max_results + 1)
        ]
        
        results = []
        for result in mock_results:
            content_result = self.extract_text_from_url(result['url'])
            if content_result['success']:
                content_result.update({
                    'search_title': result['title'],
                    'search_snippet': result['snippet']
                })
                results.append(content_result)
        
        return results
    
    def summarize_content(self, text: str, max_length: int = 200) -> str:
        """Simple content summarization"""
        sentences = text.split('.')
        if len(sentences) > 5:
            # Simple extractive summarization
            summary_sentences = sentences[:3]  # Take first 3 sentences
            summary = '. '.join(summary_sentences) + '.'
        else:
            summary = text
        
        # Truncate if too long
        if len(summary) > max_length:
            summary = summary[:max_length] + '...'
        
        return summary

# Demonstrate web browsing functionality
def demonstrate_web_browsing():
    """Demonstrate web browsing agent capabilities"""
    
    print("Web Browsing Agent Demo")
    print("=" * 40)
    
    # Initialize agent
    agent = WebBrowsingAgent()
    
    # Example query
    query = "artificial intelligence trends 2024"
    
    print(f"Searching for: {query}")
    
    # Perform search and extraction
    results = agent.search_and_extract(query, max_results=3)
    
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Title: {result.get('title', 'N/A')}")
        print(f"URL: {result.get('url', 'N/A')}")
        print(f"Word count: {result.get('word_count', 0)}")
        
        if result.get('success'):
            summary = agent.summarize_content(result['content'])
            print(f"Summary: {summary}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
    
    return agent

web_agent = demonstrate_web_browsing()
```

### When and where we use web browsing

- **Real-time information retrieval**: When current, up-to-date information is required
- **Research and fact-checking**: Verifying claims and gathering evidence from multiple sources
- **Market analysis**: Monitoring competitor websites, product prices, and market trends
- **Content aggregation**: Collecting information from multiple websites for analysis
- **News monitoring**: Tracking breaking news and trending topics
- **Academic research**: Finding relevant papers, articles, and resources

### Real-world examples

- **ChatGPT with browsing**: Accessing current information during conversations
- **Research assistants**: Automatically gathering information from academic databases
- **Price comparison tools**: Monitoring product prices across different retailers
- **Social media monitoring**: Tracking mentions and sentiment across various platforms
- **Legal research**: Finding relevant case law and legal precedents online

### Problems and Advancements

**Problem**: Handling dynamic content loaded via JavaScript.
**Advancement**: Headless browsers like Selenium and Playwright for JavaScript rendering.

**Problem**: Rate limiting and anti-bot measures.
**Advancement**: Proxy rotation, request throttling, and ethical scraping practices.

## SQL

### What is it?

SQL (Structured Query Language) integration in AI agents enables them to interact with relational databases by generating and executing SQL queries. This allows agents to retrieve, manipulate, and analyze structured data stored in databases, bridging the gap between natural language queries and structured data operations. The integration typically involves converting natural language questions into executable SQL statements.

### PyTorch Implementation

```python
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import re
import sqlparse

class SQLAgent:
    """AI agent with SQL database integration"""
    
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()
        
        # Initialize sample database for demonstration
        self._create_sample_database()
    
    def _create_sample_database(self):
        """Create sample database tables for demonstration"""
        
        # Create products table
        self.cursor.execute('''
            CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT,
                price REAL,
                stock_quantity INTEGER,
                created_date TEXT
            )
        ''')
        
        # Create customers table
        self.cursor.execute('''
            CREATE TABLE customers (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE,
                city TEXT,
                registration_date TEXT
            )
        ''')
        
        # Create orders table
        self.cursor.execute('''
            CREATE TABLE orders (
                id INTEGER PRIMARY KEY,
                customer_id INTEGER,
                product_id INTEGER,
                quantity INTEGER,
                order_date TEXT,
                total_amount REAL,
                FOREIGN KEY (customer_id) REFERENCES customers (id),
                FOREIGN KEY (product_id) REFERENCES products (id)
            )
        ''')
        
        # Insert sample data
        products_data = [
            (1, "Laptop", "Electronics", 999.99, 50, "2023-01-15"),
            (2, "Smartphone", "Electronics", 699.99, 100, "2023-02-20"),
            (3, "Desk Chair", "Furniture", 199.99, 30, "2023-03-10"),
            (4, "Coffee Maker", "Appliances", 89.99, 75, "2023-04-05"),
            (5, "Bookshelf", "Furniture", 149.99, 25, "2023-05-12")
        ]
        
        customers_data = [
            (1, "John Doe", "john@example.com", "New York", "2022-01-10"),
            (2, "Jane Smith", "jane@example.com", "Los Angeles", "2022-02-15"),
            (3, "Bob Johnson", "bob@example.com", "Chicago", "2022-03-20"),
            (4, "Alice Brown", "alice@example.com", "Houston", "2022-04-25"),
            (5, "Charlie Wilson", "charlie@example.com", "Phoenix", "2022-05-30")
        ]
        
        orders_data = [
            (1, 1, 1, 1, "2023-06-01", 999.99),
            (2, 2, 2, 1, "2023-06-02", 699.99),
            (3, 1, 3, 2, "2023-06-03", 399.98),
            (4, 3, 4, 1, "2023-06-04", 89.99),
            (5, 4, 5, 1, "2023-06-05", 149.99)
        ]
        
        self.cursor.executemany('INSERT INTO products VALUES (?, ?, ?, ?, ?, ?)', products_data)
        self.cursor.executemany('INSERT INTO customers VALUES (?, ?, ?, ?, ?)', customers_data)
        self.cursor.executemany('INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)', orders_data)
        
        self.connection.commit()
    
    def natural_to_sql(self, natural_query: str) -> str:
        """Convert natural language query to SQL"""
        
        # Simple rule-based conversion (in practice, use more sophisticated NLP)
        natural_query_lower = natural_query.lower()
        
        # Identify the main intent
        if "count" in natural_query_lower or "how many" in natural_query_lower:
            if "product" in natural_query_lower:
                return "SELECT COUNT(*) FROM products;"
            elif "customer" in natural_query_lower:
                return "SELECT COUNT(*) FROM customers;"
            elif "order" in natural_query_lower:
                return "SELECT COUNT(*) FROM orders;"
        
        elif "total" in natural_query_lower and "sales" in natural_query_lower:
            return "SELECT SUM(total_amount) FROM orders;"
        
        elif "average" in natural_query_lower and "price" in natural_query_lower:
            return "SELECT AVG(price) FROM products;"
        
        elif "top" in natural_query_lower and "expensive" in natural_query_lower:
            return "SELECT name, price FROM products ORDER BY price DESC LIMIT 5;"
        
        elif "customer" in natural_query_lower and "city" in natural_query_lower:
            # Extract city from query (simplified)
            city_match = re.search(r'in\s+(\w+)', natural_query_lower)
            if city_match:
                city = city_match.group(1).title()
                return f"SELECT * FROM customers WHERE city = '{city}';"
        
        elif "orders" in natural_query_lower and "customer" in natural_query_lower:
            # Extract customer name from query (simplified)
            name_match = re.search(r'for\s+([a-z\s]+)', natural_query_lower)
            if name_match:
                name = name_match.group(1).strip().title()
                return f"""
                SELECT o.id, o.quantity, o.order_date, o.total_amount, p.name as product_name
                FROM orders o
                JOIN customers c ON o.customer_id = c.id
                JOIN products p ON o.product_id = p.id
                WHERE c.name = '{name}';
                """
        
        # Default: return a simple query
        return "SELECT * FROM products LIMIT 5;"
    
    def execute_query(self, sql_query: str) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame"""
        try:
            # Validate query (basic security check)
            if not self._is_safe_query(sql_query):
                raise ValueError("Unsafe SQL query detected")
            
            df = pd.read_sql_query(sql_query, self.connection)
            return df
        
        except Exception as e:
            print(f"Error executing query: {e}")
            return pd.DataFrame()
    
    def _is_safe_query(self, query: str) -> bool:
        """Basic safety check for SQL queries"""
        dangerous_keywords = ['drop', 'delete', 'truncate', 'insert', 'update', 'alter', 'create']
        query_lower = query.lower()
        
        for keyword in dangerous_keywords:
            if keyword in query_lower:
                return False
        
        return True
    
    def query_database(self, natural_query: str) -> pd.DataFrame:
        """Execute natural language query against the database"""
        sql_query = self.natural_to_sql(natural_query)
        print(f"Natural Query: {natural_query}")
        print(f"Generated SQL: {sql_query}")
        
        results = self.execute_query(sql_query)
        return results

# Demonstrate SQL agent functionality
def demonstrate_sql_agent():
    """Demonstrate SQL agent capabilities"""
    
    print("SQL Agent Demo")
    print("=" * 30)
    
    # Initialize agent
    sql_agent = SQLAgent()
    
    # Test queries
    test_queries = [
        "How many products are there?",
        "What is the total sales amount?",
        "What is the average product price?",
        "Show top expensive products",
        "Show customer orders for John Doe"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = sql_agent.query_database(query)
        print("Results:")
        print(results)
        print("-" * 50)
    
    return sql_agent

sql_agent = demonstrate_sql_agent()
```

### When and where we use SQL integration

- **Business intelligence**: Generating reports and dashboards from company databases
- **Data analysis**: Analyzing structured data with natural language queries
- **Customer service**: Retrieving customer information and order history
- **Inventory management**: Tracking stock levels and product information
- **Financial analysis**: Analyzing transaction data and financial metrics
- **CRM systems**: Interacting with customer relationship management databases

### Real-world examples

- **Salesforce Einstein**: Natural language queries for CRM data
- **Microsoft Power BI**: Natural language processing for data visualization
- **Amazon QuickSight**: Conversational analytics for business intelligence
- **Google BigQuery**: Natural language to SQL conversion for data analysis
- **Customer support bots**: Retrieving customer data from databases

### Problems and Advancements

**Problem**: Generating syntactically correct and secure SQL from natural language.
**Advancement**: Pre-trained models like Text-to-SQL transformers and security validation layers.

**Problem**: Handling complex queries with multiple joins and conditions.
**Advancement**: Advanced NLP models trained specifically for complex SQL generation.

## SerpAPI

### What is it?

SerpAPI is a search engine results API that provides structured access to search results from Google, Bing, Yahoo, and other search engines. It enables AI agents to perform web searches programmatically, retrieving relevant information, featured snippets, related questions, and other search result components. This integration allows agents to access current information, trends, and real-time data from search engines.

### PyTorch Implementation

```python
import requests
import json
from typing import Dict, List, Optional
import time

class SerpAPIAgent:
    """AI agent with SerpAPI integration"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://serpapi.com/search"
    
    def search(self, query: str, num_results: int = 10, **kwargs) -> Dict:
        """Perform search using SerpAPI"""
        params = {
            'q': query,
            'api_key': self.api_key,
            'num': num_results,
            'gl': 'US',  # Geographic location
            'hl': 'en'   # Language
        }
        
        # Add any additional parameters
        params.update(kwargs)
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            print(f"Error with SerpAPI request: {e}")
            return {}
    
    def extract_relevant_info(self, search_results: Dict) -> Dict:
        """Extract relevant information from search results"""
        extracted_info = {
            'query': search_results.get('search_parameters', {}).get('q', ''),
            'organic_results': [],
            'related_questions': [],
            'featured_snippets': [],
            'knowledge_graph': {},
            'total_results': search_results.get('search_information', {}).get('total_results', 0)
        }
        
        # Extract organic results
        organic_results = search_results.get('organic_results', [])
        for result in organic_results[:5]:  # Limit to top 5
            extracted_info['organic_results'].append({
                'title': result.get('title', ''),
                'link': result.get('link', ''),
                'snippet': result.get('snippet', ''),
                'position': result.get('position', 0)
            })
        
        # Extract related questions
        related_questions = search_results.get('related_questions', [])
        for question in related_questions[:3]:  # Limit to top 3
            extracted_info['related_questions'].append({
                'question': question.get('question', ''),
                'answer': question.get('answer', ''),
                'source': question.get('source', {}).get('name', '')
            })
        
        # Extract featured snippets
        featured_snippets = search_results.get('featured_snippet', [])
        if featured_snippets:
            extracted_info['featured_snippets'].append({
                'title': featured_snippets.get('title', ''),
                'snippet': featured_snippets.get('snippet', ''),
                'source': featured_snippets.get('source', {}).get('name', '')
            })
        
        # Extract knowledge graph info
        knowledge_graph = search_results.get('knowledge_graph', {})
        if knowledge_graph:
            extracted_info['knowledge_graph'] = {
                'title': knowledge_graph.get('title', ''),
                'type': knowledge_graph.get('type', ''),
                'description': knowledge_graph.get('description', ''),
                'website': knowledge_graph.get('website', ''),
                'phone': knowledge_graph.get('phone', ''),
                'address': knowledge_graph.get('address', '')
            }
        
        return extracted_info
    
    def get_current_trends(self, query: str) -> Dict:
        """Get current trends and related information for a query"""
        # Perform search with additional parameters for trends
        results = self.search(
            query=query,
            num_results=10,
            tbs='qdr:d'  # Past day
        )
        
        return self.extract_relevant_info(results)

# Note: For demonstration, we'll create a mock implementation since we can't use actual API key
class MockSerpAPIAgent:
    """Mock implementation of SerpAPI agent for demonstration"""
    
    def __init__(self, api_key: str = "mock_key"):
        self.api_key = api_key
    
    def search(self, query: str, num_results: int = 10, **kwargs) -> Dict:
        """Mock search implementation"""
        print(f"Mock search for: {query}")
        
        # Simulate search results
        mock_results = {
            "search_parameters": {
                "q": query,
                "num": num_results
            },
            "organic_results": [
                {
                    "position": 1,
                    "title": f"Top result for {query}",
                    "link": f"https://example.com/{query.replace(' ', '-')}",
                    "snippet": f"This is a mock result snippet for {query}. It contains relevant information about the topic."
                },
                {
                    "position": 2,
                    "title": f"Second result for {query}",
                    "link": f"https://example2.com/{query.replace(' ', '-')}",
                    "snippet": f"Another mock result for {query} with additional details."
                }
            ],
            "related_questions": [
                {
                    "question": f"What is {query}?",
                    "answer": f"{query} is an important topic in the field.",
                    "source": "Example Source"
                }
            ],
            "featured_snippet": {
                "title": f"Featured: {query}",
                "snippet": f"{query} refers to the process of...",
                "source": "Featured Source"
            },
            "knowledge_graph": {
                "title": query.title(),
                "type": "Topic",
                "description": f"Information about {query}",
                "website": "https://example.com"
            },
            "search_information": {
                "total_results": 1000000
            }
        }
        
        return mock_results
    
    def extract_relevant_info(self, search_results: Dict) -> Dict:
        """Extract relevant information from mock search results"""
        extracted_info = {
            'query': search_results.get('search_parameters', {}).get('q', ''),
            'organic_results': [],
            'related_questions': [],
            'featured_snippets': [],
            'knowledge_graph': {},
            'total_results': search_results.get('search_information', {}).get('total_results', 0)
        }
        
        # Extract organic results
        organic_results = search_results.get('organic_results', [])
        for result in organic_results:
            extracted_info['organic_results'].append({
                'title': result.get('title', ''),
                'link': result.get('link', ''),
                'snippet': result.get('snippet', ''),
                'position': result.get('position', 0)
            })
        
        # Extract related questions
        related_questions = search_results.get('related_questions', [])
        for question in related_questions:
            extracted_info['related_questions'].append({
                'question': question.get('question', ''),
                'answer': question.get('answer', ''),
                'source': question.get('source', '')
            })
        
        # Extract featured snippets
        featured_snippet = search_results.get('featured_snippet', {})
        if featured_snippet:
            extracted_info['featured_snippets'].append({
                'title': featured_snippet.get('title', ''),
                'snippet': featured_snippet.get('snippet', ''),
                'source': featured_snippet.get('source', '')
            })
        
        # Extract knowledge graph info
        knowledge_graph = search_results.get('knowledge_graph', {})
        if knowledge_graph:
            extracted_info['knowledge_graph'] = {
                'title': knowledge_graph.get('title', ''),
                'type': knowledge_graph.get('type', ''),
                'description': knowledge_graph.get('description', ''),
                'website': knowledge_graph.get('website', ''),
                'phone': knowledge_graph.get('phone', ''),
                'address': knowledge_graph.get('address', '')
            }
        
        return extracted_info

def demonstrate_serpapi_agent():
    """Demonstrate SerpAPI agent functionality"""
    
    print("SerpAPI Agent Demo")
    print("=" * 30)
    
    # Initialize mock agent (in real implementation, use actual API key)
    serp_agent = MockSerpAPIAgent()
    
    # Test queries
    test_queries = [
        "artificial intelligence trends 2024",
        "machine learning applications",
        "latest developments in NLP"
    ]
    
    for query in test_queries:
        print(f"\nSearching for: {query}")
        
        # Perform search
        results = serp_agent.search(query, num_results=5)
        
        # Extract relevant information
        extracted = serp_agent.extract_relevant_info(results)
        
        print(f"Total results: {extracted['total_results']}")
        print(f"Organic results: {len(extracted['organic_results'])}")
        
        # Display top results
        for i, result in enumerate(extracted['organic_results'][:3], 1):
            print(f"{i}. {result['title']}")
            print(f"   {result['snippet'][:100]}...")
            print(f"   Link: {result['link']}")
        
        # Display related questions
        if extracted['related_questions']:
            print(f"\nRelated Questions:")
            for question in extracted['related_questions']:
                print(f"- {question['question']}")
                print(f"  {question['answer'][:100]}...")
        
        print("-" * 50)
    
    return serp_agent

serp_agent = demonstrate_serpapi_agent()
```

### When and where we use SerpAPI

- **Real-time information retrieval**: Accessing current news, events, and trending topics
- **Market research**: Gathering competitive intelligence and market data
- **Content creation**: Finding relevant information for articles and reports
- **Academic research**: Locating scholarly articles and research papers
- **Competitor analysis**: Monitoring competitor websites and strategies
- **SEO analysis**: Understanding search trends and keyword performance

### Real-world examples

- **ChatGPT with browsing**: Accessing current information during conversations
- **Market research platforms**: Automating data collection from search results
- **Content management systems**: Finding related content and resources
- **News aggregation services**: Collecting news from multiple sources
- **Research assistants**: Gathering academic and scientific information

### Problems and Advancements

**Problem**: API rate limits and costs for high-volume usage.
**Advancement**: Caching mechanisms and request optimization strategies.

**Problem**: Search result variations based on location and personalization.
**Advancement**: Geographic targeting and result standardization techniques.

## 🧪 Exercise: Use tool agents

```python
def use_tool_agents_exercise():
    """Complete exercise: Using different tool agents"""
    
    print("Tool Agents Exercise")
    print("=" * 40)
    
    # Initialize all agents
    web_agent = WebBrowsingAgent()
    sql_agent = SQLAgent()
    serp_agent = MockSerpAPIAgent()
    
    # Multi-tool query: Research a topic using all agents
    research_topic = "artificial intelligence in healthcare"
    
    print(f"Researching: {research_topic}")
    print("\n1. Using Web Browsing Agent:")
    print("-" * 30)
    
    # Web browsing - get general information
    web_results = web_agent.search_and_extract(research_topic, max_results=2)
    for result in web_results:
        if result['success']:
            print(f"Title: {result['title']}")
            print(f"Summary: {web_agent.summarize_content(result['content'])}")
            print()
    
    print("\n2. Using SerpAPI Agent:")
    print("-" * 30)
    
    # SerpAPI - get search results and trends
    serp_results = serp_agent.search(research_topic, num_results=3)
    extracted_info = serp_agent.extract_relevant_info(serp_results)
    
    print(f"Total search results: {extracted_info['total_results']}")
    print("Top results:")
    for i, result in enumerate(extracted_info['organic_results'][:3], 1):
        print(f"{i}. {result['title']}")
        print(f"   {result['snippet'][:150]}...")
    
    if extracted_info['related_questions']:
        print(f"\nRelated questions:")
        for question in extracted_info['related_questions']:
            print(f"- {question['question']}")
    
    print("\n3. Using SQL Agent (simulated database query):")
    print("-" * 30)
    
    # SQL - query related data (simulated)
    sql_query = "SELECT name, category, price FROM products WHERE category LIKE '%AI%' OR name LIKE '%AI%' LIMIT 5;"
    print(f"SQL Query: {sql_query}")
    
    try:
        results_df = sql_agent.execute_query(sql_query)
        if not results_df.empty:
            print("Query Results:")
            print(results_df.to_string(index=False))
        else:
            print("No matching records found in the database.")
    except Exception as e:
        print(f"SQL Error: {e}")
    
    # Combine insights from all agents
    print("\n4. Combined Insights:")
    print("-" * 30)
    print("Web browsing provided general information about AI in healthcare.")
    print("SerpAPI revealed current trends and related questions.")
    print("SQL database could provide structured data about AI products/services.")
    print("Combined, these tools provide a comprehensive view of the topic.")
    
    return {
        'web_agent': web_agent,
        'sql_agent': sql_agent,
        'serp_agent': serp_agent
    }

exercise_results = use_tool_agents_exercise()
```

## 📌 Project: Market research agent

```python
class MarketResearchAgent:
    """Complete market research agent using multiple tools"""
    
    def __init__(self, serpapi_key: str = "mock_key"):
        self.serp_agent = MockSerpAPIAgent(api_key=serpapi_key)
        self.sql_agent = SQLAgent()
        self.web_agent = WebBrowsingAgent()
        
        # Maintain research state
        self.research_history = []
        self.competitor_analysis = {}
        self.market_trends = {}
    
    def conduct_research(self, research_topic: str, focus_areas: List[str] = None) -> Dict:
        """Conduct comprehensive market research"""
        
        research_start_time = time.time()
        
        if focus_areas is None:
            focus_areas = ["overview", "competitors", "trends", "pricing"]
        
        results = {
            'topic': research_topic,
            'focus_areas': focus_areas,
            'research_data': {},
            'insights': [],
            'recommendations': [],
            'research_duration': 0
        }
        
        print(f"Starting market research on: {research_topic}")
        
        # 1. Get market overview
        if "overview" in focus_areas:
            print("\n1. Gathering market overview...")
            overview_results = self._get_market_overview(research_topic)
            results['research_data']['overview'] = overview_results
        
        # 2. Analyze competitors
        if "competitors" in focus_areas:
            print("\n2. Analyzing competitors...")
            competitor_results = self._analyze_competitors(research_topic)
            results['research_data']['competitors'] = competitor_results
        
        # 3. Identify market trends
        if "trends" in focus_areas:
            print("\n3. Identifying market trends...")
            trend_results = self._identify_trends(research_topic)
            results['research_data']['trends'] = trend_results
        
        # 4. Analyze pricing
        if "pricing" in focus_areas:
            print("\n4. Analyzing pricing strategies...")
            pricing_results = self._analyze_pricing(research_topic)
            results['research_data']['pricing'] = pricing_results
        
        # Generate insights
        results['insights'] = self._generate_insights(results['research_data'])
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results['insights'])
        
        # Calculate research duration
        results['research_duration'] = time.time() - research_start_time
        
        # Add to history
        self.research_history.append(results)
        
        return results
    
    def _get_market_overview(self, topic: str) -> Dict:
        """Get market overview using multiple tools"""
        print(f"  Getting overview for: {topic}")
        
        # Use SerpAPI for search results
        serp_results = self.serp_agent.search(f"market overview {topic}", num_results=5)
        overview_data = self.serp_agent.extract_relevant_info(serp_results)
        
        # Use web agent for detailed content
        if overview_data['organic_results']:
            top_url = overview_data['organic_results'][0]['link']
            content_result = self.web_agent.extract_text_from_url(top_url)
            if content_result['success']:
                content_result['summary'] = self.web_agent.summarize_content(content_result['content'])
        
        return {
            'search_results': overview_data,
            'detailed_content': content_result if 'content_result' in locals() else None
        }
    
    def _analyze_competitors(self, topic: str) -> Dict:
        """Analyze competitors in the market"""
        print(f"  Analyzing competitors for: {topic}")
        
        # Search for competitors
        competitor_search = self.serp_agent.search(f"top competitors in {topic}", num_results=10)
        competitor_data = self.serp_agent.extract_relevant_info(competitor_search)
        
        competitors = []
        for result in competitor_data['organic_results'][:5]:
            competitor_name = result['title'].split('-')[0].strip() if '-' in result['title'] else result['title']
            
            # Get more information about each competitor
            comp_info_search = self.serp_agent.search(f"{competitor_name} company information", num_results=3)
            comp_info = self.serp_agent.extract_relevant_info(comp_info_search)
            
            competitors.append({
                'name': competitor_name,
                'url': result['link'],
                'description': result['snippet'],
                'additional_info': comp_info
            })
        
        return {
            'competitors': competitors,
            'total_competitors_found': len(competitors)
        }
    
    def _identify_trends(self, topic: str) -> Dict:
        """Identify current market trends"""
        print(f"  Identifying trends for: {topic}")
        
        # Search for recent trends
        trend_search = self.serp_agent.search(f"latest trends in {topic} 2024", num_results=10)
        trend_data = self.serp_agent.extract_relevant_info(trend_search)
        
        # Look for recent news
        news_search = self.serp_agent.search(f"news about {topic}", num_results=5, tbs='qdr:d')  # Past day
        news_data = self.serp_agent.extract_relevant_info(news_search)
        
        return {
            'trends': [result['snippet'] for result in trend_data['organic_results']],
            'recent_news': news_data['organic_results'],
            'total_trends_found': len(trend_data['organic_results'])
        }
    
    def _analyze_pricing(self, topic: str) -> Dict:
        """Analyze pricing strategies in the market"""
        print(f"  Analyzing pricing for: {topic}")
        
        # Search for pricing information
        pricing_search = self.serp_agent.search(f"{topic} pricing comparison", num_results=10)
        pricing_data = self.serp_agent.extract_relevant_info(pricing_search)
        
        # Look for pricing pages of competitors
        if 'competitors' in self.competitor_analysis:
            pricing_info = []
            for competitor in self.competitor_analysis['competitors'][:3]:  # Top 3 competitors
                # In real implementation, visit competitor pricing pages
                pricing_info.append({
                    'competitor': competitor['name'],
                    'pricing_url': f"{competitor['url']}/pricing" if competitor['url'] else None,
                    'estimated_price_range': "To be determined"  # Would extract from actual pricing pages
                })
        else:
            pricing_info = []
        
        return {
            'pricing_search_results': pricing_data['organic_results'],
            'competitor_pricing': pricing_info
        }
    
    def _generate_insights(self, research_data: Dict) -> List[str]:
        """Generate insights from research data"""
        insights = []
        
        # Market size and growth
        if 'overview' in research_data:
            insights.append("Market overview gathered from multiple sources")
        
        # Competitive landscape
        if 'competitors' in research_data:
            comp_count = research_data['competitors']['total_competitors_found']
            insights.append(f"Identified {comp_count} major competitors in the market")
        
        # Current trends
        if 'trends' in research_data:
            trend_count = research_data['trends']['total_trends_found']
            insights.append(f"Identified {trend_count} current market trends")
        
        # Pricing strategies
        if 'pricing' in research_data:
            insights.append("Pricing analysis completed with competitor comparisons")
        
        return insights
    
    def _generate_recommendations(self, insights: List[str]) -> List[str]:
        """Generate recommendations based on insights"""
        recommendations = []
        
        # Add general recommendations
        recommendations.append("Focus on differentiation from top competitors")
        recommendations.append("Monitor market trends for strategic planning")
        recommendations.append("Consider competitive pricing strategies")
        
        # Add specific recommendations based on findings
        if any("competitors" in insight for insight in insights):
            recommendations.append("Develop unique value proposition to stand out")
        
        if any("trends" in insight for insight in insights):
            recommendations.append("Align product development with current trends")
        
        return recommendations
    
    def generate_report(self, research_results: Dict) -> str:
        """Generate a comprehensive market research report"""
        report = []
        report.append(f"MARKET RESEARCH REPORT: {research_results['topic'].upper()}")
        report.append("=" * 60)
        report.append(f"Research Duration: {research_results['research_duration']:.2f} seconds")
        report.append("")
        
        # Overview
        if 'overview' in research_results['research_data']:
            report.append("MARKET OVERVIEW")
            report.append("-" * 30)
            overview = research_results['research_data']['overview']
            if overview.get('detailed_content'):
                report.append(overview['detailed_content'].get('summary', 'No detailed content available'))
            report.append("")
        
        # Competitors
        if 'competitors' in research_results['research_data']:
            report.append("COMPETITOR ANALYSIS")
            report.append("-" * 30)
            competitors = research_results['research_data']['competitors']
            for i, comp in enumerate(competitors['competitors'][:3], 1):
                report.append(f"{i}. {comp['name']}")
                report.append(f"   Description: {comp['description'][:100]}...")
                report.append(f"   URL: {comp['url']}")
            report.append("")
        
        # Trends
        if 'trends' in research_results['research_data']:
            report.append("MARKET TRENDS")
            report.append("-" * 30)
            trends = research_results['research_data']['trends']
            for i, trend in enumerate(trends['trends'][:5], 1):
                report.append(f"{i}. {trend[:150]}...")
            report.append("")
        
        # Insights
        report.append("KEY INSIGHTS")
        report.append("-" * 30)
        for i, insight in enumerate(research_results['insights'], 1):
            report.append(f"{i}. {insight}")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 30)
        for i, rec in enumerate(research_results['recommendations'], 1):
            report.append(f"{i}. {rec}")
        report.append("")
        
        return "\n".join(report)

def demonstrate_market_research_agent():
    """Demonstrate the complete market research agent"""
    
    print("Market Research Agent Demo")
    print("=" * 50)
    
    # Initialize agent
    agent = MarketResearchAgent()
    
    # Conduct research on a topic
    research_topic = "artificial intelligence in healthcare"
    focus_areas = ["overview", "competitors", "trends"]
    
    print(f"Conducting market research on: {research_topic}")
    print(f"Focus areas: {', '.join(focus_areas)}")
    
    # Perform research
    research_results = agent.conduct_research(research_topic, focus_areas)
    
    # Generate and display report
    report = agent.generate_report(research_results)
    print("\n" + report)
    
    # Show research history
    print(f"\nResearch history contains {len(agent.research_history)} completed researches")
    
    return agent

# Run the market research agent demonstration
market_research_agent = demonstrate_market_research_agent()

# Advanced: Multi-modal research agent
class AdvancedMarketResearchAgent(MarketResearchAgent):
    """Enhanced market research agent with additional capabilities"""
    
    def __init__(self, serpapi_key: str = "mock_key"):
        super().__init__(serpapi_key)
        self.sentiment_analyzer = self._initialize_sentiment_analyzer()
    
    def _initialize_sentiment_analyzer(self):
        """Initialize sentiment analysis for market research"""
        # In practice, use a pre-trained sentiment model
        def mock_sentiment_analyze(text):
            # Simple mock sentiment analysis
            positive_words = ['good', 'great', 'excellent', 'positive', 'successful', 'leading', 'innovative']
            negative_words = ['bad', 'terrible', 'poor', 'negative', 'failing', 'struggling', 'declining']
            
            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count > neg_count:
                return 'positive'
            elif neg_count > pos_count:
                return 'negative'
            else:
                return 'neutral'
        
        return mock_sentiment_analyze
    
    def analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment of text"""
        return self.sentiment_analyzer(text)
    
    def conduct_sentiment_analysis(self, research_topic: str) -> Dict:
        """Conduct sentiment analysis on market research"""
        print(f"\nConducting sentiment analysis for: {research_topic}")
        
        # Get recent news and articles
        news_results = self.serp_agent.search(f"{research_topic} news", num_results=10)
        news_data = self.serp_agent.extract_relevant_info(news_results)
        
        sentiment_scores = []
        for result in news_data['organic_results']:
            sentiment = self.analyze_sentiment(result['snippet'])
            sentiment_scores.append({
                'title': result['title'],
                'snippet': result['snippet'],
                'sentiment': sentiment
            })
        
        # Calculate overall sentiment
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        for score in sentiment_scores:
            sentiment_counts[score['sentiment']] += 1
        
        overall_sentiment = max(sentiment_counts, key=sentiment_counts.get)
        
        return {
            'sentiment_analysis': sentiment_scores,
            'overall_sentiment': overall_sentiment,
            'sentiment_distribution': sentiment_counts
        }

def demonstrate_advanced_market_research():
    """Demonstrate advanced market research capabilities"""
    
    print("\nAdvanced Market Research Agent Demo")
    print("=" * 50)
    
    # Initialize advanced agent
    advanced_agent = AdvancedMarketResearchAgent()
    
    # Conduct sentiment analysis
    topic = "AI healthcare startups"
    sentiment_results = advanced_agent.conduct_sentiment_analysis(topic)
    
    print(f"Sentiment Analysis for: {topic}")
    print(f"Overall sentiment: {sentiment_results['overall_sentiment']}")
    print(f"Sentiment distribution: {sentiment_results['sentiment_distribution']}")
    
    # Show sample sentiment analysis
    print("\nSample sentiment analysis:")
    for i, analysis in enumerate(sentiment_results['sentiment_analysis'][:3], 1):
        print(f"{i}. {analysis['title']}")
        print(f"   Sentiment: {analysis['sentiment']}")
        print(f"   Snippet: {analysis['snippet'][:100]}...")

demonstrate_advanced_market_research()
```