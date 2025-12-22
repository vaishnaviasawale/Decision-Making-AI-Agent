# Decision Making Agent 

A multi-step AI agent built with **LangChain** and **LangGraph** for data-backed decision making. This agent analyzes Amazon sales data to provide actionable business insights through autonomous planning, tool execution, and iterative reasoning.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Tools Reference](#tools-reference)
- [Example Outputs](#example-outputs)
- [Design Notes](#design-notes)
- [Limitations & Future Improvements](#limitations--future-improvements)

---

## Overview

This agent helps users make data-backed decisions by:

1. **Understanding** natural language requests
2. **Planning** a sequence of steps to achieve the goal
3. **Executing** tools autonomously to gather and analyze data
4. **Iterating** based on intermediate results
5. **Synthesizing** a clear, actionable final answer

### Use Cases

- Compare product categories to identify focus areas
- Analyze customer complaints and suggest improvements
- Evaluate pricing and discount strategies
- Identify top-performing and underperforming products

---

## Architecture

### LangGraph Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    AGENT WORKFLOW GRAPH                     │
└─────────────────────────────────────────────────────────────┘

                          ┌─────────┐
                          │  START  │
                          └────┬────┘
                               │
                               ▼
                        ┌──────────────┐
                        │   PLANNER    │  ← Decomposes user goal into steps
                        └──────┬───────┘
                               │
                               ▼
                 ┌────────────────────────┐
          ┌──────│    TOOL SELECTOR       │◄─────────┐
          │      └───────────┬────────────┘          │
          │                  │                       │
          │                  ▼                       │
          │      ┌────────────────────────┐          │
          │      │    TOOL EXECUTOR       │  ← Runs LangChain tools
          │      └───────────┬────────────┘          │
          │                  │                       │
          │                  ▼                       │
          │      ┌────────────────────────┐          │
          │      │      ANALYZER          │  ← Evaluates results
          │      └───────────┬────────────┘          │
          │                  │                       │
          │         [needs_more_info?]               │
          │          YES/         \NO                │
          └─────────┘             ▼                  
                        ┌──────────────┐
                        │ SYNTHESIZER  │  ← Creates final answer
                        └──────┬───────┘
                               │
                               ▼
                          ┌─────────┐
                          │   END   │
                          └─────────┘
```

### Nodes Description

| Node | Type | Purpose |
|------|------|---------|
| **Planner** | LLM | Decomposes user goals into actionable steps |
| **Tool Selector** | LLM | Chooses appropriate tool and parameters for each step |
| **Tool Executor** | Execution | Invokes the selected LangChain tool |
| **Analyzer** | LLM | Evaluates results and decides next action |
| **Synthesizer** | LLM | Creates comprehensive, user-friendly response |

### Project Structure

```
decision-making-agent/
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
├── env.example            # Environment variables template
├── README.md              # This file
├── data/
│   └── amazon.csv  # Amazon dataset obtained from Kaggle (https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset)
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration settings
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── state.py       # Agent state definition
│   │   ├── nodes.py       # LangGraph node implementations
│   │   └── graph.py       # Graph definition and execution
│   └── tools/
│       ├── __init__.py
│       ├── search_tool.py     # Product search tool
│       ├── analysis_tool.py   # Review analysis tool
│       └── statistics_tool.py # Statistics calculation tool
└── examples/
    └── run_examples.py    # Example demonstrations
```

---

## Installation

### Prerequisites

- Python 3.9+
- CLAUDE API key

### Setup Steps

1. **Clone/Navigate to the project:**
   ```bash
   cd decision-making-agent
   ```

2. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment:**
   ```bash
   cp env.example .env
   # Edit .env and add your CLAUDE API key
   ```

5. **Run the agent:**
   ```bash
   python3 main.py
   ```

---

## Usage

### Interactive Mode (Default)

```bash
python main.py
```

This starts an interactive session where you can type queries:

```
DECISION MAKING AGENT - Interactive Mode

Your query: Compare Electronics and Clothing categories

[Agent executes and shows results...]
```

### Single Query Mode

```bash
python main.py --query "What are the top complaints for Electronics products?"
```

### Run Example Queries

```bash
python main.py --example
```

### View Workflow Graph

```bash
python main.py --graph
```

### Quiet Mode (Final Answer Only)

```bash
python main.py --query "Analyze discounts" --quiet
```

---

## Tools Reference

### 1. `search_products` - Information Retrieval Tool

Searches and retrieves products from the Amazon sales dataset.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `category` | str | Product category (e.g., "Electronics") |
| `sub_category` | str | Sub-category (e.g., "Headphones") |
| `min_price` | float | Minimum discounted price |
| `max_price` | float | Maximum discounted price |
| `min_rating` | float | Minimum rating (1.0-5.0) |
| `keyword` | str | Search in product name/description |
| `limit` | int | Maximum results (default: 10) |

**Example:**
```python
search_products.invoke({
    "category": "Electronics",
    "max_price": 2000,
    "min_rating": 4.0
})
```

### 2. `analyze_reviews` - Processing/Analysis Tool

Analyzes customer reviews to extract insights.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `category` | str | Filter by category |
| `product_name` | str | Filter by product name |
| `analysis_type` | str | "complaints", "praise", "themes", "all" |
| `min_rating` | float | Filter by minimum product rating |
| `max_rating` | float | Filter by maximum product rating |

**Example:**
```python
analyze_reviews.invoke({
    "category": "Electronics",
    "analysis_type": "complaints"
})
```

### 3. `calculate_statistics` - Data Processing Tool

Calculates statistics and performs comparisons.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `operation` | str | "category_comparison", "price_analysis", "rating_ranking", "discount_effectiveness", "summary" |
| `categories` | list | Categories to compare |
| `top_n` | int | Number of top results |
| `group_by` | str | "category" or "sub_category" |

**Example:**
```python
calculate_statistics.invoke({
    "operation": "category_comparison",
    "categories": ["Electronics", "Clothing"]
})
```

---

## Example Outputs

### Example 1: Category Comparison

**Query:** "Compare Electronics and Home & Kitchen categories. Which has better ratings?"

**Agent Plan:**
1. Search for products in Electronics category
2. Search for products in Home & Kitchen category
3. Calculate category comparison statistics
4. Analyze reviews for both categories

**Final Output:**
```
Category Analysis Summary

Electronics:
- Average Rating: 4.05
- Products: 8
- Top Issues: Battery problems, connectivity issues
- Strengths: Good value for money

Home & Kitchen:
- Average Rating: 4.15
- Products: 6
- Top Issues: Durability concerns, quality variations
- Strengths: Practical features

Recommendation:
Home & Kitchen shows slightly better customer satisfaction. 
Focus on Electronics should address:
1. Improve battery life accuracy in marketing
2. Enhance connectivity reliability
3. Better quality control for components
```

### Example 2: Complaint Analysis

**Query:** "Find products rated below 4.0 and identify main complaints"

**Agent Output:**
```
Low-Rating Product Analysis

Products with Rating < 4.0:
1. Smart Fitness Band X200 (3.8)
2. Non-Stick Cookware Set (3.5)
3. Smart LED Bulb WiFi (3.6)
4. Air Purifier HEPA (3.7)

Top Complaints Identified:

1. Quality Issues (12 mentions)
   - "Coating peels off after 2 months"
   - "Not durable, breaks easily"

2. Performance Problems (8 mentions)
   - "App crashes frequently"
   - "Inaccurate readings"

3. Connectivity Issues (6 mentions)
   - "WiFi keeps disconnecting"
   - "Bluetooth pairing problems"

Recommended Improvements:
1. Improve manufacturing QC for cookware coating
2. Invest in app stability and testing
3. Enhance wireless connectivity modules
```

---

## Design Notes

### How Tool Selection Works

The agent uses a two-phase approach:

1. **Planning Phase:** The LLM receives the user's goal and creates a step-by-step plan based on available tools.

2. **Selection Phase:** For each step, the LLM selects the appropriate tool and determines optimal parameters based on:
   - The current step's requirements
   - Previous tool results
   - The overall user goal

### Control Flow Management

LangGraph manages the agent's control flow through:

1. **Conditional Edges:** The `should_continue` function determines whether to:
   - Loop back for more tool calls (if steps remain)
   - Proceed to synthesis (if sufficient data gathered)

2. **Iteration Limits:** A maximum iteration count prevents infinite loops (default: 10)

3. **State Accumulation:** Tool results accumulate in state, available to all subsequent nodes

### Error Handling

- **Tool Errors:** Caught and logged; agent continues with available data
- **JSON Parsing:** Fallback parsing for LLM responses
- **Missing API Key:** Clear error message with setup instructions
- **Empty Results:** Graceful handling with informative messages

---

## Limitations & Future Improvements

### Current Limitations

1. **Simple Sentiment Analysis:** Keyword-based, not ML-powered
   - *Improvement:* Use transformer-based sentiment analysis

2. **No Memory Persistence:** Each session starts fresh
   - *Improvement:* Add persistent memory with LangGraph checkpointing

3. **Single LLM Model:** Fixed to CLAUDE
   - *Improvement:* Support multiple LLM providers

4. **Limited Error Recovery:** Basic error handling
   - *Improvement:* Add retry logic and alternative strategies

### Future Enhancements

- [ ] Vector store integration for semantic product search
- [ ] Multi-turn conversation memory
- [ ] Visualization generation (charts, graphs)
- [ ] Export reports to PDF/Excel
- [ ] Real-time data integration via APIs
- [ ] Support for multiple languages
- [ ] Human-in-the-loop confirmation for critical decisions


