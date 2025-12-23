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
| `group_by` | str | "category" |

**Example:**
```python
calculate_statistics.invoke({
    "operation": "category_comparison",
    "categories": ["Electronics", "Clothing"]
})
```

---

## Example Outputs
python3 main.py --example

============================================================
DECISION MAKING AGENT - Example Queries
============================================================


############################################################
# EXAMPLE 1: Compare the Home Theatre and Mobile Accessories ca...
############################################################
============================================================
DECISION MAKING AGENT
============================================================

User Query: Compare the Home Theatre and Mobile Accessories categories. Which one has better customer satisfaction and where should we focus our efforts?

------------------------------------------------------------

Node: planner
Plan: ['Step 1: Search for products in the Home Theatre and Mobile Accessories categories using the search_products tool. Retrieve key metrics like average rating, number of reviews, and price range for each category.', 'Step 2: Analyze the customer reviews for the top products in each category using the analyze_reviews tool. Look for common themes of praise and complaints to gauge customer satisfaction.', 'Step 3: Calculate and compare the overall customer satisfaction statistics for the Home Theatre and Mobile Accessories categories using the calculate_statistics tool. This could include average rating, percentage of 5-star reviews, and review sentiment analysis.', 'Step 4: Determine which category has higher customer satisfaction based on the review analysis and statistics. Identify the key drivers of satisfaction or dissatisfaction to focus product development and marketing efforts.']

Node: tool_selector

Node: tool_executor
Tool: search_products
Result Preview: Found 10 product(s) matching your criteria:

  **Amazon Basics Wireless Mouse | 2.4 GHz Connection, 1600 DPI | Type - C Adapter | Upto 12 Months of Battery Life | Ambidextrous Design | Suitable for PC...

Node: analyzer

Node: tool_selector

Node: tool_executor
Tool: analyze_reviews
Result Preview: **Review Analysis Report**
   Products Analyzed: 1337
   Total Reviews: 1465
   Analysis Type: Complaints

**Top Complaints Identified:**

**Battery/Power Issues** (722 mentions)
   • "Looks durable C...

Node: analyzer

Node: tool_selector

Node: tool_executor
Tool: calculate_statistics
Result Preview: **Dataset Summary Statistics**

**Overview:**
Total Products: 1337
Total Reviews: 1465
Categories: 211

**Rating Statistics:**
Average Rating: 4.09
Rating Range: 0.0 - 5.0
Avg Reviews per Product: 176...

Node: analyzer

Node: synthesizer
Final answer generated

============================================================
FINAL ANSWER
============================================================
Based on the research findings, here is a comprehensive analysis and recommendations for the user:

**Overview**
The analysis covers two main product categories - Home Theatre and Mobile Accessories. The Home Theatre category includes televisions, projectors, speakers, and related accessories. The Mobile Accessories category includes products like cables, chargers, power banks, cases, and other mobile device peripherals.

**Customer Satisfaction Comparison**
Across both categories, the average customer rating is quite high at 4.09 out of 5 stars. However, the Mobile Accessories category appears to have slightly better customer satisfaction overall:

- Mobile Accessories average rating: 4.16 stars
- Home Theatre average rating: 4.03 stars

The top complaints for Mobile Accessories were around battery/power issues, comfort/fit, and performance problems. For Home Theatre, the main issues were around heat, connectivity, and value concerns.

This suggests the Mobile Accessories category has fewer critical pain points for customers compared to Home Theatre.

**Recommendations**
Based on the analysis, I would recommend the following:

1. Focus product development and quality control efforts on the Home Theatre category to address the key issues around heat, connectivity, and value. This could involve improving product design, testing, and customer support in this area.

2. Continue investing in the Mobile Accessories category, as it already has strong customer satisfaction. However, monitor the top complaint areas (battery, comfort, performance) and look for opportunities to further optimize these aspects.

3. Across both categories, maintain a strong emphasis on product quality, durability, and clear communication of features/capabilities. The analysis showed some issues around misleading descriptions and quality concerns.

4. Leverage the high average discounts (46.5%) to offer competitive pricing and value, which was a pain point for some Home Theatre customers.

In summary, the Mobile Accessories category appears to have a slight edge in customer satisfaction currently. But there are opportunities to improve the Home Theatre offerings by addressing the key pain points identified in the analysis. Continued focus on quality, value, and customer experience will be critical for success in both categories.

------------------------------------------------------------

Press Enter to continue to next example...


############################################################
# EXAMPLE 2: Analyze customer complaints for products with rati...
############################################################
============================================================
DECISION MAKING AGENT
============================================================

User Query: Analyze customer complaints for products with ratings below 4.0. What are the main issues and how can we address them?

------------------------------------------------------------

Node: planner
Plan: ['Step 1: Search for products with ratings below 4.0 using the search_products tool', 'Step 2: Analyze the reviews for those low-rated products using the analyze_reviews tool to identify the main customer complaints', 'Step 3: Summarize the key issues and themes from the customer complaints', 'Step 4: Recommend potential solutions or product improvements to address the main customer complaints']

Node: tool_selector

Node: tool_executor
Tool: search_products
Result Preview: Found 10 product(s) matching your criteria:

  **Amazon Basics Wireless Mouse | 2.4 GHz Connection, 1600 DPI | Type - C Adapter | Upto 12 Months of Battery Life | Ambidextrous Design | Suitable for PC...

Node: analyzer

Node: tool_selector

Node: tool_executor
Tool: analyze_reviews
Result Preview: **Review Analysis Report**
   Products Analyzed: 1337
   Total Reviews: 1465
   Analysis Type: Complaints

**Top Complaints Identified:**

**Battery/Power Issues** (722 mentions)
   • "Looks durable C...

Node: analyzer

Node: synthesizer
Final answer generated

============================================================
FINAL ANSWER
============================================================
**Comprehensive Response to User's Goal**

**Key Findings:**
Based on the analysis of customer complaints for products with ratings below 4.0, the main issues identified are:

1. **Battery/Power Issues** (722 mentions)
   - Customers reported problems with charging speed, battery life, and power delivery.

2. **Comfort/Fit Issues** (539 mentions) 
   - Customers complained about the design, ergonomics, and overall comfort of the products.

3. **Heat Issues** (356 mentions)
   - Customers experienced overheating problems with the products.

4. **Performance Problems** (331 mentions)
   - Customers reported issues with the overall performance and functionality of the products.

5. **Value Concerns** (291 mentions)
   - Customers felt the products were overpriced and did not provide good value for money.

**Recommendations:**

1. **Address Battery/Power Issues**
   - Investigate and improve the power management and charging capabilities of the products.
   - Ensure reliable and fast charging performance across all usage scenarios.
   - Optimize battery life to meet customer expectations.

2. **Enhance Comfort and Fit**
   - Redesign the products to improve ergonomics and user comfort.
   - Conduct user testing to gather feedback on the design and make necessary adjustments.
   - Ensure the products are easy to use and integrate seamlessly into the customer's daily life.

3. **Mitigate Heat Issues**
   - Implement better thermal management solutions to prevent overheating.
   - Conduct thorough testing to identify and address any heat-related problems.
   - Educate customers on proper usage and maintenance to minimize heat-related issues.

4. **Improve Overall Performance**
   - Analyze the root causes of the performance problems and make necessary improvements.
   - Optimize the product's functionality to meet or exceed customer expectations.
   - Conduct rigorous testing to ensure consistent and reliable performance.

5. **Enhance Perceived Value**
   - Review the pricing strategy and consider adjusting prices to better align with customer expectations.
   - Highlight the unique features and benefits of the products to justify the pricing.
   - Offer additional value-added services or accessories to enhance the overall customer experience.

By addressing these key issues, you can improve customer satisfaction, increase product ratings, and drive long-term brand loyalty.

------------------------------------------------------------

Press Enter to continue to next example...


############################################################
# EXAMPLE 3: Find the top 5 products by rating and analyze what...
############################################################
============================================================
DECISION MAKING AGENT
============================================================

User Query: Find the top 5 products by rating and analyze what makes them successful.

------------------------------------------------------------

Node: planner
Plan: ['Step 1: Search for the top 5 products by rating in the Electronics category using the search_products tool.', 'Step 2: Analyze the reviews for the top 5 products using the analyze_reviews tool to identify common themes of praise and complaints.', 'Step 3: Calculate statistics for the top 5 products, such as average rating, number of reviews, and price range, using the calculate_statistics tool.', 'Step 4: Identify the key factors that contribute to the success of the top 5 products based on the review analysis and product statistics.', 'Step 5: Summarize the findings to determine what makes the top 5 products successful and how this information could be applied to improve other products in the Electronics category.']

Node: tool_selector

Node: tool_executor
Tool: search_products
Result Preview: Found 10 product(s) matching your criteria:

  **Amazon Basics Wireless Mouse | 2.4 GHz Connection, 1600 DPI | Type - C Adapter | Upto 12 Months of Battery Life | Ambidextrous Design | Suitable for PC...

Node: analyzer

Node: tool_selector

Node: tool_executor
Tool: analyze_reviews
Result Preview: **Review Analysis Report**
   Products Analyzed: 1337
   Total Reviews: 1465
   Analysis Type: Complaints

**Top Complaints Identified:**

**Battery/Power Issues** (722 mentions)
   • "Looks durable C...

Node: analyzer

Node: tool_selector

Node: tool_executor
Tool: calculate_statistics
Result Preview: **Dataset Summary Statistics**

**Overview:**
Total Products: 1337
Total Reviews: 1465
Categories: 211

**Rating Statistics:**
Average Rating: 4.09
Rating Range: 0.0 - 5.0
Avg Reviews per Product: 176...

Node: analyzer

Node: synthesizer
Final answer generated

============================================================
FINAL ANSWER
============================================================
**Key Findings:**

1. **Top 5 Highest Rated Products:**
   - Amazon Basics Wireless Mouse (5.0 stars)
   - Syncwire LTG to USB Cable (5.0 stars) 
   - REDTECH USB-C to Lightning Cable (5.0 stars)
   - Swiffer Instant Electric Water Heater (4.8 stars)
   - Instant Pot Air Fryer (4.8 stars)

2. **Common Positive Attributes:**
   - Reliable, durable, and high-quality construction
   - Fast charging and data transfer speeds
   - Innovative and convenient features
   - Good value for the price

3. **Common Complaints:**
   - Battery/power issues (e.g. slow charging, short battery life)
   - Comfort/fit problems (e.g. cables too stiff, devices uncomfortable to use)
   - Heat issues (e.g. devices getting too hot during use)
   - Performance problems (e.g. slower than expected speeds)
   - Value concerns (e.g. products perceived as overpriced)

**Recommendations:**

1. Focus on products with 4.7+ star ratings, as these tend to have the most positive reviews and satisfied customers.

2. Prioritize products that emphasize durability, fast charging/data transfer, and innovative features. Consumers value these attributes highly.

3. Address common complaints around battery life, comfort, heat management, and performance to ensure a positive user experience.

4. Offer competitive pricing and discounts to provide good value for customers, as price is an important factor.

5. Continuously monitor reviews to identify emerging issues and make product improvements accordingly.

By following these recommendations, you can identify and promote the top performing products that meet customer needs and expectations. This will help drive sales and customer satisfaction.

------------------------------------------------------------

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


