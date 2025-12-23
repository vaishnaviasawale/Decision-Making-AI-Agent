# Decision Making Agent 

A multi-step AI agent built with **LangChain** and **LangGraph** for data-backed decision making. This agent analyzes Amazon sales data to provide actionable business insights through autonomous planning, tool execution, and iterative reasoning.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Tools Reference](#tools-reference)
- [About the dataset](#about-the-dataset)
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
          ┌─────>│    TOOL SELECTOR       │
          │      └───────────┬────────────┘          
          │                  │                       
          │                  ▼                       
          │      ┌────────────────────────┐          
          │      │    TOOL EXECUTOR       │  ← Runs LangChain tools
          │      └───────────┬────────────┘          
          │                  │                       
          │                  ▼                       
          │      ┌────────────────────────┐          
          │      │      ANALYZER          │  ← Evaluates results
          │      └───────────┬────────────┘          
          │                  │                       
          │         [needs_more_info?]               
          │          YES/         \NO                
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

5. **Add the dataset:**
   Download the dataset from
   ``https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset`` and
   add it to the folder `data` with the name `amazon.csv`

6. **Run the agent:**
   ```bash
   python3 main.py
   ```

---

## Usage

### Interactive Mode (Default)

```bash
python3 main.py
```

This starts an interactive session where you can type queries:

```
DECISION MAKING AGENT - Interactive Mode

Your query: Compare Electronics and Clothing categories

[Agent executes and shows results...]
```

### Single Query Mode

```bash
python3 main.py --query "What are the top complaints for Electronics products?"
```

### Run Example Queries

```bash
python3 main.py --example
```

### View Workflow Graph

```bash
python3 main.py --graph
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
| `max_rating` | float | Maximum rating (1.0-5.0) |
| `keyword` | str | Search in product name/description |
| `limit` | int | Maximum results (optional). If omitted, returns all matches. |

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
| `product_names` | list[str] | Filter by an exact list of product names |
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
| `product_names` | list[str] | Limit analysis to an exact list of product names |
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

## About the dataset

The agent uses `data/amazon.csv` from Kaggle (`https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset`).

- **Rows**: 1465 (each row is a product + a review record; products can repeat across rows)
- **Columns (16)**: `product_id`, `product_name`, `category`, `discounted_price`, `actual_price`, `discount_percentage`, `rating`, `rating_count`, `about_product`, `user_id`, `user_name`, `review_id`, `review_title`, `review_content`, `img_link`, `product_link`
- **Unique products**: 1337 distinct `product_name` values (1351 distinct `product_id`)
- **Categories**: 211 distinct `category` values

**Important quirks the code handles:**
- **Hierarchical categories**: `category` uses a `|`-separated path (e.g., `Computers&Accessories|...|USBCables`). The search tool supports partial matches like “cables” or “speakers”.
- **Numeric fields are strings**: prices/discounts may contain currency symbols/percent signs, and `rating_count` has commas. Tools normalize these before computing statistics.

---

## Example Outputs
```
   python3 main.py --example
```
```
DECISION MAKING AGENT - Example Queries

### EXAMPLE 1: Compare the printers and speakers categories. Whic...

DECISION MAKING AGENT

**User Query: Compare the printers and speakers categories. Which one has better customer satisfaction and where should we focus our efforts?**

------------------------------------------------------------

Node: planner
Plan: ['Step 1: Search for products in the Printers and Speakers categories using the search_products tool to get a list of products in each category.', 'Step 2: Analyze the reviews for the top products in each category using the analyze_reviews tool to identify common themes, complaints, and praise from customers.', 'Step 3: Calculate and compare key statistics for the Printers and Speakers categories using the calculate_statistics tool, such as average rating, percentage of positive reviews, and any other relevant metrics.', 'Step 4: Determine which category has better overall customer satisfaction based on the review analysis and statistical comparisons. Identify any areas where one category is significantly outperforming the other, as those will be the focus areas for improvement.']

Node: tool_selector

Node: tool_executor
Tool: search_products
Result Preview: {'summary': 'Found 26 product(s) matching your criteria:\n\n  **Epson 003 65 ml for EcoTank L1110/L3100/L3101/L3110/L3115/L3116/L3150/L3151/L3152/L3156/L5190 Black Ink Bottle**\n   - Category: Compute...

Node: analyzer

Node: tool_selector

Node: tool_executor
Tool: analyze_reviews
Result Preview: **Review Analysis Report**
   Products Analyzed: 26
   Total Reviews: 26
   Analysis Type: Complaints

**Top Complaints Identified:**

**Comfort/Fit Issues** (14 mentions)
   • "After viewing the pric...

Node: analyzer

Node: tool_selector

Node: tool_executor
Tool: calculate_statistics
Result Preview: **Dataset Summary Statistics**

**Overview:**
Total Products: 26
Total Reviews: 1465
Categories: 11

**Rating Statistics:**
Average Rating: 4.00
Rating Range: 3.4 - 4.4
Avg Reviews per Product: 13902
...

Node: analyzer

Node: tool_selector

Node: tool_executor
Tool: calculate_statistics
Result Preview: **Dataset Summary Statistics**

**Overview:**
Total Products: 26
Total Reviews: 1465
Categories: 11

**Rating Statistics:**
Average Rating: 4.00
Rating Range: 3.4 - 4.4
Avg Reviews per Product: 13902
...

Node: analyzer

Node: synthesizer
Final answer generated

### FINAL ANSWER
Here is a comprehensive response addressing the user's original goal:

**Comparison of Printers and Speakers Categories**

**Key Findings:**

Customer Satisfaction:
- The printers category has an average rating of 3.8, while the speakers category has a higher average rating of 4.1. This suggests that customers are generally more satisfied with the speakers products compared to the printers.
- The top complaints for printers include performance issues, value concerns, and quality/durability concerns. For speakers, the top complaints are around connectivity, battery life, and comfort/fit.

Product Pricing and Discounts:
- Printers have a higher average original price of ₹6,339 compared to ₹2,994 for speakers.
- However, speakers have a higher average discount of 52% compared to 17% for printers. This indicates speakers may offer better value for customers.

Product Variety:
- The printers category has a narrower range of 3 subcategories, while the speakers category spans 7 subcategories like Bluetooth, outdoor, multimedia, etc. This suggests the speakers category offers more product variety for customers.

**Recommendations:**

1. Focus Efforts on Speakers Category:
   - The higher customer satisfaction ratings, better value through discounts, and broader product variety make the speakers category a stronger area to focus on.
   - Investigate ways to address the top speaker complaints around connectivity, battery life, and comfort to further improve customer satisfaction.

2. Improve Printer Offerings:
   - While the printers category lags in customer satisfaction, it still represents an important product line. 
   - Analyze the top complaints around performance, value, and quality to identify opportunities for product improvements or better customer support.
   - Consider expanding the printer subcategories to offer more variety, similar to the speakers category.

3. Leverage Data-Driven Insights:
   - Continuously monitor customer feedback and product performance data to identify emerging trends and areas for optimization.
   - Use these insights to guide product development, pricing, and marketing strategies across both the printers and speakers categories.

By focusing efforts on the stronger speakers category while also improving the printers offerings, the business can enhance overall customer satisfaction and drive growth across both product lines.
```
------------------------------------------------------------
```
Press Enter to continue to next example...


### EXAMPLE 2: Analyze customer complaints for products with rati...

DECISION MAKING AGENT

**User Query: Analyze customer complaints for products with ratings below 4.0. What are the main issues and how can we address them?**

------------------------------------------------------------

Node: planner
Plan: ['Step 1: Search for products with ratings below 4.0 using the search_products tool', 'Step 2: Analyze the reviews for those low-rated products using the analyze_reviews tool to identify the main customer complaints', 'Step 3: Summarize the key issues and themes from the customer complaints', 'Step 4: Recommend potential solutions or improvements to address the main customer complaints']

Node: tool_selector

Node: tool_executor
Tool: search_products
Result Preview: {'summary': 'Found 499 product(s) matching your criteria:\n\n  **Nokia 105 Single SIM, Keypad Mobile Phone with Wireless FM Radio | Charcoal**\n   - Category: Electronics|Mobiles&Accessories|Smartphon...

Node: analyzer

Node: tool_selector

Node: tool_executor
Tool: analyze_reviews
Result Preview: **Review Analysis Report**
   Products Analyzed: 499
   Total Reviews: 536
   Analysis Type: Complaints

**Top Complaints Identified:**

**Battery/Power Issues** (276 mentions)
   • "I also connected ...

Node: analyzer

Node: tool_selector

Node: tool_executor
Tool: analyze_reviews
Result Preview: **Review Analysis Report**
   Products Analyzed: 499
   Total Reviews: 536
   Analysis Type: Complaints

**Top Complaints Identified:**

**Battery/Power Issues** (276 mentions)
   • "I also connected ...

Node: analyzer

Node: tool_selector

Node: tool_executor
Tool: analyze_reviews
Result Preview: **Review Analysis Report**
   Products Analyzed: 499
   Total Reviews: 536
   Analysis Type: Complaints

**Top Complaints Identified:**

**Battery/Power Issues** (276 mentions)
   • "I also connected ...

Node: analyzer

Node: synthesizer
Final answer generated

### FINAL ANSWER
Based on the review analysis, here are the key insights and recommendations:

**Key Findings:**
1. Battery/Power Issues: Many customers reported issues with the charging speed and power output of the products, with some complaining that the cables do not support fast charging as advertised.
2. Comfort/Fit Issues: Customers mentioned problems with the fit and build quality of the products, such as loose connectors and parts that are prone to breaking.
3. Performance Problems: Customers experienced slow charging speeds, connectivity issues, and other performance problems with the products.
4. Value Concerns: Some customers felt the products did not offer good value for the price, with complaints about quality, durability, and functionality.

**Recommendations:**
1. Improve charging performance: Ensure the products support the advertised fast charging speeds and power output. Consider using higher quality components and optimizing the charging circuitry.
2. Enhance build quality and durability: Focus on improving the overall build quality and structural integrity of the products to address the comfort/fit and durability issues reported by customers.
3. Manage customer expectations: Clearly communicate the product capabilities and limitations in the descriptions to avoid misleading customers and setting unrealistic expectations.
4. Offer better value: Review the pricing and feature set to ensure the products provide good value for the money. Consider adding more premium features or reducing prices to better align with customer expectations.
5. Improve customer support: Respond promptly to customer complaints and provide clear guidance on troubleshooting and returns/replacements to address the various issues reported.

By addressing these key pain points, you can improve customer satisfaction, reduce negative reviews, and better position your products in the market.
```
------------------------------------------------------------
```
Press Enter to continue to next example...


### EXAMPLE 3: Find the top 5 products by rating and analyze what...

DECISION MAKING AGENT

**User Query: Find the top 5 products by rating and analyze what makes them successful.**

------------------------------------------------------------

Node: planner
Plan: ['Step 1: Search for the top 5 products by rating in the Electronics category', 'Step 2: Analyze the reviews of the top 5 products to identify common themes of praise', 'Step 3: Calculate the average rating, review count, and price for the top 5 products to understand what makes them successful']

Node: tool_selector

Node: tool_executor
Tool: search_products
Params: {'category': 'Electronics', 'limit': 5}
Result Preview: {'summary': 'Found 5 product(s) matching your criteria:\n\n  **Spigen EZ Fit Tempered Glass Screen Protector for iPhone 14 Pro Max - 2 Pack (Sensor Protection)**\n   - Category: Electronics|Mobiles&Ac...

Node: analyzer

Node: tool_selector

Node: tool_executor
Tool: analyze_reviews
Params: {'analysis_type': 'praise', 'product_names': ['Spigen EZ Fit Tempered Glass Screen Protector for iPhone 14 Pro Max - 2 Pack (Sensor Protection)', 'Sony Bravia 164 cm (65 inches) 4K Ultra HD Smart LED Google TV KD-65X74K (Black)', 'Spigen EZ Fit Tempered Glass Screen Protector Guard for iPhone 14/13/13 Pro - 2 Pack', 'Spigen EZ Fit Tempered Glass Screen Protector for iPhone 14 Pro - 2 Pack (Sensor Protection)', '10k 8k 4k HDMI Cable, Certified 48Gbps 1ms Ultra High Speed HDMI 2.1 Cable 4k 120Hz 144Hz 2k 165Hz 8k 60Hz Dynamic HDR ARC eARC DTS:X Compatible for Mac Gaming PC Soundbar TV Monitor Laptop PS5 4 Xbox']}
Result Preview: **Review Analysis Report**
   Products Analyzed: 5
   Total Reviews: 5
   Analysis Type: Praise

**Positive Feedback Highlights:**

**Quality & Build** (5 mentions)
   • "Absolutely value for money. I...

Node: analyzer

Node: tool_selector

Node: tool_executor
Tool: calculate_statistics
Params: {'operation': 'summary', 'product_names': ['Spigen EZ Fit Tempered Glass Screen Protector for iPhone 14 Pro Max - 2 Pack (Sensor Protection)', 'Sony Bravia 164 cm (65 inches) 4K Ultra HD Smart LED Google TV KD-65X74K (Black)', 'Spigen EZ Fit Tempered Glass Screen Protector Guard for iPhone 14/13/13 Pro - 2 Pack', 'Spigen EZ Fit Tempered Glass Screen Protector for iPhone 14 Pro - 2 Pack (Sensor Protection)', '10k 8k 4k HDMI Cable, Certified 48Gbps 1ms Ultra High Speed HDMI 2.1 Cable 4k 120Hz 144Hz 2k 165Hz 8k 60Hz Dynamic HDR ARC eARC DTS:X Compatible for Mac Gaming PC Soundbar TV Monitor Laptop PS5 4 Xbox']}
Result Preview: **Dataset Summary Statistics**

**Overview:**
Total Products: 5
Total Reviews: 1465
Categories: 3

**Rating Statistics:**
Average Rating: 4.64
Rating Range: 4.6 - 4.7
Avg Reviews per Product: 10022

*...

Node: analyzer

Node: synthesizer
Final answer generated

### FINAL ANSWER
# Top 5 Products by Rating Analysis

## Key Findings:

1. **High-Rated Products**: The top 5 products by rating have an average rating of 4.64 out of 5, with a rating range of 4.6 to 4.7. This indicates that these are very well-received and highly-rated products.

2. **Positive Feedback Themes**: The review analysis highlights several key positive feedback themes for these top products:
   - **Quality & Build**: Customers praise the high-quality construction and durability of the products.
   - **Value for Money**: Customers feel these products offer excellent value for the price.
   - **Performance**: Customers are satisfied with the products' performance and functionality.
   - **Comfort & Usability**: Customers find the products easy to use and integrate into their setups.
   - **Design & Aesthetics**: Customers appreciate the attractive design and appearance of the products.

3. **Pricing & Discounts**: The top products have an average price of ₹16,397, with an average discount of 60%. This suggests these are premium products that are often discounted, providing good value to customers.

4. **Product Categories**: The top products span 3 main categories: screen protectors, smart TVs, and HDMI cables. This indicates a diverse range of high-quality products across different electronics categories.

## Recommendations:

1. **Highlight Top Product Features**: Emphasize the key positive attributes that customers value, such as high quality, performance, usability, and value for money. Use customer review quotes to reinforce these points.

2. **Leverage Discounts & Pricing**: Continue offering competitive discounts on these premium products to attract price-conscious customers while maintaining profitability.

3. **Expand Product Lineup**: Consider expanding the product lineup to include more high-quality offerings in the top-performing categories (screen protectors, smart TVs, HDMI cables) as well as adjacent categories to meet a wider range of customer needs.

4. **Improve Packaging & Delivery**: Address any issues with packaging and delivery to ensure a consistently positive customer experience.

5. **Encourage Continued Positive Reviews**: Proactively engage with customers to maintain high levels of positive feedback and ratings for these top-performing products.

By focusing on the key strengths of these top-rated products, optimizing pricing and discounts, and expanding the product lineup, you can continue to deliver exceptional value to your customers and drive business growth.
```
---------------------------------------------------------------



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


