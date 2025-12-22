#!/usr/bin/env python3
"""
Example script demonstrating the Decision Making Agent.

This script shows various use cases and example queries that the agent can handle.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.graph import run_agent, visualize_graph
from src.tools import search_products, analyze_reviews, calculate_statistics


def demonstrate_tools():
    """Demonstrate individual tool usage."""
    print("\n" + "=" * 60)
    print("TOOL DEMONSTRATIONS")
    print("=" * 60)

    # Demo 1: Search Products
    print("\nTool: search_products")
    print("-" * 40)
    result = search_products.invoke(
        {"category": "Electronics", "max_price": 2000, "limit": 3}
    )
    print(result)

    # Demo 2: Analyze Reviews
    print("\nTool: analyze_reviews")
    print("-" * 40)
    result = analyze_reviews.invoke(
        {"category": "Electronics", "analysis_type": "complaints"}
    )
    print(result)

    # Demo 3: Calculate Statistics
    print("\nTool: calculate_statistics")
    print("-" * 40)
    result = calculate_statistics.invoke(
        {
            "operation": "category_comparison",
            "categories": ["Electronics", "Clothing", "Home & Kitchen"],
        }
    )
    print(result)


def demonstrate_agent():
    """Demonstrate the full agent workflow."""
    print("\n" + "=" * 60)
    print("FULL AGENT DEMONSTRATIONS")
    print("=" * 60)

    queries = [
        {
            "title": "Category Comparison",
            "query": "Compare Electronics and Home & Kitchen categories. Which has better ratings and what are the main issues in each?",
        },
        {
            "title": "Problem Identification",
            "query": "Find products with ratings below 4.0 and analyze what customers are complaining about. Suggest concrete improvements.",
        },
        {
            "title": "Discount Analysis",
            "query": "Analyze if higher discounts lead to better customer satisfaction. What's the optimal discount strategy?",
        },
    ]

    for demo in queries:
        print(f"\n\n{'#' * 60}")
        print(f"# {demo['title'].upper()}")
        print("#" * 60)
        print(f"\nQuery: {demo['query']}\n")

        try:
            run_agent(demo["query"], verbose=True)
        except Exception as e:
            print(f"Error: {e}")

        print("\n" + "-" * 60)
        input("\nPress Enter to continue to next demo...")


def show_graph():
    """Display the agent workflow graph."""
    print(visualize_graph())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run example demonstrations")
    parser.add_argument("--tools", action="store_true", help="Demo individual tools")
    parser.add_argument("--agent", action="store_true", help="Demo full agent")
    parser.add_argument("--graph", action="store_true", help="Show workflow graph")
    parser.add_argument("--all", action="store_true", help="Run all demos")

    args = parser.parse_args()

    if args.graph or args.all:
        show_graph()

    if args.tools or args.all:
        demonstrate_tools()

    if args.agent or args.all:
        demonstrate_agent()

    if not any([args.tools, args.agent, args.graph, args.all]):
        print("Usage: python run_examples.py [--tools] [--agent] [--graph] [--all]")
        print("\nOptions:")
        print("  --tools  Demonstrate individual LangChain tools")
        print("  --agent  Demonstrate full agent workflow")
        print("  --graph  Show the LangGraph workflow")
        print("  --all    Run all demonstrations")
