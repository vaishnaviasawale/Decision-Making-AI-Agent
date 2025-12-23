#!/usr/bin/env python3
"""
A multi-step decision making AI agent using LangChain + LangGraph for data-backed decision making.
It analyzes Amazon sales data to provide actionable business insights.

Usage:
    python main.py                    # Interactive mode
    python main.py --query "..."      # Single query mode
    python main.py --example          # Run example queries
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.agent.graph import run_agent, visualize_graph
from src.config import CLAUDE_API_KEY, VERBOSE


def check_api_key():
    """Check if CLAUDE API key is configured."""
    if not CLAUDE_API_KEY:
        print("Error: CLAUDE_API_KEY not found!")
        return False
    return True


def run_interactive_mode():
    """Run the agent in interactive mode."""
    print("\n" + "=" * 60)
    print("DECISION MAKING AGENT - Interactive Mode")
    print("=" * 60)
    print("\nI can help you analyze Amazon sales data to make informed decisions.")
    print("Type 'quit' to exit, 'help' for example queries, 'graph' to see workflow.\n")

    example_queries = [
        "What are the top complaints for Electronics products?",
        "Compare Electronics, Clothing, and Home & Kitchen categories",
        "Find products with high discounts but low ratings - what's going wrong?",
        "Analyze reviews for products rated below 4.0 and suggest improvements",
        "What are the best-rated products under ₹2000?",
    ]

    while True:
        try:
            user_input = input("\nYour query: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "quit":
                print("\nGoodbye!")
                break

            if user_input.lower() == "help":
                print("\nExample queries you can try:")
                for i, query in enumerate(example_queries, 1):
                    print(f"   {i}. {query}")
                continue

            if user_input.lower() == "graph":
                print(visualize_graph())
                continue

            # Run the agent
            run_agent(user_input, verbose=VERBOSE)

        except KeyboardInterrupt:
            print("\n\nKeyboard Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again with a different query.")


def run_example_queries():
    """Run a set of example queries to demonstrate the agent."""
    example_queries = [
        "Compare the printers and speakers categories. Which one has better customer satisfaction and where should we focus our efforts?",
        "Analyze customer complaints for products with ratings below 4.0. What are the main issues and how can we address them?",
        "Find the top 5 products by rating and analyze what makes them successful.",
    ]

    print("\n" + "=" * 60)
    print("DECISION MAKING AGENT - Example Queries")
    print("=" * 60)

    for i, query in enumerate(example_queries, 1):
        print(f"\n\n{'#' * 60}")
        print(f"# EXAMPLE {i}: {query[:50]}...")
        print("#" * 60)

        try:
            run_agent(query, verbose=True)
        except Exception as e:
            print(f"Error running example {i}: {e}")

        print("\n" + "-" * 60)

        # Pause between examples
        if i < len(example_queries):
            input("\nPress Enter to continue to next example...")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Decision Making Agent - LangChain + LangGraph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Interactive mode
  python main.py --query "Analyze Electronics"     # Single query
  python main.py --example                          # Run examples
  python main.py --graph                            # Show workflow graph
        """,
    )

    parser.add_argument("--query", "-q", type=str, help="Run a single query and exit")

    parser.add_argument(
        "--example",
        "-e",
        action="store_true",
        help="Run example queries to demonstrate the agent",
    )

    parser.add_argument(
        "--graph", "-g", action="store_true", help="Display the agent workflow graph"
    )

    args = parser.parse_args()

    # Show graph and exit
    if args.graph:
        print(visualize_graph())
        return

    # Check API key
    if not check_api_key():
        sys.exit(1)

    verbose = VERBOSE

    # Run single query
    if args.query:
        run_agent(args.query, verbose=verbose)
        return

    # Run examples
    if args.example:
        run_example_queries()
        return

    # Default: interactive mode
    run_interactive_mode()


if __name__ == "__main__":
    main()
