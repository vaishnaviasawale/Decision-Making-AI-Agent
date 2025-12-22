"""
LangGraph Graph Definition for the Decision Making Agent.

This module defines the complete agent workflow using LangGraph,
including nodes, edges, conditional branching, and the execution loop.

Graph Structure:
    START → Planner → Tool Selector → Tool Executor → Analyzer
                            ↑                             |
                            └──── [needs_more_info?] ─────┘
                                          ↓ (no)
                                    Synthesizer → END
"""

from typing import Literal, TYPE_CHECKING
from langgraph.graph import StateGraph, START, END  # type: ignore

# MemorySaver import with fallback for different langgraph versions.
# TYPE_CHECKING branch avoids linter complaints; runtime branch handles version differences.
if TYPE_CHECKING:
    from langgraph.checkpoint.memory import MemorySaver  # type: ignore[attr-defined]
else:  # pragma: no cover
    try:
        from langgraph.checkpoint.memory import MemorySaver  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        try:
            from langgraph.checkpoint import MemorySaver  # type: ignore[attr-defined]
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "MemorySaver not found; ensure langgraph is installed with checkpoint extras."
            ) from exc

from src.agent.state import AgentState
from src.agent.nodes import (
    planner_node,
    tool_selector_node,
    tool_executor_node,
    analyzer_node,
    synthesizer_node,
)
from src.config import MAX_ITERATIONS


def should_continue(state: AgentState) -> Literal["tool_selector", "synthesizer"]:
    """
    Conditional edge function: Determines if the agent should continue
    gathering information or synthesize the final answer.

    Returns:
        "tool_selector" - Continue with more tool calls
        "synthesizer" - Ready to create final answer
    """
    # Safety check: prevent infinite loops
    if state.get("iteration_count", 0) >= MAX_ITERATIONS:
        print(
            f"Max iterations ({MAX_ITERATIONS}) reached. Synthesizing available results."
        )
        return "synthesizer"

    # Check if more information is needed
    if state.get("needs_more_info", True):
        # Check if we still have planned steps to execute
        current_step = state.get("current_step", 0)
        plan = state.get("plan", [])

        if current_step < len(plan):
            return "tool_selector"

    return "synthesizer"


def create_agent_graph() -> StateGraph:
    """
    Creates and compiles the LangGraph workflow for the decision-making agent.

    The graph implements a planning-execution-analysis loop:
    1. Planner decomposes the user goal into steps
    2. Tool Selector chooses the right tool for each step
    3. Tool Executor runs the selected tool
    4. Analyzer evaluates results and decides next action
    5. Synthesizer creates the final answer when ready

    Returns:
        Compiled StateGraph ready for execution
    """
    # Initialize the graph with our state schema
    workflow = StateGraph(AgentState)

    # Add nodes to the graph
    workflow.add_node("planner", planner_node)
    workflow.add_node("tool_selector", tool_selector_node)
    workflow.add_node("tool_executor", tool_executor_node)
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("synthesizer", synthesizer_node)

    # Define edges (the flow between nodes)

    # Entry point: Start → Planner
    workflow.add_edge(START, "planner")

    # Planner → Tool Selector (begin executing the plan)
    workflow.add_edge("planner", "tool_selector")

    # Tool Selector → Tool Executor
    workflow.add_edge("tool_selector", "tool_executor")

    # Tool Executor → Analyzer
    workflow.add_edge("tool_executor", "analyzer")

    # Conditional edge from Analyzer
    # Either continue with more tools or synthesize final answer
    workflow.add_conditional_edges(
        "analyzer",
        should_continue,
        {
            "tool_selector": "tool_selector",  # Loop back for more tool calls
            "synthesizer": "synthesizer",  # Ready to finish
        },
    )

    # Synthesizer → End
    workflow.add_edge("synthesizer", END)

    # Compile the graph
    # Using MemorySaver for checkpointing (enables state persistence)
    memory = MemorySaver()
    compiled_graph = workflow.compile(checkpointer=memory)

    return compiled_graph


def run_agent(user_query: str, verbose: bool = True) -> str:
    """
    Run the decision-making agent with a user query.

    Args:
        user_query: The user's natural language request
        verbose: Whether to print intermediate steps

    Returns:
        The final synthesized answer
    """
    # Create the graph
    graph = create_agent_graph()

    # Initialize the state
    initial_state = {
        "messages": [],
        "user_goal": user_query,
        "plan": [],
        "current_step": 0,
        "tool_results": [],
        "intermediate_analysis": "",
        "final_answer": None,
        "iteration_count": 0,
        "needs_more_info": True,
    }

    # Configuration for the run
    config = {"configurable": {"thread_id": "agent-run-1"}}

    if verbose:
        print("=" * 60)
        print("DECISION MAKING AGENT")
        print("=" * 60)
        print(f"\nUser Query: {user_query}\n")
        print("-" * 60)

    # Run the graph
    final_state = None

    for event in graph.stream(initial_state, config):
        if verbose:
            # Print the current node being executed
            for node_name, node_output in event.items():
                print(f"\nNode: {node_name}")

                # Print key information based on node
                if node_name == "planner" and "plan" in node_output:
                    print(f"Plan: {node_output['plan']}")

                elif node_name == "tool_executor" and "tool_results" in node_output:
                    for result in node_output["tool_results"]:
                        if "error" not in result:
                            print(f"Tool: {result.get('tool')}")
                            print(
                                f"Result Preview: {str(result.get('result', ''))[:200]}..."
                            )
                        else:
                            print(f"Error: {result.get('error')}")

                elif node_name == "synthesizer" and "final_answer" in node_output:
                    print("Final answer generated")

        # Keep track of the final state
        final_state = event

    # Extract the final answer
    if final_state:
        # Get the last node's output
        for node_output in final_state.values():
            if "final_answer" in node_output and node_output["final_answer"]:
                if verbose:
                    print("\n" + "=" * 60)
                    print("FINAL ANSWER")
                    print("=" * 60)
                    print(node_output["final_answer"])
                return node_output["final_answer"]

    return "Unable to generate a response. Please try again."


def visualize_graph():
    """
    Generate a visualization of the graph structure.
    Returns a string representation of the graph.
    """
    graph_description = """
    ┌─────────────────────────────────────────────────────────────┐
    │                    AGENT WORKFLOW GRAPH                     │
    └─────────────────────────────────────────────────────────────┘
    
                              ┌─────────┐
                              │  START  │
                              └────┬────┘
                                   │
                                   ▼
                            ┌──────────────┐
                            │   PLANNER    │
                            │  (LLM Node)  │
                            └──────┬───────┘
                                   │
                                   ▼
                     ┌────────────────────────┐
              ┌──────│    TOOL SELECTOR       │◄────────┐
              │      │      (LLM Node)        │         │
              │      └───────────┬────────────┘         │
              │                  │                      │
              │                  ▼                      │
              │      ┌────────────────────────┐         │
              │      │    TOOL EXECUTOR       │         │
              │      │   (Executes Tools)     │         │
              │      └───────────┬────────────┘         │
              │                  │                      │
              │                  ▼                      │
              │      ┌────────────────────────┐         │
              │      │      ANALYZER          │         │
              │      │      (LLM Node)        │         │
              │      └───────────┬────────────┘         │
              │                  │                      │
              │         ┌───────┴───────┐               │
              │         │ needs_more_   │               │
              │         │    info?      │               │
              │         └───────┬───────┘               │
              │          YES/   │   \\NO                │
              │              ───┘    ───                │
              │             /           \\              │
              │            │             │              │
              └────────────┘             ▼              │
                                ┌──────────────┐        │
                                │ SYNTHESIZER  │        │
                                │  (LLM Node)  │        │
                                └──────┬───────┘        │
                                       │                │
                                       ▼                │
                                  ┌─────────┐           │
                                  │   END   │           │
                                  └─────────┘           │
    
    ─────────────────────────────────────────────────────────────
    NODES:
    • Planner: Decomposes user goal into actionable steps
    • Tool Selector: Chooses appropriate tool and parameters
    • Tool Executor: Runs the selected LangChain tool
    • Analyzer: Evaluates results, decides next action
    • Synthesizer: Creates final user-friendly response
    
    CONDITIONAL EDGES:
    • Analyzer → Tool Selector: If more info needed (loop)
    • Analyzer → Synthesizer: If goal achieved (finish)
    ─────────────────────────────────────────────────────────────
    """
    return graph_description
