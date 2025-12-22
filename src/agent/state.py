"""
State definition for the LangGraph agent.

The state represents all information that flows through the agent's
decision-making process, including the user's goal, planned steps,
tool results, and the final answer.
"""

from typing import TypedDict, List, Optional, Annotated
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
    """
    State schema for the decision-making agent.

    Attributes:
        messages: Conversation history including user input and AI responses
        user_goal: The original user request/goal
        plan: List of planned steps to achieve the goal
        current_step: Index of the current step being executed
        tool_results: Results from tool executions
        intermediate_analysis: Analysis of intermediate results
        final_answer: The final response to the user
        iteration_count: Number of iterations (for safety limit)
        needs_more_info: Flag indicating if more tool calls are needed
    """

    # Message history - accumulates using operator.add
    messages: Annotated[List[BaseMessage], operator.add]

    # User's original goal/request
    user_goal: str

    # Planned steps to achieve the goal
    plan: List[str]

    # Current step index in the plan
    current_step: int

    # Results from tool executions (accumulated)
    tool_results: Annotated[List[dict], operator.add]

    # Intermediate analysis and reasoning
    intermediate_analysis: str

    # Final answer to present to user
    final_answer: Optional[str]

    # Iteration counter for safety
    iteration_count: int

    # Flag to indicate if more tool calls are needed
    needs_more_info: bool
