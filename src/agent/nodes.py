"""
Node definitions for the LangGraph agent.

Each node represents a step in the agent's reasoning and execution process:
1. Planner: Decomposes user goal into steps
2. Tool Selector: Chooses which tool to use
3. Tool Executor: Executes the selected tool
4. Analyzer: Analyzes results and decides next action
5. Synthesizer: Creates the final user-friendly answer
"""

from typing import Dict, Any
import re
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from src.config import MODEL_NAME, TEMPERATURE, CLAUDE_API_KEY
from src.agent.state import AgentState
from src.tools import search_products, analyze_reviews, calculate_statistics


# Initialize LLM
def get_llm():
    """Get the LLM instance."""
    return ChatAnthropic(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        api_key=CLAUDE_API_KEY,
    )


# Available tools
TOOLS = [search_products, analyze_reviews, calculate_statistics]
TOOL_MAP = {tool.name: tool for tool in TOOLS}


def planner_node(state: AgentState) -> Dict[str, Any]:
    """
    Planner Node: Decomposes the user's goal into actionable steps.

    This node takes the user's natural language request and creates
    a structured plan of steps to achieve the goal.
    """
    llm = get_llm()

    planner_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="""You are a planning assistant for data-backed decision making.
Your job is to decompose user requests into clear, actionable steps.

Available tools for each step:
1. search_products - Find products by category, price, rating, or keywords
2. analyze_reviews - Analyze customer reviews for complaints, praise, or themes
3. calculate_statistics - Calculate statistics, comparisons, and rankings

Create a plan with 2-5 steps. Each step should be a clear action that can be executed with one of the tools.

Do NOT assume a specific category (like "Electronics") unless the user explicitly mentions it.

Respond with a JSON array of steps, like:
["Step 1: Search for products in Electronics category", "Step 2: Analyze reviews to find complaints", "Step 3: Calculate category statistics for comparison"]
"""
            ),
            HumanMessage(
                content=f"User Goal: {state['user_goal']}\n\nCreate a plan to achieve this goal."
            ),
        ]
    )

    response = llm.invoke(planner_prompt.format_messages())

    # Parse the plan from the response
    import json

    # Try to extract JSON array from response
    response_text = response.content
    json_match = re.search(r"\[.*\]", response_text, re.DOTALL)

    if json_match:
        try:
            plan = json.loads(json_match.group())
        except json.JSONDecodeError:
            # Fallback: split by newlines
            plan = [line.strip() for line in response_text.split("\n") if line.strip()]
    else:
        plan = [line.strip() for line in response_text.split("\n") if line.strip()]

    return {
        "messages": [AIMessage(content=f"Plan created: {plan}")],
        "plan": plan,
        "current_step": 0,
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def tool_selector_node(state: AgentState) -> Dict[str, Any]:
    """
    Tool Selector Node: Decides which tool to use for the current step.

    Based on the current step in the plan and previous results,
    this node selects the appropriate tool and its parameters.
    """
    llm = get_llm()

    current_step = state["current_step"]
    plan = state["plan"]

    if current_step >= len(plan):
        return {
            "messages": [AIMessage(content="All steps completed.")],
            "needs_more_info": False,
        }

    current_task = plan[current_step]
    previous_results = state.get("tool_results", [])

    selector_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="""You are a tool selection assistant. Based on the current task, select the appropriate tool and its parameters.

Dataset columns available: product_id, product_name, category, discounted_price, actual_price, discount_percentage, rating, rating_count, about_product, review_* fields. 

Available tools and their parameters:

1) search_products
   - category: str (partial match, case-insensitive)
   - min_price: float
   - max_price: float
   - min_rating: float (1.0-5.0)
   - max_rating: float (1.0-5.0)
   - keyword: str (search in product name/description)
   - limit: int (optional)

2) analyze_reviews
   - category: str
   - product_name: str (partial match)
   - product_names: list[str] (exact list)
   - analysis_type: str ("complaints", "praise", "themes", "all")
   - min_rating: float
   - max_rating: float

3) calculate_statistics
   - operation: str ("category_comparison", "price_analysis", "rating_ranking", "discount_effectiveness", "summary")
   - product_names: list[str] (exact list)
   - categories: list[str] (for comparison)
   - top_n: int (for rankings)
   - group_by: str ("category")

Important:
- If the user goal or current task mentions categories (e.g., "speakers", "printers"), include them in the category (or categories) parameter so search_products and downstream tools operate on the right subset. Do not leave category empty when categories are implied.

Respond with ONLY this JSON object (no code fences, no extra text):
{"tool": "tool_name", "parameters": { ... }}
"""
            ),
            HumanMessage(
                content=f"""
Current Task: {current_task}

Previous Results Summary: {len(previous_results)} tool calls completed

User's Original Goal: {state["user_goal"]}

Select the appropriate tool and parameters for this task.
"""
            ),
        ]
    )

    response = llm.invoke(selector_prompt.format_messages())

    return {
        "messages": [
            AIMessage(
                content=f"Tool selected for step {current_step + 1}: {response.content}"
            )
        ],
        "intermediate_analysis": response.content,
        "needs_more_info": True,
    }


def tool_executor_node(state: AgentState) -> Dict[str, Any]:
    """
    Tool Executor Node: Executes the selected tool with given parameters.

    Parses the tool selection and parameters, then invokes the appropriate
    LangChain tool to get results.
    """
    import json

    # Parse tool selection from intermediate analysis
    analysis = state.get("intermediate_analysis", "")

    # Extract JSON from the response
    json_match = re.search(r"\{[^{}]*\}", analysis, re.DOTALL)

    if not json_match:
        return {
            "messages": [AIMessage(content="Could not parse tool selection.")],
            "tool_results": [{"error": "Failed to parse tool selection"}],
            "needs_more_info": False,
        }

    try:
        tool_selection = json.loads(json_match.group())
        tool_name = tool_selection.get("tool")
        parameters = tool_selection.get("parameters", {})

        # Fallback: try to infer tool name from text if missing
        if not tool_name:
            lowered = analysis.lower()
            for candidate in TOOL_MAP.keys():
                if candidate in lowered:
                    tool_name = candidate
                    break

        # Provide a safe default operation for statistics if omitted
        if tool_name == "calculate_statistics" and "operation" not in parameters:
            parameters["operation"] = "summary"

        # Enforce pipeline: downstream tools must use the latest search subset (if available)
        if tool_name in ["analyze_reviews", "calculate_statistics"]:
            last_search = None
            for prev in reversed(state.get("tool_results", [])):
                if prev.get("tool") == "search_products":
                    last_search = prev.get("result")
                    break

            if isinstance(last_search, dict) and isinstance(
                last_search.get("products"), list
            ):
                product_names = [
                    p.get("product_name")
                    for p in last_search["products"]
                    if p.get("product_name")
                ]
                if (
                    tool_name == "analyze_reviews"
                    and "product_names" not in parameters
                    and "product_name" not in parameters
                ):
                    parameters["product_names"] = product_names
                if (
                    tool_name == "calculate_statistics"
                    and "product_names" not in parameters
                    and not parameters.get("categories")
                ):
                    parameters["product_names"] = product_names
            elif isinstance(last_search, str):
                # Search failed/was too broad; do not proceed with whole-dataset analysis
                return {
                    "messages": [
                        AIMessage(
                            content=f"Stopping: search did not produce a product subset ({last_search})"
                        )
                    ],
                    "tool_results": [
                        {
                            "error": f"Search did not produce a product subset: {last_search}"
                        }
                    ],
                    "needs_more_info": False,
                }

        # Fallback: if search_products is selected with no category/keyword, try to use user_goal as a hint
        if tool_name == "search_products":
            if not parameters.get("category") and not parameters.get("keyword"):
                parameters["category"] = state.get("user_goal", "")
            # Infer rating filter from user goal if present (e.g., "below 4.0", "under 4")
            if "max_rating" not in parameters:
                goal_text = state.get("user_goal", "")
                match = re.search(
                    r"(below|under|less than)\s*([0-9]\.?[0-9]?)",
                    goal_text,
                    re.IGNORECASE,
                )
                if match:
                    try:
                        parameters["max_rating"] = float(match.group(2))
                    except ValueError:
                        pass

        if not tool_name:
            return {
                "messages": [AIMessage(content="Tool selection missing tool name.")],
                "tool_results": [{"error": "Tool selection missing tool name"}],
                "needs_more_info": False,
            }

        if tool_name not in TOOL_MAP:
            return {
                "messages": [AIMessage(content=f"Unknown tool: {tool_name}")],
                "tool_results": [{"error": f"Unknown tool: {tool_name}"}],
                "needs_more_info": False,
            }

        # Execute the tool
        tool = TOOL_MAP[tool_name]
        result = tool.invoke(parameters)

        return {
            "messages": [
                AIMessage(content=f"Tool '{tool_name}' executed successfully.")
            ],
            "tool_results": [
                {
                    "step": state["current_step"] + 1,
                    "tool": tool_name,
                    "parameters": parameters,
                    "result": result,
                }
            ],
            "current_step": state["current_step"] + 1,
        }

    except json.JSONDecodeError as e:
        return {
            "messages": [AIMessage(content=f"JSON parsing error: {e}")],
            "tool_results": [{"error": f"JSON parsing error: {str(e)}"}],
            "needs_more_info": False,
        }
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"Tool execution error: {e}")],
            "tool_results": [{"error": f"Tool execution error: {str(e)}"}],
            "needs_more_info": False,
        }


def analyzer_node(state: AgentState) -> Dict[str, Any]:
    """
    Analyzer Node: Analyzes results and decides if more steps are needed.

    Reviews the tool execution results, determines if the goal is achieved,
    or if more information/steps are needed.
    """
    llm = get_llm()

    plan = state["plan"]
    current_step = state["current_step"]
    tool_results = state.get("tool_results", [])

    # If the latest result is an error, stop looping and synthesize with what we have
    if tool_results and "error" in tool_results[-1]:
        return {
            "messages": [
                AIMessage(
                    content=f"Halting because of tool error: {tool_results[-1]['error']}"
                )
            ],
            "needs_more_info": False,
        }

    # Check if all planned steps are completed
    if current_step >= len(plan):
        return {
            "messages": [
                AIMessage(
                    content="All planned steps completed. Ready to synthesize answer."
                )
            ],
            "needs_more_info": False,
        }

    # Analyze current results to decide next action
    analyzer_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="""You are an analysis assistant. Review the results and determine:
1. Are the results sufficient to answer the user's goal?
2. Should we continue with the next planned step?
3. Do we need to adjust our approach?

Respond with:
{
    "sufficient": true/false,
    "reasoning": "brief explanation",
    "next_action": "continue" or "synthesize" or "adjust"
}
"""
            ),
            HumanMessage(
                content=f"""
User Goal: {state["user_goal"]}

Plan: {plan}

Current Progress: Step {current_step} of {len(plan)}

Latest Results: {tool_results[-1] if tool_results else "None"}

Analyze and decide next action.
"""
            ),
        ]
    )

    response = llm.invoke(analyzer_prompt.format_messages())

    # Parse response to determine next action
    import json

    json_match = re.search(r"\{[^{}]*\}", response.content, re.DOTALL)

    needs_more = True
    if json_match:
        try:
            analysis = json.loads(json_match.group())
            if analysis.get("next_action") == "synthesize" or analysis.get(
                "sufficient"
            ):
                needs_more = False
        except json.JSONDecodeError:
            pass

    # Also stop if we've completed all steps
    if current_step >= len(plan):
        needs_more = False

    return {
        "messages": [AIMessage(content=f"Analysis: {response.content}")],
        "intermediate_analysis": response.content,
        "needs_more_info": needs_more,
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    """
    Synthesizer Node: Creates the final user-friendly answer.

    Takes all the gathered information, tool results, and analysis
    to produce a comprehensive, actionable response for the user.
    """
    llm = get_llm()

    tool_results = state.get("tool_results", [])

    # Format all results for the synthesis
    def _format_result(res: Any) -> str:
        if isinstance(res, dict) and "summary" in res:
            return str(res["summary"])
        text = str(res)
        return text if len(text) <= 3000 else text[:3000] + "\n...(truncated)"

    results_summary = "\n\n".join(
        [
            f"**Step {r.get('step', 'N/A')}: {r.get('tool', 'Unknown')}**\n{_format_result(r.get('result', 'No result'))}"
            for r in tool_results
            if "error" not in r
        ]
    )

    synthesizer_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="""You are a business analyst synthesizing research findings.

Create a comprehensive, user-friendly response that:
1. Directly addresses the user's original goal
2. Presents key findings clearly with data support
3. Provides actionable recommendations
4. Uses clear formatting with headers and bullet points
5. Highlights the most important insights

Be concise but thorough. Use data from the tool results to support your recommendations.
"""
            ),
            HumanMessage(
                content=f"""
User's Original Goal: {state["user_goal"]}

Research Results:
{results_summary}

Synthesize these findings into a clear, actionable response for the user.
"""
            ),
        ]
    )

    response = llm.invoke(synthesizer_prompt.format_messages())

    return {
        "messages": [AIMessage(content="Final answer synthesized.")],
        "final_answer": response.content,
        "needs_more_info": False,
    }
