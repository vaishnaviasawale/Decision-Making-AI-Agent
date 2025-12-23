"""
Node definitions for the LangGraph agent.

Each node represents a step in the agent's reasoning and execution process:
1. Planner: Decomposes user goal into steps
2. Tool Selector: Chooses which tool to use
3. Tool Executor: Executes the selected tool
4. Analyzer: Analyzes results and decides next action
5. Synthesizer: Creates the final user-friendly answer
"""

from typing import Dict, Any, Optional
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
Avoid redundant steps. Do not schedule the same tool with the same intent twice (e.g. calculating the same statistics twice).
Do NOT invent a number (like "top 10") unless the user explicitly asked for a number.
Do NOT say "best-selling" (we don't have a sales volume column in this dataset). Use rating/review_count/price/discount instead.
If the user asks for "top N", include that N explicitly in the step (e.g. "top 5") so the tool selector can pass it as `limit`/`top_n`.

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

Rule: Never put the entire user question sentence into the `category` field. `category` should be a short hint like "speakers" or "printers".
Rule: If the task mentions one or more categories (e.g. "Printers and Speakers"), ALWAYS pass them into `search_products.category` as a comma-separated string like "Printers, Speakers".
Rule: If the user explicitly asks for "top N", pass N into `search_products.limit` or `calculate_statistics.top_n` (especially for `rating_ranking`).
Rule: Do not repeat the same tool call with identical parameters if it has already been executed in previous steps.

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

    def _extract_category_hints(text: str) -> Optional[str]:
        """
        Extract category hints from a planning step or user goal.
        Returns a comma-separated string suitable for search_products.category, or None.
        """
        if not text:
            return None
        t = str(text).strip()
        if not t:
            return None

        # Common plan phrasing: "in the X and Y categories"
        m = re.search(
            r"\b(?:in|within)\s+(?:the\s+)?(.+?)\s+categor(?:y|ies)\b",
            t,
            re.IGNORECASE,
        )
        if m:
            raw = m.group(1)
            raw = raw.replace("&", " and ").replace("|", ",")
            raw = re.sub(r"\band\b", ",", raw, flags=re.IGNORECASE)
            parts = [p.strip(" .") for p in raw.split(",") if p.strip(" .")]
            # Keep only short-ish, category-like phrases; avoid swallowing full sentences.
            parts = [p for p in parts if 1 <= len(p.split()) <= 4 and len(p) <= 40]
            if parts:
                return ", ".join(parts)

        # Fallback: if the text contains "... Printers and Speakers ..." without "category"
        m2 = re.search(
            r"\b([A-Za-z][A-Za-z ]{2,40})\s+and\s+([A-Za-z][A-Za-z ]{2,40})\b", t
        )
        if m2:
            a, b = m2.group(1).strip(), m2.group(2).strip()
            a = a.strip(" .,:;")
            b = b.strip(" .,:;")
            if 1 <= len(a.split()) <= 4 and 1 <= len(b.split()) <= 4:
                return f"{a}, {b}"

        return None

    def _extract_top_n(text: str) -> Optional[int]:
        """Extract 'top N' or 'N products' from text."""
        if not text:
            return None
        t = str(text)
        m = re.search(r"\btop\s+(\d{1,3})\b", t, re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return None
        m2 = re.search(r"\b(\d{1,3})\s+(?:products|items)\b", t, re.IGNORECASE)
        if m2:
            try:
                return int(m2.group(1))
            except ValueError:
                return None
        return None

    def _find_duplicate_result(tool: str, params: Dict[str, Any]) -> Optional[Any]:
        """If an identical tool call already ran, reuse its result to avoid wasted looping."""

        def _norm(val: Any) -> Any:
            if isinstance(val, dict):
                return {k: _norm(val[k]) for k in sorted(val.keys())}
            if isinstance(val, list):
                normed = [_norm(v) for v in val]
                # Sort lists of scalars/dicts deterministically by their string form.
                return sorted(normed, key=lambda x: str(x))
            return val

        target = _norm(params)
        for prev in reversed(state.get("tool_results", [])):
            if prev.get("tool") != tool or "error" in prev:
                continue
            prev_params = prev.get("parameters", {})
            if _norm(prev_params) == target:
                return prev.get("result")
        return None

    try:
        tool_selection = json.loads(json_match.group())
        tool_name = tool_selection.get("tool")
        parameters = tool_selection.get("parameters", {}) or {}

        # Fallback: try to infer tool name from text if missing
        if not tool_name:
            lowered = analysis.lower()
            for candidate in TOOL_MAP.keys():
                if candidate in lowered:
                    tool_name = candidate
                    break

        # Deterministic safety: align tool choice with the current plan step intent.
        # This prevents pointless repeats like running search again when the step says "analyze reviews".
        current_task_text = ""
        try:
            current_task_text = state.get("plan", [])[state.get("current_step", 0)]
        except Exception:
            current_task_text = ""
        step_lower = str(current_task_text).lower()
        if "review" in step_lower and any(
            k in step_lower for k in ["analy", "complaint", "praise", "theme"]
        ):
            tool_name = "analyze_reviews"
        elif any(
            k in step_lower
            for k in [
                "statistic",
                "calculate",
                "compare",
                "ranking",
                "average",
                "summary",
            ]
        ):
            tool_name = "calculate_statistics"

        # Sanitize / enrich parameters for search_products
        if tool_name == "search_products":
            # If selector returns empty params, try to derive category hints from the current task / goal.
            has_any_filter = any(
                parameters.get(k) is not None
                for k in [
                    "category",
                    "keyword",
                    "min_price",
                    "max_price",
                    "min_rating",
                    "max_rating",
                    "limit",
                ]
            )
            if not has_any_filter:
                current_task = ""
                try:
                    current_task = state.get("plan", [])[state.get("current_step", 0)]
                except Exception:
                    current_task = ""
                hint = _extract_category_hints(
                    str(current_task)
                ) or _extract_category_hints(str(state.get("user_goal", "")))
                if hint:
                    parameters["category"] = hint

            # If selector accidentally puts the entire question in category, drop it.
            cat_val = parameters.get("category")
            if isinstance(cat_val, str):
                looks_like_sentence = len(cat_val) > 40 and (" " in cat_val)
                if (
                    looks_like_sentence
                    and cat_val.strip().lower()
                    == state.get("user_goal", "").strip().lower()
                ):
                    parameters.pop("category", None)

            # Infer rating constraints from user goal if present
            goal_text = state.get("user_goal", "")
            if "max_rating" not in parameters:
                m = re.search(
                    r"(below|under|less than)\s*([0-9](?:\.[0-9])?)",
                    goal_text,
                    re.IGNORECASE,
                )
                if m:
                    try:
                        parameters["max_rating"] = float(m.group(2))
                    except ValueError:
                        pass
            if "min_rating" not in parameters:
                m = re.search(
                    r"(above|over|greater than)\s*([0-9](?:\.[0-9])?)",
                    goal_text,
                    re.IGNORECASE,
                )
                if m:
                    try:
                        parameters["min_rating"] = float(m.group(2))
                    except ValueError:
                        pass

            # If the user explicitly asked for top N products, propagate to search limit
            if "limit" not in parameters:
                n = _extract_top_n(state.get("user_goal", ""))
                if n is not None and n > 0:
                    parameters["limit"] = n

        # Provide a safe default operation for statistics if omitted
        if tool_name == "calculate_statistics" and "operation" not in parameters:
            parameters["operation"] = "summary"

        # If analyze_reviews is chosen without analysis_type, infer it from the current step text.
        if tool_name == "analyze_reviews" and "analysis_type" not in parameters:
            if (
                "praise" in step_lower
                or "successful" in step_lower
                or "what makes" in step_lower
            ):
                parameters["analysis_type"] = "praise"
            elif "theme" in step_lower:
                parameters["analysis_type"] = "themes"
            elif (
                "complaint" in step_lower
                or "issue" in step_lower
                or "avoid" in step_lower
            ):
                parameters["analysis_type"] = "complaints"

        # If this is a ranking request and user asks for top N, pass it through
        if tool_name == "calculate_statistics":
            if (
                parameters.get("operation") == "rating_ranking"
                and "top_n" not in parameters
            ):
                n = _extract_top_n(state.get("user_goal", ""))
                if n is not None and n > 0:
                    parameters["top_n"] = n

        # Enforce: analysis/statistics must use the latest search subset unless explicitly requested otherwise
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

        # Avoid re-running deterministic tools with identical parameters
        dup = _find_duplicate_result(tool_name, parameters)
        if dup is not None:
            return {
                "messages": [
                    AIMessage(
                        content=f"Skipped duplicate tool call for '{tool_name}' (reused previous result)."
                    )
                ],
                "tool_results": [
                    {
                        "step": state["current_step"] + 1,
                        "tool": tool_name,
                        "parameters": parameters,
                        "result": dup,
                    }
                ],
                "current_step": state["current_step"] + 1,
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

    # Deterministic loop: execute all planned steps unless a tool error occurs.
    if current_step < len(plan):
        return {
            "messages": [
                AIMessage(
                    content=f"Continuing execution: {len(plan) - current_step} step(s) remaining."
                )
            ],
            "needs_more_info": True,
            "iteration_count": state.get("iteration_count", 0) + 1,
        }

    return {
        "messages": [
            AIMessage(
                content="All planned steps completed. Ready to synthesize answer."
            )
        ],
        "needs_more_info": False,
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
