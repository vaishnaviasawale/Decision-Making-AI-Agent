"""
Review Analysis Tool - Processing/Analysis Tool using LangChain.

This tool analyzes customer reviews to identify common complaints,
positive feedback, and extract actionable insights.
"""

import pandas as pd
from typing import Optional, List
from collections import Counter
import re
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from src.config import DATASET_PATH


class ReviewAnalysisInput(BaseModel):
    """Input schema for the review analysis tool."""

    category: Optional[str] = Field(
        default=None, description="Filter reviews by product category (partial match)"
    )
    product_name: Optional[str] = Field(
        default=None,
        description="Filter reviews for a specific product name (partial match)",
    )
    product_names: Optional[List[str]] = Field(
        default=None, description="Filter reviews for a list of product names (exact)"
    )
    analysis_type: str = Field(
        default="complaints",
        description="Type of analysis: 'complaints' (negative feedback), 'praise' (positive feedback), 'all' (comprehensive), or 'themes' (extract common themes)",
    )
    min_rating: Optional[float] = Field(
        default=None, description="Filter reviews from products with minimum rating"
    )
    max_rating: Optional[float] = Field(
        default=None, description="Filter reviews from products with maximum rating"
    )


# Common complaint keywords and patterns
COMPLAINT_PATTERNS = [
    r"\b(bad|poor|terrible|awful|worst|disappointed|disappointing|frustrat\w+)\b",
    r"\b(broke|broken|defective|faulty|damaged|doesnt work|not working)\b",
    r"\b(waste|scam|fake|misleading|false|wrong)\b",
    r"\b(return\w*|refund|exchange)\b",
    r"\b(issue|problem|complaint|bug|error|fail\w*)\b",
    r"\b(expensive|overpriced|not worth)\b",
    r"\b(slow|lag\w*|delay\w*|late)\b",
    r"\b(uncomfortable|hurts?|pain)\b",
    r"\b(leak\w*|spill\w*|break\w*|crack\w*|tear\w*)\b",
    r"\b(hot|heat\w*|overheat\w*)\b",
]

PRAISE_PATTERNS = [
    r"\b(excellent|amazing|awesome|fantastic|perfect|great|good|love|loved)\b",
    r"\b(recommend\w*|best|worth|value|satisfied)\b",
    r"\b(quality|durable|sturdy|reliable|comfortable)\b",
    r"\b(fast|quick|easy|simple|smooth)\b",
    r"\b(beautiful|gorgeous|stylish|elegant)\b",
]


def _load_dataset() -> pd.DataFrame:
    """Load and normalize the Amazon sales dataset."""
    df = pd.read_csv(DATASET_PATH)

    def _to_number(val: str) -> float:
        cleaned = re.sub(r"[^\d.]", "", str(val))
        return float(cleaned) if cleaned else 0.0

    df["discounted_price"] = df["discounted_price"].apply(_to_number)
    df["actual_price"] = df["actual_price"].apply(_to_number)
    df["discount_percentage"] = (
        df["discount_percentage"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .apply(_to_number)
    )
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0)
    df["rating_count"] = (
        df["rating_count"].astype(str).str.replace(",", "", regex=False)
    )
    df["rating_count"] = (
        pd.to_numeric(df["rating_count"], errors="coerce").fillna(0).astype(int)
    )

    if "sub_category" not in df.columns:
        df["sub_category"] = (
            df["category"]
            .astype(str)
            .apply(lambda x: x.split("|")[-1] if "|" in x else "")
        )

    return df


def _extract_issues(reviews: List[str]) -> dict:
    """Extract and categorize issues from reviews."""
    issue_categories = {
        "Quality Issues": [
            "broke",
            "broken",
            "defective",
            "faulty",
            "damaged",
            "peel",
            "tear",
            "crack",
        ],
        "Performance Problems": [
            "slow",
            "lag",
            "crash",
            "freeze",
            "bug",
            "error",
            "not working",
            "doesnt work",
        ],
        "Comfort/Fit Issues": [
            "uncomfortable",
            "hurt",
            "pain",
            "tight",
            "loose",
            "small",
            "big",
            "shrink",
        ],
        "Battery/Power Issues": [
            "battery",
            "drain",
            "charge",
            "charging",
            "power",
            "dies",
        ],
        "Connectivity Issues": [
            "disconnect",
            "connection",
            "wifi",
            "bluetooth",
            "pair",
            "sync",
        ],
        "Value Concerns": ["expensive", "overpriced", "not worth", "waste", "return"],
        "Durability Issues": ["dent", "scratch", "wear", "fade", "rust"],
        "Misleading Description": [
            "misleading",
            "false",
            "fake",
            "advertised",
            "expected",
        ],
        "Heat Issues": ["hot", "heat", "overheat", "warm"],
        "Usability Issues": [
            "confusing",
            "difficult",
            "complicated",
            "hard to use",
            "assembly",
        ],
    }

    found_issues = {cat: [] for cat in issue_categories}

    for review in reviews:
        review_lower = review.lower()
        for category, keywords in issue_categories.items():
            for keyword in keywords:
                if keyword in review_lower:
                    # Extract the sentence containing the issue
                    sentences = review.split(".")
                    for sentence in sentences:
                        if keyword in sentence.lower():
                            found_issues[category].append(sentence.strip())
                            break
                    break

    # Remove empty categories
    return {k: v for k, v in found_issues.items() if v}


def _identify_sentiment(review: str) -> str:
    """Identify if a review is positive, negative, or mixed."""
    complaint_count = sum(
        1 for pattern in COMPLAINT_PATTERNS if re.search(pattern, review.lower())
    )
    praise_count = sum(
        1 for pattern in PRAISE_PATTERNS if re.search(pattern, review.lower())
    )

    if complaint_count > praise_count:
        return "negative"
    elif praise_count > complaint_count:
        return "positive"
    else:
        return "mixed"


@tool(args_schema=ReviewAnalysisInput)
def analyze_reviews(
    category: Optional[str] = None,
    product_name: Optional[str] = None,
    product_names: Optional[List[str]] = None,
    analysis_type: str = "complaints",
    min_rating: Optional[float] = None,
    max_rating: Optional[float] = None,
) -> str:
    """
    Analyze customer reviews to identify complaints, praise, or common themes.

    Use this tool to:
    - Find top customer complaints for products/categories
    - Identify what customers love about products
    - Extract common themes and patterns in feedback
    - Understand pain points for low-rated products

    Examples:
    - Analyze complaints for Electronics products
    - Find what customers praise about products rated above 4.0
    - Extract themes from all Headphones reviews
    """
    try:
        df = _load_dataset()

        # Apply filters
        if category:
            df = df[df["category"].str.contains(category, case=False, na=False)]

        if product_names:
            df = df[df["product_name"].isin(product_names)]
        elif product_name:
            df = df[
                df["product_name"]
                .str.lower()
                .str.contains(product_name.lower(), na=False)
            ]

        if min_rating is not None:
            df = df[df["rating"] >= min_rating]

        if max_rating is not None:
            df = df[df["rating"] <= max_rating]

        if df.empty:
            return "No reviews found matching the specified criteria."

        reviews = df["review_content"].dropna().tolist()
        review_titles = df["review_title"].dropna().tolist()
        products_analyzed = df["product_name"].unique().tolist()

        result_parts = [
            f"**Review Analysis Report**",
            f"   Products Analyzed: {len(products_analyzed)}",
            f"   Total Reviews: {len(reviews)}",
            f"   Analysis Type: {analysis_type.title()}",
            "",
        ]

        if analysis_type == "complaints":
            # Focus on negative aspects
            negative_reviews = [
                r for r in reviews if _identify_sentiment(r) == "negative"
            ]
            issues = _extract_issues(reviews)

            result_parts.append("**Top Complaints Identified:**\n")

            if issues:
                for category_name, examples in sorted(
                    issues.items(), key=lambda x: len(x[1]), reverse=True
                ):
                    result_parts.append(
                        f"**{category_name}** ({len(examples)} mentions)"
                    )
                    for example in examples[:2]:  # Show top 2 examples
                        result_parts.append(f'   • "{example}"')
                    result_parts.append("")
            else:
                result_parts.append("No significant complaints found.")

            # Add summary
            result_parts.append(f"\n**Complaint Summary:**")
            result_parts.append(
                f"   - {len(negative_reviews)} reviews with negative sentiment"
            )
            result_parts.append(
                f"   - {len(issues)} distinct issue categories identified"
            )

        elif analysis_type == "praise":
            # Focus on positive aspects
            positive_reviews = [
                r for r in reviews if _identify_sentiment(r) == "positive"
            ]

            result_parts.append("**Positive Feedback Highlights:**\n")

            praise_themes = {
                "Quality & Build": ["quality", "durable", "sturdy", "premium", "solid"],
                "Value for Money": [
                    "worth",
                    "value",
                    "price",
                    "affordable",
                    "great deal",
                ],
                "Performance": ["fast", "quick", "efficient", "works great", "perfect"],
                "Comfort & Usability": ["comfortable", "easy", "simple", "convenient"],
                "Design & Aesthetics": [
                    "beautiful",
                    "stylish",
                    "looks great",
                    "design",
                ],
            }

            for theme, keywords in praise_themes.items():
                matching_reviews = []
                for review in positive_reviews:
                    if any(kw in review.lower() for kw in keywords):
                        matching_reviews.append(review)

                if matching_reviews:
                    result_parts.append(
                        f"**{theme}** ({len(matching_reviews)} mentions)"
                    )
                    result_parts.append(f'   • "{matching_reviews[0][:100]}..."')
                    result_parts.append("")

            result_parts.append(f"\n**Positive Summary:**")
            result_parts.append(
                f"   - {len(positive_reviews)} reviews with positive sentiment"
            )

        elif analysis_type == "themes":
            # Extract common themes
            all_text = " ".join(reviews + review_titles).lower()

            # Common product-related terms to find
            theme_keywords = [
                "battery",
                "quality",
                "price",
                "comfortable",
                "sound",
                "fit",
                "durable",
                "fast",
                "easy",
                "value",
                "recommend",
                "size",
                "design",
                "performance",
                "customer service",
                "delivery",
            ]

            word_counts = Counter()
            for keyword in theme_keywords:
                count = all_text.count(keyword)
                if count > 0:
                    word_counts[keyword] = count

            result_parts.append("**Common Themes in Reviews:**\n")
            for word, count in word_counts.most_common(10):
                result_parts.append(f"   • {word.title()}: {count} mentions")

        else:  # 'all' - comprehensive analysis
            negative_reviews = [
                r for r in reviews if _identify_sentiment(r) == "negative"
            ]
            positive_reviews = [
                r for r in reviews if _identify_sentiment(r) == "positive"
            ]
            mixed_reviews = [r for r in reviews if _identify_sentiment(r) == "mixed"]

            result_parts.append("**Sentiment Distribution:**")
            result_parts.append(
                f" Positive: {len(positive_reviews)} ({len(positive_reviews) * 100 // len(reviews)}%)"
            )
            result_parts.append(
                f"Negative: {len(negative_reviews)} ({len(negative_reviews) * 100 // len(reviews)}%)"
            )
            result_parts.append(
                f"Mixed: {len(mixed_reviews)} ({len(mixed_reviews) * 100 // len(reviews)}%)"
            )
            result_parts.append("")

            # Top issues
            issues = _extract_issues(reviews)
            if issues:
                result_parts.append("**Key Issues:**")
                for cat, examples in list(issues.items())[:3]:
                    result_parts.append(f"   • {cat}")

            result_parts.append("")
            result_parts.append("**Sample Positive Feedback:**")
            if positive_reviews:
                result_parts.append(f'   "{positive_reviews[0][:150]}..."')

        return "\n".join(result_parts)

    except Exception as e:
        return f"Error analyzing reviews: {str(e)}"
