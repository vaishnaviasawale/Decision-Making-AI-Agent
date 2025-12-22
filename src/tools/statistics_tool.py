"""
Statistics Tool - Data Processing/Calculation Tool using LangChain.

This tool calculates statistics, performs comparisons, rankings, and
generates data-driven insights from the sales dataset.
"""

import pandas as pd
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from src.config import DATASET_PATH


class StatisticsInput(BaseModel):
    """Input schema for the statistics tool."""

    operation: str = Field(
        description="Type of statistical operation: 'category_comparison', 'price_analysis', 'rating_ranking', 'discount_effectiveness', 'summary'"
    )
    categories: Optional[List[str]] = Field(
        default=None,
        description="List of categories to compare (for category_comparison)",
    )
    top_n: int = Field(
        default=5, description="Number of top results to return for rankings"
    )
    group_by: Optional[str] = Field(
        default=None,
        description="Field to group results by: 'category', 'sub_category'",
    )


def _load_dataset() -> pd.DataFrame:
    """Load the Amazon sales dataset."""
    df = pd.read_csv(DATASET_PATH)
    df["discount_percentage"] = (
        df["discount_percentage"].str.replace("%", "").astype(float)
    )
    return df


def _get_unique_products(df: pd.DataFrame) -> pd.DataFrame:
    """Get unique products (remove duplicate entries from reviews)."""
    return df.drop_duplicates(subset=["product_name"])


@tool(args_schema=StatisticsInput)
def calculate_statistics(
    operation: str,
    categories: Optional[List[str]] = None,
    top_n: int = 5,
    group_by: Optional[str] = None,
) -> str:
    """
    Calculate statistics and perform data analysis on the sales dataset.

    Available operations:
    - 'category_comparison': Compare metrics across categories
    - 'price_analysis': Analyze pricing and discounts
    - 'rating_ranking': Rank products/categories by ratings
    - 'discount_effectiveness': Analyze relationship between discounts and ratings
    - 'summary': Overall dataset summary statistics

    Examples:
    - Compare Electronics, Clothing, and Home & Kitchen categories
    - Find top 5 products by rating
    - Analyze if high discounts correlate with better ratings
    """
    try:
        df = _load_dataset()
        unique_products = _get_unique_products(df)

        if operation == "category_comparison":
            # Compare metrics across categories
            if categories:
                filtered = unique_products[
                    unique_products["category"]
                    .str.lower()
                    .isin([c.lower() for c in categories])
                ]
            else:
                filtered = unique_products

            comparison = (
                filtered.groupby("category")
                .agg(
                    {
                        "rating": ["mean", "min", "max", "count"],
                        "discounted_price": "mean",
                        "discount_percentage": "mean",
                        "rating_count": "sum",
                    }
                )
                .round(2)
            )

            result_parts = ["**Category Comparison Analysis**\n"]

            for category in comparison.index:
                cat_data = comparison.loc[category]
                result_parts.append(f"**{category}**")
                result_parts.append(f"Products: {int(cat_data[('rating', 'count')])}")
                result_parts.append(
                    f"Avg Rating: {cat_data[('rating', 'mean')]:.2f} (Range: {cat_data[('rating', 'min')]:.1f} - {cat_data[('rating', 'max')]:.1f})"
                )
                result_parts.append(
                    f"Avg Price: ₹{cat_data[('discounted_price', 'mean')]:.0f}"
                )
                result_parts.append(
                    f"Avg Discount: {cat_data[('discount_percentage', 'mean')]:.1f}%"
                )
                result_parts.append(
                    f"Total Reviews: {int(cat_data[('rating_count', 'sum')])}"
                )
                result_parts.append("")

            # Add insights
            best_rated = comparison[("rating", "mean")].idxmax()
            most_discounted = comparison[("discount_percentage", "mean")].idxmax()
            most_reviewed = comparison[("rating_count", "sum")].idxmax()

            result_parts.append("**Key Insights:**")
            result_parts.append(f"• Highest rated category: {best_rated}")
            result_parts.append(f"• Highest discounts: {most_discounted}")
            result_parts.append(f"• Most reviewed: {most_reviewed}")

            return "\n".join(result_parts)

        elif operation == "price_analysis":
            result_parts = ["**Price Analysis Report**\n"]

            if group_by:
                price_stats = (
                    unique_products.groupby(group_by)
                    .agg(
                        {
                            "actual_price": ["mean", "min", "max"],
                            "discounted_price": ["mean", "min", "max"],
                            "discount_percentage": "mean",
                        }
                    )
                    .round(2)
                )

                for group in price_stats.index:
                    g_data = price_stats.loc[group]
                    result_parts.append(f"**{group}**")
                    result_parts.append(
                        f"Original: ₹{g_data[('actual_price', 'min')]:.0f} - ₹{g_data[('actual_price', 'max')]:.0f} (Avg: ₹{g_data[('actual_price', 'mean')]:.0f})"
                    )
                    result_parts.append(
                        f"Discounted: ₹{g_data[('discounted_price', 'min')]:.0f} - ₹{g_data[('discounted_price', 'max')]:.0f} (Avg: ₹{g_data[('discounted_price', 'mean')]:.0f})"
                    )
                    result_parts.append(
                        f"Avg Discount: {g_data[('discount_percentage', 'mean')]:.1f}%"
                    )
                    result_parts.append("")
            else:
                result_parts.append("**Overall Price Statistics:**")
                result_parts.append(
                    f"Price Range: ₹{unique_products['discounted_price'].min():.0f} - ₹{unique_products['discounted_price'].max():.0f}"
                )
                result_parts.append(
                    f"Average Price: ₹{unique_products['discounted_price'].mean():.0f}"
                )
                result_parts.append(
                    f"Discount Range: {unique_products['discount_percentage'].min():.0f}% - {unique_products['discount_percentage'].max():.0f}%"
                )
                result_parts.append(
                    f"Average Discount: {unique_products['discount_percentage'].mean():.1f}%"
                )
                result_parts.append(
                    f"Total Savings: ₹{(unique_products['actual_price'] - unique_products['discounted_price']).sum():.0f}"
                )

            return "\n".join(result_parts)

        elif operation == "rating_ranking":
            result_parts = ["**Rating Rankings**\n"]

            if group_by == "category":
                ranking = (
                    unique_products.groupby("category")["rating"]
                    .mean()
                    .sort_values(ascending=False)
                )
                result_parts.append("**Categories Ranked by Average Rating:**")
                for i, (cat, rating) in enumerate(ranking.head(top_n).items(), 1):
                    result_parts.append(f"{i}. {cat}: {rating:.2f}")
            elif group_by == "sub_category":
                ranking = (
                    unique_products.groupby("sub_category")["rating"]
                    .mean()
                    .sort_values(ascending=False)
                )
                result_parts.append("**Sub-categories Ranked by Average Rating:**")
                for i, (subcat, rating) in enumerate(ranking.head(top_n).items(), 1):
                    result_parts.append(f"{i}. {subcat}: {rating:.2f}")
            else:
                # Rank individual products
                top_products = unique_products.nlargest(top_n, "rating")
                result_parts.append(f"**Top {top_n} Products by Rating:**")
                for i, (_, row) in enumerate(top_products.iterrows(), 1):
                    result_parts.append(f"{i}. {row['product_name'][:40]}...")
                    result_parts.append(
                        f"Rating: {row['rating']} | Price: ₹{row['discounted_price']}"
                    )

                result_parts.append(f"\n**Bottom {top_n} Products by Rating:**")
                bottom_products = unique_products.nsmallest(top_n, "rating")
                for i, (_, row) in enumerate(bottom_products.iterrows(), 1):
                    result_parts.append(f"{i}. {row['product_name'][:40]}...")
                    result_parts.append(
                        f"Rating: {row['rating']} | Price: ₹{row['discounted_price']}"
                    )

            return "\n".join(result_parts)

        elif operation == "discount_effectiveness":
            result_parts = ["**Discount Effectiveness Analysis**\n"]

            # Categorize discount levels
            unique_products_copy = unique_products.copy()
            unique_products_copy["discount_level"] = pd.cut(
                unique_products_copy["discount_percentage"],
                bins=[0, 35, 45, 100],
                labels=["Low (≤35%)", "Medium (35-45%)", "High (>45%)"],
            )

            effectiveness = (
                unique_products_copy.groupby("discount_level", observed=True)
                .agg(
                    {"rating": "mean", "rating_count": "mean", "product_name": "count"}
                )
                .round(2)
            )

            result_parts.append("**Impact of Discount Level on Performance:**\n")

            for level in effectiveness.index:
                data = effectiveness.loc[level]
                result_parts.append(f"**{level}**")
                result_parts.append(f"Products: {int(data['product_name'])}")
                result_parts.append(f"Avg Rating: {data['rating']:.2f}⭐")
                result_parts.append(f"Avg Review Count: {data['rating_count']:.0f}")
                result_parts.append("")

            # Correlation insight
            correlation = unique_products_copy["discount_percentage"].corr(
                unique_products_copy["rating"]
            )

            result_parts.append("**Insights:**")
            if correlation > 0.1:
                result_parts.append(
                    f"• Higher discounts show positive correlation with ratings ({correlation:.2f})"
                )
            elif correlation < -0.1:
                result_parts.append(
                    f"   • Higher discounts show negative correlation with ratings ({correlation:.2f})"
                )
                result_parts.append(
                    "   • Customers may perceive heavily discounted items as lower quality"
                )
            else:
                result_parts.append(
                    f"   • Discount level has minimal correlation with ratings ({correlation:.2f})"
                )

            return "\n".join(result_parts)

        elif operation == "summary":
            result_parts = ["**Dataset Summary Statistics**\n"]

            result_parts.append("**Overview:**")
            result_parts.append(f"Total Products: {len(unique_products)}")
            result_parts.append(f"Total Reviews: {len(df)}")
            result_parts.append(f"Categories: {unique_products['category'].nunique()}")
            result_parts.append(
                f"Sub-categories: {unique_products['sub_category'].nunique()}"
            )
            result_parts.append("")

            result_parts.append("**Rating Statistics:**")
            result_parts.append(
                f"Average Rating: {unique_products['rating'].mean():.2f}"
            )
            result_parts.append(
                f"Rating Range: {unique_products['rating'].min():.1f} - {unique_products['rating'].max():.1f}"
            )
            result_parts.append(
                f"Avg Reviews per Product: {unique_products['rating_count'].mean():.0f}"
            )
            result_parts.append("")

            result_parts.append("**Price Statistics:**")
            result_parts.append(
                f"Price Range: ₹{unique_products['discounted_price'].min():.0f} - ₹{unique_products['discounted_price'].max():.0f}"
            )
            result_parts.append(
                f"Average Price: ₹{unique_products['discounted_price'].mean():.0f}"
            )
            result_parts.append(
                f"Average Discount: {unique_products['discount_percentage'].mean():.1f}%"
            )
            result_parts.append("")

            result_parts.append("**Categories Available:**")
            for cat in unique_products["category"].unique():
                count = len(unique_products[unique_products["category"] == cat])
                result_parts.append(f"- {cat}: {count} products")

            return "\n".join(result_parts)

        else:
            return f"Unknown operation: {operation}. Available operations: category_comparison, price_analysis, rating_ranking, discount_effectiveness, summary"

    except Exception as e:
        return f"Error calculating statistics: {str(e)}"
