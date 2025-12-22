"""
Product Search Tool - Information Retrieval Tool using LangChain.

This tool searches and retrieves products from the Amazon sales dataset
based on various criteria like category, price range, ratings, or keywords.
"""

import pandas as pd
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from src.config import DATASET_PATH


class ProductSearchInput(BaseModel):
    """Input schema for the product search tool."""

    category: Optional[str] = Field(
        default=None,
        description="Product category to filter by (e.g., 'Electronics', 'Clothing', 'Home & Kitchen')",
    )
    sub_category: Optional[str] = Field(
        default=None,
        description="Sub-category to filter by (e.g., 'Headphones', 'Wearables', 'Footwear')",
    )
    min_price: Optional[float] = Field(
        default=None, description="Minimum price filter (discounted price)"
    )
    max_price: Optional[float] = Field(
        default=None, description="Maximum price filter (discounted price)"
    )
    min_rating: Optional[float] = Field(
        default=None, description="Minimum rating filter (1.0 to 5.0)"
    )
    keyword: Optional[str] = Field(
        default=None, description="Keyword to search in product name or description"
    )
    limit: int = Field(default=10, description="Maximum number of products to return")


def _load_dataset() -> pd.DataFrame:
    """Load the Amazon sales dataset."""
    try:
        df = pd.read_csv(DATASET_PATH)
        # Clean discount percentage column
        df["discount_percentage"] = (
            df["discount_percentage"].str.replace("%", "").astype(float)
        )
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")


@tool(args_schema=ProductSearchInput)
def search_products(
    category: Optional[str] = None,
    sub_category: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    min_rating: Optional[float] = None,
    keyword: Optional[str] = None,
    limit: int = 10,
) -> str:
    """
    Search and retrieve products from the Amazon sales dataset.

    Use this tool to find products based on category, price range, ratings,
    or keywords. Returns product details including name, price, discount,
    rating, and descriptions.

    Examples:
    - Find all Electronics products
    - Find products under ₹2000 with rating above 4.0
    - Search for 'wireless' products in Electronics category
    """
    try:
        df = _load_dataset()

        # Apply filters
        if category:
            df = df[df["category"].str.lower() == category.lower()]

        if sub_category:
            df = df[df["sub_category"].str.lower() == sub_category.lower()]

        if min_price is not None:
            df = df[df["discounted_price"] >= min_price]

        if max_price is not None:
            df = df[df["discounted_price"] <= max_price]

        if min_rating is not None:
            df = df[df["rating"] >= min_rating]

        if keyword:
            keyword_lower = keyword.lower()
            df = df[
                df["product_name"].str.lower().str.contains(keyword_lower, na=False)
                | df["about_product"].str.lower().str.contains(keyword_lower, na=False)
            ]

        if df.empty:
            return "No products found matching the specified criteria."

        # Get unique products (since reviews create duplicates)
        unique_products = df.drop_duplicates(subset=["product_name"]).head(limit)

        # Format results
        results = []
        for _, row in unique_products.iterrows():
            product_info = f"""
     **{row["product_name"]}**
   - Category: {row["category"]} > {row["sub_category"]}
   - Original Price: ₹{row["actual_price"]}
   - Discounted Price: ₹{row["discounted_price"]}
   - Discount: {row["discount_percentage"]}%
   - Rating: {row["rating"]}⭐ ({row["rating_count"]} reviews)
   - Description: {row["about_product"][:150]}...
"""
            results.append(product_info)

        header = f"Found {len(unique_products)} product(s) matching your criteria:\n"
        return header + "\n".join(results)

    except Exception as e:
        return f"Error searching products: {str(e)}"
