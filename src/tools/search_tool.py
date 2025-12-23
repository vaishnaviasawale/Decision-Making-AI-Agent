"""
Product Search Tool - Information Retrieval Tool using LangChain.

This tool searches and retrieves products from the Amazon sales dataset
based on various criteria like category, price range, ratings, or keywords.
"""

import pandas as pd
import re
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from src.config import DATASET_PATH


class ProductSearchInput(BaseModel):
    """Input schema for the product search tool."""

    category: Optional[str] = Field(
        default=None,
        description="Product category to filter by (partial, case-insensitive match)",
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
    max_rating: Optional[float] = Field(
        default=None, description="Maximum rating filter (1.0 to 5.0)"
    )
    keyword: Optional[str] = Field(
        default=None, description="Keyword to search in product name or description"
    )
    limit: Optional[int] = Field(
        default=None, description="Maximum number of products to return (optional)"
    )


def _load_dataset() -> pd.DataFrame:
    """Load the Amazon sales dataset."""
    try:
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

        # Normalized category for robust partial matching (lowercase, alphanumeric)
        def _norm(text: str) -> str:
            return re.sub(r"[^a-z0-9]", "", str(text).lower())

        df["category_norm"] = df["category"].apply(_norm)
        df["category_segments_norm"] = df["category"].apply(
            lambda x: [_norm(seg) for seg in str(x).split("|") if _norm(seg)]
        )

        return df
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}") from exc


@tool(args_schema=ProductSearchInput)
def search_products(
    category: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    min_rating: Optional[float] = None,
    max_rating: Optional[float] = None,
    keyword: Optional[str] = None,
    limit: Optional[int] = None,
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

        # Guard: avoid returning the entire dataset when no meaningful filter is provided.
        # Exception: if `limit` is provided, treat it as a request for "top N" products overall.
        if (
            not category
            and not keyword
            and min_price is None
            and max_price is None
            and min_rating is None
            and max_rating is None
            and limit is None
        ):
            return "Please specify a category or keyword to narrow the search."

        # Apply filters
        if category:
            # Allow multiple category hints separated by comma/and/&/|
            raw_parts = (
                category.replace("&", " and ")
                .replace("|", ",")
                .replace(" and ", ",")
                .split(",")
            )
            parts = [p.strip() for p in raw_parts if p.strip()]
            if parts:
                mask = False
                for part in parts:
                    norm_part = re.sub(r"[^a-z0-9]", "", part.lower())
                    # Match against full normalized category and any normalized segments
                    mask = mask | df["category_norm"].str.contains(
                        norm_part, case=False, na=False
                    )
                    mask = mask | df["category_segments_norm"].apply(
                        lambda segs, np=norm_part: any(np in seg for seg in segs)
                    )
                if mask.any():
                    df = df[mask]
                else:
                    return "No products found matching the specified category filters."

        if min_price is not None:
            df = df[df["discounted_price"] >= min_price]

        if max_price is not None:
            df = df[df["discounted_price"] <= max_price]

        if min_rating is not None:
            df = df[df["rating"] >= min_rating]

        if max_rating is not None:
            df = df[df["rating"] <= max_rating]

        if keyword:
            keyword_lower = keyword.lower()
            df = df[
                df["product_name"].str.lower().str.contains(keyword_lower, na=False)
                | df["about_product"].str.lower().str.contains(keyword_lower, na=False)
            ]

        if df.empty:
            return "No products found matching the specified criteria."

        # Get unique products (since reviews create duplicates), sorted by rating then review count
        unique_products = df.drop_duplicates(subset=["product_name"]).sort_values(
            by=["rating", "rating_count"], ascending=False
        )
        if limit is not None:
            unique_products = unique_products.head(limit)

        products_payload = unique_products.to_dict(orient="records")

        # Format results
        results = []
        for _, row in unique_products.iterrows():
            product_info = f"""
  **{row["product_name"]}**
   - Category: {row["category"]}
   - Original Price: ₹{row["actual_price"]:.0f}
   - Discounted Price: ₹{row["discounted_price"]:.0f}
   - Discount: {row["discount_percentage"]}%
   - Rating: {row["rating"]} ({row["rating_count"]} reviews)
   - Description: {row["about_product"][:150]}...
"""
            results.append(product_info)

        header = f"Found {len(unique_products)} product(s) matching your criteria:\n"
        return {"summary": header + "\n".join(results), "products": products_payload}

    except Exception as e:
        return f"Error searching products: {str(e)}"
