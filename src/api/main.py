"""
Customer Recommendation Engine — FastAPI Application
Serves product recommendations and customer segment profiles.
"""

import os
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Load models and data at startup ---

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load recommendation models
with open(os.path.join(BASE_DIR, "models", "recommendation_models.pkl"), "rb") as f:
    rec_data = pickle.load(f)

als_model = rec_data["als_model"]
user_item = rec_data["user_item"]
user_to_idx = rec_data["user_to_idx"]
idx_to_product = rec_data["idx_to_product"]
product_to_idx = rec_data["product_to_idx"]
prod_content_scaled = rec_data["prod_content_scaled"]
prod_id_to_content_idx = rec_data["prod_id_to_content_idx"]
prod_features = rec_data["prod_features"]
interactions = rec_data["interactions"]

# Load customer features
customer_features = pd.read_csv(
    os.path.join(BASE_DIR, "data", "processed", "customer_features.csv")
)

# --- FastAPI app ---

app = FastAPI(
    title="Customer Recommendation Engine",
    description="Product recommendations and customer segmentation for Olist e-commerce data.",
    version="1.0.0",
)


# --- Response models ---

class Recommendation(BaseModel):
    product_id: str
    category: str
    hybrid_score: float
    cf_score: float
    content_score: float


class RecommendationResponse(BaseModel):
    customer_id: str
    segment: str
    recommendations: list[Recommendation]


class SegmentResponse(BaseModel):
    customer_id: str
    segment: str
    recency: float
    frequency: float
    monetary: float
    avg_order_value: float
    num_products: float
    avg_review_score: float
    avg_sentiment: float


# --- Helper functions ---

def get_content_recommendations(product_id, top_n=5):
    if product_id not in prod_id_to_content_idx:
        return []
    idx = prod_id_to_content_idx[product_id]
    sim_scores = cosine_similarity(
        prod_content_scaled[idx : idx + 1], prod_content_scaled
    )[0]
    top_indices = sim_scores.argsort()[::-1][1 : top_n + 1]
    results = []
    for i in top_indices:
        results.append(
            {
                "product_id": prod_features.iloc[i]["product_id"],
                "category": prod_features.iloc[i]["product_category_name_english"]
                or "unknown",
                "similarity": float(sim_scores[i]),
            }
        )
    return results


def get_hybrid_recommendations(customer_unique_id, top_n=5):
    recommendations = {}

    if customer_unique_id in user_to_idx:
        user_idx = user_to_idx[customer_unique_id]
        rec_idx, rec_scores = als_model.recommend(
            user_idx, user_item[user_idx], N=top_n * 2, filter_already_liked_items=True
        )
        for idx, score in zip(rec_idx, rec_scores):
            pid = idx_to_product[idx]
            recommendations[pid] = {"cf_score": float(score), "content_score": 0.0}

    user_products = interactions[
        interactions["customer_unique_id"] == customer_unique_id
    ]["product_id"].unique()
    for bought_pid in user_products:
        for rec in get_content_recommendations(bought_pid, top_n=top_n):
            pid = rec["product_id"]
            if pid not in user_products:
                if pid not in recommendations:
                    recommendations[pid] = {"cf_score": 0.0, "content_score": 0.0}
                recommendations[pid]["content_score"] = max(
                    recommendations[pid]["content_score"], rec["similarity"]
                )

    for pid in recommendations:
        recommendations[pid]["hybrid_score"] = (
            0.6 * recommendations[pid]["cf_score"]
            + 0.4 * recommendations[pid]["content_score"]
        )

    sorted_recs = sorted(
        recommendations.items(), key=lambda x: x[1]["hybrid_score"], reverse=True
    )[:top_n]

    result = []
    for pid, scores in sorted_recs:
        cat = prod_features[prod_features["product_id"] == pid][
            "product_category_name_english"
        ].values
        cat = cat[0] if len(cat) > 0 else "unknown"
        result.append(
            {
                "product_id": pid,
                "category": cat,
                "hybrid_score": round(scores["hybrid_score"], 4),
                "cf_score": round(scores["cf_score"], 4),
                "content_score": round(scores["content_score"], 4),
            }
        )
    return result


# --- Endpoints ---

@app.get("/")
def root():
    return {"message": "Customer Recommendation Engine API", "docs": "/docs"}


@app.get("/recommendations/{customer_id}", response_model=RecommendationResponse)
def get_recommendations(customer_id: str, top_n: int = 5):
    """Get top-N product recommendations for a customer."""
    row = customer_features[customer_features["customer_unique_id"] == customer_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")

    segment = row.iloc[0].get("segment_name", "unknown")
    recs = get_hybrid_recommendations(customer_id, top_n=top_n)

    return RecommendationResponse(
        customer_id=customer_id,
        segment=str(segment),
        recommendations=[Recommendation(**r) for r in recs],
    )


@app.get("/segments/{customer_id}", response_model=SegmentResponse)
def get_segment(customer_id: str):
    """Get customer segment profile."""
    row = customer_features[customer_features["customer_unique_id"] == customer_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")

    r = row.iloc[0]
    return SegmentResponse(
        customer_id=customer_id,
        segment=str(r.get("segment_name", "unknown")),
        recency=float(r["recency"]),
        frequency=float(r["frequency"]),
        monetary=float(r["monetary"]),
        avg_order_value=float(r["avg_order_value"]),
        num_products=float(r["num_products"]),
        avg_review_score=float(r["avg_review_score"]),
        avg_sentiment=float(r.get("avg_sentiment", 0)),
    )


@app.get("/customers/sample")
def get_sample_customers(n: int = 5):
    """Get a sample of customer IDs (useful for testing)."""
    sample = customer_features["customer_unique_id"].sample(n).tolist()
    return {"customers": sample}
