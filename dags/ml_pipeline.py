"""
Customer Recommendation Engine — Airflow DAG
Orchestrates the full ML pipeline: data processing → NLP → segmentation → recommendations.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

# --- Default arguments for all tasks ---
default_args = {
    "owner": "daniel",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


# --- Task functions ---

def ingest_data(**kwargs):
    """Step 1: Load and validate raw data files."""
    import os
    data_dir = "/opt/airflow/data/raw"
    files = os.listdir(data_dir) if os.path.exists(data_dir) else []
    csv_files = [f for f in files if f.endswith(".csv")]
    print(f"Found {len(csv_files)} CSV files in {data_dir}")
    if len(csv_files) < 8:
        raise ValueError(f"Expected 8+ CSV files, found {len(csv_files)}")
    print("Data ingestion complete.")


def process_features(**kwargs):
    """Step 2: Build customer feature table from raw data."""
    import pandas as pd
    import numpy as np

    raw = "/opt/airflow/data/raw/"
    customers = pd.read_csv(raw + "olist_customers_dataset.csv")
    orders = pd.read_csv(raw + "olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"])
    items = pd.read_csv(raw + "olist_order_items_dataset.csv")
    reviews = pd.read_csv(raw + "olist_order_reviews_dataset.csv")
    products = pd.read_csv(raw + "olist_products_dataset.csv")
    categories = pd.read_csv(raw + "product_category_name_translation.csv")

    # Join tables
    df = (
        orders
        .merge(customers, on="customer_id", how="left")
        .merge(items, on="order_id", how="left")
        .merge(reviews, on="order_id", how="left")
        .merge(products, on="product_id", how="left")
        .merge(categories, on="product_category_name", how="left")
    )
    df["review_score"] = pd.to_numeric(df["review_score"], errors="coerce")

    reference_date = df["order_purchase_timestamp"].max()

    customer_features = (
        df.groupby("customer_unique_id")
        .agg(
            recency=("order_purchase_timestamp", lambda x: (reference_date - x.max()).days),
            frequency=("order_id", "nunique"),
            monetary=("price", "sum"),
            num_products=("product_id", "nunique"),
            avg_review_score=("review_score", "mean"),
            delivery_diff=("order_delivered_customer_date", lambda x: (
                (df.loc[x.index, "order_delivered_customer_date"] - df.loc[x.index, "order_estimated_delivery_date"]).dt.days.mean()
            )),
        )
        .reset_index()
    )
    customer_features["avg_order_value"] = customer_features["monetary"] / customer_features["frequency"]
    customer_features = customer_features.fillna(0)

    import os
    os.makedirs("/opt/airflow/data/processed", exist_ok=True)
    customer_features.to_csv("/opt/airflow/data/processed/customer_features.csv", index=False)
    print(f"Saved {len(customer_features):,} customer features.")


def train_segmentation(**kwargs):
    """Step 3: Run K-means segmentation on customer features."""
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    df = pd.read_csv("/opt/airflow/data/processed/customer_features.csv")

    feature_cols = [c for c in df.columns if c != "customer_unique_id"]
    X = df[feature_cols].fillna(0)
    X_scaled = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=5, random_state=42, n_init=10)
    df["segment"] = km.fit_predict(X_scaled)

    segment_names = {
        0: "Silent Satisfied",
        1: "High-Value Buyers",
        2: "Engaged Enthusiasts",
        3: "Dissatisfied / At Risk",
        4: "Repeat Loyalists",
    }
    df["segment_name"] = df["segment"].map(segment_names)

    df.to_csv("/opt/airflow/data/processed/customer_features.csv", index=False)
    print(f"Segmentation complete. {len(df):,} customers assigned to {df['segment'].nunique()} segments.")


def train_recommendations(**kwargs):
    """Step 4: Train recommendation models and save."""
    import pandas as pd
    import numpy as np
    import pickle
    from scipy import sparse
    from implicit.als import AlternatingLeastSquares
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    raw = "/opt/airflow/data/raw/"
    orders = pd.read_csv(raw + "olist_orders_dataset.csv")
    items = pd.read_csv(raw + "olist_order_items_dataset.csv")
    products = pd.read_csv(raw + "olist_products_dataset.csv")
    customers = pd.read_csv(raw + "olist_customers_dataset.csv")
    categories = pd.read_csv(raw + "product_category_name_translation.csv")

    interactions = (
        items[["order_id", "product_id", "price"]]
        .merge(orders[["order_id", "customer_id"]], on="order_id")
        .merge(customers[["customer_id", "customer_unique_id"]], on="customer_id")
    )

    user_ids = interactions["customer_unique_id"].unique()
    product_ids = interactions["product_id"].unique()
    user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    product_to_idx = {pid: i for i, pid in enumerate(product_ids)}
    idx_to_product = {i: pid for pid, i in product_to_idx.items()}
    idx_to_user = {i: uid for uid, i in user_to_idx.items()}

    rows = interactions["customer_unique_id"].map(user_to_idx).values
    cols = interactions["product_id"].map(product_to_idx).values
    user_item = sparse.csr_matrix((np.ones(len(interactions)), (rows, cols)), shape=(len(user_ids), len(product_ids)))

    als_model = AlternatingLeastSquares(factors=50, regularization=0.1, iterations=20, random_state=42)
    als_model.fit(user_item)

    prod_features = products.merge(categories, on="product_category_name", how="left")
    le = LabelEncoder()
    prod_features["category_encoded"] = le.fit_transform(prod_features["product_category_name_english"].fillna("unknown"))
    content_cols = ["category_encoded", "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm", "product_photos_qty"]
    prod_content_scaled = StandardScaler().fit_transform(prod_features[content_cols].fillna(0))
    prod_id_to_content_idx = {pid: i for i, pid in enumerate(prod_features["product_id"].values)}

    import os
    os.makedirs("/opt/airflow/models", exist_ok=True)
    rec_data = {
        "als_model": als_model,
        "user_item": user_item,
        "user_to_idx": user_to_idx,
        "idx_to_product": idx_to_product,
        "idx_to_user": idx_to_user,
        "product_to_idx": product_to_idx,
        "prod_content_scaled": prod_content_scaled,
        "prod_id_to_content_idx": prod_id_to_content_idx,
        "prod_features": prod_features[["product_id", "product_category_name_english"]],
        "interactions": interactions[["customer_unique_id", "product_id"]],
    }
    with open("/opt/airflow/models/recommendation_models.pkl", "wb") as f:
        pickle.dump(rec_data, f)
    print("Recommendation models saved.")


def pipeline_complete(**kwargs):
    """Final step: log completion."""
    print(f"Pipeline completed successfully at {datetime.now()}")


# --- DAG definition ---

with DAG(
    dag_id="customer_recommendation_pipeline",
    default_args=default_args,
    description="End-to-end ML pipeline: data processing, segmentation, and recommendations",
    schedule="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "recommendations"],
) as dag:

    t1 = PythonOperator(task_id="ingest_data", python_callable=ingest_data)
    t2 = PythonOperator(task_id="process_features", python_callable=process_features)
    t3 = PythonOperator(task_id="train_segmentation", python_callable=train_segmentation)
    t4 = PythonOperator(task_id="train_recommendations", python_callable=train_recommendations)
    t5 = PythonOperator(task_id="pipeline_complete", python_callable=pipeline_complete)

    # Task dependencies: each step depends on the previous one
    t1 >> t2 >> [t3, t4] >> t5
