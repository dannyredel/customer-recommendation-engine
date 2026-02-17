# Customer Recommendation Engine

An end-to-end ML system that segments e-commerce customers and generates personalized product recommendations. Built with real-world data, deployed as a live API, and orchestrated with automated pipelines.

**Live API:** [customer-rec-engine-89681103679.europe-west1.run.app](https://customer-rec-engine-89681103679.europe-west1.run.app/docs)

## Architecture

```
Raw Data (9 CSVs)
    |
    v
Feature Engineering (PySpark) --> Customer Features (96K customers)
    |                                    |
    v                                    v
NLP Sentiment Analysis          K-Means Segmentation (5 segments)
(HuggingFace BERT)                       |
    |                                    v
    +-----------+------------> Hybrid Recommendation Engine
                               (Collaborative + Content-Based)
                                         |
                                         v
                               FastAPI REST API --> GCP Cloud Run
                                         |
                                         v
                               Airflow DAG (weekly retraining)
                               GitHub Actions CI/CD
```

## Tech Stack

| Layer | Technology |
|---|---|
| Data Processing | PySpark, Pandas |
| NLP | HuggingFace Transformers, BERT (multilingual sentiment) |
| Segmentation | Scikit-learn (K-Means, StandardScaler, PCA) |
| Recommendations | Implicit (ALS), Scikit-learn (Cosine Similarity) |
| API | FastAPI, Pydantic, Uvicorn |
| Containerization | Docker, Docker Compose |
| Orchestration | Apache Airflow |
| Deployment | Google Cloud Run, Artifact Registry |
| CI/CD | GitHub Actions |
| Language | Python 3.11 |

## Dataset

[Olist Brazilian E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — 100K+ orders from 2016-2018 across 9 relational tables covering customers, orders, products, reviews, sellers, and geolocation.

## Customer Segments

| Segment | Description |
|---|---|
| Silent Satisfied | Low engagement, decent reviews, infrequent buyers |
| High-Value Buyers | High monetary value, large order sizes |
| Engaged Enthusiasts | Active reviewers, moderate spending |
| Dissatisfied / At Risk | Low review scores, delivery issues |
| Repeat Loyalists | High frequency, consistent purchasing |

## Recommendation Approach

The system uses a **hybrid approach** combining two models:

- **Collaborative Filtering (60%)** — Alternating Least Squares (ALS) on the user-item interaction matrix, identifying patterns from similar users' purchase behavior
- **Content-Based Filtering (40%)** — Cosine similarity on product features (category, dimensions, weight, photos), recommending products similar to what the customer has purchased

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/recommendations/{customer_id}?top_n=5` | Get personalized product recommendations |
| GET | `/segments/{customer_id}` | Get customer segment profile and RFM metrics |
| GET | `/customers/sample?n=5` | Get sample customer IDs for testing |

### Example Response

```json
GET /recommendations/{customer_id}

{
  "customer_id": "abc123",
  "segment": "High-Value Buyers",
  "recommendations": [
    {
      "product_id": "prod_456",
      "category": "electronics",
      "hybrid_score": 0.85,
      "cf_score": 0.92,
      "content_score": 0.74
    }
  ]
}
```

## Project Structure

```
customer-recommendation-engine/
├── .github/workflows/
│   └── ci.yml                  # GitHub Actions CI pipeline
├── dags/
│   └── ml_pipeline.py          # Airflow DAG (5-task ML pipeline)
├── data/
│   └── processed/
│       └── customer_features.csv
├── models/
│   └── recommendation_models.pkl
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_pyspark_exploration.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_nlp_sentiment.ipynb
│   ├── 05_segmentation.ipynb
│   └── 06_recommendations.ipynb
├── src/
│   └── api/
│       └── main.py             # FastAPI application
├── tests/
│   └── test_api.py
├── Dockerfile                  # API container
├── Dockerfile.airflow          # Airflow container with ML deps
├── docker-compose.yml          # Airflow orchestration stack
└── requirements.txt
```

## Getting Started

### Prerequisites

- Python 3.11
- Docker & Docker Compose
- Git LFS (for model files)

### Run Locally

```bash
# Clone the repository
git clone https://github.com/dannyredel/customer-recommendation-engine.git
cd customer-recommendation-engine

# Install dependencies
pip install -r requirements.txt

# Start the API
uvicorn src.api.main:app --reload
```

### Run with Docker

```bash
# Build and run the API container
docker build -t customer-rec-engine .
docker run -p 8000:8000 customer-rec-engine
```

### Run Airflow Pipeline

```bash
# Start the full Airflow stack (PostgreSQL + Webserver + Scheduler)
docker-compose up -d

# Access Airflow UI at http://localhost:8080 (admin/admin)
```

### Run Tests

```bash
pip install pytest
pytest tests/ -v
```

## Pipeline (Airflow DAG)

The ML pipeline runs weekly and executes 5 tasks:

```
ingest_data --> process_features --> train_segmentation -----> pipeline_complete
                                 --> train_recommendations -/
```

1. **ingest_data** — Validate raw CSV files
2. **process_features** — Build customer feature table (RFM + extras)
3. **train_segmentation** — K-Means clustering (5 segments)
4. **train_recommendations** — Train ALS model + content-based features
5. **pipeline_complete** — Log completion

Segmentation and recommendations run in parallel for efficiency.

## CI/CD

Every push to `main` triggers the GitHub Actions pipeline:

1. **test** — Installs dependencies, runs pytest
2. **docker-build** — Verifies the Docker image builds successfully
