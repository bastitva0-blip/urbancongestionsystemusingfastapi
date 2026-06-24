# Urban Congestion Prediction API

A production-grade, asynchronous REST API that ingests real-time vehicle GPS telemetry, predicts urban traffic congestion via a pre-trained Scikit-Learn model, and asynchronously persists readings for continuous retraining loops.

---

## 🏗️ System Architecture

The service optimizes throughput by decoupling feature extraction and ML inference from the database persistence layer using FastAPI's `BackgroundTask`.

```
[Vehicle / Simulation Client]
        │
        │  POST /api/v1/traffic/predict  { lat, lng, timestamp }
        ▼
[FastAPI — auth (X-API-Key) + rate limit (slowapi)]
        │
        ├─► PostGIS ST_Contains → Spatial zone lookup
        │
        ├─► Feature extraction: (hour, day_of_week, zone_id)
        │
        ├─► GradientBoostingClassifier.predict_proba() (Cached in memory)
        │
        ├─► Response: { congestion_probability, congestion_label, zone_id }
        │
        └─► BackgroundTask: async INSERT into traffic_history (partitioned)

```

### Key Design Patterns

* **Singleton Model Lifecycle:** The Scikit-Learn model is loaded exactly once at startup via the FastAPI `lifespan` handler and stored globally in `app.state.predictor`.
* **Graceful Degradation:** If the serialized model (`.pkl`) is corrupted or missing, the server still boots successfully; the `/predict` endpoint safely degrades to a `503 Service Unavailable` response.
* **Partitioned Writes:** The `traffic_history` table uses native Postgres `PARTITION BY RANGE (reported_at)`. Historical months can be instantly unlinked or dropped without table-level locking.
* **Non-Blocking I/O:** Every database operation leverages fully asynchronous execution paths via `asyncpg` and SQLAlchemy 2.0 async sessions.

---

## 📊 Partitioning Strategy: Native Postgres vs. TimescaleDB

| Criterion | Native Postgres RANGE | TimescaleDB Hypertable |
| --- | --- | --- |
| **Extensions Required** | PostGIS only | PostGIS + TimescaleDB |
| **Docker Base Image** | `postgis/postgis` (Official) | Custom or `timescale/timescaledb-ha` |
| **Auto-Partition Creation** | ❌ Manual/Scripted migration per month | ✅ Automatic chunks |
| **Partition Dropping** | `DROP TABLE ...` (Instant metadata op) | `drop_chunks()` (Instant metadata op) |
| **Compression** | ❌ Manual `pg_partman` needed | ✅ Built-in columnar compression |
| **Retention Policy** | ❌ Manual cron / worker script | ✅ Built-in `add_retention_policy()` |
| **Query Planning** | Standard Postgres planner | Transparent, chunk-aware planner |
| **Cloud Managed Support** | Ubiquitous (RDS, Cloud SQL, Supabase) | Timescale Cloud or self-hosted |

### Rationale for Native Postgres Range Partitioning

1. **Minimized Dependency Footprint:** Adding TimescaleDB introduces custom image maintenance or specialized cloud hosting costs. PostGIS is already required for spatial lookups (`ST_Contains`).
2. **Scale Alignment:** For mid-tier volumes (millions of rows/month), standard range partitions are highly performant and easily managed via monthly migration scripts.
3. **Future-Proof Schema:** The schema layout remains 100% compatible with TimescaleDB. If volume scales drastically, tables can be migrated to hypertables transparently down the line.

---

## 🚀 Getting Started

### Prerequisites

* Docker 24+ and Docker Compose v2
* Python 3.11+ (for local exploration/training)

### 1. Environment Setup

Clone the repository and instantiate your configuration file:

```bash
git clone <repo-url>
cd urban-congestion-api
cp .env.example .env

```

Generate a secure API key and its corresponding SHA-256 hash for your `.env`:

```bash
python -c "
import hashlib, secrets
k = secrets.token_hex(32)
print('API_KEY (Put in X-API-Key header):  ', k)
print('API_KEY_HASH (Put in your .env):   ', hashlib.sha256(k.encode()).hexdigest())
"

```

Paste the generated `API_KEY_HASH` value into your `.env` file.

### 2. Train the Predictive Model

Generate synthetic spatial data and train the initial `GradientBoostingClassifier`:

```bash
pip install scikit-learn joblib numpy
python -m ml_pipeline.train

```

*This outputs `ml_pipeline/models/congestion_model.pkl` and `model_meta.json`.*

### 3. Spin Up the Infrastructure

Build and run the primary containers:

```bash
docker compose up --build -d

```

| Component | Target URL |
| --- | --- |
| **FastAPI Core Engine** | [http://localhost:8000](http://localhost:8000) |
| **Interactive OpenAPI/Swagger** | [http://localhost:8000/docs](http://localhost:8000/docs) |
| **Prometheus Exporter** | [http://localhost:8000/metrics](http://localhost:8000/metrics) |
| **Prometheus Dashboard** | [http://localhost:9090](http://localhost:9090) |

### 4. Execute Spatial Database Migrations

Initialize your PostGIS geometry extensions, seed static zones, and create initial table partitions:

```bash
docker compose exec api alembic upgrade head

```

---

## 📡 Core API Specification

### Health Check

Verify application readiness and dependency connectivity.

* **Request:** `GET /health`
* **Response (`200 OK`):**

```json
{
  "status": "ok",
  "model_loaded": true,
  "db_reachable": true,
  "version": "1.0.0"
}

```

### Predict Congestion

Ingests coordinate metrics and calculates current traffic probabilities.

* **Request:** `POST /api/v1/traffic/predict`
* **Headers:** `X-API-Key: <YOUR_API_KEY>`

```bash
curl -X POST http://localhost:8000/api/v1/traffic/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "lat": 40.7128,
    "lng": -74.0060,
    "timestamp": "2026-06-24T23:55:00Z"
  }'

```

* **Response (`200 OK`):**

```json
{
  "congestion_probability": 0.872,
  "congestion_label": "high",
  "zone_id": 1,
  "zone_name": "downtown",
  "model_version": "a3f9b1c2d4e5",
  "processed_at": "2026-06-24T23:55:01.234Z"
}

```

> 💡 **Congestion Level Thresholds:** `low` ($< 0.4$) · `medium` ($0.4 \le p < 0.7$) · `high` ($\ge 0.7$)

---

## 🧪 Testing Suite

### Unit Tests

Executed via mocked database sessions for instant continuous integration feedback:

```bash
pip install -r requirements.txt
pytest tests/unit

```

### Integration Tests

Runs contract and database constraints against a live, ephemeral PostGIS instance:

```bash
docker compose -f docker-compose.test.yml up -d
DATABASE_URL=postgresql+asyncpg://test_user:test_secret@localhost:5433/test_congestion pytest tests/integration
docker compose -f docker-compose.test.yml down

```

### Load Testing

Evaluate concurrency limits and endpoint saturation points via Locust:

```bash
pip install locust
locust -f locustfile.py --host http://localhost:8000

```

Navigate to [http://localhost:8089](http://localhost:8089) for real-time graphs. Alternatively, execute a headless 60-second smoke test:

```bash
locust -f locustfile.py --host http://localhost:8000 --users 200 --spawn-rate 20 --run-time 60s --headless

```

---

## 🛠️ Operations: Database Partition Management

Monthly partitions are pre-provisioned via the baseline Alembic migrations.

### Manual Partition Additions

To attach future timeline ranges before rollover occurrences:

```sql
CREATE TABLE traffic_history_2026_07
  PARTITION OF traffic_history
  FOR VALUES FROM ('2026-07-01') TO ('2026-08-01');

```

### Pruning Historical Records

To purge ancient metrics instantly without table locks or transaction-log bloating:

```sql
DROP TABLE traffic_history_2025_01;

```

---

## 📂 Project Anatomy

```
urban-congestion-api/
├── src/
│   ├── main.py                  # Entrypoint: Configures lifespan hooks & global app state
│   ├── api/
│   │   ├── dependencies.py      # Middleware: API key evaluation, rate limits, DB sessions
│   │   └── v1/traffic.py        # Controllers: Inference and spatial retrieval routes
│   ├── core/config.py           # Infrastructure: Pydantic-settings environment validation
│   ├── db/                      # Persistence: Async connection pools & factories
│   ├── models/                  # Declarative Layer: PostGIS/SQLAlchemy tables
│   ├── schemas/traffic.py       # Validation: Strict request/response Pydantic schemas
│   └── services/predictor.py    # ML Core: Thread-pooled inferences & feature extraction
├── ml_pipeline/
│   ├── train.py                 # Routine: Pipeline generation & synthetic training runs
│   └── models/                  # Artifact Store: Serialized .pkl files (Git ignored)
├── alembic/                     # Migrations: DB schema iterations & initial partitions
├── tests/                       # Validation Suite: Unit and integration paradigms
└── .github/workflows/ci.yml     # Automation: Quality control linting & building pipelines

```
