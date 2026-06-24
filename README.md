# Urban Congestion Prediction API

A production-grade async REST API that ingests real-time vehicle GPS telemetry, predicts traffic congestion using a pre-trained Scikit-Learn model, and persists readings for continuous retraining.

---

## Architecture

```
[Vehicle / Simulation Client]
        │
        │  POST /api/v1/traffic/predict  { lat, lng, timestamp }
        ▼
[FastAPI — auth (X-API-Key) + rate limit (slowapi)]
        │
        ├─► PostGIS ST_Contains → zone lookup
        │
        ├─► Feature extraction: (hour, day_of_week, zone_id)
        │
        ├─► GradientBoostingClassifier.predict_proba()   (loaded once at startup)
        │
        ├─► Response: { congestion_probability, congestion_label, zone_id }
        │
        └─► BackgroundTask: async INSERT into traffic_history (partitioned)
```

Key design points:
- **Model loaded once** at startup via FastAPI lifespan; stored in `app.state.predictor`
- **Graceful degradation**: if the model file is missing, the server still boots and `/predict` returns `503`
- **Partitioned writes**: `traffic_history` uses native Postgres `PARTITION BY RANGE (reported_at)` — old months can be dropped without locking the parent table
- **Zero blocking DB calls**: all ORM work goes through `asyncpg` + SQLAlchemy 2.0 async sessions

---

## Partitioning Strategy: Native Postgres vs TimescaleDB

| Criterion | Native Postgres RANGE | TimescaleDB Hypertable |
|---|---|---|
| Extensions needed | PostGIS only | PostGIS + TimescaleDB |
| Docker image | `postgis/postgis` (official) | custom or `timescale/timescaledb-ha` |
| Auto-partition creation | ❌ manual migration per month | ✅ automatic |
| Partition dropping | `DROP TABLE traffic_history_YYYY_MM` — instant metadata op | `drop_chunks()` — same speed |
| Compression | ❌ manual pg_partman needed | ✅ built-in columnar compression |
| Retention policy | ❌ manual cron | ✅ `add_retention_policy()` |
| Query planning | Standard Postgres planner | Transparent — planner-aware |
| Managed cloud support | All Postgres providers | Timescale Cloud or self-host |

**Decision: native Postgres range partitioning.**

Rationale:
1. Adding TimescaleDB would require a custom Docker image or a paid managed service — PostGIS is already the only extension dependency.
2. At this data volume (millions of rows/month) manual partitioning is maintainable with a simple monthly migration.
3. TimescaleDB's auto-compression and retention policies are compelling but can be layered on later by pointing the same schema at a TimescaleDB instance — the table structure is compatible.
4. Accepted trade-off: a cron job or Alembic migration must create next month's partition before month rollover.

---

## Setup

### Prerequisites
- Docker 24+ and Docker Compose v2
- Python 3.11+ (for local dev / training)

### 1. Clone and configure

```bash
git clone <repo>
cd urban-congestion-api
cp .env.example .env
```

Generate an API key and its hash:

```bash
python -c "
import hashlib, secrets
k = secrets.token_hex(32)
print('API_KEY (put in X-API-Key header):', k)
print('API_KEY_HASH (put in .env):', hashlib.sha256(k.encode()).hexdigest())
"
```

Edit `.env` with your `API_KEY_HASH`.

### 2. Train the model

```bash
pip install scikit-learn joblib numpy
python -m ml_pipeline.train
```

This writes `ml_pipeline/models/congestion_model.pkl` and `model_meta.json`.

### 3. Start the stack

```bash
docker compose up --build
```

Services:
| Service | URL |
|---|---|
| FastAPI | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| Prometheus metrics | http://localhost:8000/metrics |
| Prometheus server | http://localhost:9090 |

### 4. Run migrations

```bash
# From inside the api container, or with DATABASE_URL set locally:
alembic upgrade head
```

---

## API Usage

### Health check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "db_reachable": true,
  "version": "1.0.0"
}
```

### Predict congestion

```bash
curl -X POST http://localhost:8000/api/v1/traffic/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "lat": 40.7128,
    "lng": -74.0060,
    "timestamp": "2024-08-17T08:30:00Z"
  }'
```

```json
{
  "congestion_probability": 0.872,
  "congestion_label": "high",
  "zone_id": 1,
  "zone_name": "downtown",
  "model_version": "a3f9b1c2d4e5...",
  "processed_at": "2024-08-17T08:30:01.234Z"
}
```

**Labels:** `low` (< 0.4) · `medium` (0.4–0.7) · `high` (≥ 0.7)

### List zones

```bash
curl http://localhost:8000/api/v1/traffic/zones \
  -H "X-API-Key: YOUR_API_KEY"
```

---

## Running Tests

```bash
pip install -r requirements.txt
pytest
```

Tests mock the database session — no live Postgres required for the unit test suite.

For integration tests against a real PostGIS DB:

```bash
docker compose -f docker-compose.test.yml up -d
DATABASE_URL=postgresql+asyncpg://test_user:test_secret@localhost:5433/test_congestion pytest
```

---

## Load Testing

```bash
pip install locust
locust -f locustfile.py --host http://localhost:8000 --users 200 --spawn-rate 20
```

Open http://localhost:8089 for the Locust web UI.

Headless run (60-second burst):

```bash
locust -f locustfile.py --host http://localhost:8000 \
       --users 200 --spawn-rate 20 --run-time 60s --headless
```

---

## Database Partitions

Monthly partitions are pre-created in the initial migration for a rolling 18-month window. To add a new month:

```sql
CREATE TABLE traffic_history_2025_07
  PARTITION OF traffic_history
  FOR VALUES FROM ('2025-07-01') TO ('2025-08-01');
```

To drop an old partition (instant, no lock on parent):

```sql
DROP TABLE traffic_history_2024_01;
```

---

## Project Structure

```
urban-congestion-api/
├── src/
│   ├── main.py                  # App init, lifespan (model load + cleanup)
│   ├── api/
│   │   ├── dependencies.py      # Auth, rate limit, DB session
│   │   └── v1/traffic.py        # Predict + zones endpoints
│   ├── core/config.py           # pydantic-settings typed config
│   ├── db/{base,session}.py     # Async engine + session factory
│   ├── models/{zone,telemetry}  # SQLAlchemy ORM models
│   ├── schemas/traffic.py       # Pydantic request/response schemas
│   └── services/predictor.py   # Model load, feature extraction, inference
├── ml_pipeline/
│   ├── train.py                 # Synthetic data + GBClassifier training
│   └── models/                 # congestion_model.pkl (gitignored)
├── alembic/                    # Migrations (PostGIS, partitioned schema)
├── tests/                      # pytest suite (unit + API contract)
├── locustfile.py               # Load test
├── docker-compose.yml
├── Dockerfile
└── .github/workflows/ci.yml    # Lint + test + Docker build on push
```
