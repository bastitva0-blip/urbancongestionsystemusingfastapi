"""
Locust load test — simulates concurrent vehicle GPS clients.

Run:
    locust -f locustfile.py --host http://localhost:8000 --users 200 --spawn-rate 20

Or headless:
    locust -f locustfile.py --host http://localhost:8000 \
           --users 200 --spawn-rate 20 --run-time 60s --headless
"""

import random
from datetime import datetime, timezone

from locust import HttpUser, between, task

# NYC bounding box for synthetic coords
LAT_MIN, LAT_MAX = 40.68, 40.78
LNG_MIN, LNG_MAX = -74.05, -73.93

# Pre-generate an API key — replace with real key in production
API_KEY = "dev_locust_test_key"


class VehicleClient(HttpUser):
    """Simulates a vehicle sending GPS pings to the prediction endpoint."""

    wait_time = between(0.5, 2.0)  # seconds between requests per user

    def on_start(self) -> None:
        self.headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    @task(10)
    def predict_congestion(self) -> None:
        payload = {
            "lat": round(random.uniform(LAT_MIN, LAT_MAX), 6),
            "lng": round(random.uniform(LNG_MIN, LNG_MAX), 6),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with self.client.post(
            "/api/v1/traffic/predict",
            json=payload,
            headers=self.headers,
            catch_response=True,
            name="/predict",
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            elif resp.status_code == 503:
                resp.failure("Model not available")
            elif resp.status_code == 429:
                resp.failure("Rate limited")
            else:
                resp.failure(f"Unexpected {resp.status_code}")

    @task(1)
    def health_check(self) -> None:
        self.client.get("/health", name="/health")

    @task(1)
    def metrics(self) -> None:
        self.client.get("/metrics", name="/metrics")
