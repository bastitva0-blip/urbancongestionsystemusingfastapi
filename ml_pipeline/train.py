"""
Train a congestion prediction model on synthetic GPS telemetry data.

Features:
  - hour         (0-23): rush hour at 8 and 17 drives congestion
  - day_of_week  (0-6):  weekdays more congested than weekends
  - zone_id      (int):  proxy for spatial density

Run:
    python -m ml_pipeline.train
"""

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

OUTPUT_DIR = Path(__file__).parent / "models"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_PATH = OUTPUT_DIR / "congestion_model.pkl"


# ── Synthetic data generation ─────────────────────────────────────────────────

def _congestion_probability(hour: int, dow: int, zone_id: int) -> float:
    """Rule-based ground truth used to label synthetic rows."""
    prob = 0.1
    # Rush hours: 7-9 AM and 16-18 PM
    if hour in {7, 8, 9, 16, 17, 18}:
        prob += 0.45
    # Slightly elevated mid-day
    elif 11 <= hour <= 14:
        prob += 0.15
    # Weekday boost
    if dow < 5:
        prob += 0.1
    # Denser zones (low zone_id = downtown-like)
    if zone_id <= 2:
        prob += 0.15
    return min(prob, 1.0)


def generate_dataset(n_samples: int = 50_000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    hours = rng.integers(0, 24, size=n_samples)
    dows = rng.integers(0, 7, size=n_samples)
    zone_ids = rng.integers(1, 10, size=n_samples)

    probs = np.array([
        _congestion_probability(h, d, z)
        for h, d, z in zip(hours, dows, zone_ids)
    ])
    # Add noise
    probs = np.clip(probs + rng.normal(0, 0.05, size=n_samples), 0, 1)
    labels = (probs > 0.5).astype(int)

    X = np.column_stack([hours, dows, zone_ids])
    return X, labels


# ── Training ──────────────────────────────────────────────────────────────────

def train() -> None:
    print("Generating synthetic dataset…")
    X, y = generate_dataset(50_000)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")
    print(f"Class balance — congested: {y.mean():.1%}")

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    print("Training GradientBoostingClassifier…")
    model.fit(X_train, y_train)

    # ── Evaluation ────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    print("\n── Evaluation ─────────────────────────────")
    print(classification_report(y_test, y_pred, target_names=["clear", "congested"]))
    print(f"ROC-AUC: {auc:.4f}")

    feature_names = ["hour", "day_of_week", "zone_id"]
    importances = dict(zip(feature_names, model.feature_importances_))
    print("\nFeature importances:")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"  {feat:15s}: {imp:.4f}")

    # ── Persist ───────────────────────────────────────────────────────────────
    joblib.dump(model, MODEL_PATH)
    model_hash = hashlib.md5(MODEL_PATH.read_bytes()).hexdigest()

    meta = {
        "model_type": type(model).__name__,
        "n_estimators": model.n_estimators,
        "roc_auc": round(auc, 4),
        "feature_importances": {k: round(v, 4) for k, v in importances.items()},
        "md5": model_hash,
        "train_samples": len(X_train),
    }
    (OUTPUT_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nModel saved → {MODEL_PATH}")
    print(f"MD5: {model_hash}")
    print("Metadata → ml_pipeline/models/model_meta.json")


if __name__ == "__main__":
    train()
    sys.exit(0)
