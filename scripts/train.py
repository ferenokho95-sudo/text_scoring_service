import argparse
import csv
import hashlib
import json
import os
import platform
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from app.model import Scorer


def train(data_path, output_dir, tracking_uri=None, track=True):
    data_path, output_dir = Path(data_path), Path(output_dir)
    with data_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"text", "label"}.issubset(rows[0]):
        raise ValueError("Expected a CSV with text,label columns")
    texts = [row["text"].strip() for row in rows]
    labels = [row["label"].strip() for row in rows]
    if any(not text for text in texts) or len({t.casefold() for t in texts}) != len(texts):
        raise ValueError("Empty or duplicate texts are not allowed")
    if set(labels) != {"positive", "negative"} or min(labels.count(label) for label in set(labels)) < 10:
        raise ValueError("Need positive and negative classes with at least 10 rows each")
    x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=0.25,
                                                       random_state=42, stratify=labels)
    params = {"C": 3.0, "ngram_range": "1,2", "random_state": 42, "test_size": 0.25}
    pipeline = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                         ("classifier", LogisticRegression(C=3.0, max_iter=1000, random_state=42))])
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    positive_index = list(pipeline.classes_).index("positive")
    scores = pipeline.predict_proba(x_test)[:, positive_index]
    metrics = {"accuracy": float(accuracy_score(y_test, predictions)),
               "f1_positive": float(f1_score(y_test, predictions, pos_label="positive")),
               "roc_auc": float(roc_auc_score(np.asarray(y_test) == "positive", scores))}
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_dir / "model.joblib")
    checksum = hashlib.sha256((output_dir / "model.joblib").read_bytes()).hexdigest()
    metadata = {"model_version": checksum[:12], "model_sha256": checksum,
                "created_at": datetime.now(timezone.utc).isoformat(), "task": "russian_sentiment",
                "algorithm": "TF-IDF + LogisticRegression", "python": platform.python_version(),
                "sklearn": sklearn.__version__, "train_rows": len(x_train), "test_rows": len(x_test),
                "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                "metrics": metrics, "parameters": params,
                "limitations": "Small hand-written demo dataset; metrics do not establish real-world quality. Scores are not calibrated."}
    # Build training-only reference through the same feature extractor as serving.
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(output_dir / "reference.npy", np.zeros((1, 5)))
    scorer = Scorer(output_dir)
    _, features = scorer.predict(x_train)
    np.save(output_dir / "reference.npy", features)
    (output_dir / "evaluation.json").write_text(json.dumps([
        {"text": t, "expected": y, "predicted": str(p), "positive_score": float(s)}
        for t, y, p, s in zip(x_test, y_test, predictions, scores)
    ], ensure_ascii=False, indent=2), encoding="utf-8")
    if track:
        import mlflow
        # Remote URI in Docker; local SQLite when running on a developer machine.
        mlflow.set_tracking_uri(tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
        mlflow.set_experiment("text-scoring")
        with mlflow.start_run(run_name="tfidf-logreg") as run:
            mlflow.log_params(params)
            mlflow.log_params({"train_rows": len(x_train), "test_rows": len(x_test)})
            mlflow.log_metrics(metrics)
            mlflow.set_tags({"dataset": data_path.name, "model_version": metadata["model_version"],
                             "dataset_sha256": metadata["data_sha256"], "purpose": "portfolio-demo"})
            metadata["mlflow_run_id"] = run.info.run_id
            (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            mlflow.log_artifacts(str(output_dir), artifact_path="model_bundle")
            mlflow.log_artifact(str(data_path), artifact_path="training_data")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/demo_reviews.csv")
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--tracking-uri")
    parser.add_argument("--no-track", action="store_true", help="Skip MLflow, e.g. during tests")
    args = parser.parse_args()
    train(args.data, args.output, args.tracking_uri, not args.no_track)


if __name__ == "__main__":
    main()
