from contextlib import asynccontextmanager
from pathlib import Path
import json
import logging
import os
import time
import uuid
from typing import Annotated

from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.concurrency import run_in_threadpool

from app.model import Scorer
from app.drift import DriftMonitor

logger = logging.getLogger("text_scoring")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=3000)]


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: Text


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    texts: list[Text] = Field(min_length=1, max_length=32)


class Prediction(BaseModel):
    label: str
    positive_score: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0.5, le=1)
    model_version: str


def create_app(model_dir=None, min_samples=None):
    registry = CollectorRegistry()
    requests = Counter("http_requests_total", "HTTP requests", ["route", "method", "status"], registry=registry)
    latency = Histogram("http_request_duration_seconds", "Server request duration", ["route"],
                        buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1, 5), registry=registry)
    scored = Counter("texts_scored_total", "Scored texts", ["label"], registry=registry)
    psi_metric = Gauge("data_drift_psi", "PSI of recent inputs against training baseline", ["feature"], registry=registry)
    drift_flag = Gauge("data_drift_detected", "At least one PSI >= threshold; -1 until ready", registry=registry)
    sample_metric = Gauge("data_drift_window_samples", "Samples in in-memory window", registry=registry)
    drift_flag.set(-1)

    @asynccontextmanager
    async def lifespan(application):
        directory = Path(model_dir or os.getenv("MODEL_DIR", "artifacts"))
        if not (directory / "model.joblib").exists():
            raise RuntimeError("Model missing. Run: python -m scripts.train")
        application.state.scorer = Scorer(directory)
        application.state.monitor = DriftMonitor(application.state.scorer.reference,
            window_size=int(os.getenv("DRIFT_WINDOW_SIZE", "1000")),
            min_samples=min_samples if min_samples is not None else int(os.getenv("DRIFT_MIN_SAMPLES", "50")))
        logger.info(json.dumps({"event": "model_loaded", "model_version": application.state.scorer.version}))
        yield

    api = FastAPI(title="Text Scoring Service", version="1.0.0", lifespan=lifespan,
                  description="Demo Russian sentiment classifier. positive_score is not a calibrated probability.")

    @api.middleware("http")
    async def observe(request, call_next):
        request_id = uuid.uuid4().hex
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            elapsed = time.perf_counter() - started
            route = getattr(request.scope.get("route"), "path", "unmatched")
            # Never log text, raw URL, query strings, headers or validation bodies.
            method = request.method if request.method in {"GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"} else "OTHER"
            requests.labels(route, method, str(status)).inc()
            latency.labels(route).observe(elapsed)
            logger.info(json.dumps({"event": "http_request", "request_id": request_id,
                "route": route, "method": method, "status": status, "duration_ms": round(elapsed * 1000, 3)}))

    def drift_snapshot():
        report = api.state.monitor.report()
        sample_metric.set(report["samples"])
        drift_flag.set(-1 if report["drift_detected"] is None else int(report["drift_detected"]))
        for name, values in report["features"].items():
            psi_metric.labels(name).set(values["psi"])
        return report

    def score_many(texts):
        scores, features = api.state.scorer.predict(texts)
        api.state.monitor.add(features)
        drift_snapshot()
        predictions = []
        for score in scores:
            score = float(score)
            label = "positive" if score >= 0.5 else "negative"
            scored.labels(label).inc()
            predictions.append(Prediction(label=label, positive_score=score,
                confidence=max(score, 1 - score), model_version=api.state.scorer.version))
        return predictions

    @api.get("/health")
    def health():
        return {"status": "ok"}

    @api.get("/ready")
    def ready():
        return {"status": "ready", "model_version": api.state.scorer.version}

    @api.get("/model")
    def model_info():
        return api.state.scorer.metadata

    @api.post("/score", response_model=Prediction)
    async def score(payload: ScoreRequest):
        return (await run_in_threadpool(score_many, [payload.text]))[0]

    @api.post("/score/batch", response_model=list[Prediction])
    async def batch(payload: BatchRequest):
        return await run_in_threadpool(score_many, payload.texts)

    @api.get("/drift")
    def drift():
        return drift_snapshot()

    @api.get("/metrics", include_in_schema=False)
    def metrics():
        return Response(generate_latest(registry), headers={"Content-Type": CONTENT_TYPE_LATEST})

    return api


app = create_app()
