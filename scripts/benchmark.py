"""Real HTTP round-trip benchmark; server must already be running."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import time
import httpx
import numpy as np

TEXTS = ("Отличный сервис, всё понравилось", "Товар сломался, качество плохое", "Спасибо за быструю доставку")


def benchmark(url, count=300, warmup=30, concurrency=1):
    if count < 1 or concurrency < 1 or warmup < 0:
        raise ValueError("Positive count/concurrency and nonnegative warmup required")
    with httpx.Client(base_url=url, timeout=30, trust_env=False,
                      limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)) as client:
        model = client.get("/model")
        model.raise_for_status()
        for i in range(warmup):
            response = client.post("/score", json={"text": TEXTS[i % len(TEXTS)]})
            response.raise_for_status()

        def request(i):
            started = time.perf_counter()
            try:
                response = client.post("/score", json={"text": TEXTS[i % len(TEXTS)]})
                response.raise_for_status()
                response.json()
                return (time.perf_counter() - started) * 1000, None
            except Exception as error:
                return None, type(error).__name__

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(request, range(count)))
        seconds = time.perf_counter() - started
    timings = [duration for duration, error in results if error is None]
    errors = [error for duration, error in results if error is not None]
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), cpu)
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "url": url,
              "measurement": "HTTP round trip, persistent client, full JSON response; training/startup excluded",
              "model_version": model.json()["model_version"], "requests": count,
              "warmup_requests": warmup, "concurrency": concurrency,
              "successful_requests": len(timings), "failed_requests": len(errors),
              "error_types": sorted(set(errors)), "wall_seconds": seconds,
              "successful_requests_per_second": len(timings) / seconds,
              "environment": {"python": platform.python_version(), "platform": platform.system(),
                              "cpu": cpu, "logical_cpus": os.cpu_count()},
              "latency_ms": {"mean": statistics.mean(timings), "p50": float(np.percentile(timings, 50)),
                  "p95": float(np.percentile(timings, 95)), "p99": float(np.percentile(timings, 99)),
                  "min": min(timings), "max": max(timings)} if timings else None}
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output", default="reports/benchmark.json")
    parser.add_argument("--log-mlflow", action="store_true")
    args = parser.parse_args()
    report = benchmark(args.url, args.requests, args.warmup, args.concurrency)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.log_mlflow:
        import mlflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
        mlflow.set_experiment("text-scoring-benchmark")
        with mlflow.start_run(run_name="http-latency"):
            mlflow.log_params({"requests": args.requests, "concurrency": args.concurrency,
                               "warmup": args.warmup, "model_version": report["model_version"]})
            mlflow.log_metrics({f"latency_{name}_ms": value for name, value in (report["latency_ms"] or {}).items()})
            mlflow.log_metrics({"failed_requests": report["failed_requests"],
                                "requests_per_second": report["successful_requests_per_second"]})
            mlflow.log_artifact(str(output))
    if report["failed_requests"]:
        raise SystemExit("Benchmark had failed requests. Inspect the report.")


if __name__ == "__main__":
    main()
