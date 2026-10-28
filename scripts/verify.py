"""Start a real local HTTP server, benchmark it and test a drift scenario."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import httpx
from scripts.benchmark import benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18000)
    args = parser.parse_args()
    url = f"http://127.0.0.1:{args.port}"
    Path("reports").mkdir(exist_ok=True)
    with tempfile.TemporaryFile(mode="w+") as log:
        process = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1",
                                    "--port", str(args.port), "--no-access-log"], stdout=log, stderr=log)
        try:
            with httpx.Client(timeout=2, trust_env=False) as client:
                deadline = time.monotonic() + 30
                while True:
                    if process.poll() is not None:
                        log.seek(0)
                        raise RuntimeError(log.read())
                    try:
                        response = client.get(url + "/ready")
                        if response.status_code == 200:
                            break
                    except httpx.RequestError:
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError("API startup timeout")
                    time.sleep(0.1)
                result = benchmark(url, count=300, warmup=30, concurrency=1)
                Path("reports/benchmark.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                concurrent = benchmark(url, count=200, warmup=10, concurrency=8)
                Path("reports/benchmark_concurrent.json").write_text(json.dumps(concurrent, ensure_ascii=False, indent=2), encoding="utf-8")
                for _ in range(10):
                    response = client.post(url + "/score/batch", json={"texts": ["xyzzy spaceship " * 100] * 32})
                    response.raise_for_status()
                drift = client.get(url + "/drift").json()
                metrics = client.get(url + "/metrics")
                assert result["failed_requests"] == concurrent["failed_requests"] == 0
                assert drift["status"] == "ready" and drift["drift_detected"] is True
                assert metrics.status_code == 200 and "data_drift_psi" in metrics.text
                assert client.get(url + "/docs").status_code == 200
                Path("reports/drift_demo.json").write_text(json.dumps(drift, indent=2), encoding="utf-8")
                print(json.dumps({"http_smoke": "passed", "single_request_latency_ms": result["latency_ms"],
                                  "concurrent_failed_requests": concurrent["failed_requests"], "drift_demo": "detected"}, indent=2))
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
