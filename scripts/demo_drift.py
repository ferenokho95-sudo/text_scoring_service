"""Send intentionally shifted inputs, then inspect monitoring endpoint."""
import argparse
import json
import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count must be positive")
    with httpx.Client(base_url=args.url, timeout=30, trust_env=False) as client:
        for _ in range(args.count):
            response = client.post("/score", json={"text": "Unseen vocabulary xyzzy quantum spaceship " * 45})
            response.raise_for_status()
        response = client.get("/drift")
        response.raise_for_status()
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
