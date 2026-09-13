from __future__ import annotations

import argparse
import random
import time

import httpx

from triage_api.metrics import HEALTH_PATH

REPORTS = [
    "CT scan of the chest: no acute abnormality identified. Routine review.",
    "The laboratory panel demonstrates mildly elevated inflammatory markers.",
    "Impression: acute intracranial hemorrhage on CT scan of the cranium. "
    "Immediate clinical attention required.",
    "MRI of the lumbar spine: mild degenerative changes consistent with age.",
    "Reported signs of septic shock with rising lactate. Rapid deterioration noted.",
    "Radiograph of the right shoulder: no evidence of fracture or dislocation.",
    "Large pulmonary embolism with right heart strain. Patient is hemodynamically unstable.",
    "Non-obstructing calculus without hydronephrosis. Non-emergent.",
]

INVALID_PAYLOADS = [{"text": ""}, {"text": "   "}, {}]

ERROR_RATE = 0.1


def fire(client: httpx.Client) -> None:
    if random.random() < ERROR_RATE:
        client.post("/predict", json=random.choice(INVALID_PAYLOADS))
        return
    client.post("/predict", json={"text": random.choice(REPORTS)})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fire a stream of requests at the API so the Grafana dashboard has data."
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--interval", type=float, default=0.05, help="seconds between requests")
    args = parser.parse_args()

    with httpx.Client(base_url=args.base_url, timeout=10.0) as client:
        client.get(HEALTH_PATH).raise_for_status()
        for i in range(args.requests):
            fire(client)
            if (i + 1) % 20 == 0:
                print(f"sent {i + 1}/{args.requests} requests")
            time.sleep(args.interval)

    print(f"done: sent {args.requests} requests to {args.base_url}")


if __name__ == "__main__":
    main()
