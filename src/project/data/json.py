import json
from pathlib import Path


def save_performance_json(perf_path: Path, mode: str, report: dict):
    with open(perf_path / f"performance_{mode}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)
