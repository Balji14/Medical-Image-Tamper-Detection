"""Prediction history — persisted as JSON on disk."""
import json
import os
from datetime import datetime

HISTORY_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "history.json")


def _load() -> list:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE) as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save(records: list) -> None:
    with open(HISTORY_FILE, "w") as f:
        json.dump(records, f, indent=2)


def add_record(username: str, filename: str, verdict: str,
               similarity: float, roi_ratio: float, metrics: dict) -> None:
    records = _load()
    records.insert(0, {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "username":  username,
        "filename":  filename,
        "verdict":   verdict,
        "similarity": round(similarity, 4),
        "roi_ratio":  round(roi_ratio, 4),
        "metrics":    metrics,
    })
    records = records[:500]   # cap at 500 entries
    _save(records)


def get_records(username=None) -> list:
    records = _load()
    if username:
        records = [r for r in records if r.get("username") == username]
    return records


def clear_records(username=None) -> None:
    if username is None:
        _save([])
    else:
        _save([r for r in _load() if r.get("username") != username])
