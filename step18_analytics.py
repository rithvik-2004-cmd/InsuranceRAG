
import csv
from pathlib import Path
from datetime import datetime

# Project folder
BASE_DIR = Path(__file__).resolve().parent

# Analytics CSV file
ANALYTICS_FILE = BASE_DIR / "analytics.csv"

# Columns that will be stored for every question
FIELDNAMES = [
    "timestamp",
    "question",
    "input_tokens",
    "output_tokens",
    "embedding_time_seconds",
    "retrieval_time_seconds",
    "chat_time_seconds",
    "total_time_seconds",
    "retrieved_sections",
    "answer_status",
    "accuracy_status",
]


def initialize_analytics():
    """Create the CSV file with headers if it does not exist."""

    if not ANALYTICS_FILE.exists():
        with open(
            ANALYTICS_FILE,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=FIELDNAMES,
            )
            writer.writeheader()

        print("Analytics file created:", ANALYTICS_FILE.name)
    else:
        print("Analytics file already exists:", ANALYTICS_FILE.name)


def save_analytics(record):
    """Append one question's metrics to the analytics CSV file."""

    with open(
        ANALYTICS_FILE,
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES,
        )
        writer.writerow(record)

    print("Analytics record saved.")


if __name__ == "__main__":
    initialize_analytics()

    test_record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "question": "What is a deductible?",
        "input_tokens": 420,
        "output_tokens": 185,
        "embedding_time_seconds": 0.65,
        "retrieval_time_seconds": 0.03,
        "chat_time_seconds": 1.42,
        "total_time_seconds": 2.10,
        "retrieved_sections": 3,
        "answer_status": "answered",
        "accuracy_status": "not_evaluated",
    }

    save_analytics(test_record)
