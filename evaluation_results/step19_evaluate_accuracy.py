
import csv
import time
from pathlib import Path
from datetime import datetime

import requests

import step12_chroma_rag_chat as rag


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "evaluation_sample.csv"
OUTPUT_FILE = BASE_DIR / "evaluation_results.csv"

FIELDNAMES = [
    "timestamp",
    "question",
    "expected_source",
    "retrieved_sources",
    "expected_source_found",
    "answer",
    "answer_status",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "embedding_time_seconds",
    "retrieval_time_seconds",
    "chat_time_seconds",
    "total_time_seconds",
]


def save_results(results):
    """Save all results, including progress made so far."""
    with OUTPUT_FILE.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)


def main():
    if not INPUT_FILE.exists():
        print("Missing file:", INPUT_FILE.name)
        return

    with INPUT_FILE.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        test_cases = list(csv.DictReader(file))

    if not test_cases:
        print("No evaluation questions found.")
        return

    # Load previous results so successful answers are preserved.
    previous_results = {}

    if OUTPUT_FILE.exists():
        with OUTPUT_FILE.open(
            "r", encoding="utf-8-sig", newline=""
        ) as file:
            for row in csv.DictReader(file):
                key = (
                    row["question"].strip(),
                    row["expected_source"].strip(),
                )
                previous_results[key] = row

    results = []
    total = len(test_cases)
    newly_evaluated = 0
    skipped = 0

    print("Evaluation questions:", total)
    print("Existing successful results will be preserved.")
    print("Only new questions and previous errors will be retried.")
    print("Progress will be saved after each question.\n")

    for index, test in enumerate(test_cases, start=1):
        question = test["question"].strip()
        expected_source = test["expected_source"].strip()
        key = (question, expected_source)

        previous = previous_results.get(key)

        # Keep a previously successful result without another API call.
        if previous and previous["answer_status"] != "error":
            results.append(previous)
            skipped += 1
            print(
                f"[{index}/{total}] Already completed; skipping: "
                f"{question}"
            )
            continue

        metrics = {
            "embedding_time_seconds": 0,
            "retrieval_time_seconds": 0,
            "retrieved_sections": 0,
        }

        chat_metrics = {
            "input_tokens": "",
            "output_tokens": "",
            "total_tokens": "",
            "chat_time_seconds": 0,
        }

        answer = ""
        answer_status = "error"
        retrieved_sources = []
        source_found = False

        total_start = time.perf_counter()

        print(f"\n[{index}/{total}] Retrying/evaluating: {question}")

        try:
            documents = rag.retrieve_documents(question, metrics)

            retrieved_sources = sorted({
                doc["source"] for doc in documents
            })

            source_found = expected_source in retrieved_sources

            if documents:
                answer, chat_metrics = rag.generate_answer(
                    question, documents
                )
                answer_status = "answered"
                print("Answer:", answer)
            else:
                answer = (
                    "The provided policy documents do not contain "
                    "enough information to answer that."
                )
                answer_status = "no_relevant_documents"
                print("No relevant documents retrieved.")

            print("Expected source:", expected_source)
            print("Retrieved sources:", retrieved_sources)
            print("Expected source found:", source_found)

        except (requests.RequestException, ValueError, KeyError) as error:
            answer_status = "error"
            answer = f"Evaluation error: {type(error).__name__}: {error}"
            print(answer)

        except Exception as error:
            answer_status = "error"
            answer = f"Evaluation error: {type(error).__name__}: {error}"
            print(answer)

        total_time = round(time.perf_counter() - total_start, 2)

        result = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "question": question,
            "expected_source": expected_source,
            "retrieved_sources": "; ".join(retrieved_sources),
            "expected_source_found": source_found,
            "answer": answer,
            "answer_status": answer_status,
            "input_tokens": chat_metrics["input_tokens"],
            "output_tokens": chat_metrics["output_tokens"],
            "total_tokens": chat_metrics["total_tokens"],
            "embedding_time_seconds": metrics["embedding_time_seconds"],
            "retrieval_time_seconds": metrics["retrieval_time_seconds"],
            "chat_time_seconds": chat_metrics["chat_time_seconds"],
            "total_time_seconds": total_time,
        }

        results.append(result)
        newly_evaluated += 1

        # Save progress immediately, even if a later question fails.
        remaining_results = []
        processed_keys = {(
            row["question"].strip(),
            row["expected_source"].strip(),
        ) for row in results}

        for remaining_test in test_cases[index:]:
            remaining_key = (
                remaining_test["question"].strip(),
                remaining_test["expected_source"].strip(),
            )
            if remaining_key not in processed_keys:
                old_row = previous_results.get(remaining_key)
                if old_row:
                    remaining_results.append(old_row)

        save_results(results + remaining_results)

    # Calculate metrics using completed evaluations only.
    completed = [
        row for row in results
        if row["answer_status"] != "error"
    ]

    correct_sources = sum(
        str(row["expected_source_found"]).strip().lower() == "true"
        for row in completed
    )

    retrieval_rate = (
        correct_sources / len(completed) * 100
        if completed else 0
    )

    print("\n========== EVALUATION SUMMARY ==========")
    print("Test cases:", total)
    print("Newly evaluated/retried:", newly_evaluated)
    print("Previous results skipped:", skipped)
    print("Completed without errors:", len(completed))
    print("Still in error:", total - len(completed))
    print("Expected source retrieved:", correct_sources)
    print(
        "Source retrieval rate among completed evaluations:",
        f"{retrieval_rate:.1f}%"
    )
    print("Results saved to:", OUTPUT_FILE.name)
    print("========================================")


if __name__ == "__main__":
    main()
