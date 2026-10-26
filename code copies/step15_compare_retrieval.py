
from step14_retrieval_evaluation import (
    evaluation_questions,
    get_query_embedding,
    collection,
)

# Test retrieving 3 sections versus 5 sections
for top_k in [3, 5]:
    scores = []

    print(f"\n{'=' * 50}")
    print(f"Testing retrieval with top_k = {top_k}")
    print("=" * 50)

    for test in evaluation_questions:
        question = test["question"]
        relevant_sections = test["relevant_sections"]

        query_embedding = get_query_embedding(question)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=["metadatas", "distances"],
        )

        retrieved_metadata = results["metadatas"][0]

        relevant_count = sum(
            metadata.get("section_title", "Unknown")
            in relevant_sections
            for metadata in retrieved_metadata
        )

        total_retrieved = len(retrieved_metadata)

        precision = (
            relevant_count / total_retrieved
            if total_retrieved
            else 0
        )

        scores.append(precision)

        print(
            f"{question}\n"
            f"  Relevant: {relevant_count}/{total_retrieved}"
            f" | Precision: {precision:.2%}"
        )

    average_precision = (
        sum(scores) / len(scores)
        if scores
        else 0
    )

    print(f"\nAverage Precision@{top_k}: {average_precision:.2%}")
