
def load_document(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def split_into_chunks(text):
    sections = text.split("\n\n")
    chunks = []

    for section in sections:
        section = section.strip()

        if section:
            chunks.append(section)

    return chunks


if __name__ == "__main__":
    text = load_document("insurance_policy.txt")
    chunks = split_into_chunks(text)

    print("Total chunks:", len(chunks))

    for i, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {i} ---")
        print(chunk)
