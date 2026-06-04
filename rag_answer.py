import os
import pickle
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

VECTORSTORE_DIR = "vectorstore"

index = faiss.read_index(os.path.join(VECTORSTORE_DIR, "faiss_index.index"))

with open(os.path.join(VECTORSTORE_DIR, "metadata.pkl"), "rb") as f:
    metadata = pickle.load(f)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
llm_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


def detect_category(query):
    query = query.lower()

    if any(word in query for word in ["business loan", "business lending", "sme", "commercial loan", "business finance"]):
        return "business_loans"

    if any(word in query for word in ["home loan", "mortgage", "property loan", "housing loan"]):
        return "home_loans"

    if any(word in query for word in ["credit card", "card fee", "card limit", "minimum repayment"]):
        return "credit_cards"

    if any(word in query for word in ["transaction account", "savings", "bank account", "everyday account"]):
        return "transaction_accounts"

    if any(word in query for word in ["insurance", "claim", "cover", "pds", "protection"]):
        return "insurance"

    if any(word in query for word in ["terms", "conditions", "fees", "policy", "privacy"]):
        return "policies_terms"

    return None


def detect_bank(query):
    query = query.lower()

    if "commbank" in query or "commonwealth bank" in query or "cba" in query:
        return "CommBank"

    if "anz" in query:
        return "ANZ"

    if "nab" in query:
        return "NAB"

    if "westpac" in query:
        return "Westpac"

    return None


def is_relevant_file_for_query(item, query):
    query = query.lower()

    file_name = item.get("file_name", "").lower()
    file_path = item.get("file_path", "").lower()
    text = file_name + " " + file_path

    if any(word in query for word in ["insurance", "claim", "cover", "pds", "protection"]):
        return any(word in text for word in [
            "insurance",
            "claim",
            "claims",
            "pds",
            "protection",
            "cover"
        ])

    if any(word in query for word in ["business loan", "business lending", "commercial loan", "business finance"]):
        if "consumer" in text:
            return False

        return any(word in text for word in [
            "loan",
            "lending",
            "facility",
            "finance",
            "business-lending",
            "business-loan",
            "commercial",
            "sme"
        ])

    if any(word in query for word in ["home loan", "mortgage"]):
        return any(word in text for word in [
            "home-loan",
            "home loan",
            "home-lending",
            "mortgage",
            "housing"
        ])

    if "credit card" in query:
        return any(word in text for word in [
            "credit-card",
            "credit card",
            "card"
        ])

    return True


def retrieve_chunks(query, top_k=5, initial_k=250):
    detected_category = detect_category(query)
    detected_bank = detect_bank(query)

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    distances, indices = index.search(query_embedding, initial_k)

    results = []

    for i, idx in enumerate(indices[0]):
        item = metadata[idx].copy()
        item["distance"] = float(distances[0][i])

        if detected_category:
            file_name = item.get("file_name", "").lower()

            if item.get("category") != detected_category:
                if detected_category == "insurance" and "insurance" in file_name:
                    pass
                else:
                    continue

        if detected_bank:
            if item.get("bank") != detected_bank:
                continue

        if not is_relevant_file_for_query(item, query):
            continue

        results.append(item)

        if len(results) == top_k:
            break

    return results, detected_category, detected_bank


def build_context(chunks, max_chars=2500):
    context_parts = []

    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"""
Source {i}
Bank: {chunk.get("bank")}
Category: {chunk.get("category")}
File: {chunk.get("file_name")}

Content:
{chunk.get("chunk_text")}
"""
        )

    context = "\n".join(context_parts)
    return context[:max_chars]


def generate_answer(query, chunks):
    if not chunks:
        return "The available documents do not provide enough information."

    context = build_context(chunks)

    prompt = f"""
Use the context below to answer the question.
Write a short and direct answer.
Do not include source labels in the answer.
Do not copy unnecessary text.

Question:
{query}

Context:
{context}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )

    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=180,
            num_beams=4,
            do_sample=False
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer


def main():
    query = input("Ask a financial question: ")

    chunks, category, bank = retrieve_chunks(query)

    print(f"\nDetected category: {category}")
    print(f"Detected bank: {bank}")

    answer = generate_answer(query, chunks)

    print("\nAnswer:\n")
    print(answer)

    print("\nSources used:\n")
    if not chunks:
        print("No sources found.")
    else:
        for i, chunk in enumerate(chunks, start=1):
            print(f"{i}. {chunk.get('bank')} | {chunk.get('category')} | {chunk.get('file_name')}")


if __name__ == "__main__":
    main()