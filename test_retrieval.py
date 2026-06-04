import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

VECTORSTORE_DIR = "vectorstore"

index = faiss.read_index(os.path.join(VECTORSTORE_DIR, "faiss_index.index"))

with open(os.path.join(VECTORSTORE_DIR, "metadata.pkl"), "rb") as f:
    metadata = pickle.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")


def detect_category(query):
    query = query.lower()

    if any(word in query for word in [
        "business loan",
        "business lending",
        "sme",
        "commercial loan",
        "business finance"
    ]):
        return "business_loans"

    if any(word in query for word in [
        "home loan",
        "mortgage",
        "property loan",
        "housing loan"
    ]):
        return "home_loans"

    if any(word in query for word in [
        "credit card",
        "card fee",
        "card limit",
        "minimum repayment"
    ]):
        return "credit_cards"

    if any(word in query for word in [
        "transaction account",
        "savings",
        "bank account",
        "everyday account"
    ]):
        return "transaction_accounts"

    if any(word in query for word in [
        "insurance",
        "claim",
        "cover",
        "pds"
    ]):
        return "insurance"

    if any(word in query for word in [
        "terms",
        "conditions",
        "fees",
        "policy",
        "privacy"
    ]):
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

    # BUSINESS LOANS
    if any(word in query for word in [
        "business loan",
        "business lending",
        "commercial loan",
        "business finance"
    ]):

        # Exclude consumer lending docs
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

    # HOME LOANS
    if any(word in query for word in [
        "home loan",
        "mortgage"
    ]):

        return any(word in text for word in [
            "home-loan",
            "home loan",
            "home-lending",
            "mortgage",
            "housing"
        ])

    # CREDIT CARDS
    if "credit card" in query:

        return any(word in text for word in [
            "credit-card",
            "credit card",
            "card"
        ])

    return True


def search_documents(query, top_k=5, initial_k=200):

    detected_category = detect_category(query)
    detected_bank = detect_bank(query)

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    distances, indices = index.search(query_embedding, initial_k)

    results = []

    for i, idx in enumerate(indices[0]):

        item = metadata[idx].copy()

        item["distance"] = float(distances[0][i])

        # CATEGORY FILTER
        if detected_category:
            if item.get("category") != detected_category:
                continue

        # BANK FILTER
        if detected_bank:
            if item.get("bank") != detected_bank:
                continue

        # QUERY-SPECIFIC FILE FILTER
        if not is_relevant_file_for_query(item, query):
            continue

        results.append(item)

        if len(results) == top_k:
            break

    return results, detected_category, detected_bank


query = input("Ask a financial question: ")

results, detected_category, detected_bank = search_documents(query)

print(f"\nDetected category: {detected_category}")
print(f"Detected bank: {detected_bank}")

print("\nTop relevant chunks:\n")

if not results:
    print("No relevant chunks found.")
else:

    for i, result in enumerate(results, start=1):

        print("=" * 80)

        print(f"Result {i}")
        print(f"Bank: {result.get('bank')}")
        print(f"Category: {result.get('category')}")
        print(f"File: {result.get('file_name')}")
        print(f"File path: {result.get('file_path')}")
        print(f"Chunk ID: {result.get('chunk_id')}")
        print(f"Distance: {result.get('distance')}")

        print("-" * 80)

        print(result.get("chunk_text", "")[:1000])