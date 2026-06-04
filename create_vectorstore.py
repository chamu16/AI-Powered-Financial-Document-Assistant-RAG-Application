import os
import pickle
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INPUT_FILE = "document_chunks.csv"
VECTORSTORE_DIR = "vectorstore"

os.makedirs(VECTORSTORE_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

df = df.dropna(subset=["chunk_text"])
df["chunk_text"] = df["chunk_text"].astype(str)

def extract_bank(file_path):
    file_path = str(file_path).lower()

    if "commbank" in file_path:
        return "CommBank"
    elif "anz" in file_path:
        return "ANZ"
    elif "nab" in file_path:
        return "NAB"
    elif "westpac" in file_path:
        return "Westpac"
    else:
        return "Unknown"


def extract_category(file_path):
    text = str(file_path).lower()

    # HOME LOANS
    if any(word in text for word in [
        "home-loan",
        "home loan",
        "home-lending",
        "home lending",
        "mortgage",
        "housing"
    ]):
        return "home_loans"

    # CREDIT CARDS
    if any(word in text for word in [
        "credit-card",
        "credit card",
        "cards",
        "visa",
        "mastercard"
    ]):
        return "credit_cards"

    # INSURANCE
    if any(word in text for word in [
        "insurance",
        "pds",
        "product-disclosure",
        "product disclosure"
    ]):
        return "insurance"

    # TRANSACTION / SAVINGS ACCOUNTS
    if any(word in text for word in [
        "transaction",
        "savings",
        "bank-account",
        "bank account",
        "everyday account",
        "deposit account",
        "account"
    ]):
        return "transaction_accounts"

    # POLICIES / TERMS
    if any(word in text for word in [
        "terms",
        "conditions",
        "fees",
        "policy",
        "privacy",
        "fsg",
        "financial-services-guide"
    ]):
        return "policies_terms"

    # BUSINESS LOANS
    if any(word in text for word in [
        "business",
        "commercial",
        "sme",
        "business-loan",
        "business loan",
        "business-lending",
        "business lending",
        "finance",
        "equipment-finance"
    ]):
        return "business_loans"

    return "other_financial_docs"


df["bank"] = df["file_path"].apply(extract_bank)
df["category"] = df["file_path"].apply(extract_category)

print(f"Total chunks loaded: {len(df)}")
print("\nChunks by bank:")
print(df["bank"].value_counts())

print("\nChunks by category:")
print(df["category"].value_counts())

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = df["chunk_text"].tolist()

print("\nCreating embeddings...")
embeddings = model.encode(
    texts,
    show_progress_bar=True,
    convert_to_numpy=True
)

embeddings = embeddings.astype("float32")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, os.path.join(VECTORSTORE_DIR, "faiss_index.index"))

metadata = df[
    [
        "file_name",
        "file_path",
        "bank",
        "category",
        "chunk_id",
        "chunk_text"
    ]
].to_dict(orient="records")

with open(os.path.join(VECTORSTORE_DIR, "metadata.pkl"), "wb") as f:
    pickle.dump(metadata, f)

print("\nFAISS vectorstore created successfully.")
print(f"Embedding dimension: {dimension}")
print(f"Total vectors stored: {index.ntotal}")
print(f"Metadata records stored: {len(metadata)}")