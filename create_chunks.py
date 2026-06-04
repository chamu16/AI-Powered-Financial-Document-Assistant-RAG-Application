import pandas as pd
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

INPUT_FILE = "processed_documents.csv"
OUTPUT_FILE = "document_chunks.csv"

df = pd.read_csv(INPUT_FILE)

def clean_text(text):
    text = str(text)

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove repeated dots or strange spacing
    text = re.sub(r"\.{3,}", ".", text)

    # Strip leading/trailing spaces
    text = text.strip()

    return text

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " ", ""]
)

chunks_data = []

for _, row in df.iterrows():
    file_name = row["file_name"]
    file_path = row["file_path"]
    text = clean_text(row["text"])

    chunks = splitter.split_text(text)

    for i, chunk in enumerate(chunks):
        chunks_data.append({
            "file_name": file_name,
            "file_path": file_path,
            "chunk_id": i,
            "chunk_text": chunk
        })

chunk_df = pd.DataFrame(chunks_data)
chunk_df.to_csv(OUTPUT_FILE, index=False)

print(f"Total documents: {len(df)}")
print(f"Total chunks created: {len(chunk_df)}")
print(f"Saved chunks to: {OUTPUT_FILE}")