import os
import fitz
import pandas as pd

PDF_DIR = "data"
OUTPUT_FILE = "processed_documents.csv"

documents = []

for root, dirs, files in os.walk(PDF_DIR):
    for file in files:
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(root, file)

            try:
                doc = fitz.open(pdf_path)
                text = ""

                for page in doc:
                    text += page.get_text()

                if len(text.strip()) > 200:
                    documents.append({
                        "file_name": file,
                        "file_path": pdf_path,
                        "text": text
                    })

                    print(f"Extracted: {file}")

                else:
                    print(f"Skipped, too little text: {file}")

            except Exception as e:
                print(f"Failed: {file} | {e}")

df = pd.DataFrame(documents)
df.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved extracted text to {OUTPUT_FILE}")
print(f"Total documents extracted: {len(df)}")