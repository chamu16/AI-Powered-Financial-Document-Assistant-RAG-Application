# Financial Document Intelligence Platform

## Overview

Financial Document Intelligence Platform is a Retrieval-Augmented Generation (RAG) system that enables users to ask natural language questions about Australian banking documents. The system automatically collects financial PDFs, processes and indexes their content, and generates source-grounded answers using semantic search and open-source language models.

## Features

* Automated collection of banking PDF documents
* Semantic document retrieval using FAISS
* Source-grounded question answering
* Support for business loans, home loans, insurance, credit cards, and banking policies
* Interactive Streamlit chatbot interface

## Technology Stack

* Python
* Streamlit
* FAISS
* Sentence Transformers
* Hugging Face Transformers
* PyMuPDF
* LangChain

## Workflow

**Step 1:** Collect financial PDF documents from Australian banks

**Step 2:** Extract and clean text from PDFs

**Step 3:** Split documents into overlapping chunks

**Step 4:** Generate embeddings using Sentence Transformers

**Step 5:** Store embeddings in a FAISS vector database

**Step 6:** Retrieve relevant document chunks based on user queries

**Step 7:** Generate answers using FLAN-T5

**Step 8:** Display responses through a Streamlit chatbot

## Banks Included

* Commonwealth Bank
* ANZ
* NAB
* Westpac

## Running the Project

```bash
pip install -r requirements.txt

python download_bank_pdfs.py
python extract_text.py
python create_chunks.py
python create_vectorstore.py

python -m streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## Example Questions

* What documents are required for a business loan?
* How can I make an insurance claim?
* What are NAB business loan fees?
* Compare ANZ and NAB business lending requirements.

## Future Improvements

* Stronger open-source LLMs
* Hybrid search (semantic + keyword)
* Conversational memory
* Cloud deployment
