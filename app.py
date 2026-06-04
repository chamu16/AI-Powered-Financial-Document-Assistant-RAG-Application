import os
import pickle
import faiss
import torch
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

VECTORSTORE_DIR = "vectorstore"
MODEL_NAME = "google/flan-t5-large"


@st.cache_resource
def load_resources():
    index = faiss.read_index(os.path.join(VECTORSTORE_DIR, "faiss_index.index"))

    with open(os.path.join(VECTORSTORE_DIR, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    llm_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    return index, metadata, embedding_model, tokenizer, llm_model


def detect_category(query):
    query = query.lower()

    if any(w in query for w in ["business loan", "business lending", "sme", "commercial loan", "business finance"]):
        return "business_loans"
    if any(w in query for w in ["home loan", "mortgage", "property loan", "housing loan"]):
        return "home_loans"
    if any(w in query for w in ["credit card", "card fee", "card limit", "minimum repayment"]):
        return "credit_cards"
    if any(w in query for w in ["transaction account", "savings", "bank account", "everyday account"]):
        return "transaction_accounts"
    if any(w in query for w in ["insurance", "claim", "cover", "pds", "protection"]):
        return "insurance"
    if any(w in query for w in ["terms", "conditions", "fees", "policy", "privacy"]):
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

    if any(w in query for w in ["insurance", "claim", "cover", "pds", "protection"]):
        return any(w in text for w in ["insurance", "claim", "claims", "pds", "protection", "cover"])

    if any(w in query for w in ["business loan", "business lending", "commercial loan", "business finance"]):
        if "consumer" in text:
            return False
        return any(w in text for w in [
            "loan", "lending", "facility", "finance",
            "business-lending", "business-loan", "commercial", "sme"
        ])

    if any(w in query for w in ["home loan", "mortgage"]):
        return any(w in text for w in [
            "home-loan", "home loan", "home-lending", "mortgage", "housing"
        ])

    if "credit card" in query:
        return any(w in text for w in ["credit-card", "credit card", "card"])

    return True


def retrieve_chunks(query, index, metadata, embedding_model, selected_banks, top_k=5, initial_k=300):
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

        if selected_banks and item.get("bank") not in selected_banks:
            continue

        if detected_bank and item.get("bank") != detected_bank:
            continue

        if detected_category:
            file_name = item.get("file_name", "").lower()

            if item.get("category") != detected_category:
                if detected_category == "insurance" and "insurance" in file_name:
                    pass
                else:
                    continue

        if not is_relevant_file_for_query(item, query):
            continue

        results.append(item)

        if len(results) == top_k:
            break

    return results


def build_context(chunks, max_chars=3500):
    context_parts = []

    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"""
Bank: {chunk.get("bank")}
Category: {chunk.get("category")}
File: {chunk.get("file_name")}
Content: {chunk.get("chunk_text")}
"""
        )

    return "\n\n".join(context_parts)[:max_chars]


def clean_answer(answer):
    remove_phrases = [
        "Only answer using the context provided.",
        "Only answer using the context.",
        "Use the context below to answer the question.",
        "Answer the question using the retrieved document information.",
        "Response:",
        "Answer:"
    ]

    for phrase in remove_phrases:
        answer = answer.replace(phrase, "")

    answer = answer.replace("DONE DONE DONE", "")
    answer = answer.replace("DONE", "")
    answer = answer.replace("IF YOU’RE", "\n\nIF YOU’RE")
    answer = answer.replace("IF YOU'RE", "\n\nIF YOU'RE")
    answer = answer.strip()

    if len(answer) < 10:
        return "The available documents do not provide enough information."

    return answer


def generate_answer(query, chunks, tokenizer, llm_model):
    if not chunks:
        return "The available documents do not provide enough information."

    context = build_context(chunks)

    prompt = f"""
You are a financial document assistant.

Answer the question using the retrieved document information.

Rules:
- Do not repeat these instructions.
- Do not mention the word context.
- Do not include source labels in the answer.
- Do not make up information.
- Use short bullet points when useful.
- If comparing banks, clearly separate similarities and differences.
- If the information is insufficient, say: The available documents do not provide enough information.

Question:
{query}

Retrieved document information:
{context}

Final answer:
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
            max_new_tokens=220,
            num_beams=5,
            do_sample=False,
            early_stopping=True
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_answer(answer)


def get_project_stats(metadata):
    df = pd.DataFrame(metadata)

    total_banks = df["bank"].nunique() if "bank" in df.columns else 0
    total_docs = df["file_name"].nunique() if "file_name" in df.columns else 0
    total_chunks = len(df)

    return total_banks, total_docs, total_chunks


st.set_page_config(
    page_title="Financial Document Intelligence Platform",
    page_icon="🏦",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #020617 0%, #0f172a 45%, #111827 100%);
        color: #f8fafc;
    }

    div[data-testid="stSidebar"] {
        background: #020617;
        border-right: 1px solid rgba(148, 163, 184, 0.2);
    }

    .hero-card {
        background: rgba(15, 23, 42, 0.92);
        padding: 34px;
        border-radius: 26px;
        border: 1px solid rgba(148, 163, 184, 0.25);
        box-shadow: 0 20px 65px rgba(0, 0, 0, 0.35);
        margin-bottom: 24px;
    }

    .hero-title {
        font-size: 44px;
        font-weight: 850;
        color: #f8fafc;
        margin-bottom: 10px;
    }

    .hero-subtitle {
        font-size: 17px;
        color: #cbd5e1;
        max-width: 950px;
        line-height: 1.6;
    }

    .metric-card {
        background: rgba(30, 41, 59, 0.88);
        padding: 22px;
        border-radius: 18px;
        border: 1px solid rgba(148, 163, 184, 0.22);
        text-align: center;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.22);
    }

    .metric-number {
        font-size: 30px;
        font-weight: 850;
        color: #38bdf8;
    }

    .metric-label {
        font-size: 14px;
        color: #cbd5e1;
    }

    .source-card {
        background: rgba(30, 41, 59, 0.88);
        padding: 18px;
        border-radius: 16px;
        border: 1px solid rgba(148, 163, 184, 0.25);
        margin-bottom: 12px;
    }

    .example-card {
        background: rgba(30, 41, 59, 0.75);
        padding: 14px 16px;
        border-radius: 14px;
        border: 1px solid rgba(148, 163, 184, 0.2);
        color: #e2e8f0;
        margin-bottom: 10px;
        font-size: 14px;
    }

    .stButton > button {
        background: linear-gradient(90deg, #0284c7, #38bdf8);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.65rem 1.4rem;
        font-weight: 700;
        width: 100%;
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #0369a1, #0ea5e9);
        color: white;
    }

    .stTextInput > div > div > input {
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

index, metadata, embedding_model, tokenizer, llm_model = load_resources()
total_banks, total_docs, total_chunks = get_project_stats(metadata)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.title("🏦 Assistant Panel")
    st.caption("Financial RAG System")

    st.markdown("---")
    st.subheader("Bank Filter")

    available_banks = sorted(list(set([m.get("bank") for m in metadata if m.get("bank")])))
    selected_banks = st.multiselect(
        "Choose banks to search",
        options=available_banks,
        default=available_banks
    )

    st.markdown("---")
    st.subheader("Example Questions")

    st.markdown('<div class="example-card">What documents are required for a business loan?</div>', unsafe_allow_html=True)
    st.markdown('<div class="example-card">How can I make an insurance claim?</div>', unsafe_allow_html=True)
    st.markdown('<div class="example-card">What are NAB business loan fees?</div>', unsafe_allow_html=True)
    st.markdown('<div class="example-card">Compare ANZ and NAB business lending requirements.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Technology")
    st.write("FAISS Vector Search")
    st.write("Sentence Transformers")
    st.write("Hugging Face FLAN-T5")
    st.write("Streamlit UI")

    st.markdown("---")
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Financial Document Intelligence Platform</div>
        <div class="hero-subtitle">
            A Retrieval-Augmented Generation assistant for Australian banking documents.
            Ask questions across business loans, home loans, credit cards, insurance,
            policy documents and financial terms. Responses are grounded in retrieved source documents.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-number">{total_banks}</div>
            <div class="metric-label">Banks Covered</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-number">{total_docs}</div>
            <div class="metric-label">Documents Indexed</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-number">{total_chunks}</div>
            <div class="metric-label">Document Chunks</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("")

query = st.text_input(
    "Ask a financial question",
    placeholder="Example: What documents are required for a business loan?"
)

ask_button = st.button("Generate Answer")

if ask_button:
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching financial documents and generating answer..."):
            chunks = retrieve_chunks(
                query=query,
                index=index,
                metadata=metadata,
                embedding_model=embedding_model,
                selected_banks=selected_banks
            )

            answer = generate_answer(
                query=query,
                chunks=chunks,
                tokenizer=tokenizer,
                llm_model=llm_model
            )

        st.session_state.chat_history.append(
            {
                "question": query,
                "answer": answer,
                "sources": chunks
            }
        )

for chat in reversed(st.session_state.chat_history):
    with st.chat_message("user"):
        st.write(chat["question"])

    with st.chat_message("assistant"):
        st.write(chat["answer"])

    with st.expander("📄 Sources"):
        if not chat["sources"]:
            st.write("No sources found.")
        else:
            for i, chunk in enumerate(chat["sources"], start=1):
                st.markdown(
                    f"""
                    <div class="source-card">
                        <b>Source {i}</b><br>
                        <b>Bank:</b> {chunk.get("bank")}<br>
                        <b>Category:</b> {chunk.get("category")}<br>
                        <b>File:</b> {chunk.get("file_name")}<br>
                        <b>Similarity Distance:</b> {round(chunk.get("distance"), 4)}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.write(chunk.get("chunk_text", "")[:1500])