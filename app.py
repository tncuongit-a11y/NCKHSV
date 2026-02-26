import streamlit as st
import faiss
import pickle
import numpy as np
import os
from sentence_transformers import SentenceTransformer, CrossEncoder

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="RAG Medical Assistant",
    page_icon="🧠",
    layout="wide"
)

# =========================================================
# CUSTOM CSS (Modern UI)
# =========================================================
st.markdown("""
<style>
.chat-user {
    background-color: #DCF8C6;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 10px;
}
.chat-bot {
    background-color: #F1F0F0;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 10px;
}
.small-text {
    font-size: 13px;
    color: gray;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODELS & INDEX
# =========================================================
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("intfloat/multilingual-e5-base")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return embedder, reranker


@st.cache_resource
def load_index():
    base_path = os.path.dirname(__file__)
    index_path = os.path.join(base_path, "vector.index")
    metadata_path = os.path.join(base_path, "metadata.pkl")

    index = faiss.read_index(index_path)

    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata


embedder, reranker = load_models()
index, metadata = load_index()

# =========================================================
# SEARCH FUNCTIONS
# =========================================================
def retrieve(query, top_k=5):
    q_emb = embedder.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    scores, ids = index.search(q_emb, top_k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        item = metadata[idx]
        item["score"] = float(score)
        results.append(item)

    return results


def retrieve_with_rerank(query, faiss_top_k=20, final_top_k=5):
    q_emb = embedder.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    scores, ids = index.search(q_emb, faiss_top_k)

    candidates = [metadata[i] for i in ids[0]]
    pairs = [(query, c["text"]) for c in candidates]

    rerank_scores = reranker.predict(pairs)

    reranked = sorted(
        zip(candidates, rerank_scores),
        key=lambda x: x[1],
        reverse=True
    )

    results = []
    for item, score in reranked[:final_top_k]:
        item["rerank_score"] = float(score)
        results.append(item)

    return results


# =========================================================
# SIMPLE ANSWER GENERATOR (Placeholder)
# =========================================================
def generate_answer(query, contexts):
    context_text = "\n\n".join([c["text"] for c in contexts])

    return f"""
Dựa trên hồ sơ bệnh án truy xuất được, thông tin liên quan đến câu hỏi:

{query}

Cho thấy các dấu hiệu được ghi nhận trong hồ sơ. 
(Bạn có thể tích hợp OpenAI API để sinh câu trả lời thực tế.)
"""


# =========================================================
# SIDEBAR CONFIGURATION
# =========================================================
st.sidebar.title("⚙️ Cấu hình hệ thống")

top_k = st.sidebar.slider("Top K Retrieval", 1, 10, 5)
use_rerank = st.sidebar.checkbox("Sử dụng CrossEncoder Rerank", value=True)
generate_llm = st.sidebar.checkbox("Sinh câu trả lời (LLM)", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Thông tin hệ thống")
st.sidebar.markdown("""
- Embedding: multilingual-e5-base  
- Vector DB: FAISS (Inner Product)  
- Reranker: MiniLM CrossEncoder  
""")

# =========================================================
# MAIN UI
# =========================================================
st.title("🧠 RAG Medical Assistant")
st.markdown("Hệ thống truy xuất & sinh câu trả lời từ hồ sơ bệnh án")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.chat_input("Nhập câu hỏi của bạn...")

if query:
    st.session_state.chat_history.append(("user", query))

    with st.spinner("🔎 Đang truy xuất dữ liệu..."):
        if use_rerank:
            results = retrieve_with_rerank(
                query,
                faiss_top_k=20,
                final_top_k=top_k
            )
        else:
            results = retrieve(query, top_k)

    if generate_llm:
        answer = generate_answer(query, results)
    else:
        answer = "Đã truy xuất xong. Xem context bên dưới."

    st.session_state.chat_history.append(("bot", answer))

# =========================================================
# DISPLAY CHAT
# =========================================================
for role, message in st.session_state.chat_history:
    if role == "user":
        st.markdown(f'<div class="chat-user">🧑‍⚕️ {message}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bot">🤖 {message}</div>', unsafe_allow_html=True)

# =========================================================
# DISPLAY CONTEXT PANEL
# =========================================================
if query:
    st.markdown("---")
    st.subheader("📄 Retrieved Context")

    for i, r in enumerate(results):
        with st.expander(f"Chunk {i+1} | Doc {r['doc_id']}"):
            st.write(r["text"])
            if "score" in r:
                st.caption(f"Similarity score: {r['score']:.4f}")
            if "rerank_score" in r:
                st.caption(f"Rerank score: {r['rerank_score']:.4f}")