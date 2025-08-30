# app.py — StudyMate with Google Gemini API + Extra Functions + Styled Background
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import os
from dotenv import load_dotenv
from io import BytesIO
import re

# Load .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.warning("⚠ Set GOOGLE_API_KEY in a .env file or your environment.")
genai.configure(api_key=GOOGLE_API_KEY)

# ---- Gemini model ----
model_gemini = genai.GenerativeModel("gemini-1.5-flash")

# Streamlit settings + Styling
st.set_page_config(page_title="StudyMate — PDF Q&A with Gemini", layout="wide")
st.markdown("""
<style>

/* Headings in blue with subtle black overlay */
h1, h2, h3 {
    color: #2E86AB;
    background-color: rgba(0,0,0,0.5);
    padding: 5px 10px;
    border-radius: 5px;
}

/* Buttons */
div.stButton > button {
    background-color: #2E86AB;
    color: white;
    border-radius: 10px;
    padding: 0.6em 1.2em;
    font-size: 1em;
    font-weight: bold;
}
div.stButton > button:hover {
    background-color: #1B4F72;
    color: #f1f1f1;
}

/* Input box */
.stTextInput input {
    border: 2px solid #2E86AB;
    border-radius: 8px;
}

/* Answer / Notes / Quiz / Chunks overlay styling */
.text-overlay {
    background-color: rgba(0,0,0,0.5);
    color: white;
    padding: 12px;
    border-radius: 8px;
    margin: 5px 0;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.text-overlay:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.5);
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1 style='color:#1B4F72; text-align:center;' class='text-overlay'> 📚 StudyMate — PDF Q&A </h1>",
    unsafe_allow_html=True
)


# Cache embedding model
@st.cache_resource(show_spinner=False)
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")
embed_model = load_embed_model()

# ---- Helpers ----
def extract_text_from_pdf(uploaded_file) -> str:
    uploaded_file.seek(0)
    raw = uploaded_file.read()
    reader = PdfReader(BytesIO(raw))
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
    return "\n".join(text_parts)

def chunk_text(text, max_chars=1000, overlap=200):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) <= max_chars:
            current += " " + sent
        else:
            if current:
                chunks.append(current.strip())
            if len(sent) > max_chars:
                for i in range(0, len(sent), max_chars - overlap):
                    chunks.append(sent[i:i + max_chars].strip())
                current = ""
            else:
                current = sent
    if current:
        chunks.append(current.strip())
    return chunks

def embed_texts(texts):
    embeddings = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms==0] = 1e-9
    embeddings = embeddings / norms
    return embeddings

def retrieve(query, embeddings, chunks, top_k=4):
    query_embedding = embed_model.encode([query], convert_to_numpy=True, show_progress_bar=False)
    similarities = cosine_similarity(query_embedding, embeddings)
    top_indices = similarities[0].argsort()[-top_k:][::-1]
    return [(chunks[i], similarities[0][i]) for i in top_indices]

def ask_gemini(prompt):
    response = model_gemini.generate_content(prompt)
    return response.text if response else "⚠ No response from Gemini."

# ---- File upload ----
uploaded_file = st.file_uploader("📂 Upload a PDF", type=["pdf"])
if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    chunks = chunk_text(text)
    embeddings = embed_texts(chunks)

    st.success("✅ PDF processed successfully!")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # User Query
    query = st.text_input("🔎 Ask a question about your PDF:")
    if query:
        retrieved_chunks = retrieve(query, embeddings, chunks)
        context = "\n".join([r[0] for r in retrieved_chunks])
        prompt = f"Answer the question based on the following PDF context:\n\n{context}\n\nQuestion: {query}"
        answer = ask_gemini(prompt)

        # Save to history
        st.session_state["chat_history"].append({"q": query, "a": answer})

        st.subheader("💡 Answer")
        st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)

    # ---- Extra Functions ----
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("📌 Summarize PDF"):
            all_text = " ".join(chunks)
            summary = ask_gemini(f"Summarize this PDF in concise points:\n\n{all_text[:5000]}")
            st.subheader("📌 Summary")
            st.markdown(f"<div class='answer-box'>{summary}</div>", unsafe_allow_html=True)

    with col2:
        if st.button("🗒 Separate Notes"):
            all_text = " ".join(chunks)
            notes = ask_gemini(f"Extract bullet-point study notes from this PDF:\n\n{all_text[:5000]}")
            st.subheader("🗒 Notes")
            st.markdown(f"<div class='answer-box'>{notes}</div>", unsafe_allow_html=True)

    with col3:
        if st.button("📝 Generate Quiz from PDF"):
            quiz_chunks = retrieve("Generate quiz questions", embeddings, chunks)
            quiz_text = ask_gemini(f"""
                Create 5 multiple-choice quiz questions with options and answers from the following text:
                { ' '.join([q[0] for q in quiz_chunks]) }
            """)
            st.subheader("📝 Quiz")
            st.markdown(f"<div class='answer-box'>{quiz_text}</div>", unsafe_allow_html=True)

    with col4:
        if st.button("📖 Retrieve from PDF"):
            if query:
                retrieved_chunks = retrieve(query, embeddings, chunks, top_k=5)
                st.subheader("📖 Retrieved Chunks")
                for i, (chunk, score) in enumerate(retrieved_chunks, 1):
                    st.markdown(f"<div class='answer-box'><b>Score:</b> {score:.2f}<br>{chunk}</div>", unsafe_allow_html=True)
            else:
                st.warning("⚠ Please enter a query above to retrieve from PDF.")