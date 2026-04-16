# app_ollama.py — Streamlit RAG Chatbot using Ollama embeddings + FAISS + Groq

import os
import traceback

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(page_title="AI RAG Chatbot", layout="wide")

# ---------------- ENV ----------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY missing in your .env file.")
    st.stop()

# ---------------- DATA LOADING ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("rag_dataset.csv")

    # Remove columns we don't need if they exist
    for c in ["_id", "id", "ups", "subreddit", "created_utc", "num_comments", "url", "response"]:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    # Ensure the text columns exist and have no NaNs
    for c in ["title", "selftext", "comments"]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("")

    df["combined_text"] = df["title"] + " " + df["selftext"] + " " + df["comments"]
    return df


df = load_data()
docs = [Document(page_content=t) for t in df["combined_text"].tolist()]

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = splitter.split_documents(docs)

# ---------------- EMBEDDINGS + FAISS (Ollama) ----------------
INDEX_DIR = "faiss_index"

# Requires:
#   ollama serve
#   ollama pull nomic-embed-text
embeddings = OllamaEmbeddings(model="nomic-embed-text")


def build_and_save_index():
    """Create a new FAISS index from documents and save it locally."""
    st.sidebar.info("🧱 Building new FAISS index using Ollama embeddings...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(INDEX_DIR)
    st.sidebar.success("✅ FAISS index built and saved.")
    return vectorstore


# Try to use existing FAISS index; if it’s broken or missing, rebuild it.
try:
    vectorstore = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    st.sidebar.success("✅ Loaded FAISS index successfully.")
except Exception as e:
    st.sidebar.warning(f"⚠️ Could not load FAISS index: {e}")

    # If you think the existing index is corrupted, uncomment this to force rebuild:
    # import shutil
    # if os.path.exists(INDEX_DIR):
    #     shutil.rmtree(INDEX_DIR)

    vectorstore = build_and_save_index()

retriever = vectorstore.as_retriever()

# ---------------- LLM (Groq) ----------------
llm = ChatGroq(model="qwen/qwen3-32b")

# If you later want to use an Ollama LLM instead of Groq:
# from langchain_community.chat_models import ChatOllama
# llm = ChatOllama(model="deepseek-r1")

# ---------------- RAG PIPELINE ----------------
system_prompt = (
    "You are an AI assistant for entrepreneurs. Use the retrieved context as your main source, "
    "but you can also use your own general knowledge if the context is not sufficient. "
    "Always provide practical, insightful, and actionable answers.\n\nContext:\n{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}"),
    ]
)


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


def get_response(q: str) -> str:
    try:
        return rag_chain.invoke(q)
    except Exception:
        st.error("⚠️ Inference error:\n\n" + traceback.format_exc())
        return "Sorry, I hit an error while generating the answer."


# ---------------- UI ----------------
st.title("🔍 AI-Powered RAG Chatbot for Startups")

query = st.chat_input("Ask me anything about startups or AI...")

if query:
    with st.chat_message("user"):
        st.markdown(query)

    answer = get_response(query)

    with st.chat_message("assistant"):
        st.markdown(answer)