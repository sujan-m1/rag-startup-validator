# app.py
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# LangChain imports
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ---------------- ENV ----------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY missing in .env")
    st.stop()

INDEX_DIR = "faiss_index"

# ---------------- DATA + INDEX ----------------
@st.cache_resource
def get_retriever():
    # 1) Load dataset
    df = pd.read_csv("rag_dataset.csv")
    df["combined_text"] = (
        df["selftext"].fillna("") + " " +
        df["title"].fillna("") + " " +
        df["comments"].fillna("")
    )
    docs = [Document(page_content=text) for text in df["combined_text"].tolist()]

    # 2) Split documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = splitter.split_documents(docs)

    # 3) Embeddings (same model every time!)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4) Try to LOAD existing FAISS index, otherwise BUILD + SAVE
    try:
        vectorstore = FAISS.load_local(
            INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        st.sidebar.success("✅ Loaded FAISS index from disk.")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Could not load FAISS index: {e}")
        st.sidebar.info("Building new FAISS index from CSV...")
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(INDEX_DIR)
        st.sidebar.success("✅ FAISS index built and saved to disk.")

    return vectorstore.as_retriever()


retriever = get_retriever()

# ---------------- LLM + RAG CHAIN ----------------
chat = ChatGroq(model="llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_messages([
    ("system", "Use the context to answer. If unsure, say so.\n\nContext:\n{context}"),
    ("human", "{question}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | chat
    | StrOutputParser()
)

def get_response(user_input: str) -> str:
    return rag_chain.invoke(user_input)

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("Ask me anything about Startups")

if "messages" not in st.session_state:
    st.session_state.messages = []

# show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# new message
if user_msg := st.chat_input("Type your input here..."):
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.spinner("Thinking..."):
        response = get_response(user_msg)

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)