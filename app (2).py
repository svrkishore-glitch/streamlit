import os
import re
import glob
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

st.set_page_config(page_title="Zyro HR Help Desk", page_icon="💬", layout="wide")

st.title("Zyro Dynamics HR Help Desk")
st.caption("RAG chatbot for HR policy questions")

LLM_MODEL = st.sidebar.text_input("Groq model", value="llama-3.1-8b-instant")
CORPUS_PATH = st.sidebar.text_input("Corpus folder", value="zyro-dynamics-hr-corpus")

REFUSAL_MESSAGE = "I can only answer HR-related questions from Zyro Dynamics policy documents."
OOS_KEYWORDS = {
    "weather", "news", "sports", "movie", "song", "lyrics", "recipe",
    "bitcoin", "crypto", "stock", "share price", "game", "joke",
    "python", "java", "code", "programming", "holiday destinations",
    "capital of", "prime minister", "president", "celebrity"
}

def is_likely_out_of_scope(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in OOS_KEYWORDS)

@st.cache_resource
def build_pipeline():
    if not os.path.exists(CORPUS_PATH):
        raise FileNotFoundError(f"Corpus folder not found: {CORPUS_PATH}")

    loader = PyPDFDirectoryLoader(CORPUS_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.6},
    )

    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0.1,
        max_tokens=256,
        api_key=os.getenv("GROQ_API_KEY", st.secrets.get("GROQ_API_KEY", "")),
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are Zyro Dynamics' HR Help Desk assistant.

Use ONLY the context below to answer the question.
If the answer is not explicitly supported by the context, say:
"I can only answer HR-related questions from Zyro Dynamics policy documents."

Keep the answer short, direct, and policy-grounded.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    return retriever, llm, prompt, vectorstore

def format_docs(docs):
    parts = []
    for i, doc in enumerate(docs, 1):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page", "NA")
        parts.append(f"[{i}] {source} | page {page}\n{doc.page_content}")
    return "\n\n".join(parts)

def best_relevance(vectorstore, question: str) -> float:
    try:
        scored = vectorstore.similarity_search_with_relevance_scores(question, k=1)
        if scored:
            return float(scored[0][1])
    except Exception:
        pass
    return 0.0

def answer_question(question: str):
    retriever, llm, prompt, vectorstore = build_pipeline()

    if is_likely_out_of_scope(question):
        return REFUSAL_MESSAGE, []

    if best_relevance(vectorstore, question) < 0.20:
        return REFUSAL_MESSAGE, []

    docs = retriever.invoke(question)
    context = format_docs(docs)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question}).strip()
    return answer, docs

question = st.text_input("Ask an HR question", placeholder="Example: What is the leave policy for sick leave?")

col1, col2 = st.columns([1, 1])
with col1:
    ask = st.button("Get Answer", use_container_width=True)
with col2:
    clear = st.button("Clear", use_container_width=True)

if clear:
    st.session_state["question"] = ""
    st.rerun()

if ask and question.strip():
    try:
        answer, docs = answer_question(question.strip())
        st.subheader("Answer")
        st.write(answer)

        if docs:
            with st.expander("Sources"):
                for i, doc in enumerate(docs, 1):
                    st.markdown(f"**Source {i}:** `{os.path.basename(doc.metadata.get('source', 'unknown'))}`")
                    st.caption(f"Page: {doc.metadata.get('page', 'NA')}")
                    st.write(doc.page_content[:1200] + ("..." if len(doc.page_content) > 1200 else ""))
    except Exception as e:
        st.error(str(e))

st.sidebar.markdown("### Setup")
st.sidebar.write("Put your Groq API key in Streamlit secrets or environment variables.")
st.sidebar.write("Keep the corpus folder next to `app.py` when deploying.")
