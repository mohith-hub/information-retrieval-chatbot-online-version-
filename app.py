import streamlit as st
import os
import logging
import warnings

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

# -------------------------------------------------
# Embeddings
# -------------------------------------------------
class LocalSentenceTransformerEmbeddings:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text]).tolist()[0]

# -------------------------------------------------
# Load PDFs
# -------------------------------------------------
def load_pdfs():
    pdf_dir = "missile_pdfs"
    os.makedirs(pdf_dir, exist_ok=True)

    pdfs = [
        os.path.join(pdf_dir, f)
        for f in os.listdir(pdf_dir)
        if f.endswith(".pdf")
    ]

    return pdfs

# -------------------------------------------------
# Vector Store
# -------------------------------------------------
@st.cache_resource
def setup_vector_store():
    embeddings = LocalSentenceTransformerEmbeddings()
    persist_dir = "chroma_db"

    pdfs = load_pdfs()
    docs = []

    for pdf in pdfs:
        loader = PyMuPDFLoader(pdf)
        docs.extend(loader.load())

    if not docs:
        st.warning("No PDFs found. Please add PDFs to missile_pdfs/")
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    return vector_store.as_retriever(search_kwargs={"k": 3})

# -------------------------------------------------
# LLM
# -------------------------------------------------
@st.cache_resource
def get_llm():
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY missing in Streamlit secrets")
        st.stop()

    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant"
    )

# -------------------------------------------------
# Prompt
# -------------------------------------------------
prompt = PromptTemplate(
    template="""
Answer ONLY using the context below.
If the answer is not present, say:
"I can only answer based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)

# -------------------------------------------------
# QA Chain
# -------------------------------------------------
@st.cache_resource
def init_qa():
    retriever = setup_vector_store()
    if retriever is None:
        return None

    llm = get_llm()

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )

# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("🚀 DRDO Missile Systems Chatbot")

qa = init_qa()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask about DRDO missile systems"):
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if qa:
                answer = qa.run(question)
            else:
                answer = "Please upload PDFs to missile_pdfs/"
        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()
