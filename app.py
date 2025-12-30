import streamlit as st
import os
import logging
import warnings

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

from sentence_transformers import SentenceTransformer

# -------------------------------------------------
# Basic setup
# -------------------------------------------------
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Chatbot System for information retrieval [online-version]",
    page_icon="🚀",
    layout="centered"
)

# -------------------------------------------------
# Embeddings
# -------------------------------------------------
class LocalSentenceTransformerEmbeddings:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text):
        return self.model.encode([text], show_progress_bar=False)[0].tolist()

# -------------------------------------------------
# Load PDFs
# -------------------------------------------------
def load_pdfs():
    pdf_dir = "missile_pdfs"
    os.makedirs(pdf_dir, exist_ok=True)

    pdfs = []
    for file in os.listdir(pdf_dir):
        if file.endswith(".pdf"):
            pdfs.append(os.path.join(pdf_dir, file))
    return pdfs

# -------------------------------------------------
# Vector store
# -------------------------------------------------
@st.cache_resource
def setup_vector_store():
    embeddings = LocalSentenceTransformerEmbeddings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    persist_dir = "chroma_db"
    pdfs = load_pdfs()

    if not pdfs:
        st.warning("No PDFs found. General questions will still work.")
        return None

    docs = []
    for pdf in pdfs:
        loader = PyMuPDFLoader(pdf)
        docs.extend(loader.load())

    chunks = splitter.split_documents(docs)

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    st.success("Vector store ready")
    return db.as_retriever(search_kwargs={"k": 3})

# -------------------------------------------------
# LLM (Groq)
# -------------------------------------------------
@st.cache_resource
def get_llm():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not set in Streamlit Secrets")
        st.stop()

    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant"
    )

# -------------------------------------------------
# Prompts
# -------------------------------------------------
rag_prompt = PromptTemplate(
    template="""
You are a DRDO missile systems expert.
Answer ONLY using the provided context.
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

fallback_prompt = PromptTemplate(
    template="""
Answer the following question using general knowledge.
Be concise and accurate.

Question:
{question}

Answer:
""",
    input_variables=["question"]
)

# -------------------------------------------------
# Response Logic (HYBRID)
# -------------------------------------------------
def get_bot_response(question, retriever, llm):
    # If no PDFs exist → direct general answer
    if retriever is None:
        return llm.invoke(
            fallback_prompt.format(question=question)
        ).content

    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = rag_prompt.format(
        context=context,
        question=question
    )

    response = llm.invoke(prompt).content

    # If PDF doesn't contain answer → fallback
    if "I can only answer based on the provided documents" in response:
        return llm.invoke(
            fallback_prompt.format(question=question)
        ).content

    return response

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.title("🚀 DRDO Missile Systems Chatbot")
st.write("Ask questions based on missile-related PDFs or general knowledge.")

with st.sidebar:
    st.header("Setup")
    if st.button("Rebuild Vector Store"):
        st.cache_resource.clear()
        st.rerun()
    st.info("Ensure PDFs exist in `missile_pdfs/` folder.")

if "messages" not in st.session_state:
    st.session_state.messages = []

retriever = setup_vector_store()
llm = get_llm()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask a question about missiles or general topics"):
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = get_bot_response(user_input, retriever, llm)
        st.markdown(reply)

    st.session_state.messages.append(
        {"role": "assistant", "content": reply}
    )

if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()
