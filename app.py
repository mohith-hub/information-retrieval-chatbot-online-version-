import streamlit as st
import os
from dotenv import load_dotenv
import logging
import time
import warnings

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.prompts import PromptTemplate

# -------------------------------------------------
# Basic setup
# -------------------------------------------------
warnings.filterwarnings("ignore")
load_dotenv()
logging.basicConfig(level=logging.INFO)

# -------------------------------------------------
# Embeddings
# -------------------------------------------------
class LocalSentenceTransformerEmbeddings:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        start = time.time()
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        logging.info(f"Embeddings loaded in {time.time() - start:.2f}s")

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text):
        return self.model.encode([text], show_progress_bar=False).tolist()[0]

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

    if not pdfs:
        st.warning("No PDFs found in missile_pdfs/")
    return pdfs

# -------------------------------------------------
# Text splitter
# -------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# -------------------------------------------------
# Vector store
# -------------------------------------------------
@st.cache_resource
def setup_vector_store():
    embeddings = LocalSentenceTransformerEmbeddings()
    persist_dir = "chroma_db"

    if os.path.exists(persist_dir):
         
        vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
    else:
        st.info("Building new Chroma DB...")
        docs = []
        for pdf in load_pdfs():
            loader = PyMuPDFLoader(pdf)
            docs.extend(loader.load())

        if docs:
            chunks = text_splitter.split_documents(docs)
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=persist_dir
            )
            vector_store.persist()
        else:
            vector_store = Chroma(
                embedding_function=embeddings,
                persist_directory=persist_dir
            )

    st.success("Vector store ready")
    return vector_store.as_retriever(search_kwargs={"k": 3})

# -------------------------------------------------
# LLM (Groq)
# -------------------------------------------------
@st.cache_resource
def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found. Add it to .env or Streamlit secrets.")
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
Use the following context to answer the question.
If the answer is not in the context, say:
"I can only answer based on the provided documents."

Context:
{context}

Question:
{input}

Answer:
""",
    input_variables=["context", "input"]
)

fallback_prompt = PromptTemplate(
    template="""
Answer the question using general knowledge.
If unsure, say you do not know.

Question:
{question}

Answer:
""",
    input_variables=["question"]
)

# -------------------------------------------------
# RAG Chain (LCEL – FINAL)
# -------------------------------------------------
@st.cache_resource
def initialize_qa_chain():
    retriever = setup_vector_store()
    llm = get_llm()

    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=rag_prompt
    )

    return create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=document_chain
    )

# -------------------------------------------------
# Response logic
# -------------------------------------------------
def get_bot_response(question, qa_chain):
    llm = get_llm()

    try:
        result = qa_chain.invoke({"input": question})
        answer = result.get("answer", "")

        if "I can only answer based on the provided documents" in answer:
            msg = llm.invoke(
                fallback_prompt.format(question=question)
            )
            return msg.content

        return answer

    except Exception as e:
        st.error(f"Error: {e}")
        msg = llm.invoke(
            fallback_prompt.format(question=question)
        )
        return msg.content


# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.title("🚀 DRDO Missile Systems Chatbot")
st.write("Ask questions based on missile-related PDFs.")

with st.sidebar:
    st.header("Setup")
    if st.button("Rebuild Vector Store"):
        st.cache_resource.clear()
        st.rerun()
    st.info("Add PDFs to missile_pdfs/ and redeploy.")

if "messages" not in st.session_state:
    st.session_state.messages = []

qa_chain = initialize_qa_chain()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask a question about missiles"):
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = get_bot_response(user_input, qa_chain)
        st.markdown(reply)

    st.session_state.messages.append(
        {"role": "assistant", "content": reply}
    )

if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()
