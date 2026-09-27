import streamlit as st
from dotenv import load_dotenv
from hashlib import sha256
import os
from pathlib import Path
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
#from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
 
from Ingest import COLLECTION, DB_DIR, add_uploaded_document, get_embeddings
 
load_dotenv()  # reads GROQ_API_KEY from .env
groq_api_key = st.secrets["GROQ_API_KEY"]
 
st.set_page_config(page_title="Climate Chatbot", page_icon="🌍")
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(rgba(8, 32, 45, 0.88), rgba(8, 32, 45, 0.94)),
            url("https://images.unsplash.com/photo-1534088568595-a066f410bcda?auto=format&fit=crop&w=2200&q=85");
        background-size: cover;
        background-attachment: fixed;
    }
    [data-testid="stSidebar"] {
        background: rgba(7, 27, 39, 0.96);
    }
    .hero {
        padding: 1.6rem 0 1rem;
    }
    .hero h1 {
        color: #f4fbff;
        font-size: 2.5rem;
        margin-bottom: 0.25rem;
    }
    .hero p {
        color: #b8dce4;
        font-size: 1.05rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="hero">
        <h1>🌍 Climate Compass</h1>
        <p>Ask questions and explore insights from your climate knowledge base 🌱</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("📚 Knowledge base")
    st.caption("Add a climate document to expand the chatbot's sources.")
    uploaded_file = st.file_uploader(
        "Upload a PDF or CSV",
        type=["pdf", "csv"],
        accept_multiple_files=False,
    )

    if uploaded_file is not None:
        upload_bytes = uploaded_file.getvalue()
        upload_key = f"{uploaded_file.name}:{sha256(upload_bytes).hexdigest()}"
        processed_uploads = st.session_state.setdefault("processed_uploads", set())

        if upload_key not in processed_uploads:
            upload_dir = Path(__file__).resolve().parent.parent / "data" / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            saved_path = upload_dir / Path(uploaded_file.name).name
            saved_path.write_bytes(upload_bytes)

            try:
                with st.spinner("Indexing your document... 📖"):
                    chunk_count = add_uploaded_document(saved_path)
                processed_uploads.add(upload_key)
                st.cache_resource.clear()
                st.success(f"Added {chunk_count} chunks to the knowledge base! ✅")
                st.rerun()
            except Exception as error:
                st.error(f"Could not index this document: {error}")
 
 
@st.cache_resource
def build_chain():
    vectordb = Chroma(
        persist_directory=DB_DIR,
        embedding_function=get_embeddings(),
        collection_name=COLLECTION,
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, groq_api_key=groq_api_key)
 
    # Rewrites follow-up questions into standalone ones using chat history
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite the latest user question as a standalone question "
                   "using the chat history. Do not answer it."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_retriever = create_history_aware_retriever(llm, retriever, rewrite_prompt)
 
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You answer questions about climate change using ONLY the "
                   "context below. If the answer isn't in the context, say you "
                   "don't know.\n\nContext:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_retriever, qa_chain)
 
 
chain = build_chain()
 
if "messages" not in st.session_state:
    st.session_state.messages = []
 
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
 
if question := st.chat_input("Ask about your climate documents..."):
    history = [
        HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
        for m in st.session_state.messages
    ]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
 
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = chain.invoke({"input": question, "chat_history": history})
        st.markdown(result["answer"])
        with st.expander("Sources"):
            for d in result["context"]:
                page = d.metadata.get("page")
                label = d.metadata.get("source", "unknown")
                st.caption(f"{label}" + (f" (page {page + 1})" if page is not None else ""))
                st.write(d.page_content[:300] + "...")
 
    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})