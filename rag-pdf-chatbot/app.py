import os
import time
import hashlib
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
PINECONE_CLOUD = os.environ.get("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.environ.get("PINECONE_REGION", "us-east-1")
HF_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN", None)

INDEX_NAME = "rag-pdf-chatbot"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

pc = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
    )
    time.sleep(5)

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"token": HF_TOKEN} if HF_TOKEN else {}
)

def get_namespace_from_file(file_bytes, filename):
    content_hash = hashlib.md5(file_bytes).hexdigest()
    return f"{filename}_{content_hash}"

def process_pdf(uploaded_file):
    temp_path = f"/tmp/{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader(temp_path)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = text_splitter.split_documents(pages)

    file_bytes = uploaded_file.getvalue()
    namespace = get_namespace_from_file(file_bytes, uploaded_file.name)

    index = pc.Index(INDEX_NAME)
    stats = index.describe_index_stats()
    if namespace not in stats.get("namespaces", {}):
        PineconeVectorStore.from_documents(
            documents=docs,
            embedding=embeddings,
            index_name=INDEX_NAME,
            namespace=namespace
        )
        time.sleep(3)

    return namespace

def get_rag_chain(namespace):
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        namespace=namespace
    )
    retriever = vectorstore.as_retriever()

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama3-70b-8192",
        temperature=0
    )

    system_prompt = (
        "You are a helpful assistant. Answer the question based only on the provided PDF content. "
        "If you cannot find the answer, say so.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return rag_chain

st.set_page_config(page_title="PDF RAG Chatbot")
st.title("Chat with your PDF")
st.write("Upload a PDF and ask questions about its content.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "namespace" not in st.session_state:
    st.session_state.namespace = None

uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        try:
            namespace = process_pdf(uploaded_file)
            st.session_state.namespace = namespace
            st.success("PDF processed successfully! Ask your questions below.")
        except Exception as e:
            st.error(f"Failed to process PDF: {e}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.namespace:
    user_input = st.chat_input("Ask a question about the PDF...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    rag_chain = get_rag_chain(st.session_state.namespace)
                    result = rag_chain.invoke({"input": user_input})
                    answer = result["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
else:
    st.info("Upload a PDF to start chatting.")
