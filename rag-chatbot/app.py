# import streamlit as st
# from typing import List
# from langchain_community.document_loaders import PyMuPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
# from langchain_groq import ChatGroq
# from langchain.prompts import PromptTemplate
# import os
# from dotenv import load_dotenv
# import uuid
# from db import save_chat, get_chat_history

# # Assign a session ID if not already present
# if "session_id" not in st.session_state:
#     st.session_state.session_id = str(uuid.uuid4())


# # Load environment variables
# load_dotenv()

# # Initialize LLM
# llm = ChatGroq(
#     api_key=os.getenv("GROQ_API_KEY"),
#     model="openai/gpt-oss-120b",
#     temperature=0
# )


# # Initialize LLM
# llm = ChatGroq(
#     api_key=os.getenv("GROQ_API_KEY"),
#     model="openai/gpt-oss-120b",
#     temperature=0
# )


# # -----------------------------
# # Helpers
# # -----------------------------
# def load_pdfs(files: List[str]):
#     all_docs = []
#     for file in files:
#         reader = PyMuPDFLoader(file)
#         docs = reader.load()
#         all_docs.extend(docs)
#     return all_docs

# def split_docs(docs, chunk_size=800, chunk_overlap=150):
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap
#     )
#     return splitter.split_documents(docs)

# def create_vectorstore(chunks):
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     return FAISS.from_documents(chunks, embeddings)

# # -----------------------------
# # Custom system prompt
# # -----------------------------
# qa_prompt = PromptTemplate(
#     template="""
# You are a helpful assistant specialized in answering questions from documents.  
# Always use the retrieved context when answering.
# Do not answer anything from outside the provided. If the context is not relevant, politely say:  
# "I could not find that in the provided documents."

# Context:
# {context}

# Question:
# {question}

# Answer in a clear and structured way:
# """,
#     input_variables=["context", "question"]
# )


# # -----------------------------
# # Create chatbot with prompt
# # -----------------------------
# def create_chatbot(vectorstore):
#     retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

#     qa = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=retriever,
#         memory=memory,
#         verbose=True,
#         combine_docs_chain_kwargs={"prompt": qa_prompt}  # 👈 inject custom prompt
#     )
#     return qa

# # -----------------------------
# # Streamlit UI
# # -----------------------------
# st.set_page_config(page_title="RAG Chatbot", layout="wide")
# st.title("📑 RAG-based Chatbot with Groq")

# # Sidebar for file upload
# st.sidebar.header("Upload and Process Documents")
# uploaded_files = st.sidebar.file_uploader(
#     "Upload your PDF(s)", type=["pdf"], accept_multiple_files=True
# )

# if uploaded_files and st.sidebar.button("Process Documents"):
#     with st.spinner("Processing documents..."):
#         # Save uploaded files temporarily
#         file_paths = []
#         for uploaded_file in uploaded_files:
#             file_path = os.path.join("temp", uploaded_file.name)
#             os.makedirs("temp", exist_ok=True)
#             with open(file_path, "wb") as f:
#                 f.write(uploaded_file.getbuffer())
#             file_paths.append(file_path)

#         # Load → Split → Embed
#         docs = load_pdfs(file_paths)
#         chunks = split_docs(docs)
#         vectorstore = create_vectorstore(chunks)

#         # Create chatbot and store in session
#         st.session_state.chatbot = create_chatbot(vectorstore)
#         st.session_state.history = []  # reset chat history
#         st.sidebar.success("✅ Documents processed and embeddings created!")

# # -----------------------------
# # Chat Interface
# # -----------------------------
# if "chatbot" in st.session_state:
#     st.subheader("💬 Chat with your documents")

#     # Show chat history
#     if "history" not in st.session_state:
#         st.session_state.history = []

#     for role, message in st.session_state.history:
#         with st.chat_message(role):
#             st.markdown(message)

#     # Chat input
#     user_query = st.chat_input("Ask a question about the uploaded documents...")

#     if user_query:
#         with st.chat_message("user"):
#             st.markdown(user_query)
#         st.session_state.history.append(("user", user_query))

#         with st.spinner("Thinking..."):
#             bot_answer = st.session_state.chatbot.run(user_query)

#         with st.chat_message("assistant"):
#             st.markdown(bot_answer)
#         st.session_state.history.append(("assistant", bot_answer))

#         # Save Q&A to MongoDB
#         save_chat(
#             session_id=st.session_state.session_id,
#             user_query=user_query,
#             bot_answer=bot_answer,
#             source_files=[f.name for f in uploaded_files] if uploaded_files else []
#         )


# else:
#     st.info("👈 Upload and process documents from the sidebar to start chatting.")


import streamlit as st
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import uuid
from db import save_chat, get_chat_history
from azure.storage.blob import BlobServiceClient

# Assign a session ID if not already present
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="openai/gpt-oss-120b",
    temperature=0
)

# Initialize Azure Blob Service
blob_service_client = BlobServiceClient.from_connection_string(
    os.getenv("AZURE_STORAGE_CONNECTION_STRING")
)
container_name = os.getenv("AZURE_BLOB_CONTAINER")
container_client = blob_service_client.get_container_client(container_name)

# Create container if not exists
try:
    container_client.create_container()
except Exception:
    pass  # already exists


# -----------------------------
# Helpers
# -----------------------------
# def upload_file_to_blob(file_data, filename: str):
#     """Upload a file to Azure Blob Storage"""
#     blob_client = blob_service_client.get_blob_client(container=container_name, blob=filename)
#     blob_client.upload_blob(file_data, overwrite=True)
#     return filename  # store only filename, not path

def upload_file_to_blob(file_data, file_name):
    blob_client = container_client.get_blob_client(file_name)
    blob_client.upload_blob(file_data, overwrite=True)
    return file_name



def load_pdfs(file_names: List[str]):
    """Download PDFs from Azure Blob and load with PyMuPDFLoader"""
    all_docs = []
    os.makedirs("azure_docs", exist_ok=True)

    for blob_name in file_names:
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

        local_path = os.path.join("azure_docs", blob_name)
        with open(local_path, "wb") as f:
            f.write(blob_client.download_blob().readall())

        reader = PyMuPDFLoader(local_path)
        docs = reader.load()
        all_docs.extend(docs)

    return all_docs


def split_docs(docs, chunk_size=800, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)


# -----------------------------
# Custom system prompt
# -----------------------------
qa_prompt = PromptTemplate(
    template="""
You are a helpful assistant specialized in answering questions from documents.  
Always use the retrieved context when answering.
Do not answer anything from outside the provided. If the context is not relevant, politely say:  
"I could not find that in the provided documents."

Context:
{context}

Question:
{question}

Answer in a clear and structured way:
""",
    input_variables=["context", "question"]
)


# -----------------------------
# Create chatbot with prompt
# -----------------------------
def create_chatbot(vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt}  # 👈 inject custom prompt
    )
    return qa


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📑 RAG-based Chatbot with Groq + Azure Blob Storage")

# Sidebar for file upload
st.sidebar.header("Upload and Process Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload your PDF(s)", type=["pdf"], accept_multiple_files=True
)

if uploaded_files and st.sidebar.button("Process Documents"):
    with st.spinner("Processing documents..."):
        # Upload files to Azure Blob
        blob_file_names = []
        for uploaded_file in uploaded_files:
            # blob_name = upload_file_to_blob(uploaded_file.getbuffer(), uploaded_file.name)
            blob_name = upload_file_to_blob(bytes(uploaded_file.getbuffer()), uploaded_file.name)

            blob_file_names.append(blob_name)

        # Load → Split → Embed
        docs = load_pdfs(blob_file_names)
        chunks = split_docs(docs)
        vectorstore = create_vectorstore(chunks)

        # Create chatbot and store in session
        st.session_state.chatbot = create_chatbot(vectorstore)
        st.session_state.history = []  # reset chat history
        st.sidebar.success("✅ Documents uploaded to Azure Blob and embeddings created!")


# -----------------------------
# Chat Interface
# -----------------------------
if "chatbot" in st.session_state:
    st.subheader("💬 Chat with your documents")

    # Show chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    for role, message in st.session_state.history:
        with st.chat_message(role):
            st.markdown(message)

    # Chat input
    user_query = st.chat_input("Ask a question about the uploaded documents...")

    if user_query:
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.history.append(("user", user_query))

        with st.spinner("Thinking..."):
            bot_answer = st.session_state.chatbot.run(user_query)

        with st.chat_message("assistant"):
            st.markdown(bot_answer)
        st.session_state.history.append(("assistant", bot_answer))

        # Save Q&A to MongoDB
        save_chat(
            session_id=st.session_state.session_id,
            user_query=user_query,
            bot_answer=bot_answer,
            source_files=[f.name for f in uploaded_files] if uploaded_files else []
        )
else:
    st.info("👈 Upload and process documents from the sidebar to start chatting.")
