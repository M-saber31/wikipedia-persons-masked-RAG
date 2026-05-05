import streamlit as st

st.title("Import Diagnostic")

try:
    import chromadb
    st.success("chromadb OK")
except Exception as e:
    st.error(f"chromadb FAILED: {e}")

try:
    from llama_index.llms.ollama import Ollama
    st.success("llama_index Ollama OK")
except Exception as e:
    st.error(f"llama_index Ollama FAILED: {e}")

try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    st.success("HuggingFaceEmbedding OK")
except Exception as e:
    st.error(f"HuggingFaceEmbedding FAILED: {e}")

try:
    from llama_index.vector_stores.chroma import ChromaVectorStore
    st.success("ChromaVectorStore OK")
except Exception as e:
    st.error(f"ChromaVectorStore FAILED: {e}")

try:
    from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader, Settings
    st.success("llama_index core OK")
except Exception as e:
    st.error(f"llama_index core FAILED: {e}")

st.write("All imports attempted.")
