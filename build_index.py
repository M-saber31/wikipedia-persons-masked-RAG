import os
import pickle
import chromadb
from tqdm import tqdm
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    SimpleDirectoryReader,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter

INDEX_DIR = 'wiki_rag'
DATA_DIR = 'wiki_data_filtered'
NODES_FILE = 'nodes.pkl'
EMBED_MODEL = 'nomic-embed-text'
BATCH_SIZE = 50
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

if __name__ == '__main__':
    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)

    db = chromadb.PersistentClient(path=INDEX_DIR)
    collection = db.get_or_create_collection("wiki_pages")

    if collection.count() > 0:
        print(f"Index already exists ({collection.count()} vectors). Delete '{INDEX_DIR}/' to rebuild.")
        exit(0)

    files = sorted([
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.endswith('.txt')
    ])

    if not files:
        print(f"No .txt files found in '{DATA_DIR}/'. Run export_to_txt.py first.")
        exit(1)

    total = len(files)
    batches = [files[i:i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]

    print(f"Files   : {total}")
    print(f"Batches : {len(batches)}  (batch size = {BATCH_SIZE})")
    print(f"Model   : {EMBED_MODEL}  (GPU)")
    print()

    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    all_nodes = []

    with tqdm(total=total, unit="file", dynamic_ncols=True) as bar:
        for batch in batches:
            docs = SimpleDirectoryReader(input_files=batch).load_data()
            nodes = splitter.get_nodes_from_documents(docs)
            all_nodes.extend(nodes)
            VectorStoreIndex(nodes, storage_context=storage_context, show_progress=False)
            bar.update(len(batch))
            bar.set_postfix(vectors=collection.count())

    print(f"\nSaving {len(all_nodes)} nodes to '{NODES_FILE}' for BM25...")
    with open(NODES_FILE, 'wb') as f:
        pickle.dump(all_nodes, f)

    print(f"Done! {collection.count()} vectors in ChromaDB, {len(all_nodes)} nodes saved.")
    print("Run the app with:  streamlit run main.py")
