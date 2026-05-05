import nest_asyncio
nest_asyncio.apply()

import re
import streamlit as st
import chromadb
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.core.postprocessor import SimilarityPostprocessor, LLMRerank

INDEX_DIR = 'wiki_rag'
EMBED_MODEL = 'nomic-embed-text'
OLLAMA_MODEL = 'qwen2.5:3b'

QA_PROMPT = PromptTemplate(
    "You are a precise assistant answering questions about football players.\n"
    "Answer using ONLY what is explicitly stated in the context below.\n"
    "Rules:\n"
    "- Do not infer, assume, or add any information not directly written in the context.\n"
    "- If the context is insufficient to answer fully, say so clearly.\n"
    "- 'Survived' means the person did NOT die from that event — do not include them.\n"
    "- Only include a player as having died if the context explicitly confirms their death AND the cause matches the question.\n\n"
    "Context:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    "Question: {query_str}\n\n"
    "Answer: "
)

RERANK_PROMPT = PromptTemplate(
    "You are ranking documents by relevance to a question.\n"
    "Rate each document's relevance on a scale from 1 to 10.\n\n"
    "Documents:\n"
    "{context_str}\n\n"
    "Question: {query_str}\n\n"
    "Reply ONLY with lines in this exact format, no other text:\n"
    "Doc: 1, Relevance: 7\n"
    "Doc: 2, Relevance: 3\n"
    "(one line per document, numbers only)\n\n"
    "Answer:\n"
)


def robust_rerank_parser(answer: str, num_choices: int):
    choices, relevances = [], []

    pattern = re.findall(
        r'[Dd]oc(?:ument)?[\s:]*(\d+)[\s,]+[Rr]elevance[\s:]*(\d+)',
        answer
    )
    if pattern:
        for doc_str, rel_str in pattern:
            n = int(doc_str)
            if 1 <= n <= num_choices:
                choices.append(n)
                relevances.append(float(rel_str))
        if choices:
            return choices, relevances

    for line in answer.strip().split('\n'):
        nums = re.findall(r'\d+', line)
        if len(nums) >= 2:
            n, score = int(nums[0]), float(nums[1])
            if 1 <= n <= num_choices and 1 <= score <= 10:
                choices.append(n)
                relevances.append(score)

    # Fallback: if parsing fails entirely, return all docs with neutral score
    # so the pipeline always has chunks to synthesize from
    if not choices:
        choices = list(range(1, num_choices + 1))
        relevances = [5.0] * num_choices

    return choices, relevances


@st.cache_resource
def get_query_engine():
    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)
    Settings.llm = Ollama(model=OLLAMA_MODEL, request_timeout=300.0, context_window=4096, temperature=0)

    db = chromadb.PersistentClient(path=INDEX_DIR)
    collection = db.get_or_create_collection("wiki_pages")
    if collection.count() == 0:
        st.error("Index not found. Run `python build_index.py` first.")
        st.stop()

    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(vector_store)

    reranker = LLMRerank(
        choice_select_prompt=RERANK_PROMPT,
        parse_choice_select_answer_fn=robust_rerank_parser,
        choice_batch_size=5,
        top_n=2,
    )

    return index.as_query_engine(
        similarity_top_k=10,
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=0.50),
            reranker,
        ],
        text_qa_template=QA_PROMPT,
    )


def main():
    st.title('Wikipedia RAG — Football Players')

    question = st.text_input('Ask a question about a football player')

    if st.button('Submit') and question:
        with st.spinner('Thinking...'):
            qa = get_query_engine()
            response = qa.query(question)

        st.subheader('Answer')
        st.write(response.response)

        st.subheader('Retrieved context')
        for src in response.source_nodes:
            score = round(src.score, 3) if src.score is not None else 'N/A'
            st.caption(f"**{src.node.metadata.get('file_name', '')}** — score: {score}")
            st.markdown(src.node.get_content())
            st.divider()


if __name__ == '__main__':
    main()
