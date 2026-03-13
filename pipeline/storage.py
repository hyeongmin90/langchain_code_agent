import threading

from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

PERSIST_DIRECTORY = "./chroma_db"

_vectorstore_lock = threading.Lock()
_vectorstores: dict = {}

def get_vectorstore(collection_name: str = "spring_docs") -> Chroma:
    """Chroma vectorstore Singleton을 반환합니다. Thread-safe."""
    global _vectorstores

    if collection_name not in _vectorstores:
        with _vectorstore_lock:
            if collection_name not in _vectorstores:
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                _vectorstores[collection_name] = Chroma(
                    collection_name=collection_name,
                    embedding_function=embeddings,
                    persist_directory=PERSIST_DIRECTORY,
                )

    return _vectorstores[collection_name]


def add_documents(documents: list, collection_name: str = "spring_docs") -> None:
    """Document 리스트를 vectorstore에 upsert합니다 (source 기준 중복 제거)."""
    if not documents:
        tqdm.write("No documents to add.")
        return

    vectorstore = get_vectorstore(collection_name)
    tqdm.write(f"Adding {len(documents)} documents to ChromaDB ({collection_name})...")

    url_link = documents[0].metadata["source"]
    result = vectorstore.get(where={"source": url_link})
    if result["ids"]:
        vectorstore.delete(ids=result["ids"])

    ids = [doc.metadata["chunk_id"] for doc in documents]
    vectorstore.add_documents(documents=documents, ids=ids)
    tqdm.write("Documents added successfully.")



