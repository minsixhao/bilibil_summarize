# coding: utf-8
from typing import List, Dict
from collections import defaultdict

from langchain.docstore.document import Document
from langchain.storage import LocalFileStore
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.indexes import SQLRecordManager, index
from langchain.vectorstores import Chroma
from langchain.indexes._api import _batch
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.manager import Callbacks




def get_record_manager(namespace: str = "bilibili") -> SQLRecordManager:
    return SQLRecordManager(
        f"chroma/{namespace}", db_url="sqlite:///bilibili_record_manager_cache.sql"
    )


def clear_vectorstore(collection_name: str = "bilibili") -> None:
    record_manager = get_record_manager(collection_name)
    vectorstore = get_vectorstore(collection_name)

    index([], record_manager, vectorstore, cleanup="full", source_id_key="source")


def get_vectorstore(collection_name: str = "bilibili") -> Chroma:
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=get_cached_embedder(),
        collection_name=collection_name)

    return vectorstore

def get_cached_embedder() -> CacheBackedEmbeddings:
    fs = LocalFileStore("./.cache/embeddings")
    underlying_embeddings = OpenAIEmbeddings()

    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings, fs, namespace=underlying_embeddings.model
    )
    return cached_embedder


def bilibli_index(docs: List[Document], show_progress: bool = True) -> Dict:
    info = defaultdict(int)

    record_manager = get_record_manager("bilibili")
    vectorstore = get_vectorstore("bilibili")

    pbar = None
    if show_progress:
        from tqdm import tqdm
        pbar = tqdm(total=len(docs))

    for docs in _batch(100, docs):
        result = index(
            docs,
            record_manager,
            vectorstore,
            cleanup=None,
            # cleanup="full",
            source_id_key="source",
        )
        for k, v in result.items():
            info[k] += v

        if pbar:
            pbar.update(len(docs))

    if pbar:
        pbar.close()

    return dict(info)