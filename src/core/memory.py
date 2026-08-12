import uuid
import logging
from datetime import datetime
from typing import Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import settings, CHROMA_DIR

# Suppress ChromaDB's noisy telemetry warnings
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


class ChunkType:
    RESEARCH     = "research"
    CODE         = "code"
    ANALYSIS     = "analysis"
    REPORT_DRAFT = "report_draft"
    CRITIQUE     = "critique"
    PLAN         = "plan"
    TOOL_OUTPUT  = "tool_output"


class AgentMemory:
    """
    Shared vector memory backed by ChromaDB.
    All agents use the same instance via get_memory() singleton.

    Improvement: ChromaDB telemetry warnings suppressed via logging config.
    """

    def __init__(self):
        self._client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"AgentMemory initialised — {self._collection.count()} existing chunks"
        )

    def store(self, content: str, agent_name: str, task_id: str,
              chunk_type: str, metadata: Optional[dict] = None,
              chunk_id: Optional[str] = None) -> str:
        if not content or not content.strip():
            logger.warning(f"[{agent_name}] Empty content, skipping store")
            return ""
        chunk_id = chunk_id or f"{agent_name}_{task_id}_{uuid.uuid4().hex[:8]}"
        full_metadata = {
            "agent_name": agent_name, "task_id": task_id,
            "chunk_type": chunk_type,
            "timestamp": datetime.utcnow().isoformat(),
            **(metadata or {}),
        }
        self._collection.upsert(ids=[chunk_id], documents=[content],
                                metadatas=[full_metadata])
        logger.debug(f"[memory] Stored chunk {chunk_id} ({chunk_type}, {len(content)} chars)")
        return chunk_id

    def store_many(self, chunks: list[dict]) -> list[str]:
        ids, documents, metadatas = [], [], []
        for chunk in chunks:
            content = chunk.get("content", "").strip()
            if not content:
                continue
            chunk_id = chunk.get("chunk_id") or (
                f"{chunk['agent_name']}_{chunk['task_id']}_{uuid.uuid4().hex[:8]}"
            )
            full_metadata = {
                "agent_name": chunk["agent_name"], "task_id": chunk["task_id"],
                "chunk_type": chunk["chunk_type"],
                "timestamp": datetime.utcnow().isoformat(),
                **(chunk.get("metadata") or {}),
            }
            ids.append(chunk_id)
            documents.append(content)
            metadatas.append(full_metadata)
        if ids:
            self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            logger.debug(f"[memory] Batch stored {len(ids)} chunks")
        return ids

    def retrieve(self, query: str, task_id: Optional[str] = None,
                 agent_name: Optional[str] = None, chunk_type: Optional[str] = None,
                 k: Optional[int] = None) -> list[dict]:
        k = k or settings.agent_memory_k
        conditions = []
        if task_id:
            conditions.append({"task_id": {"$eq": task_id}})
        if agent_name:
            conditions.append({"agent_name": {"$eq": agent_name}})
        if chunk_type:
            conditions.append({"chunk_type": {"$eq": chunk_type}})

        where = {}
        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}

        total = self._collection.count()
        if total == 0:
            return []

        query_kwargs = {
            "query_texts": [query],
            "n_results": min(k, total),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        results = self._collection.query(**query_kwargs)
        chunks = []
        if results["ids"] and results["ids"][0]:
            for i, cid in enumerate(results["ids"][0]):
                chunks.append({
                    "id": cid,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                })
        logger.debug(f"[memory] Retrieved {len(chunks)} chunks for query '{query[:50]}'")
        return chunks

    def get_task_summary(self, task_id: str) -> dict[str, list[dict]]:
        results = self._collection.get(
            where={"task_id": {"$eq": task_id}},
            include=["documents", "metadatas"],
        )
        grouped: dict[str, list[dict]] = {}
        if results["ids"]:
            for i, chunk_id in enumerate(results["ids"]):
                agent = results["metadatas"][i].get("agent_name", "unknown")
                if agent not in grouped:
                    grouped[agent] = []
                grouped[agent].append({
                    "id": chunk_id,
                    "content": results["documents"][i],
                    "metadata": results["metadatas"][i],
                })
        return grouped

    def clear_task(self, task_id: str) -> int:
        results = self._collection.get(
            where={"task_id": {"$eq": task_id}}, include=[],
        )
        ids = results.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
            logger.info(f"[memory] Cleared {len(ids)} chunks for task {task_id}")
        return len(ids)

    def clear_all(self):
        self._client.delete_collection(settings.chroma_collection)
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning("[memory] All memory cleared!")

    @property
    def count(self) -> int:
        return self._collection.count()


_memory_instance: Optional[AgentMemory] = None

def get_memory() -> AgentMemory:
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = AgentMemory()
    return _memory_instance
