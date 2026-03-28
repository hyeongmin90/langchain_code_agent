import json
import os
from typing import cast
import numpy as np
from redis import Redis
from langchain_openai import OpenAIEmbeddings

CACHE_KEY_PREFIX = "rag:semantic:"
DEFAULT_THRESHOLD = 0.92
DEFAULT_TTL = 60 * 60 * 24 * 7  # 7일


class SemanticCache:
    client: Redis
    embeddings: OpenAIEmbeddings
    threshold: float
    available: bool

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.client = Redis.from_url(redis_url, decode_responses=False)  # type: ignore[arg-type]
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.threshold = threshold
        self.available = self._check_connection()

    def _check_connection(self) -> bool:
        try:
            self.client.ping()
            return True
        except Exception:
            return False

    def _embed(self, text: str) -> list[float]:
        return self.embeddings.embed_query(text)

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        a_arr = np.array(a, dtype=np.float32)
        b_arr = np.array(b, dtype=np.float32)
        denom = float(np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
        if denom == 0:
            return 0.0
        return float(np.dot(a_arr, b_arr)) / denom

    def get(self, question: str) -> str | None:
        """유사 질문의 캐시된 답변을 반환합니다. 없으면 None."""
        if not self.available:
            return None

        try:
            query_emb = self._embed(question)
            keys = cast(list[bytes], self.client.keys(f"{CACHE_KEY_PREFIX}*"))

            best_score = -1.0
            best_answer = None

            for key in keys:
                key_str = key.decode("utf-8") if isinstance(key, bytes) else str(key)
                data = cast(dict[bytes, bytes], self.client.hgetall(key_str))
                if not data or b"embedding" not in data or b"answer" not in data:
                    continue

                cached_emb: list[float] = json.loads(data[b"embedding"])
                score = self._cosine_similarity(query_emb, cached_emb)

                if score > best_score:
                    best_score = score
                    best_answer = data[b"answer"].decode("utf-8")

            if best_score >= self.threshold:
                return best_answer
            return None

        except Exception:
            return None

    def set(self, question: str, answer: str) -> None:
        """질문과 답변을 임베딩과 함께 Redis에 저장합니다."""
        if not self.available:
            return

        try:
            emb = self._embed(question)
            key = f"{CACHE_KEY_PREFIX}{hash(question) & 0xFFFFFFFFFFFFFFFF}"
            self.client.hset(key, mapping={  # type: ignore[arg-type]
                "question": question,
                "embedding": json.dumps(emb),
                "answer": answer,
            })
            self.client.expire(key, DEFAULT_TTL)
        except Exception:
            pass

    def clear(self) -> int:
        """모든 캐시 항목을 삭제합니다. 삭제된 키 수를 반환합니다."""
        if not self.available:
            return 0

        keys = cast(list[bytes], self.client.keys(f"{CACHE_KEY_PREFIX}*"))
        if keys:
            return cast(int, self.client.delete(*keys))
        return 0
