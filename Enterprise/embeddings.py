from .deps import *
from .logger import StructuredLogger

class RealEmbeddingModel:
    """
    Uses sentence-transformers for real dense embeddings.
    Falls back to a random projection if library not installed.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if _ST_OK:
            self.model = SentenceTransformer(model_name)
            self.dim   = self.model.get_sentence_embedding_dimension()
            self._real = True
        else:
            # Fallback: deterministic random projection (not real embeddings)
            self._real = False
            self.dim   = 384
            rng = np.random.default_rng(42)
            self._proj = rng.standard_normal((10_000, self.dim)).astype(np.float32)

    def encode(self, texts: List[str]) -> np.ndarray:
        if self._real:
            return self.model.encode(texts, normalize_embeddings=True)
        # Fallback: hash-based pseudo-embedding
        out = []
        for t in texts:
            idx = int(hashlib.md5(t.encode()).hexdigest(), 16) % 10_000
            out.append(self._proj[idx])
        return np.array(out)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


# ═══════════════════════════════════════
#  2. REAL VECTOR DATABASE
# ══════════════════════════════════════

class RealVectorDatabase:
    """
    ChromaDB (persistent) → FAISS (in-memory) → pure-numpy fallback.
    """
    def __init__(
        self,
        embedder: RealEmbeddingModel,
        collection_name: str = "knowledge",
        persist_dir: str = "./chroma_db",
    ):
        self.embedder = embedder
        self.collection_name = collection_name

        if _CHROMA_OK:
            self._backend = "chroma"
            self._client  = chromadb.PersistentClient(path=persist_dir)
            self._col     = self._client.get_or_create_collection(collection_name)

        elif _FAISS_OK:
            self._backend = "faiss"
            self._index   = faiss.IndexFlatIP(embedder.dim)   # inner-product (cosine after norm)
            self._docs: List[Dict] = []

        else:
            self._backend = "numpy"
            self._docs: List[Dict] = []

    # ── write ───────────────────────────────────────
    def add(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        metadatas = metadatas or [{} for _ in texts]
        embeddings = self.embedder.encode(texts)

        if self._backend == "chroma":
            ids = [str(uuid.uuid4()) for _ in texts]
            self._col.add(
                documents=texts,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
                ids=ids,
            )

        elif self._backend == "faiss":
            self._index.add(embeddings)
            for t, m in zip(texts, metadatas):
                self._docs.append({"text": t, "meta": m})

        else:
            for t, e, m in zip(texts, embeddings, metadatas):
                self._docs.append({"text": t, "embedding": e, "meta": m})

    # ── read ───────────────────────────────────────
    def query(self, query_text: str, top_k: int = 5) -> List[Dict]:
        q_emb = self.embedder.encode_one(query_text)

        if self._backend == "chroma":
            res = self._col.query(
                query_embeddings=[q_emb.tolist()],
                n_results=min(top_k, self._col.count() or 1),
            )
            docs = res["documents"][0]
            metas = res["metadatas"][0]
            return [{"text": d, "meta": m} for d, m in zip(docs, metas)]

        elif self._backend == "faiss":
            if self._index.ntotal == 0:
                return []
            q = q_emb[np.newaxis, :].astype(np.float32)
            _, idxs = self._index.search(q, min(top_k, self._index.ntotal))
            return [self._docs[i] for i in idxs[0] if i < len(self._docs)]

        else:
            if not self._docs:
                return []
            scores = [
                (float(np.dot(q_emb, d["embedding"])), d)
                for d in self._docs
            ]
            scores.sort(key=lambda x: x[0], reverse=True)
            return [d for _, d in scores[:top_k]]

    def __len__(self):
        if self._backend == "chroma":  return self._col.count()
        return len(self._docs)
