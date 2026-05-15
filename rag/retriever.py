"""Top-k retrieval against a prebuilt index (CLIP or ColPali)."""
from pathlib import Path
import pandas as pd
import torch
from PIL import Image

from src.config import VECTOR_DB


class Retriever:
    def __init__(self, model, backend: str, index_dir: Path = VECTOR_DB):
        self.model = model
        self.backend = backend
        index_dir = Path(index_dir) / backend
        self.embeddings = torch.load(index_dir / "embeddings.pt")
        self.meta = pd.read_parquet(index_dir / "metadata.parquet")

    def search(self, query_image: Image.Image | None = None,
               query_text: str | None = None, top_k: int = 3) -> list[dict]:
        if self.backend == "clip":
            scores = self._search_clip(query_image, query_text)
        else:
            scores = self._search_colpali(query_image, query_text)

        top = torch.topk(scores, k=min(top_k, len(self.meta)))
        return [
            {**self.meta.iloc[int(i)].to_dict(), "score": float(s)}
            for s, i in zip(top.values, top.indices)
        ]

    def _search_clip(self, image, text) -> torch.Tensor:
        if image is not None:
            q = self.model.embed_image([image])
        else:
            q = self.model.embed_text([text])
        return (q.cpu() @ self.embeddings.T).squeeze(0)

    def _search_colpali(self, image, text) -> torch.Tensor:
        q = self.model.embed_query(text) if text else self.model.embed_images([image])
        return self.model.score(q.cpu(), self.embeddings).squeeze(0)
