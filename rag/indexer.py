"""Build a vector index of the mini dataset for RAG retrieval."""
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from src.config import MINI_CSV, VECTOR_DB


def build_index(retriever, backend: str, mini_csv: Path = MINI_CSV,
                out_dir: Path = VECTOR_DB, batch_size: int = 4) -> Path:
    """Encode every image in the mini set; save embeddings + metadata.

    retriever : object with embed_image() (CLIP) or embed_images() (ColPali).
    backend   : 'clip' or 'colpali' (used as subdir name).
    """
    df = pd.read_csv(mini_csv)
    out_dir = Path(out_dir) / backend
    out_dir.mkdir(parents=True, exist_ok=True)

    embed_fn = retriever.embed_image if backend == "clip" else retriever.embed_images

    chunks = []
    for i in tqdm(range(0, len(df), batch_size), desc=f"Indexing ({backend})"):
        batch_rows = df.iloc[i:i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_rows["local_path"]]
        chunks.append(embed_fn(images).cpu())
        if torch.cuda.is_available() and i % 50 == 0:
            torch.cuda.empty_cache()

    embeddings = torch.cat(chunks, dim=0)
    torch.save(embeddings, out_dir / "embeddings.pt")

    df[["study_id", "local_path", "cleaned_report"]].rename(
        columns={"cleaned_report": "report"}
    ).to_parquet(out_dir / "metadata.parquet", index=False)

    return out_dir
