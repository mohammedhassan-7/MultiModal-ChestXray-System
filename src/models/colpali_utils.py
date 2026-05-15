"""ColPali retriever: multi-vector image embeddings with late-interaction scoring."""
import torch
from PIL import Image
from colpali_engine.models import ColPali, ColPaliProcessor

from src.config import COLPALI_ID, DEVICE


class ColPaliRetriever:
    def __init__(self, model_id: str = COLPALI_ID):
        self.model = ColPali.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map=DEVICE
        ).eval()
        self.processor = ColPaliProcessor.from_pretrained(model_id)

    @torch.inference_mode()
    def embed_images(self, images: list[Image.Image]) -> torch.Tensor:
        batch = self.processor.process_images(images).to(self.model.device)
        return self.model(**batch)  # [B, num_patches, dim]

    @torch.inference_mode()
    def embed_query(self, text: str) -> torch.Tensor:
        batch = self.processor.process_queries([text]).to(self.model.device)
        return self.model(**batch)  # [1, num_tokens, dim]

    def score(self, query_emb: torch.Tensor, doc_embs: torch.Tensor) -> torch.Tensor:
        """Late-interaction MaxSim score for each doc against the query."""
        return self.processor.score_multi_vector(query_emb, doc_embs)
