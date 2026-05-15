"""CLIP (BiomedCLIP) wrapper: single-vector image/text embeddings, baseline retriever."""
import torch
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer

from src.config import CLIP_ID, DEVICE


class CLIPModelWrapper:
    def __init__(self, model_id: str = CLIP_ID):
        self.model, self.preprocess = create_model_from_pretrained(f"hf-hub:{model_id}")
        self.tokenizer = get_tokenizer(f"hf-hub:{model_id}")
        self.model = self.model.to(DEVICE).eval()

    @torch.inference_mode()
    def embed_image(self, images: list[Image.Image]) -> torch.Tensor:
        batch = torch.stack([self.preprocess(img) for img in images]).to(DEVICE)
        feats = self.model.encode_image(batch)
        return torch.nn.functional.normalize(feats, dim=-1)

    @torch.inference_mode()
    def embed_text(self, texts: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(texts).to(DEVICE)
        feats = self.model.encode_text(tokens)
        return torch.nn.functional.normalize(feats, dim=-1)
