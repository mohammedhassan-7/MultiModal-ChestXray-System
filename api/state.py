"""Lazy global model registry. Loads on first request to keep startup fast."""
loaded: dict = {}


def get(name: str):
    if name in loaded:
        return loaded[name]
    loaded[name] = _build(name)
    return loaded[name]


def _build(name: str):
    if name == "medgemma":
        from src.models.medgemma_utils import MedGemmaVLM
        return MedGemmaVLM()

    if name == "retriever_clip":
        from src.models.clip_utils import CLIPModelWrapper
        from rag.retriever import Retriever
        return Retriever(CLIPModelWrapper(), backend="clip")

    if name == "retriever_colpali":
        from src.models.colpali_utils import ColPaliRetriever
        from rag.retriever import Retriever
        return Retriever(ColPaliRetriever(), backend="colpali")

    raise KeyError(name)
