"""FastAPI routes for report generation and QA."""
import io
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from PIL import Image

from api.schemas import ReportResponse, QAResponse, RetrievedItem
from api import state

router = APIRouter()


def _load_image(upload: UploadFile) -> Image.Image:
    return Image.open(io.BytesIO(upload.file.read())).convert("RGB")


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "loaded": list(state.loaded.keys())}


@router.post("/report", response_model=ReportResponse)
def report(image: UploadFile = File(...)) -> ReportResponse:
    vlm = state.get("medgemma")
    text = vlm.generate_report(_load_image(image))
    return ReportResponse(report=text)


@router.post("/qa", response_model=QAResponse)
def qa(
    question: str = Form(...),
    image: UploadFile = File(...),
    use_rag: bool = Form(True),
    retriever_name: str = Form("colpali"),  # "colpali" | "clip"
) -> QAResponse:
    if retriever_name not in {"colpali", "clip"}:
        raise HTTPException(400, "retriever_name must be 'colpali' or 'clip'")

    img = _load_image(image)
    vlm = state.get("medgemma")

    retrieved: list[dict] = []
    context = None
    if use_rag:
        retriever = state.get(f"retriever_{retriever_name}")
        retrieved = retriever.search(query_image=img, top_k=3)
        context = "\n\n".join(r["report"] for r in retrieved)

    answer = vlm.answer_question(img, question, context=context)
    return QAResponse(
        answer=answer,
        retrieved=[RetrievedItem(**r) for r in retrieved],
        retriever=retriever_name if use_rag else None,
    )
