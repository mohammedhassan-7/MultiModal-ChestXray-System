"""HTTP client for the FastAPI backend."""
import os
import requests

API_URL = os.environ.get("API_URL", "http://localhost:8000")


def generate_report(image_bytes: bytes, filename: str = "xray.jpg") -> dict:
    r = requests.post(
        f"{API_URL}/report",
        files={"image": (filename, image_bytes, "image/jpeg")},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def answer_question(image_bytes: bytes, question: str,
                    use_rag: bool = True, retriever_name: str = "colpali",
                    filename: str = "xray.jpg") -> dict:
    r = requests.post(
        f"{API_URL}/qa",
        files={"image": (filename, image_bytes, "image/jpeg")},
        data={"question": question, "use_rag": str(use_rag),
              "retriever_name": retriever_name},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()
