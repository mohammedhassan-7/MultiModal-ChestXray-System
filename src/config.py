"""Central configuration. Import paths and model IDs from here."""
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

ARCHIVE_ZIP = DATA / "archive.zip"
IMAGES_DIR  = DATA / "images"
MINI_CSV    = DATA / "processed" / "mini_dataset.csv"
QA_CSV      = DATA / "synthetic_qa" / "qa_dataset.csv"
VECTOR_DB   = DATA / "vector_db"

MEDGEMMA_ID = "google/medgemma-1.5-4b-it"
COLPALI_ID  = "vidore/colpali-v1.2-merged"  # LoRA pre-merged into base weights
CLIP_ID     = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
USE_4BIT = True  # RTX 4060 has 8GB VRAM
