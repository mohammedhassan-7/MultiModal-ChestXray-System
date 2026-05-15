"""Build CLIP and ColPali retrieval indexes over the mini dataset.

Usage:
    python -m rag.build_indexes              # both backends
    python -m rag.build_indexes --only clip
    python -m rag.build_indexes --only colpali
"""
import argparse

from rag.indexer import build_index


def build_clip():
    from src.models.clip_utils import CLIPModelWrapper
    print("=== CLIP ===")
    build_index(CLIPModelWrapper(), backend="clip")


def build_colpali():
    from src.models.colpali_utils import ColPaliRetriever
    print("=== ColPali ===")
    build_index(ColPaliRetriever(), backend="colpali", batch_size=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["clip", "colpali"],
                    help="Build just one backend (default: both)")
    args = ap.parse_args()

    if args.only in (None, "clip"):
        build_clip()
    if args.only in (None, "colpali"):
        build_colpali()

    print("Done.")


if __name__ == "__main__":
    main()
