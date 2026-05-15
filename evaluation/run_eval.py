"""Run model comparison and save tables to results/.

Usage:
    python -m evaluation.run_eval                 # default: 50 QA samples + 20 report samples
    python -m evaluation.run_eval --qa 100 --rg 30
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.config import MINI_CSV, QA_CSV
from evaluation.metrics import score_text, recall_at_k, compare_models


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _open_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _qa_modes(qa_df: pd.DataFrame, mini_df: pd.DataFrame, vlm, clip_retr, colpali_retr):
    """Run each mode (no-RAG / CLIP / ColPali) on the same eval slice."""
    img_by_study = dict(zip(mini_df["study_id"], mini_df["local_path"]))

    eval_rows = qa_df[qa_df["study_id"].isin(img_by_study)].copy()
    eval_rows["local_path"] = eval_rows["study_id"].map(img_by_study)

    all_preds = []
    summary = {}
    for mode, retriever in [
        ("no_rag",  None),
        ("clip",    clip_retr),
        ("colpali", colpali_retr),
    ]:
        preds, golds, retrieved_lists = [], [], []
        for _, row in tqdm(eval_rows.iterrows(), total=len(eval_rows), desc=f"QA [{mode}]"):
            img = _open_image(row["local_path"])
            retrieved_ids: list = []
            context = None
            if retriever is not None:
                retrieved = retriever.search(query_image=img, top_k=3)
                retrieved_ids = [r["study_id"] for r in retrieved]
                context = "\n\n".join(r["report"] for r in retrieved)

            answer = vlm.answer_question(img, row["question"], context=context)
            preds.append(answer)
            golds.append(row["answer"])
            retrieved_lists.append(retrieved_ids)

            all_preds.append({
                "study_id": row["study_id"],
                "question": row["question"],
                "gold": row["answer"],
                "mode": mode,
                "prediction": answer,
                "retrieved_ids": retrieved_ids,
            })

        metrics = score_text(preds, golds)
        if retriever is not None:
            metrics["recall_at_3"] = recall_at_k(
                retrieved_lists, eval_rows["study_id"].tolist(), k=3
            )
        summary[mode] = metrics

    return summary, pd.DataFrame(all_preds)


def _report_gen(mini_df: pd.DataFrame, vlm, n: int):
    sample = mini_df.sample(n=min(n, len(mini_df)), random_state=7).reset_index(drop=True)
    preds, rows = [], []
    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="Report gen"):
        img = _open_image(row["local_path"])
        pred = vlm.generate_report(img)
        preds.append(pred)
        rows.append({"study_id": row["study_id"], "gold": row["cleaned_report"], "prediction": pred})

    metrics = score_text(preds, sample["cleaned_report"].tolist())
    return {"report_gen_medgemma": metrics}, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", type=int, default=50,  help="QA eval sample size")
    ap.add_argument("--rg", type=int, default=20,  help="Report-gen eval sample size")
    ap.add_argument("--skip-rg", action="store_true", help="Skip report-gen eval")
    ap.add_argument("--skip-qa", action="store_true", help="Skip QA eval")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    from src.models.medgemma_utils import MedGemmaVLM

    mini = pd.read_csv(MINI_CSV)
    qa = pd.read_csv(QA_CSV)
    print(f"mini: {len(mini)} rows | qa: {len(qa)} pairs")

    vlm = MedGemmaVLM()

    all_summary = {}

    if not args.skip_qa:
        from src.models.clip_utils import CLIPModelWrapper
        from src.models.colpali_utils import ColPaliRetriever
        from rag.retriever import Retriever

        clip_retr    = Retriever(CLIPModelWrapper(),    backend="clip")
        colpali_retr = Retriever(ColPaliRetriever(),    backend="colpali")

        qa_eval = qa.sample(n=min(args.qa, len(qa)), random_state=7)
        qa_summary, qa_preds = _qa_modes(qa_eval, mini, vlm, clip_retr, colpali_retr)
        qa_preds.to_csv(RESULTS_DIR / "qa_predictions.csv", index=False)
        all_summary.update(qa_summary)

    if not args.skip_rg:
        rg_summary, rg_preds = _report_gen(mini, vlm, args.rg)
        rg_preds.to_csv(RESULTS_DIR / "report_predictions.csv", index=False)
        all_summary.update(rg_summary)

    table = compare_models(all_summary)
    table.to_csv(RESULTS_DIR / "comparison.csv")
    (RESULTS_DIR / "comparison.json").write_text(json.dumps(all_summary, indent=2))

    print("\n=== Comparison ===")
    print(table.to_string())
    print(f"\nSaved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
