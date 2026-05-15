"""Generate synthetic VQA pairs from MIMIC-CXR reports with MedGemma.

Run as a script:
    python -m src.data.qa_generator
"""
import json
import re
import pandas as pd
from tqdm import tqdm

from src.config import MINI_CSV, QA_CSV


PROMPT = """Convert this radiology report into 2-3 chest X-ray VQA pairs.
Output ONLY the JSON list below. No analysis, no preamble, no thinking text.

Each pair: {{"type": one of presence/location/severity/comparison/normality,
            "question": image-grounded question (do NOT mention "the report"),
            "answer":  1-sentence clinical statement}}

Example:
Report: "Small left pleural effusion. No pneumothorax."
JSON: [{{"type":"presence","question":"Is there a pleural effusion?","answer":"Yes, a small left pleural effusion is seen."}},{{"type":"presence","question":"Is there a pneumothorax?","answer":"No pneumothorax is identified."}}]

Report: \"\"\"{report}\"\"\"
JSON:"""


ALLOWED_TYPES = {"presence", "location", "severity", "comparison", "normality"}
LEAKAGE_PATTERNS = (
    "the report", "according to the report", "report states",
    "report mentions", "as mentioned in the report",
)


class QAGenerator:
    def __init__(self, vlm, temperature: float = 0.7):
        self.vlm = vlm
        self.temperature = temperature

    def generate_triplet(self, report_text: str) -> tuple[list[dict] | None, str]:
        raw = self.vlm.chat_text(
            PROMPT.format(report=report_text),
            temperature=self.temperature,
            max_new_tokens=384,
            force_prefix="[",   # bypass thinking mode — model starts the JSON list directly
        )
        # Find the JSON list directly — works whether or not thinking is present.
        # Pattern: [ ... { ... } ... ]  with at least one object inside.
        match = re.search(r"\[\s*\{.*\}\s*\]", raw, flags=re.DOTALL)
        if not match:
            return None, raw
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, list) else None, raw
        except json.JSONDecodeError:
            return None, raw

    @staticmethod
    def _is_clean(pair: dict) -> bool:
        q = str(pair.get("question", "")).strip()
        a = str(pair.get("answer", "")).strip()
        t = str(pair.get("type", "")).strip().lower()
        if t not in ALLOWED_TYPES:
            return False
        if len(q.split()) < 3 or len(a.split()) < 3:
            return False
        if any(p in a.lower() for p in LEAKAGE_PATTERNS):
            return False
        return True


def _already_done(qa_csv) -> set:
    if not qa_csv.exists():
        return set()
    return set(pd.read_csv(qa_csv)["study_id"].unique())


def _append_rows(qa_csv, rows: list[dict]):
    df = pd.DataFrame(rows)
    write_header = not qa_csv.exists()
    df.to_csv(qa_csv, mode="a", header=write_header, index=False)


def _log_rejected(path, study_id, raw):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"study_id": str(study_id), "raw": raw}) + "\n")


def main():
    from src.models.medgemma_utils import MedGemmaVLM

    df = pd.read_csv(MINI_CSV)
    done = _already_done(QA_CSV)
    df = df[~df["study_id"].isin(done)].reset_index(drop=True)
    print(f"{len(done)} studies already done. Processing {len(df)} new reports.")
    if df.empty:
        return

    QA_CSV.parent.mkdir(parents=True, exist_ok=True)
    rejected_log = QA_CSV.parent / "_rejected.jsonl"

    vlm = MedGemmaVLM()
    gen = QAGenerator(vlm)

    n_kept = n_rejected = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating QA"):
        triplet, raw = gen.generate_triplet(row["cleaned_report"])
        if not triplet:
            _log_rejected(rejected_log, row["study_id"], raw)
            n_rejected += 1
            continue

        clean = [p for p in triplet if QAGenerator._is_clean(p)]
        if not clean:
            _log_rejected(rejected_log, row["study_id"], raw)
            n_rejected += 1
            continue

        rows = [{**p, "subject_id": row["subject_id"],
                 "study_id": row["study_id"], "image_path": row["image_path"]}
                for p in clean]
        _append_rows(QA_CSV, rows)
        n_kept += len(rows)

    print(f"Done. Kept {n_kept} QA pairs. Rejected {n_rejected} reports "
          f"(see {rejected_log if n_rejected else 'no log'}).")


if __name__ == "__main__":
    main()
