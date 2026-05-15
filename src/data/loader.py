"""MIMIC-CXR loader. Run as a script to prepare the mini dataset and images:

    python -m src.data.loader
"""
import re
import pandas as pd
import zipfile
import ast
from pathlib import Path
from tqdm import tqdm

class MimicLoader:
    def __init__(self, archive_path="../data/archive.zip", img_prefix="official_data_iccv_final/"):
        self.archive_path = Path(archive_path)
        self.img_prefix = img_prefix

    def load_and_clean(self, csv_name="mimic_cxr_aug_train.csv"):
        """Loads the raw CSV from the zip and extracts single reports/paths."""
        if not self.archive_path.exists():
            raise FileNotFoundError(f"Archive not found at {self.archive_path}")

        with zipfile.ZipFile(self.archive_path) as zf:
            with zf.open(csv_name) as f:
                df = pd.read_csv(f)

        # Drop unnamed columns
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

        # Parse stringified lists found in 'text' and 'image' columns
        df["raw_text"] = df["text"].apply(self._safe_eval_first)
        df["image_path"] = df["image"].apply(
            lambda x: self.img_prefix + self._safe_eval_first(x) if x else None
        )
        
        return df

    def _safe_eval_first(self, val):
        try:
            result = ast.literal_eval(val)
            return result[0] if isinstance(result, list) and len(result) > 0 else ""
        except:
            return ""

    def create_mini_dataset(self, df, n=2000, min_len=80):
        """Filters and samples the dataset for local testing."""
        mini = df[
            df["image_path"].notna() &
            (df["raw_text"].str.len() > min_len)
        ].sample(n=n, random_state=42).copy()

        return mini.reset_index(drop=True)

    def extract_images(self, df, dest_dir):
        """Extract only the images listed in df from the zip into dest_dir.

        Adds a 'local_path' column. Already-extracted files are skipped.
        Rows whose image isn't in the zip get local_path = None so the caller
        can .dropna(subset=["local_path"]).
        """
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        local_paths = []
        missing = 0

        with zipfile.ZipFile(self.archive_path) as zf:
            names = set(zf.namelist())
            for path in tqdm(df["image_path"], desc="Extracting images", total=len(df)):
                out = dest_dir / Path(path).name
                if out.exists():
                    local_paths.append(str(out))
                elif path in names:
                    with zf.open(path) as src, open(out, "wb") as dst:
                        dst.write(src.read())
                    local_paths.append(str(out))
                else:
                    local_paths.append(None)
                    missing += 1

        if missing:
            print(f"[extract_images] {missing} images not found in zip; local_path=None")

        df = df.copy()
        df["local_path"] = local_paths
        return df


def clean_report(text: str) -> str:
    """Strip 'Findings:' / 'Impression:' labels and collapse whitespace."""
    text = re.sub(r"(?i)\b(findings|impression)\s*:", "", text)
    return re.sub(r"\s+", " ", text).strip()


TARGET_N = 2000        # final number of rows we want in the mini set
OVERSAMPLE = 1.5       # zip only contains ~73% of referenced images; oversample to compensate


def main():
    from src.config import ARCHIVE_ZIP, IMAGES_DIR, MINI_CSV, QA_CSV

    loader = MimicLoader(archive_path=ARCHIVE_ZIP)
    df = loader.load_and_clean()
    df["study_id"] = df["image_path"].str.extract(r"/(s\d+)/")
    print(f"Loaded {len(df)} rows from archive")

    # If a QA dataset already exists, prioritize its studies so the new mini
    # set stays aligned with existing QA pairs (avoids regenerating QA).
    qa_studies: set = set()
    if QA_CSV.exists():
        qa_studies = set(pd.read_csv(QA_CSV)["study_id"].unique())
        print(f"Found existing QA for {len(qa_studies)} studies - prioritizing them")

    candidates = df[df["image_path"].notna() & (df["raw_text"].str.len() > 80)]

    if qa_studies:
        priority = candidates[candidates["study_id"].isin(qa_studies)]
        others = candidates[~candidates["study_id"].isin(qa_studies)]
        sample_others = max(int(TARGET_N * OVERSAMPLE) - len(priority), 0)
        sample_others = min(sample_others, len(others))
        filler = others.sample(n=sample_others, random_state=42)
        mini = pd.concat([priority, filler], ignore_index=True)
    else:
        sample_n = min(int(TARGET_N * OVERSAMPLE), len(candidates))
        mini = candidates.sample(n=sample_n, random_state=42).reset_index(drop=True)

    print(f"Sampled {len(mini)} candidate rows ({len(qa_studies)} QA-priority + filler)")

    mini["cleaned_report"] = mini["raw_text"].map(clean_report)
    mini["report_len"] = mini["cleaned_report"].str.split().str.len()

    mini = loader.extract_images(mini, IMAGES_DIR)
    mini = mini.dropna(subset=["local_path"]).reset_index(drop=True)
    print(f"{len(mini)} rows survived after image extraction")

    mini = mini.head(TARGET_N).reset_index(drop=True)
    print(f"Trimmed to target: {len(mini)} rows in {IMAGES_DIR}")

    MINI_CSV.parent.mkdir(parents=True, exist_ok=True)
    mini.to_csv(MINI_CSV, index=False)
    print(f"Saved {MINI_CSV}")


if __name__ == "__main__":
    main()