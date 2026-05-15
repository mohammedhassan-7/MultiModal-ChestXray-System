"""Simple text-similarity and retrieval metrics."""
from collections import Counter
import pandas as pd
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
_smooth = SmoothingFunction().method1


def token_f1(pred: str, gold: str) -> float:
    p, g = pred.lower().split(), gold.lower().split()
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    precision, recall = n / len(p), n / len(g)
    return 2 * precision * recall / (precision + recall)


def score_text(predictions: list[str], references: list[str]) -> dict:
    """BLEU-4, ROUGE-L, token-F1 averaged over the pairs."""
    bleu, rouge, f1 = [], [], []
    for pred, ref in zip(predictions, references):
        bleu.append(sentence_bleu([ref.split()], pred.split(), smoothing_function=_smooth))
        rouge.append(_rouge.score(ref, pred)["rougeL"].fmeasure)
        f1.append(token_f1(pred, ref))
    n = max(len(predictions), 1)
    return {"bleu": sum(bleu) / n, "rougeL": sum(rouge) / n, "token_f1": sum(f1) / n}


def recall_at_k(retrieved_ids: list[list], gold_ids: list, k: int = 3) -> float:
    """Fraction of queries where the gold id appears in top-k retrieved."""
    hits = sum(1 for retr, gold in zip(retrieved_ids, gold_ids) if gold in retr[:k])
    return hits / max(len(gold_ids), 1)


def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """results: {'model_name': metric_dict, ...} -> tidy DataFrame."""
    return pd.DataFrame(results).T.round(4)
