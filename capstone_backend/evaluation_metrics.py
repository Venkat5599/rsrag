import json
from tqdm import tqdm


from v1_pipeline import run_pipeline_on_text as run_v1
from v2_pipeline import run_pipeline_on_text as run_v2
from v3_pipeline import run_pipeline_on_text as run_v3

from rouge_score import rouge_scorer
from bert_score import score as bertscore_fn
import string


DATA_PATH = "CUAD_v1/CUAD_v1.json"
MAX_SAMPLES = 3


def map_label(question):
    q = question.lower()

    if "govern" in q and "law" in q:
        return "Governing Law"
    if "parties" in q:
        return "Parties"
    if "effective" in q:
        return "Effective Date"
    if "expire" in q or "expiration" in q:
        return "Expiration Date"
    if "payment" in q:
        return "Payment Terms"
    if "confidential" in q:
        return "Confidentiality"
    if "terminate" in q:
        return "Termination for Convenience"
    if "liability" in q:
        return "Limitation of Liability"
    if "indemn" in q:
        return "Indemnification"
    if "intellectual property" in q or " ip " in f" {q} ":
        return "IP Ownership Assignment"
    if "license" in q:
        return "License Grant"
    if "dispute" in q:
        return "Dispute Resolution"
    if "arbitration" in q:
        return "Arbitration"
    if "force majeure" in q:
        return "Force Majeure"
    if "notice" in q:
        return "Notice"
    if "amend" in q:
        return "Amendment"
    if "renew" in q:
        return "Renewal Term"
    if "severability" in q:
        return "Severability"
    if "entire agreement" in q:
        return "Entire Agreement"
    if "waiver" in q:
        return "Waiver"

    return None


def normalize(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()

def compute_span_f1(pred, gold):
    p = set(normalize(pred))
    g = set(normalize(gold))

    if not p or not g:
        return 0.0

    common = len(p & g)
    precision = common / len(p)
    recall = common / len(g)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)

def compute_macro_f1(pred_spans, gold_spans):
    scores = []

    common = set(pred_spans.keys()) & set(gold_spans.keys())

    for c in common:
        pred = pred_spans[c].get("span", "")
        gold = gold_spans[c]

        scores.append(compute_span_f1(pred, gold))

    return sum(scores) / len(scores) if scores else 0.0

def compute_rouge_l(summary, gold_text):
    try:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        return scorer.score(gold_text, summary)['rougeL'].fmeasure
    except:
        return 0.0

def compute_bertscore(summary, gold_text):
    try:
        _, _, F1 = bertscore_fn([summary], [gold_text], lang="en")
        return F1.mean().item()
    except:
        return 0.0

def compute_all_metrics(pred_spans, gold_spans, summary):
    gold_text = " ".join(gold_spans.values()) if gold_spans else ""

    return {
        "macro_f1": round(compute_macro_f1(pred_spans, gold_spans), 4),
        "rougeL": round(compute_rouge_l(summary, gold_text), 4),
        "bert_score": round(compute_bertscore(summary, gold_text), 4),
        "n_clauses_found": len(pred_spans)
    }


def load_data(path):
    with open(path, "r") as f:
        raw = json.load(f)

    samples = []

    for doc in raw["data"][:MAX_SAMPLES]:
        for para in doc.get("paragraphs", []):
            context = para.get("context", "").strip()

            if not context:
                continue

            gold_spans = {}

            for qa in para.get("qas", []):
                question = qa.get("question", "").strip()
                label = map_label(question)

                if not label:
                    continue

                if qa.get("answers"):
                    gold_spans[label] = qa["answers"][0]["text"]

            samples.append({
                "id": doc.get("title", "sample"),
                "text": context,
                "gold_spans": gold_spans
            })

    return samples


def evaluate_model(run_fn, dataset, name="model"):
    print(f"\nRunning {name}")

    all_metrics = []

    for sample in tqdm(dataset):
        text = sample.get("text", "")
        gold_spans = sample.get("gold_spans", {})

        if not text:
            continue

        try:
            result = run_fn(
                text,
                filename=sample["id"],
                gold_spans=gold_spans
            )

            metrics = compute_all_metrics(
                result.get("clauses", {}),
                gold_spans,
                result.get("summary", "")
            )

            all_metrics.append(metrics)

        except Exception as e:
            print(f"Error in {name}: {e}")

    return aggregate_metrics(all_metrics)


def aggregate_metrics(metrics_list):
    if not metrics_list:
        return {}

    def avg(key):
        vals = [m.get(key, 0) for m in metrics_list]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    return {
        "macro_f1": avg("macro_f1"),
        "rougeL": avg("rougeL"),
        "bert_score": avg("bert_score"),
        "avg_clauses_found": avg("n_clauses_found"),
    }


def main():
    dataset = load_data(DATA_PATH)

    print(f"Loaded {len(dataset)} samples")

    results = {}

    results["v1"] = evaluate_model(run_v1, dataset, "v1 (baseline)")
    results["v2"] = evaluate_model(run_v2, dataset, "v2 (intermediate)")
    results["v3"] = evaluate_model(run_v3, dataset, "v3 (final)")

    print("\nFINAL COMPARISON\n")
    print_table(results)


def print_table(results):
    headers = ["Model", "Macro F1", "ROUGE-L", "BERTScore"]

    print(f"{headers[0]:<10} {headers[1]:<12} {headers[2]:<12} {headers[3]:<12}")
    print("-" * 60)

    for model, metrics in results.items():
        print(f"{model:<10} "
              f"{metrics.get('macro_f1',0):<12} "
              f"{metrics.get('rougeL',0):<12} "
              f"{metrics.get('bert_score',0):<12}")


if __name__ == "__main__":
    main()
