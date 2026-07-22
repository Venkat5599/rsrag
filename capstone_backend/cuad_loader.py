import logging
import json
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)


DATA_PATH = Path(r"/Users/vanshikatyagi/cspstone/CUAD_v1/CUAD_v1.json")


QUESTION_MAP = {
    "what law governs":                          "Governing Law",
    "who are the parties":                       "Parties",
    "when does this agreement become effective": "Effective Date",
    "when does this agreement expire":           "Expiration Date",
    "can either party terminate":                "Termination for Convenience",
    "what confidentiality":                      "Confidentiality",
    "non-compete":                               "Non-Compete",
    "who must indemnify":                        "Indemnification",
    "what is the cap":                           "Limitation of Liability",
    "what is the limitation":                    "Limitation of Liability",
    "who owns intellectual":                     "IP Ownership Assignment",
    "what license rights":                       "License Grant",
    "what are the payment":                      "Payment Terms",
    "audit right":                               "Audit Rights",
    "what insurance":                            "Insurance",
    "is assignment":                             "Anti-Assignment",
    "what happens upon a change of control":     "Change of Control",
    "force majeure":                             "Force Majeure",
    "how are disputes":                          "Dispute Resolution",
    "what jurisdiction":                         "Governing Jurisdiction",
    "what are the notice":                       "Notice",
    "does this contract automatically renew":    "Renewal Term",
    "what notice period":                        "Notice Period to Terminate Renewal",
    "revenue or profit sharing":                 "Revenue or Profit Sharing",
    "minimum purchase":                          "Minimum Commitment",
    "volume restriction":                        "Volume Restriction",
    "non-disparagement":                         "Non-Disparagement",
    "third party beneficiary":                   "Third Party Beneficiary",
    "source code escrow":                        "Source Code Escrow",
    "covenant not to sue":                       "Covenant Not to Sue",
    "most favored nation":                       "Most Favored Nation",
    "liquidated damages":                        "Liquidated Damages",
    "scope of the license":                      "License Scope",
    "exclusive or non-exclusive":                "License Exclusivity",
    "price restriction":                         "Price Restrictions",
    "severability":                              "Severability",
    "entire agreement":                          "Entire Agreement",
    "waiver":                                    "Waiver",
    "arbitration":                               "Arbitration",
    "how can this agreement be amended":         "Amendment",
}


@lru_cache(maxsize=1)
def _load_dataset():
    logger.info("Loading CUAD v1 dataset from local JSON...")
    if not DATA_PATH.exists():
        logger.error(f"Dataset not found at {DATA_PATH}")
        return []


    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)


    if isinstance(raw, dict) and "data" in raw:
        data = raw["data"]
    else:
        data = raw

    logger.info(f"Loaded {len(data)} contracts from local file")
    print("DATA PATH:", DATA_PATH.resolve())
    print("EXISTS:", DATA_PATH.exists())
    return data


def _group_by_contract(data) -> list:
    contracts = []

    for item in data:
        title = item.get("title", "unknown")

        for para in item.get("paragraphs", []):
            context = para.get("context", "")
            qas = para.get("qas", [])

            if len(context) > 500:
                contracts.append({
                    "title": title,
                    "context": context,
                    "qas": qas
                })

    logger.info(f"Grouped into {len(contracts)} contracts")
    return contracts


def _extract_gold_spans(qas: list) -> dict:
    gold = {}
    for qa in qas:
        question = qa.get("question", "").lower()
        answers = qa.get("answers", [])

        if not answers:
            continue

        text = answers[0].get("text", "").strip()
        if not text:
            continue

        for fragment, clause_type in QUESTION_MAP.items():
            if fragment in question:
                if clause_type not in gold:
                    gold[clause_type] = text
                break

    return gold


def list_cuad_samples(n: int = 20) -> list:
    try:
        data = _load_dataset()
        contracts = _group_by_contract(data)

        return [
            {"index": i, "title": c["title"], "length": len(c["context"])}
            for i, c in enumerate(contracts[:n])
        ]

    except Exception as e:
        logger.error(f"CUAD list error: {e}")
        return []


def load_cuad_sample(index: int = 0) -> dict:
    try:
        data = _load_dataset()
        contracts = _group_by_contract(data)

        if index >= len(contracts):
            logger.error(f"CUAD index {index} out of range")
            return {}

        contract = contracts[index]
        gold_spans = _extract_gold_spans(contract["qas"])

        logger.info(f"Loaded CUAD contract [{index}]")

        return {
            "title": contract["title"],
            "context": contract["context"],
            "gold_spans": gold_spans,
        }

    except Exception as e:
        logger.error(f"CUAD load error: {e}")
        return {}
