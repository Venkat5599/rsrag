import os, uuid, logging
from datetime import datetime
from pathlib import Path
from typing import Optional
import re

logger = logging.getLogger(__name__)

MODEL_CACHE_DIR = Path(
    os.getenv("CLNEA_MODEL_CACHE", Path(".hf-cache"))
).resolve()
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(MODEL_CACHE_DIR / "hub"))


_qa_pipeline  = None
_ner_pipeline = None
_tokenizer    = None
_model        = None
_nlp_v1       = None
_summ_v1      = None


def preload_models():
    global _qa_pipeline, _ner_pipeline, _tokenizer, _model

    from transformers import (
        pipeline,
        AutoModelForQuestionAnswering,
        AutoModelForSeq2SeqLM,
        AutoModelForTokenClassification,
        AutoTokenizer,
    )
    import torch

    logger.info("Loading DeBERTa QA (Stage 2)...")
    qa_tokenizer = AutoTokenizer.from_pretrained(
        "deepset/deberta-v3-base-squad2",
        cache_dir=str(MODEL_CACHE_DIR),
    )
    qa_model = AutoModelForQuestionAnswering.from_pretrained(
        "deepset/deberta-v3-base-squad2",
        cache_dir=str(MODEL_CACHE_DIR),
    )
    _qa_pipeline = pipeline(
        "question-answering",
        model=qa_model,
        tokenizer=qa_tokenizer,
        device=-1,
    )

    logger.info("Loading LegalBERT NER (Stage 3)...")
    ner_tokenizer = AutoTokenizer.from_pretrained(
        "nlpaueb/legal-bert-base-uncased",
        cache_dir=str(MODEL_CACHE_DIR),
    )
    ner_model = AutoModelForTokenClassification.from_pretrained(
        "nlpaueb/legal-bert-base-uncased",
        cache_dir=str(MODEL_CACHE_DIR),
    )
    _ner_pipeline = pipeline(
        "ner",
        model=ner_model,
        tokenizer=ner_tokenizer,
        aggregation_strategy="simple",
        device=-1,
    )

    logger.info("Loading Legal-Pegasus (Stage 4)...")
    _tokenizer = AutoTokenizer.from_pretrained(
        "nsi319/legal-pegasus",
        use_fast=False,
        cache_dir=str(MODEL_CACHE_DIR),
    )
    _model     = AutoModelForSeq2SeqLM.from_pretrained(
        "nsi319/legal-pegasus",
        torch_dtype=torch.float32,
        cache_dir=str(MODEL_CACHE_DIR),
    )
    _model.eval()

    logger.info("All CLNEA models loaded.")


def is_legal_text(text: str) -> bool:
    text_lower = text.lower()


    legal_keywords = [
        "agreement", "contract", "party", "parties",
        "shall", "hereby", "whereas", "liability",
        "indemnify", "termination", "confidential",
        "governing law", "jurisdiction", "obligation"
    ]


    non_legal_keywords = [
        "hey", "hello", "thanks", "regards", "lol",
        "please find attached", "hi team", "good morning"
    ]

    legal_hits = sum(1 for kw in legal_keywords if kw in text_lower)
    non_legal_hits = sum(1 for kw in non_legal_keywords if kw in text_lower)


    has_sections = bool(re.search(r"\n\s*\d+[\.\)]", text))
    has_caps_headers = bool(re.search(r"\n[A-Z\s]{5,}\n", text))
    has_long_sentences = any(len(s.split()) > 20 for s in text.split("."))

    word_count = len(text.split())

    score = 0
    score += legal_hits * 2
    score += 2 if has_sections else 0
    score += 2 if has_caps_headers else 0
    score += 1 if has_long_sentences else 0
    score -= non_legal_hits * 2

    return score >= 4 and word_count > 80


def stage1_ocr(pdf_bytes: bytes) -> str:
    import pytesseract
    from pdf2image import convert_from_bytes

    logger.info("Stage 1: OCR...")
    images    = convert_from_bytes(pdf_bytes, dpi=200)
    pages     = [pytesseract.image_to_string(img, lang="eng") for img in images]
    full_text = "\n".join(pages)
    logger.info(f"Stage 1 done: {len(full_text)} chars, {len(images)} pages")
    return full_text


CUAD_QUESTIONS = {
    "Governing Law":               "What law governs this contract?",
    "Parties":                     "Who are the parties to this agreement?",
    "Effective Date":              "When does this agreement become effective?",
    "Expiration Date":             "When does this agreement expire?",
    "Termination for Convenience": "Can either party terminate without cause?",
    "Confidentiality":             "What confidentiality obligations exist?",
    "Non-Compete":                 "Is there a non-compete restriction?",
    "Indemnification":             "Who must indemnify whom?",
    "Limitation of Liability":     "What is the cap on liability?",
    "IP Ownership Assignment":     "Who owns intellectual property created?",
    "License Grant":               "What license rights are granted?",
    "Payment Terms":               "What are the payment terms and amounts?",
    "Audit Rights":                "Does either party have audit rights?",
    "Warranty Duration":           "What is the warranty duration?",
    "Insurance":                   "What insurance requirements exist?",
    "Anti-Assignment":             "Is assignment of this agreement restricted?",
    "Change of Control":           "What happens upon a change of control?",
    "Force Majeure":               "Is there a force majeure clause?",
    "Dispute Resolution":          "How are disputes resolved?",
    "Governing Jurisdiction":      "What jurisdiction governs disputes?",
    "Notice":                      "What are the notice requirements?",
    "Amendment":                   "How can this agreement be amended?",
    "Renewal Term":                "Does this contract automatically renew?",
    "Notice Period to Terminate Renewal": "What notice prevents automatic renewal?",
    "Revenue or Profit Sharing":   "Is there a revenue or profit sharing arrangement?",
    "Minimum Commitment":          "Is there a minimum purchase commitment?",
    "Volume Restriction":          "Are there volume restrictions?",
    "Non-Disparagement":           "Is there a non-disparagement clause?",
    "Third Party Beneficiary":     "Are there third party beneficiaries?",
    "Source Code Escrow":          "Is there a source code escrow obligation?",
    "Covenant Not to Sue":         "Is there a covenant not to sue?",
    "Most Favored Nation":         "Is there a most favored nation clause?",
    "Cap on Liability":            "What is the maximum monetary liability cap?",
    "Liquidated Damages":          "Are there liquidated damages provisions?",
    "License Scope":               "What is the scope of the license?",
    "License Exclusivity":         "Is the license exclusive or non-exclusive?",
    "Price Restrictions":          "Are there price restrictions?",
    "Severability":                "Is there a severability clause?",
    "Entire Agreement":            "Is there an entire agreement clause?",
    "Waiver":                      "Is there a waiver provision?",
    "Arbitration":                 "Is arbitration required?",
}


def stage2_clause_segmentation(text: str, threshold: float = 0.30) -> dict:
    logger.info("Stage 2: Clause segmentation...")


    if _qa_pipeline is None:
        raise RuntimeError("QA pipeline not initialized. Call preload_models() first.")

    context = text[:4096]
    spans   = {}
    used_spans = set()

    for clause_type, question in CUAD_QUESTIONS.items():
        try:
            res = _qa_pipeline(
                question=question,
                context=context,
                max_answer_len=200,
                handle_impossible_answer=True
            )

            score = res.get("score", 0.0)
            ans   = res.get("answer", "").strip()

            if score >= threshold and len(ans) > 5:

                span = ans.strip()


                if span in used_spans:
                    continue


                if len(span.split()) < 3:
                    continue

                used_spans.add(span)

                spans[clause_type] = {
                    "span":  span,
                    "score": round(score, 4),
                    "start": res.get("start", -1),
                    "end":   res.get("end",   -1),
                }

        except Exception as e:
            logger.debug(f"  Stage 2 skip '{clause_type}': {e}")

    logger.info(f"Stage 2 done: {len(spans)} clauses found")
    return spans

CLAUSE_SCHEMA = {
    "Governing Law":               ["GPE", "LOC", "DATE", "ORG"],
    "Parties":                     ["ORG", "PERSON"],
    "Effective Date":              ["DATE", "TIME"],
    "Expiration Date":             ["DATE", "TIME"],
    "Termination for Convenience": ["DATE", "TIME", "CARDINAL", "QUANTITY"],
    "Confidentiality":             ["DATE", "TIME", "CARDINAL", "QUANTITY", "ORG"],
    "Non-Compete":                 ["DATE", "TIME", "GPE", "LOC", "ORG"],
    "Non-Disparagement":           ["ORG", "PERSON"],
    "Indemnification":             ["ORG", "PERSON"],
    "Limitation of Liability":     ["MONEY", "PERCENT", "CARDINAL"],
    "IP Ownership Assignment":     ["ORG", "PERSON"],
    "License Grant":               ["ORG", "PERSON"],
    "License Scope":               ["ORG", "PERSON", "GPE"],
    "License Exclusivity":         ["ORG", "PERSON"],
    "Payment Terms":               ["MONEY", "DATE", "CARDINAL", "PERCENT"],
    "Audit Rights":                ["DATE", "TIME", "CARDINAL", "ORG"],
    "Warranty Duration":           ["DATE", "TIME", "CARDINAL", "QUANTITY"],
    "Price Restrictions":          ["MONEY", "PERCENT", "CARDINAL", "ORG"],
    "Minimum Commitment":          ["MONEY", "CARDINAL", "QUANTITY", "PERCENT"],
    "Volume Restriction":          ["CARDINAL", "QUANTITY", "PERCENT", "MONEY"],
    "Insurance":                   ["MONEY", "CARDINAL", "ORG"],
    "Anti-Assignment":             ["ORG", "PERSON"],
    "Change of Control":           ["ORG", "PERSON"],
    "Revenue or Profit Sharing":   ["PERCENT", "MONEY", "CARDINAL"],
    "Source Code Escrow":          ["ORG"],
    "Covenant Not to Sue":         ["ORG", "PERSON", "DATE"],
    "Third Party Beneficiary":     ["ORG", "PERSON"],
    "Most Favored Nation":         ["ORG", "PERSON"],
    "Cap on Liability":            ["MONEY", "PERCENT", "CARDINAL"],
    "Liquidated Damages":          ["MONEY", "CARDINAL", "DATE"],
    "Force Majeure":               ["DATE", "TIME", "GPE", "CARDINAL"],
    "Dispute Resolution":          ["GPE", "LOC", "ORG"],
    "Arbitration":                 ["GPE", "LOC", "ORG"],
    "Governing Jurisdiction":      ["GPE", "LOC", "ORG"],
    "Notice":                      ["DATE", "TIME", "CARDINAL", "ORG"],
    "Amendment":                   ["DATE", "CARDINAL", "ORG"],
    "Entire Agreement":            ["DATE", "ORG"],
    "Severability":                ["ORG"],
    "Waiver":                      ["DATE", "ORG"],
    "Renewal Term":                ["DATE", "TIME", "CARDINAL"],
    "Notice Period to Terminate Renewal": ["DATE", "TIME", "CARDINAL"],
}
DEFAULT_SCHEMA    = ["ORG", "PERSON", "DATE", "GPE", "MONEY", "CARDINAL"]
NER_THRESHOLD     = 0.75


def stage3_ner(clause_spans: dict) -> dict:
    logger.info("Stage 3: Clause-conditioned NER...")


    if _ner_pipeline is None:
        raise RuntimeError("NER pipeline not initialized. Call preload_models() first.")

    results = {}

    for clause_type, span_data in clause_spans.items():
        span_text    = span_data["span"] if isinstance(span_data, dict) else span_data
        valid_labels = CLAUSE_SCHEMA.get(clause_type, DEFAULT_SCHEMA)

        try:
            raw_ents = _ner_pipeline(span_text) or []
        except Exception as e:
            logger.debug(f"  Stage 3 NER error '{clause_type}': {e}")
            results[clause_type] = []
            continue

        filtered = []
        seen     = set()

        for ent in raw_ents:
            if not isinstance(ent, dict):
                continue

            label = ent.get("entity_group", "")
            text  = ent.get("word", "").replace(" ##", "").replace("##", "").strip()
            score = ent.get("score", 0.0)
            key   = (text.lower(), label)

            if not text or score <= NER_THRESHOLD:
                continue
            if label not in valid_labels:
                continue
            if key in seen:
                continue

            seen.add(key)
            filtered.append({
                "text": text,
                "label": label,
                "score": round(score, 4)
            })

        results[clause_type] = filtered

    total = sum(len(v) for v in results.values())
    logger.info(f"Stage 3 done: {total} entities across {len(results)} clauses")
    return results


PRIORITY_CLAUSES = [
    "Parties", "Effective Date", "Governing Law",
    "Termination for Convenience", "Confidentiality",
    "Payment Terms", "Limitation of Liability",
    "Indemnification", "IP Ownership Assignment",
]


def _build_entity_header(entities_per_clause: dict) -> str:
    parts   = []
    ordered = [k for k in PRIORITY_CLAUSES if k in entities_per_clause]
    rest    = [k for k in entities_per_clause if k not in PRIORITY_CLAUSES]
    for k in ordered + rest:
        ents = entities_per_clause.get(k, [])
        if ents:
            vals = ", ".join(e["text"] for e in ents)
            parts.append(f"{k}: {vals}")
    return ". ".join(parts)


def _build_clause_body(clause_spans: dict) -> str:
    ordered = [k for k in PRIORITY_CLAUSES if k in clause_spans]
    rest    = [k for k in clause_spans if k not in PRIORITY_CLAUSES]
    spans   = []
    for k in ordered + rest:
        val = clause_spans[k]
        spans.append(val["span"] if isinstance(val, dict) else val)
    return " ".join(spans)


def stage4_summarize(clause_spans: dict, entities_per_clause: dict) -> dict:
    import torch

    logger.info("Stage 4: Entity-aware summarization...")

    if _tokenizer is None:
        raise ValueError("Tokenizer not loaded. Ensure preload_models() was called successfully.")

    header = _build_entity_header(entities_per_clause)
    body   = _build_clause_body(clause_spans)
    prompt = (header + ". " + body) if header else body

    input_ids = _tokenizer.encode(
        prompt,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
    )
    prompt_tokens = input_ids.shape[1]
    logger.info(f"Stage 4: prompt = {prompt_tokens} tokens")

    with torch.no_grad():
        out = _model.generate(
            input_ids,
            num_beams=4,
            no_repeat_ngram_size=3,
            length_penalty=2.0,
            min_length=60,
            max_length=200,
            early_stopping=True,
        )

    summary = _tokenizer.decode(out[0], skip_special_tokens=True,
                                 clean_up_tokenization_spaces=True)
    logger.info(f"Stage 4 done: {len(summary)} chars")
    return {"summary": summary, "prompt_tokens": prompt_tokens, "prompt": prompt}


def run_pipeline(pdf_bytes: bytes, filename: str) -> dict:
    text = stage1_ocr(pdf_bytes)
    return run_pipeline_on_text(text, filename)


def run_pipeline_on_text(
    text: str,
    filename: str,
    gold_spans: Optional[dict] = None,
) -> dict:
    preload_models()
    if not text or len(text.strip()) < 50:
        return {"error": "Text too short"}

    if not is_legal_text(text):
        return {"error": "Uploaded content does not appear to be a legal document"}


    clause_spans_full = stage2_clause_segmentation(text)


    entities = stage3_ner(clause_spans_full)


    summary_result = stage4_summarize(clause_spans_full, entities)


    clauses_clean = {}
    for clause_type in set(list(clause_spans_full.keys()) + list(entities.keys())):
        span_data = clause_spans_full.get(clause_type, {})
        clauses_clean[clause_type] = {
            "span":     span_data.get("span", "") if isinstance(span_data, dict) else "",
            "score":    span_data.get("score", 0.0) if isinstance(span_data, dict) else 0.0,
            "entities": entities.get(clause_type, []),
        }

    return {
        "id":          str(uuid.uuid4())[:8],
        "filename":    filename,
        "timestamp":   datetime.utcnow().isoformat(),
        "clauses":     clauses_clean,
        "summary":     summary_result["summary"],
        "prompt":      summary_result["prompt"],
        "promptTokens": summary_result["prompt_tokens"],
        "pipeline":    "v2",

    }
