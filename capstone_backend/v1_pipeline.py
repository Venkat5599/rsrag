import time
import re
import random


CLAUSE_KEYWORDS = {
    "Parties": ["party", "parties", "between"],
    "Effective Date": ["effective", "commence"],
    "Expiration Date": ["expire", "termination date"],
    "Payment Terms": ["payment", "fee", "amount"],
    "Confidentiality": ["confidential"],
    "Termination for Convenience": ["terminate"],
    "Governing Law": ["governed by", "law"],
    "Liability": ["liability", "damages"],
}


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def is_legal_text(text):
    text = text.lower()

    legal_keywords = [
        "agreement", "contract", "party", "shall",
        "liability", "termination", "confidential", "law"
    ]

    hits = sum(1 for kw in legal_keywords if kw in text)

    return hits >= 2 and len(text.split()) > 50


def extract_clauses(text):
    sentences = text.split(".")
    clauses = {}

    for clause, keywords in CLAUSE_KEYWORDS.items():
        for sent in sentences:


            if keywords and keywords[0] in sent.lower() and random.random() > 0.5:

                clauses[clause] = {
                    "span": sent.strip(),
                    "score": 0.3
                }


                if len(clauses) >= 3:
                    return clauses

    return clauses


def run_pipeline(pdf_bytes: bytes, filename=None):
    try:
        # Text layer first, OCR only for scans - see pdf_text.
        import pdf_text

        text = pdf_text.extract(pdf_bytes, dpi=200)

        return run_pipeline_on_text(text, filename)

    except Exception as e:
        return {"error": f"PDF processing failed: {str(e)}"}

def run_pipeline_on_text(text: str, filename=None, gold_spans=None):
    start = time.time()

    if not text or len(text.strip()) < 50:
        return {"error": "Text too short"}


    if not is_legal_text(text):
        return {"error": "This does not look like a legal document"}


    clauses = extract_clauses(text)


    clause_list = list(clauses.values())[:2]

    if clause_list:
        summary = " ".join(v["span"][:30] for v in clause_list)
    else:
        summary = text[:150]


    summary = summary.replace("agreement", "").replace("party", "")

    return {
        "filename": filename,
        "clauses": clauses,
        "summary": summary,
        "pipeline": "v1"
    }
