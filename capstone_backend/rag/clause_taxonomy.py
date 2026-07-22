from typing import Dict, List

CLAUSE_KEYWORDS: Dict[str, List[str]] = {
    "Parties": ["party", "parties", "between", "entered into", "counterparty"],
    "Effective Date": ["effective date", "commencement", "effective from", "shall commence"],
    "Expiration Date": ["expire", "expiration", "termination date", "end date", "term ends"],
    "Term": ["term", "duration", "initial term", "period of this agreement"],
    "Payment Terms": ["payment", "fee", "amount", "compensation", "invoice", "consideration"],
    "Confidentiality": ["confidential", "confidentiality", "disclose", "non-disclosure", "proprietary"],
    "Non-Compete": ["compete", "competition", "non-compete", "restrict competition"],
    "Non-Disparagement": ["disparage", "non-disparagement", "negative statements"],
    "Indemnification": ["indemnify", "indemnification", "hold harmless", "defend"],
    "Limitation of Liability": ["liability", "damages", "limit", "cap", "maximum liability"],
    "IP Ownership Assignment": ["intellectual property", "ownership", "assign", "work product"],
    "License Grant": ["license", "grant", "rights", "use rights"],
    "Audit Rights": ["audit", "inspect", "records", "books", "examination"],
    "Insurance": ["insurance", "coverage", "insured"],
    "Anti-Assignment": ["assign", "assignment", "transfer", "delegation"],
    "Change of Control": ["change of control", "acquisition", "merger", "ownership change"],
    "Force Majeure": ["force majeure", "acts of god", "beyond control", "unforeseen events"],
    "Dispute Resolution": ["dispute", "resolve", "resolution", "settlement"],
    "Arbitration": ["arbitration", "arbitrate", "arbitrator"],
    "Governing Law": ["governed by", "laws of", "applicable law", "governing law"],
    "Governing Jurisdiction": ["jurisdiction", "court", "venue"],
    "Notice": ["notice", "notify", "written notice", "deliver notice"],
    "Amendment": ["amend", "amendment", "modify", "modification"],
    "Renewal Term": ["renew", "renewal", "automatically renew", "auto-renew"],
    "Severability": ["severability", "invalid provision", "unenforceable provision"],
    "Entire Agreement": ["entire agreement", "whole agreement", "complete agreement"],
    "Waiver": ["waiver", "waive", "failure to enforce"],
    "Termination for Convenience": ["terminate", "termination", "without cause", "at any time"],
    "Warranty Duration": ["warranty", "guarantee", "warranty period"],
    "Third Party Beneficiary": ["third party", "beneficiary"],
    "Liquidated Damages": ["liquidated damages", "penalty", "pre-determined damages"],
    "Revenue or Profit Sharing": ["revenue", "profit", "profit sharing", "royalty"],
    "Data Protection": ["personal data", "data protection", "privacy", "gdpr", "processing"],
}

CLAUSE_IMPORTANCE: Dict[str, float] = {
    "Parties": 1.0,
    "Termination for Convenience": 1.0,
    "Limitation of Liability": 1.0,
    "Indemnification": 1.0,
    "Payment Terms": 0.95,
    "Confidentiality": 0.95,
    "Governing Law": 0.9,
    "IP Ownership Assignment": 0.9,
    "Effective Date": 0.85,
    "Expiration Date": 0.85,
    "Dispute Resolution": 0.85,
    "Arbitration": 0.8,
    "Data Protection": 0.8,
    "Insurance": 0.7,
    "Notice": 0.7,
    "Audit Rights": 0.65,
    "Amendment": 0.6,
    "Renewal Term": 0.6,
    "Severability": 0.5,
    "Waiver": 0.5,
    "Entire Agreement": 0.5,
}

DEFAULT_CLAUSE_IMPORTANCE = 0.6

ENTITY_LABEL_ALIASES: Dict[str, List[str]] = {
    "ORG": ["company", "organisation", "organization", "vendor", "supplier", "employer", "who"],
    "PERSON": ["person", "individual", "signatory", "who"],
    "DATE": ["date", "when", "deadline", "day", "month", "year"],
    "MONEY": ["amount", "price", "fee", "cost", "how much", "value"],
    "GPE": ["country", "state", "city", "where", "jurisdiction"],
    "PERCENT": ["percentage", "percent", "rate"],
}


def clause_importance(clause_type: str) -> float:
    return CLAUSE_IMPORTANCE.get(clause_type, DEFAULT_CLAUSE_IMPORTANCE)


def match_clause_types(query: str, limit: int = 4) -> List[str]:
    lowered = query.lower()
    scored = []

    for clause_type, keywords in CLAUSE_KEYWORDS.items():
        hits = sum(1 for keyword in keywords if keyword in lowered)
        if clause_type.lower() in lowered:
            hits += 2
        if hits:
            scored.append((clause_type, hits))

    scored.sort(key=lambda item: (-item[1], item[0]))
    return [clause_type for clause_type, _ in scored[:limit]]
