"""Rule-based clause risk classification.

Spec section 7 lists ``risk`` as a field on every indexed chunk, and section 11
routes *Risk Explanation* queries to the grounded LLM. Without this module the
field exists but is always empty, so the risk route reasons over raw clause text
with no risk metadata behind it.

The classifier is deliberately rule-based rather than learned. Two reasons:

1. There is no labelled risk corpus in this project, so a learned classifier would
   be trained on guesses.
2. Risk has to be *explainable* to a lawyer. Every level produced here comes with
   the concrete markers that produced it (``risk_factors``), so the level can be
   argued with rather than merely asserted.

Scoring is a baseline per clause type (some clauses are simply higher stakes than
others) raised by language markers that shift risk onto the reader - uncapped
liability, unilateral discretion, irrevocable or perpetual grants, short cure
periods, automatic renewal.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

from rag.schemas import RetrievedChunk

HIGH = "high"
MEDIUM = "medium"
LOW = "low"

# Baseline risk by clause type. A limitation-of-liability clause is high stakes
# even when its language is unremarkable; a severability clause is not.
CLAUSE_BASELINE: Dict[str, float] = {
    "Limitation of Liability": 0.55,
    "Indemnification": 0.55,
    "Liquidated Damages": 0.5,
    "Non-Compete": 0.5,
    "IP Ownership Assignment": 0.45,
    "Termination for Convenience": 0.45,
    "Data Protection": 0.45,
    "Confidentiality": 0.4,
    "Payment Terms": 0.4,
    "Change of Control": 0.4,
    "Renewal Term": 0.35,
    "Arbitration": 0.35,
    "Revenue or Profit Sharing": 0.35,
    "Anti-Assignment": 0.3,
    "Governing Law": 0.3,
    "Governing Jurisdiction": 0.3,
    "Insurance": 0.3,
    "Audit Rights": 0.3,
    "Warranty Duration": 0.3,
    "Dispute Resolution": 0.3,
    "License Grant": 0.3,
    "Force Majeure": 0.25,
    "Notice": 0.2,
    "Term": 0.2,
    "Amendment": 0.2,
    "Waiver": 0.2,
    "Third Party Beneficiary": 0.2,
    "Non-Disparagement": 0.2,
    "Parties": 0.15,
    "Expiration Date": 0.15,
    "Effective Date": 0.1,
    "Severability": 0.1,
    "Entire Agreement": 0.1,
}

DEFAULT_BASELINE = 0.25

# (label, weight, pattern). Weights are additive and the total is clamped, so a
# clause hitting several markers lands high without any single marker dominating.
RISK_MARKERS: Sequence[Tuple[str, float, str]] = (
    ("unlimited liability", 0.30, r"\bunlimited\s+liabilit"),
    ("no liability cap", 0.28, r"\b(?:no|without)\s+(?:any\s+)?(?:cap|limit)\b"),
    ("sole discretion", 0.22, r"\bsole\s+(?:and\s+absolute\s+)?discretion\b"),
    ("termination without cause", 0.20, r"\bwithout\s+cause\b"),
    ("irrevocable grant", 0.20, r"\birrevocabl"),
    ("indemnify and hold harmless", 0.20, r"\bhold\s+harmless\b"),
    ("perpetual term", 0.18, r"\bperpetu"),
    ("waiver of jury trial", 0.18, r"\bwaive[sd]?\s+(?:any\s+)?(?:right\s+to\s+a?\s*)?jury\b"),
    ("broad indemnity", 0.16, r"\bindemnif\w*\s+(?:and\s+defend|against\s+all|for\s+any)"),
    ("liquidated damages", 0.16, r"\bliquidated\s+damages\b"),
    ("automatic renewal", 0.16, r"\bautomatical\w*\s+renew|\bauto-?renew"),
    (
        "unilateral amendment",
        0.16,
        r"\b(?:may|reserves?\s+the\s+right\s+to)\s+(?:unilaterally\s+)?(?:amend|modify|change)\b",
    ),
    ("exclusive rights", 0.14, r"\bexclusiv"),
    (
        "assignment without consent",
        0.14,
        r"\bwithout\s+(?:the\s+)?(?:prior\s+)?(?:written\s+)?consent\b",
    ),
    (
        "consequential damages excluded",
        0.14,
        r"\b(?:no|not)\s+(?:be\s+)?liable\s+for\s+(?:any\s+)?(?:indirect|consequential|special|punitive)",
    ),
    ("personal data processing", 0.12, r"\bpersonal\s+data\b|\bgdpr\b"),
    ("injunctive relief", 0.10, r"\binjunctive\s+relief\b"),
    ("survives termination", 0.10, r"\bsurviv\w*\s+(?:any\s+)?(?:termination|expiration)"),
)

# Markers that reduce risk: the clause explicitly bounds the exposure.
MITIGATION_MARKERS: Sequence[Tuple[str, float, str]] = (
    (
        "liability capped",
        0.15,
        r"\b(?:shall\s+not\s+exceed|capped\s+at|maximum\s+(?:aggregate\s+)?liabilit)",
    ),
    ("mutual obligation", 0.08, r"\bmutual\w*\b|\beach\s+party\s+shall\b"),
    ("cure period granted", 0.08, r"\bcure\s+period\b|\bopportunity\s+to\s+cure\b"),
)

# A cure or notice period shorter than this many days is itself a risk marker.
SHORT_CURE_DAYS = 15
SHORT_CURE_WEIGHT = 0.14

HIGH_THRESHOLD = 0.60
MEDIUM_THRESHOLD = 0.35


@dataclass
class RiskAssessment:
    level: str
    score: float
    factors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "risk": self.level,
            "risk_score": round(self.score, 4),
            "risk_factors": list(self.factors),
        }


def _short_cure_factor(text: str) -> List[str]:
    """Find a cure or notice window shorter than ``SHORT_CURE_DAYS``.

    Contracts spell the number both ways round - "30 days" and "thirty (30) days" -
    so the digits may sit inside the parenthetical rather than before it. Both forms
    have to match or the marker silently never fires on real drafting.
    """

    for match in re.finditer(
        r"\b(?:(\d{1,3})\s*(?:\([^)]*\)\s*)?|\w+\s*\((\d{1,3})\)\s*)days?\b"
        r"[^.]{0,80}?\b(?:cure|remedy|notice|notify)\b",
        text,
        re.IGNORECASE,
    ):
        raw = match.group(1) or match.group(2)

        if not raw:
            continue

        days = int(raw)

        if 0 < days <= SHORT_CURE_DAYS:
            return [f"short cure or notice period ({days} days)"]

    return []


def assess(text: str, clause_type: str = "") -> RiskAssessment:
    """Score one clause. Pure function - no model, no I/O, deterministic."""

    body = (text or "").strip()

    if not body:
        return RiskAssessment(level=LOW, score=0.0, factors=[])

    lowered = body.lower()
    score = CLAUSE_BASELINE.get(clause_type, DEFAULT_BASELINE)
    factors: List[str] = []

    for label, weight, pattern in RISK_MARKERS:
        if re.search(pattern, lowered, re.IGNORECASE):
            score += weight
            factors.append(label)

    for factor in _short_cure_factor(lowered):
        score += SHORT_CURE_WEIGHT
        factors.append(factor)

    for label, weight, pattern in MITIGATION_MARKERS:
        if re.search(pattern, lowered, re.IGNORECASE):
            score -= weight
            factors.append(f"mitigated: {label}")

    score = max(0.0, min(1.0, score))

    if score >= HIGH_THRESHOLD:
        level = HIGH
    elif score >= MEDIUM_THRESHOLD:
        level = MEDIUM
    else:
        level = LOW

    if not factors:
        factors.append(
            f"baseline risk for {clause_type}" if clause_type else "no risk markers found"
        )

    return RiskAssessment(level=level, score=score, factors=factors)


def annotate(chunk: RetrievedChunk) -> RetrievedChunk:
    """Populate ``risk``, ``risk_score`` and ``risk_factors`` in place.

    An existing non-empty ``risk`` from the upstream v3 pipeline is respected: the
    pipeline saw the whole document, this classifier only sees one clause.
    """

    if chunk.risk:
        return chunk

    assessment = assess(chunk.text, chunk.clause_type)

    chunk.risk = assessment.level
    chunk.risk_score = assessment.score
    chunk.risk_factors = assessment.factors

    return chunk


def annotate_many(chunks: Iterable[RetrievedChunk]) -> List[RetrievedChunk]:
    return [annotate(chunk) for chunk in chunks]
