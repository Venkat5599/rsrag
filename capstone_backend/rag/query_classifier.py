import re
from typing import Dict, List, Optional, Tuple

from . import semantic
from .schemas import QueryIntent

INTENT_PATTERNS: Dict[QueryIntent, List[Tuple[str, float]]] = {
    QueryIntent.SUMMARIZATION: [
        (r"\bsummar(y|ise|ize|ising|izing)\b", 3.0),
        (r"\boverview\b", 2.2),
        (r"\bkey (points|terms|highlights)\b", 2.2),
        (r"\bbrief me\b", 2.0),
        (r"\bwhat is this (contract|agreement|document) about\b", 2.4),
    ],
    QueryIntent.CLAUSE_COMPARISON: [
        (r"\bcompare\b", 3.0),
        (r"\bdifference(s)? between\b", 3.0),
        (r"\bversus\b|\bvs\.?\b", 2.4),
        (r"\bdiffer\b", 2.0),
        (r"\bwhich (one )?is (stronger|better|stricter|more favourable|more favorable)\b", 2.6),
        (r"\bboth (contracts|agreements|clauses)\b", 2.2),
    ],
    QueryIntent.RISK_EXPLANATION: [
        (r"\brisk(s|y)?\b", 3.0),
        (r"\bexposure\b", 2.4),
        (r"\bred flag(s)?\b", 2.6),
        (r"\bunfavou?rable\b", 2.2),
        (r"\bliabilit(y|ies) exposure\b", 2.4),
        (r"\bwhat could go wrong\b", 2.2),
    ],
    QueryIntent.LEGAL_REASONING: [
        (r"\bwhat happens if\b", 3.0),
        (r"\bare we (allowed|permitted|obligated|required)\b", 2.8),
        (r"\bcan (we|i|the \w+) (still |legally )?\b", 2.2),
        (r"\bis (it|this) (enforceable|valid|binding|legal)\b", 2.8),
        (r"\bdoes this mean\b", 2.4),
        (r"\bimplication(s)?\b", 2.2),
        (r"\bexplain why\b", 2.4),
        (r"\bwould (we|the \w+) breach\b", 2.6),
    ],
    QueryIntent.ENTITY_LOOKUP: [
        (r"\bwho (is|are|signed|owns|indemnifies)\b", 2.8),
        (r"\bname the (part(y|ies)|compan(y|ies)|signator(y|ies))\b", 3.0),
        (r"\bwhich (company|organisation|organization|entity)\b", 2.6),
        (r"\bhow much\b", 2.2),
        (r"\bwhat (date|amount|percentage)\b", 2.2),
        (r"\blist the (part(y|ies)|entities|names|dates|amounts)\b", 2.8),
    ],
    QueryIntent.CLAUSE_RETRIEVAL: [
        (r"\b(show|find|fetch|retrieve|pull up|give me|display|quote)\b[^.?!]{0,48}\b(clause|section|provision|article)\b", 3.0),
        (r"\bclause(s)? (about|on|regarding|covering)\b", 2.8),
        (r"\bwhere does (the|this) (contract|agreement) (say|state|mention)\b", 2.6),
        (r"\bsection (on|about)\b", 2.4),
        (r"\bfull text of\b", 2.4),
        (r"\bquote the\b", 2.2),
    ],
    QueryIntent.FACT_LOOKUP: [
        (r"\bwhat is the\b", 1.8),
        (r"\bhow many (days|months|years)\b", 2.4),
        (r"\bnotice period\b", 2.0),
        (r"\bwhen (does|is|will)\b", 1.8),
        (r"\bgoverning law\b", 1.6),
        (r"\bwhat (are|does) the .* (terms|term|limit|cap)\b", 1.8),
    ],
}

INTENT_PROTOTYPES: Dict[QueryIntent, List[str]] = {
    QueryIntent.SUMMARIZATION: [
        "Give me a summary of this agreement.",
        "Provide an overview of the key contract terms.",
    ],
    QueryIntent.CLAUSE_COMPARISON: [
        "Compare the termination clauses of the two agreements.",
        "How does the indemnity clause differ from the liability clause?",
    ],
    QueryIntent.RISK_EXPLANATION: [
        "What are the main risks in this contract for the supplier?",
        "Explain the liability exposure created by this clause.",
    ],
    QueryIntent.LEGAL_REASONING: [
        "What happens if the customer fails to pay on time?",
        "Are we permitted to subcontract our obligations under this agreement?",
    ],
    QueryIntent.ENTITY_LOOKUP: [
        "Who are the parties to this agreement?",
        "Which company is responsible for indemnification?",
    ],
    QueryIntent.CLAUSE_RETRIEVAL: [
        "Show me the confidentiality clause.",
        "Retrieve the section covering dispute resolution.",
    ],
    QueryIntent.FACT_LOOKUP: [
        "What is the notice period for termination?",
        "What law governs this agreement?",
    ],
}

INTENT_PRIORITY: List[QueryIntent] = [
    QueryIntent.CLAUSE_COMPARISON,
    QueryIntent.RISK_EXPLANATION,
    QueryIntent.SUMMARIZATION,
    QueryIntent.LEGAL_REASONING,
    QueryIntent.CLAUSE_RETRIEVAL,
    QueryIntent.ENTITY_LOOKUP,
    QueryIntent.FACT_LOOKUP,
]

_RULE_WEIGHT = 0.72
_SEMANTIC_WEIGHT = 0.28
_MAX_RULE_SCORE = 6.0


def _rule_scores(query: str) -> Dict[QueryIntent, float]:
    lowered = query.lower()
    scores: Dict[QueryIntent, float] = {}

    for intent, patterns in INTENT_PATTERNS.items():
        total = 0.0
        for pattern, weight in patterns:
            if re.search(pattern, lowered):
                total += weight
        scores[intent] = min(total / _MAX_RULE_SCORE, 1.0)

    return scores


def _semantic_scores(query: str, embedding_model: Optional[str]) -> Dict[QueryIntent, float]:
    scores: Dict[QueryIntent, float] = {}

    for intent, prototypes in INTENT_PROTOTYPES.items():
        scores[intent] = semantic.max_similarity(query, prototypes, embedding_model)

    return scores


def classify_query(
    query: str,
    embedding_model: Optional[str] = None,
) -> Tuple[QueryIntent, float, Dict[str, float]]:
    query = (query or "").strip()

    if not query:
        return QueryIntent.FACT_LOOKUP, 0.0, {}

    rule = _rule_scores(query)
    semantic_score = _semantic_scores(query, embedding_model)

    combined: Dict[QueryIntent, float] = {}
    for intent in INTENT_PRIORITY:
        combined[intent] = (
            _RULE_WEIGHT * rule.get(intent, 0.0)
            + _SEMANTIC_WEIGHT * semantic_score.get(intent, 0.0)
        )

    ordered = sorted(
        combined.items(),
        key=lambda item: (-item[1], INTENT_PRIORITY.index(item[0])),
    )

    best_intent, best_score = ordered[0]
    runner_up_score = ordered[1][1] if len(ordered) > 1 else 0.0

    if best_score <= 0.0:
        return QueryIntent.FACT_LOOKUP, 0.2, {k.value: v for k, v in combined.items()}

    margin = (best_score - runner_up_score) / best_score
    confidence = max(0.0, min(1.0, 0.5 * best_score + 0.5 * margin))

    return best_intent, confidence, {k.value: v for k, v in combined.items()}
