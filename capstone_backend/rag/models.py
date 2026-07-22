import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MODEL_CACHE_DIR = Path(os.getenv("CLNEA_MODEL_CACHE", ".hf-cache")).resolve()
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(MODEL_CACHE_DIR / "hub"))

_qa_pipeline = None
_qa_model_name = ""
_qa_failed = False

_ner_pipeline = None
_ner_model_name = ""
_ner_failed = False


def get_qa_pipeline(model_name: str):
    global _qa_pipeline, _qa_model_name, _qa_failed

    if _qa_pipeline is not None and _qa_model_name == model_name:
        return _qa_pipeline

    if _qa_failed and _qa_model_name == model_name:
        return None

    try:
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(MODEL_CACHE_DIR))
        model = AutoModelForQuestionAnswering.from_pretrained(
            model_name, cache_dir=str(MODEL_CACHE_DIR)
        )
        _qa_pipeline = pipeline(
            "question-answering", model=model, tokenizer=tokenizer, device=-1
        )
        _qa_model_name = model_name
        _qa_failed = False
    except Exception as error:
        logger.warning("Extractive QA model unavailable: %s", error)
        _qa_pipeline = None
        _qa_model_name = model_name
        _qa_failed = True

    return _qa_pipeline


def get_ner_pipeline(model_name: str):
    global _ner_pipeline, _ner_model_name, _ner_failed

    if _ner_pipeline is not None and _ner_model_name == model_name:
        return _ner_pipeline

    if _ner_failed and _ner_model_name == model_name:
        return None

    try:
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(MODEL_CACHE_DIR))
        model = AutoModelForTokenClassification.from_pretrained(
            model_name, cache_dir=str(MODEL_CACHE_DIR)
        )
        _ner_pipeline = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=-1,
        )
        _ner_model_name = model_name
        _ner_failed = False
    except Exception as error:
        logger.warning("Entity model unavailable: %s", error)
        _ner_pipeline = None
        _ner_model_name = model_name
        _ner_failed = True

    return _ner_pipeline


def reset_model_cache() -> None:
    global _qa_pipeline, _qa_model_name, _qa_failed
    global _ner_pipeline, _ner_model_name, _ner_failed

    _qa_pipeline = None
    _qa_model_name = ""
    _qa_failed = False
    _ner_pipeline = None
    _ner_model_name = ""
    _ner_failed = False
