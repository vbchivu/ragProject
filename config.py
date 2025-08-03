from typing import Dict, Any

# -- Data and Embedding Configuration --
DATA_DIRECTORY: str = "data"
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200
TOP_K: int = 3

# -- Model Configuration --
MODEL_OPTIONS: Dict[str, Dict[str, Any]] = {
    "distilgpt2": {
        "model_name": "distilgpt2",
        "tokenizer_name": "distilgpt2",
        "model_class_str": "AutoModelForCausalLM",
        "requires_attention_mask": False,
    },
    "flan-t5-small": {
        "model_name": "google/flan-t5-small",
        "tokenizer_name": "google/flan-t5-small",
        "model_class_str": "AutoModelForSeq2SeqLM",
        "requires_attention_mask": True,
    },
    "flan-t5-base": {
        "model_name": "google/flan-t5-base",
        "tokenizer_name": "google/flan-t5-base",
        "model_class_str": "AutoModelForSeq2SeqLM",
        "requires_attention_mask": True,
    },
    "flan-t5-large": {
        "model_name": "google/flan-t5-large",
        "tokenizer_name": "google/flan-t5-large",
        "model_class_str": "AutoModelForSeq2SeqLM",
        "requires_attention_mask": True,
    },
}

# -- Generation Parameters --
GENERATION_PREFERENCES: Dict[str, Dict[str, Any]] = {
    "Precision": {
        "max_new_tokens": 150,
        "num_beams": 5,
        "early_stopping": True,
        "no_repeat_ngram_size": 2,
    },
    "Creativity": {
        "max_new_tokens": 150,
        "do_sample": True,
        "top_p": 0.95,
        "temperature": 0.9,
        "num_return_sequences": 1,
        "early_stopping": True,
    },
    "Conciseness": {
        "max_new_tokens": 50,
        "num_beams": 5,
        "early_stopping": True,
        "no_repeat_ngram_size": 2,
    },
    "Default": {"max_new_tokens": 100, "num_beams": 5, "early_stopping": True},
}
