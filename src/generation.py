import torch
import streamlit as st
from transformers import AutoTokenizer

# We import the transformers library and use getattr to dynamically get the model class
import transformers
from typing import List, Dict, Any, Tuple

# Import configuration from our config file
from config import MODEL_OPTIONS, GENERATION_PREFERENCES

# --- Model Loading ---

# Determine the appropriate device for PyTorch (GPU if available, otherwise CPU).
device = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def load_model(model_choice: str) -> Tuple[Any, Any, bool]:
    """
    Loads and caches a language model and its tokenizer from Hugging Face.

    Args:
        model_choice: The key corresponding to the model in the MODEL_OPTIONS config.

    Returns:
        A tuple containing the tokenizer, the model, and a boolean indicating
        if an attention mask is required.
    """
    # Retrieve model details from our configuration dictionary.
    model_info = MODEL_OPTIONS[model_choice]

    # Load the tokenizer associated with the model.
    tokenizer = AutoTokenizer.from_pretrained(model_info["tokenizer_name"])

    # Get the model class dynamically from the transformers library using its string name.
    model_class = getattr(transformers, model_info["model_class_str"])

    # Load the pre-trained model and move it to the determined device (GPU/CPU).
    model = model_class.from_pretrained(model_info["model_name"]).to(device)

    # Some models don't have a padding token, so we set it to the end-of-sentence token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model, model_info["requires_attention_mask"]


# --- Prompt Engineering ---


def format_context(retrieved_docs: List[Dict[str, Any]]) -> str:
    """Formats the retrieved documents into a single string for the prompt context."""
    # Combine the content of each retrieved document, separated by a line.
    # Include the source metadata for potential citation in the response.
    return "\n---\n".join(
        [
            f"Source: {doc['metadata']['file_name']} ({doc['metadata']['source_id']})\nContent: {doc['content']}"
            for doc in retrieved_docs
        ]
    )


def construct_prompt(retrieved_docs: List[Dict[str, Any]], user_query: str) -> str:
    """Constructs the final prompt to be sent to the language model."""
    # Create the context string from the retrieved documents.
    context = format_context(retrieved_docs)

    # This is the prompt template. It instructs the model on its role, provides the context,
    # and presents the user's question.
    return (
        f"You are a helpful assistant. Based on the context provided below from various documents, "
        f"answer the user's question. If the context does not contain the answer, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {user_query}\nAnswer:"
    )


# --- Response Generation ---


def generate_response(
    prompt: str,
    tokenizer: Any,
    model: Any,
    requires_attention_mask: bool,
    preference: str,
) -> str:
    """
    Generates a response from the language model based on the given prompt.

    Args:
        prompt: The full prompt including context and the user's question.
        tokenizer: The tokenizer for the model.
        model: The language model.
        requires_attention_mask: A boolean for the model's generation call.
        preference: The user's selected preference for the response style.

    Returns:
        The generated text from the model.
    """
    # Get the generation parameters (e.g., for creativity, conciseness) from our config.
    generation_args = GENERATION_PREFERENCES.get(
        preference, GENERATION_PREFERENCES["Default"]
    )

    # Tokenize the prompt and move the resulting tensors to the active device.
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, padding=True, max_length=2048
    ).to(device)

    # Generate the output from the model.
    # We unpack the inputs dictionary and the generation arguments.
    if requires_attention_mask:
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            **generation_args,
        )
    else:
        outputs = model.generate(input_ids=inputs.input_ids, **generation_args)

    # Decode the generated token IDs back into a string.
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Clean up the output by splitting on the "Answer:" delimiter to remove the prompt part.
    if "Answer:" in decoded_output:
        return decoded_output.split("Answer:")[-1].strip()

    # Fallback in case the model doesn't follow the prompt format perfectly.
    return decoded_output.strip()
