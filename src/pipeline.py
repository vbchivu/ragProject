import faiss
from typing import List, Dict, Any

# Import configuration and modules from our project structure
from config import TOP_K
from src.retrieval import process_query, search_index
from src.generation import construct_prompt, generate_response


def rag_system(
    user_query: str,
    index: faiss.Index,
    metadata: List[Dict[str, Any]],
    tokenizer: Any,
    model: Any,
    requires_attention_mask: bool,
    preference: str,
) -> str:
    """
    Orchestrates the end-to-end Retrieval-Augmented Generation pipeline.

    Args:
        user_query: The question asked by the user.
        index: The FAISS index of the document embeddings.
        metadata: The metadata associated with the documents.
        tokenizer: The pre-loaded tokenizer.
        model: The pre-loaded language model.
        requires_attention_mask: A boolean for the model's generation call.
        preference: The user's selected response style preference.

    Returns:
        The final, generated answer from the language model.
    """
    # Step 1: Process the user's query into an embedding.
    query_embedding = process_query(user_query)

    # Step 2: Retrieve the most relevant document chunks from the FAISS index.
    retrieved_docs = search_index(query_embedding, index, metadata, top_k=TOP_K)

    # Step 3: Construct a detailed prompt for the LLM, including the retrieved context.
    prompt = construct_prompt(retrieved_docs, user_query)

    # Step 4: Generate a response from the LLM based on the prompt.
    answer = generate_response(
        prompt, tokenizer, model, requires_attention_mask, preference
    )

    return answer
