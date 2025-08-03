import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple

# Import constants and utility functions from other modules
from config import EMBEDDING_MODEL_NAME
from src.data_processing import clean_text

# --- Embedding Model Initialization ---

# Load the SentenceTransformer model from Hugging Face.
# This model is specifically trained to create meaningful embeddings for sentences and paragraphs.
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# --- Embedding Generation ---


def generate_embeddings(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generates vector embeddings for a list of documents.

    Args:
        documents: A list of document dictionaries.

    Returns:
        The same list of documents, with an "embedding" key added to each.
    """
    # Clean the content of each document before creating embeddings.
    content_batch = [clean_text(doc["content"]) for doc in documents]

    # Use the model's encode method to generate embeddings for all documents in a single batch.
    # This is highly efficient. show_progress_bar provides visual feedback.
    embedding_vectors = embedding_model.encode(content_batch, show_progress_bar=True)

    # Add the generated embedding vector back into its corresponding document dictionary.
    for i, doc in enumerate(documents):
        doc["embedding"] = embedding_vectors[i]

    return documents


# --- FAISS Indexing ---


def create_faiss_index(
    embeddings_data: List[Dict[str, Any]],
) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    """
    Creates a FAISS index for fast similarity searching of embeddings.

    Args:
        embeddings_data: A list of documents, each containing an "embedding".

    Returns:
        A tuple containing the created FAISS index and the associated metadata.
    """
    if not embeddings_data:
        raise ValueError("No embeddings were provided to create the index.")

    # Stack all embedding vectors into a single 2D numpy matrix.
    # FAISS requires this matrix to be of type float32.
    embedding_matrix = np.vstack(
        [item["embedding"] for item in embeddings_data]
    ).astype("float32")

    # The metadata list will store the original content and source information.
    # We map it by index to the rows in the embedding_matrix.
    metadata = [
        {"content": item["content"], "metadata": item["metadata"]}
        for item in embeddings_data
    ]

    # Get the dimensionality of the embeddings (e.g., 384 for all-MiniLM-L6-v2).
    dimension = embedding_matrix.shape[1]

    # Create a flat FAISS index. IndexFlatL2 performs an exact search using L2 distance (Euclidean distance).
    index = faiss.IndexFlatL2(dimension)

    # Add the embedding matrix to the index.
    index.add(embedding_matrix)

    return index, metadata


# --- Query Processing and Searching ---


def process_query(query: str) -> np.ndarray:
    """Cleans and generates an embedding for a single user query."""
    return embedding_model.encode(clean_text(query))


def search_index(
    query_embedding: np.ndarray,
    index: faiss.Index,
    metadata: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Searches the FAISS index for the most similar documents to the query.

    Args:
        query_embedding: The vector embedding of the user's query.
        index: The FAISS index to search.
        metadata: The list of metadata associated with the documents in the index.
        top_k: The number of top results to return.

    Returns:
        A list of the top_k most relevant document dictionaries.
    """
    # The query embedding needs to be a 2D array for the search method.
    query_vector = np.array([query_embedding]).astype("float32")

    # The search method returns distances and indices of the top_k results.
    _, indices = index.search(query_vector, top_k)

    # Use the returned indices to retrieve the corresponding metadata for the top documents.
    return [metadata[idx] for idx in indices[0]]
