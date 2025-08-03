import os
import json
import re
from typing import List, Dict, Any, Tuple

import faiss
import numpy as np
import pandas as pd
import streamlit as st
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

# --- Constants for Configuration ---

# -- Data and Embedding Settings --
DATA_DIRECTORY: str = "data"  # Specifies the folder where input documents are stored.
EMBEDDING_MODEL_NAME: str = (
    "all-MiniLM-L6-v2"  # The pre-trained model for converting text to vector embeddings.
)
CHUNK_SIZE: int = 1000  # The maximum size (in characters) of a text chunk.
CHUNK_OVERLAP: int = (
    200  # The number of characters to overlap between consecutive chunks to maintain context.
)
TOP_K: int = (
    3  # The number of most relevant document chunks to retrieve for a given query.
)

# -- Language Model (LLM) Options --
# A dictionary to define and configure different generative models that can be used.
MODEL_OPTIONS: Dict[str, Dict[str, Any]] = {
    "distilgpt2": {
        "model_name": "distilgpt2",
        "tokenizer_name": "distilgpt2",
        "model_class": AutoModelForCausalLM,  # Causal LM, good for text completion.
        "requires_attention_mask": False,  # Some models don't require an explicit attention mask.
    },
    "flan-t5-small": {
        "model_name": "google/flan-t5-small",
        "tokenizer_name": "google/flan-t5-small",
        "model_class": AutoModelForSeq2SeqLM,  # Sequence-to-sequence LM, good for instruction-following tasks.
        "requires_attention_mask": True,
    },
    "flan-t5-base": {
        "model_name": "google/flan-t5-base",
        "tokenizer_name": "google/flan-t5-base",
        "model_class": AutoModelForSeq2SeqLM,
        "requires_attention_mask": True,
    },
    "flan-t5-large": {
        "model_name": "google/flan-t5-large",
        "tokenizer_name": "google/flan-t5-large",
        "model_class": AutoModelForSeq2SeqLM,
        "requires_attention_mask": True,
    },
}

# -- Generation Parameter Presets --
# Defines different "personalities" for the LLM's response generation.
GENERATION_PREFERENCES: Dict[str, Dict[str, Any]] = {
    "Precision": {
        "max_new_tokens": 150,
        "num_beams": 5,  # Uses beam search to find more optimal sequences.
        "early_stopping": True,
        "no_repeat_ngram_size": 2,  # Prevents the model from repeating the same 2-word phrases.
    },
    "Creativity": {
        "max_new_tokens": 150,
        "do_sample": True,  # Enables sampling to generate more diverse text.
        "top_p": 0.95,  # Nucleus sampling: considers only the most probable tokens with a cumulative probability of 95%.
        "temperature": 0.9,  # Higher temperature (e.g., 0.9) makes output more random and creative.
        "num_return_sequences": 1,
        "early_stopping": True,
    },
    "Conciseness": {
        "max_new_tokens": 50,  # Limits the response to be very short.
        "num_beams": 5,
        "early_stopping": True,
        "no_repeat_ngram_size": 2,
    },
    "Default": {"max_new_tokens": 100, "num_beams": 5, "early_stopping": True},
}

# --- Step 1: Data Ingestion and Preprocessing ---

# Initialize a text splitter to break large documents into smaller, overlapping chunks.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len
)


# Defines functions to extract text from different file formats.
def process_text_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def process_csv_file(file_path: str) -> List[str]:
    df = pd.read_csv(file_path)
    # Converts each row of the CSV into a single string, with column names as keys.
    return [
        " ".join([f"{col}: {str(val)}" for col, val in row.items()])
        for _, row in df.iterrows()
    ]


def process_json_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return json.dumps(data, indent=2)


def process_pdf_file(file_path: str) -> str:
    reader = PdfReader(file_path)
    # Extracts text from each page of the PDF and concatenates it.
    return "".join(page.extract_text() for page in reader.pages if page.extract_text())


# A dictionary that maps file extensions to their corresponding processing function and processing type.
FILE_PROCESSORS: Dict[str, Dict[str, Any]] = {
    ".txt": {
        "processor": process_text_file,
        "type": "chunk",
    },  # "chunk" type will be split by the text_splitter.
    ".csv": {
        "processor": process_csv_file,
        "type": "row",
    },  # "row" type treats each CSV row as a separate document.
    ".json": {"processor": process_json_file, "type": "chunk"},
    ".pdf": {"processor": process_pdf_file, "type": "chunk"},
}


def ingest_data(data_directory: str) -> List[Dict[str, Any]]:
    """Walks through the data directory, processes each file, and returns a list of document chunks."""
    documents: List[Dict[str, Any]] = []
    for root, _, files in os.walk(data_directory):  # Traverse the specified directory.
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()

            if file_ext in FILE_PROCESSORS:
                handler = FILE_PROCESSORS[file_ext]
                processor = handler["processor"]
                processing_type = handler["type"]
                content = processor(file_path)

                # For file types like CSV, each row is treated as a separate document.
                if processing_type == "row":
                    for i, item in enumerate(content):
                        documents.append(
                            {
                                "content": item,
                                "metadata": {
                                    "file_name": file,
                                    "source_id": f"row_{i+1}",
                                },
                            }
                        )
                # For text-heavy files, the content is split into chunks.
                elif processing_type == "chunk":
                    text_chunks = text_splitter.split_text(content)
                    for i, chunk in enumerate(text_chunks):
                        documents.append(
                            {
                                "content": chunk,
                                "metadata": {
                                    "file_name": file,
                                    "source_id": f"chunk_{i+1}",
                                },
                            }
                        )

    if not documents:
        raise ValueError("No processable documents found in the data directory.")
    return documents


def clean_text(text: str) -> str:
    """A simple text cleaning function to normalize text before embedding."""
    text = text.lower()  # Convert to lowercase.
    text = re.sub(
        r"\s+", " ", text
    )  # Replace multiple whitespace characters with a single space.
    return text.strip()


# --- Step 2: Embedding Generation and Indexing ---

# Load the sentence transformer model specified in the constants. This model turns text into vectors.
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


def generate_embeddings(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generates vector embeddings for a list of document chunks."""
    content_batch = [clean_text(doc["content"]) for doc in documents]
    # The `encode` method efficiently converts a batch of texts into embeddings.
    embedding_vectors = embedding_model.encode(content_batch, show_progress_bar=True)
    for i, doc in enumerate(documents):
        doc["embedding"] = embedding_vectors[
            i
        ]  # Add the generated embedding to each document dictionary.
    return documents


def create_faiss_index(
    embeddings_data: List[Dict[str, Any]],
) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    """Creates a FAISS index for fast similarity searching of embeddings."""
    if not embeddings_data:
        raise ValueError("No embeddings were generated.")
    # Stack all individual embedding vectors into a single 2D NumPy matrix.
    embedding_matrix = np.vstack(
        [item["embedding"] for item in embeddings_data]
    ).astype("float32")
    # Separate metadata from embeddings, as the index only stores vectors.
    metadata = [
        {"content": item["content"], "metadata": item["metadata"]}
        for item in embeddings_data
    ]
    dimension = embedding_matrix.shape[
        1
    ]  # Get the dimensionality of the embeddings (e.g., 384 for MiniLM).
    index = faiss.IndexFlatL2(dimension)  # Create a flat L2 (Euclidean) distance index.
    index.add(embedding_matrix)  # Add the embedding matrix to the index.
    return index, metadata


# --- Step 3: Retrieval Component ---


def process_query(query: str) -> np.ndarray:
    """Converts the user's text query into an embedding vector using the same model."""
    return embedding_model.encode(clean_text(query))


def search_index(
    query_embedding: np.ndarray,
    index: faiss.Index,
    metadata: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Searches the FAISS index for the `top_k` most similar document embeddings."""
    query_vector = np.array([query_embedding]).astype("float32")
    # The `search` method returns the distances and the indices of the nearest neighbors.
    _, indices = index.search(query_vector, top_k)
    # Retrieve the original content and metadata for the top matching documents using their indices.
    return [metadata[idx] for idx in indices[0]]


# --- Step 4 & 5: Generation and Streamlit App ---

# Set the computation device to GPU if available, otherwise CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource  # Streamlit decorator to cache the model loading, preventing reloads on every interaction.
def load_model(model_choice: str) -> Tuple[Any, Any, bool]:
    """Loads the selected language model and tokenizer from Hugging Face."""
    model_info = MODEL_OPTIONS[model_choice]
    tokenizer = AutoTokenizer.from_pretrained(model_info["tokenizer_name"])
    model = (
        model_info["model_class"].from_pretrained(model_info["model_name"]).to(device)
    )
    # Some models require a padding token to be explicitly set.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model, model_info["requires_attention_mask"]


def format_context(retrieved_docs: List[Dict[str, Any]]) -> str:
    """Formats the retrieved documents into a single string to be used as context in the prompt."""
    return "\n---\n".join(
        [
            f"Source: {doc['metadata']['file_name']} ({doc['metadata']['source_id']})\nContent: {doc['content']}"
            for doc in retrieved_docs
        ]
    )


def construct_prompt(retrieved_docs: List[Dict[str, Any]], user_query: str) -> str:
    """Builds the final prompt for the LLM by combining instructions, context, and the user query."""
    context = format_context(retrieved_docs)
    return (
        f"You are a civil engineering assistant. Based on the context provided below from various documents, answer the user's question. "
        f"Cite the source file and id for the information you use.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {user_query}\nAnswer:"
    )


def generate_response(
    prompt: str,
    tokenizer: Any,
    model: Any,
    requires_attention_mask: bool,
    preference: str,
) -> str:
    """Generates a response from the LLM based on the constructed prompt and generation preferences."""
    # Get the generation parameters (e.g., for "Creativity" or "Precision") from the constants.
    generation_args = GENERATION_PREFERENCES.get(
        preference, GENERATION_PREFERENCES["Default"]
    )
    # Convert the prompt string into tokens the model can understand.
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, padding=True, max_length=2048
    ).to(device)

    # The generate call differs slightly depending on the model's requirements for an attention mask.
    if requires_attention_mask:
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            **generation_args,
        )
    else:
        outputs = model.generate(input_ids=inputs.input_ids, **generation_args)

    # Convert the generated token IDs back into a readable string.
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Robustly parse the output to extract only the text following "Answer:".
    if "Answer:" in decoded_output:
        return decoded_output.split("Answer:")[-1].strip()
    return decoded_output.strip()


def rag_system(
    user_query: str,
    index: faiss.Index,
    metadata: List[Dict[str, Any]],
    tokenizer: Any,
    model: Any,
    requires_attention_mask: bool,
    preference: str,
) -> str:
    """Orchestrates the entire RAG pipeline from query to final answer."""
    query_embedding = process_query(user_query)  # 1. Embed the user's query.
    retrieved_docs = search_index(
        query_embedding, index, metadata, top_k=TOP_K
    )  # 2. Retrieve relevant docs.
    prompt = construct_prompt(retrieved_docs, user_query)  # 3. Construct the prompt.
    # 4. Generate the final response.
    return generate_response(
        prompt, tokenizer, model, requires_attention_mask, preference
    )


def main() -> None:
    """The main function that runs the Streamlit web application."""
    st.title("Retrieval-Augmented Generation System")
    st.header("Final Version: All Code Quality Cycles Complete ✨")
    st.info(
        "This app incorporates performance optimizations, structural refactors, and code quality improvements."
    )

    data_directory = st.text_input(
        "Enter the data directory path:", value=DATA_DIRECTORY
    )

    if st.button("Ingest Data"):
        # Clear previous index and metadata if re-ingesting data.
        if "index" in st.session_state:
            st.session_state.pop("index", None)
            st.session_state.pop("metadata", None)

        with st.spinner("Ingesting and chunking documents..."):
            try:
                documents = ingest_data(data_directory)
                st.success(
                    f"Successfully ingested and chunked {len(documents)} documents."
                )
                with st.spinner(
                    f"Generating embeddings for all chunks using '{EMBEDDING_MODEL_NAME}'..."
                ):
                    embeddings_data = generate_embeddings(documents)
                    index, metadata = create_faiss_index(embeddings_data)
                    # Store the index and metadata in Streamlit's session state to persist them across interactions.
                    st.session_state["index"] = index
                    st.session_state["metadata"] = metadata
                    st.success("Indexing complete.")
            except Exception as e:
                st.error(f"Error during ingestion: {e}")

    # UI for selecting the language model.
    model_choice = st.selectbox(
        "Select the language model:", options=list(MODEL_OPTIONS.keys()), index=2
    )

    try:
        with st.spinner(f"Loading {model_choice}..."):
            tokenizer, model, requires_attention_mask = load_model(model_choice)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()  # Stop the app if the model fails to load.

    # UI for selecting the generation preference.
    preference = st.selectbox(
        "Select your preference for the response:",
        options=list(GENERATION_PREFERENCES.keys()),
    )

    user_query = st.text_input("Enter your question:")
    if st.button("Get Response"):
        # Ensure data has been ingested and indexed before proceeding.
        if "index" in st.session_state and "metadata" in st.session_state:
            with st.spinner("Retrieving context and generating response..."):
                try:
                    # Call the main RAG pipeline function.
                    answer = rag_system(
                        user_query,
                        st.session_state["index"],
                        st.session_state["metadata"],
                        tokenizer,
                        model,
                        requires_attention_mask,
                        preference,
                    )
                    st.subheader("Assistant's Response:")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Error generating response: {e}")
        else:
            st.warning("Please ingest data and create the index first.")


if __name__ == "__main__":
    main()
