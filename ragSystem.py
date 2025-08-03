import os
import pandas as pd
import json
import re
import numpy as np
import faiss
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import streamlit as st

# Step 1: Data Ingestion and Preprocessing


def process_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    return content


def process_csv_file(file_path):
    df = pd.read_csv(file_path)
    content_list = []
    for _, row in df.iterrows():
        content = " ".join([str(value) for value in row.values])
        content_list.append(content)
    return content_list


def process_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    content = json.dumps(data)
    return content


def process_pdf_file(file_path):
    reader = PdfReader(file_path)
    text = "".join(page.extract_text() for page in reader.pages)
    return text


def ingest_data(data_directory):
    documents = []
    for root, _, files in os.walk(data_directory):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".txt"):
                content = process_text_file(file_path)
            elif file.endswith(".csv"):
                content = " ".join(process_csv_file(file_path))
            elif file.endswith(".json"):
                content = process_json_file(file_path)
            elif file.endswith(".pdf"):
                content = process_pdf_file(file_path)
            else:
                continue
            documents.append({"content": content, "metadata": {"file_name": file}})
    if not documents:
        raise ValueError("No documents found in the data directory.")
    return documents


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


# Step 2: Embedding Generation and Indexing

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(documents):
    embeddings = []
    for doc in documents:
        cleaned_content = clean_text(doc["content"])
        embedding = embedding_model.encode(cleaned_content)
        embeddings.append(
            {
                "embedding": embedding,
                "metadata": doc["metadata"],
                "content": doc["content"],
            }
        )
    return embeddings


def create_faiss_index(embeddings):
    if not embeddings:
        raise ValueError(
            "No embeddings were generated. Ensure valid documents are provided."
        )
    embedding_vectors = [item["embedding"] for item in embeddings]
    metadata = [
        {"metadata": item["metadata"], "content": item["content"]}
        for item in embeddings
    ]
    embedding_matrix = np.vstack(embedding_vectors).astype("float32")
    dimension = embedding_matrix.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_matrix)
    return index, metadata


# Step 3: Retrieval Component


def process_query(query):
    cleaned_query = clean_text(query)
    return embedding_model.encode(cleaned_query)


def search_index(query_embedding, index, metadata, top_k=3):
    query_vector = np.array([query_embedding]).astype("float32")
    distances, indices = index.search(query_vector, top_k)
    return [metadata[idx] for idx in indices[0]]


# Step 4: Generative AI Model Integration

device = "cuda" if torch.cuda.is_available() else "cpu"

# Model options
MODEL_OPTIONS = {
    "distilgpt2": {
        "model_name": "distilgpt2",
        "tokenizer_name": "distilgpt2",
        "model_class": AutoModelForCausalLM,
        "requires_attention_mask": False,
    },
    "flan-t5-small": {
        "model_name": "google/flan-t5-small",
        "tokenizer_name": "google/flan-t5-small",
        "model_class": AutoModelForSeq2SeqLM,
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


def load_model(model_choice):
    model_info = MODEL_OPTIONS[model_choice]
    tokenizer = AutoTokenizer.from_pretrained(model_info["tokenizer_name"])
    model = (
        model_info["model_class"].from_pretrained(model_info["model_name"]).to(device)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model, model_info["requires_attention_mask"]


def summarize_documents(retrieved_docs, max_length=512):
    return " ".join(doc["content"][:max_length] for doc in retrieved_docs)


def construct_prompt(retrieved_docs, user_query):
    summarized_content = summarize_documents(retrieved_docs)
    return (
        f"You are a civil engineering assistant. Based on the documents below:\n\n"
        f"{summarized_content}\n\n"
        f"Question: {user_query}\nAnswer:"
    )


def generate_response(prompt, tokenizer, model, requires_attention_mask, preference):
    # Adjust generation parameters based on user preference
    if preference == "Precision":
        # Parameters for precise and accurate responses
        generation_args = {
            "max_new_tokens": 150,
            "num_beams": 5,
            "early_stopping": True,
            "no_repeat_ngram_size": 2,
        }
    elif preference == "Creativity":
        # Parameters for more creative responses
        generation_args = {
            "max_new_tokens": 150,
            "do_sample": True,
            "top_p": 0.95,
            "temperature": 0.9,
            "num_return_sequences": 1,
            "early_stopping": True,
        }
    elif preference == "Conciseness":
        # Parameters for concise responses
        generation_args = {
            "max_new_tokens": 50,
            "num_beams": 5,
            "early_stopping": True,
            "no_repeat_ngram_size": 2,
        }
    else:
        # Default parameters
        generation_args = {
            "max_new_tokens": 100,
            "num_beams": 5,
            "early_stopping": True,
        }

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, padding=True, max_length=512
    ).to(device)

    if requires_attention_mask:
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            **generation_args,
        )
    else:
        outputs = model.generate(input_ids=inputs.input_ids, **generation_args)

    return (
        tokenizer.decode(outputs[0], skip_special_tokens=True)
        .split("Answer:")[-1]
        .strip()
    )


# Step 5: End-to-End RAG Pipeline


def rag_system(user_query, index, metadata, model_choice, preference):
    tokenizer, model, requires_attention_mask = load_model(model_choice)
    query_embedding = process_query(user_query)
    retrieved_docs = search_index(query_embedding, index, metadata)
    prompt = construct_prompt(retrieved_docs, user_query)
    return generate_response(
        prompt, tokenizer, model, requires_attention_mask, preference
    )


# Streamlit App


def main():
    st.title("Retrieval-Augmented Generation System")
    st.subheader("Ingest, Embed, Retrieve, and Generate Responses")

    # Data directory input
    data_directory = st.text_input("Enter the data directory path:", value="data")

    # Ingest data and create index
    if st.button("Ingest Data"):
        if "index" not in st.session_state:
            try:
                documents = ingest_data(data_directory)
                embeddings = generate_embeddings(documents)
                index, metadata = create_faiss_index(embeddings)
                st.session_state["index"] = index
                st.session_state["metadata"] = metadata
                st.success("Ingestion and indexing complete.")
            except Exception as e:
                st.error(f"Error during ingestion: {e}")
        else:
            st.info("Data already ingested and indexed.")

    # Model selection input
    model_choice = st.selectbox(
        "Select the language model:",
        options=list(MODEL_OPTIONS.keys()),
        index=2,  # Default to 'flan-t5-base'
    )

    # User preference input
    preference = st.selectbox(
        "Select your preference for the response:",
        ("Default", "Precision", "Creativity", "Conciseness"),
    )

    # Query input and response generation
    user_query = st.text_input("Enter your question:")
    if st.button("Get Response"):
        if "index" in st.session_state and "metadata" in st.session_state:
            try:
                answer = rag_system(
                    user_query,
                    st.session_state["index"],
                    st.session_state["metadata"],
                    model_choice,
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
