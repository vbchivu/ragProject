import streamlit as st
from typing import Dict, Any, List

# Import all necessary functions and constants from our modules
from config import (
    DATA_DIRECTORY,
    EMBEDDING_MODEL_NAME,
    MODEL_OPTIONS,
    GENERATION_PREFERENCES,
)
from src.data_processing import ingest_data
from src.retrieval import generate_embeddings, create_faiss_index
from src.generation import load_model
from src.pipeline import rag_system


def run_app() -> None:
    """
    Main function to run the Streamlit application.
    """
    st.title("Retrieval-Augmented Generation System ⚙️")
    st.info(
        "This application uses a modular structure to ingest documents, retrieve relevant context, and generate answers."
    )

    # --- 1. Data Ingestion Section ---
    st.header("1. Ingest Your Documents")
    data_directory = st.text_input(
        "Enter the path to your data directory:", value=DATA_DIRECTORY
    )

    if st.button("Ingest Data"):
        # Clear any old index from the session state
        if "index" in st.session_state:
            st.session_state.pop("index", None)
            st.session_state.pop("metadata", None)

        # Start the ingestion process with user feedback
        with st.spinner("Ingesting and chunking documents..."):
            try:
                documents = ingest_data(data_directory)
                st.success(
                    f"Successfully ingested and chunked {len(documents)} documents."
                )

                with st.spinner(
                    f"Generating embeddings using '{EMBEDDING_MODEL_NAME}'..."
                ):
                    embeddings_data = generate_embeddings(documents)
                    index, metadata = create_faiss_index(embeddings_data)

                    # Store the index and metadata in the session state to persist them
                    st.session_state["index"] = index
                    st.session_state["metadata"] = metadata
                    st.success("Indexing complete and ready for queries.")
            except Exception as e:
                st.error(f"Error during ingestion: {e}")

    # --- 2. Query and Response Section ---
    st.header("2. Ask a Question")

    # Check if the index is ready before allowing questions
    if "index" in st.session_state:
        # Model and preference selection
        model_choice = st.selectbox(
            "Select a language model:",
            options=list(MODEL_OPTIONS.keys()),
            index=2,  # Default to flan-t5-base
        )
        preference = st.selectbox(
            "Select a response style:", options=list(GENERATION_PREFERENCES.keys())
        )

        # Load the selected model (will be cached by Streamlit)
        try:
            with st.spinner(f"Loading {model_choice}..."):
                tokenizer, model, requires_attention_mask = load_model(model_choice)
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.stop()

        # User query input
        user_query = st.text_input("Enter your question:")
        if st.button("Get Response"):
            if user_query:
                with st.spinner("Retrieving context and generating response..."):
                    try:
                        # Call the main RAG pipeline function
                        answer = rag_system(
                            user_query=user_query,
                            index=st.session_state["index"],
                            metadata=st.session_state["metadata"],
                            tokenizer=tokenizer,
                            model=model,
                            requires_attention_mask=requires_attention_mask,
                            preference=preference,
                        )
                        st.subheader("Assistant's Response:")
                        st.write(answer)
                    except Exception as e:
                        st.error(f"Error generating response: {e}")
            else:
                st.warning("Please enter a question.")
    else:
        st.warning("Please ingest data first to build the search index.")


if __name__ == "__main__":
    run_app()
