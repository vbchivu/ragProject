import os
import json
import re
from typing import List, Dict, Any

import pandas as pd
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import constants from our configuration file
from config import CHUNK_SIZE, CHUNK_OVERLAP

# --- Text Splitting Strategy ---

# Instantiate a text splitter for chunking large documents.
# This ensures that no single document is too long for the model's context window.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len
)

# --- Individual File Processors ---


def process_text_file(file_path: str) -> str:
    """Reads and returns the content of a text file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def process_csv_file(file_path: str) -> List[str]:
    """
    Reads a CSV file and converts each row into a descriptive string.
    Each row is treated as a separate document.
    """
    df = pd.read_csv(file_path)
    # Convert each row to a string format like "column1: value1, column2: value2, ..."
    return [
        ", ".join([f"{col}: {str(val)}" for col, val in row.items()])
        for _, row in df.iterrows()
    ]


def process_json_file(file_path: str) -> str:
    """Reads a JSON file and returns its content as a formatted string."""
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    # Use dumps to create a string representation of the JSON object
    return json.dumps(data, indent=2)


def process_pdf_file(file_path: str) -> str:
    """Extracts and returns all text from a PDF file."""
    reader = PdfReader(file_path)
    # Concatenate text from all pages that contain extractable text
    return "".join(page.extract_text() for page in reader.pages if page.extract_text())


# --- Text Cleaning ---


def clean_text(text: str) -> str:
    """
    Cleans text by lowercasing and normalizing whitespace.
    This helps standardize the text for more consistent embeddings.
    """
    # Convert text to lowercase
    text = text.lower()
    # Replace multiple whitespace characters (spaces, tabs, newlines) with a single space
    text = re.sub(r"\s+", " ", text)
    # Remove leading/trailing whitespace
    return text.strip()


# --- Main Ingestion Logic ---

# A dictionary that maps file extensions to their corresponding processing function and type.
# This makes it easy to add support for new file types in the future.
FILE_PROCESSORS: Dict[str, Dict[str, Any]] = {
    ".txt": {"processor": process_text_file, "type": "chunk"},
    ".csv": {"processor": process_csv_file, "type": "row"},
    ".json": {"processor": process_json_file, "type": "chunk"},
    ".pdf": {"processor": process_pdf_file, "type": "chunk"},
}


def ingest_data(data_directory: str) -> List[Dict[str, Any]]:
    """
    Walks through a directory, processes all supported files, and returns a list of documents.
    Each document is a dictionary containing the content and metadata.
    """
    documents: List[Dict[str, Any]] = []

    # Recursively walk through the provided data directory
    for root, _, files in os.walk(data_directory):
        for file in files:
            file_path = os.path.join(root, file)
            # Get the file extension and convert to lowercase for consistent matching
            file_ext = os.path.splitext(file)[1].lower()

            # Check if we have a processor for this file type
            if file_ext in FILE_PROCESSORS:
                handler = FILE_PROCESSORS[file_ext]
                processor = handler["processor"]
                processing_type = handler["type"]

                # Call the appropriate function to get the file's content
                content = processor(file_path)

                # Handle documents that should be processed row-by-row (like CSVs)
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
                # Handle documents that should be chunked (like PDFs and TXTs)
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

    # Raise an error if no processable documents were found
    if not documents:
        raise ValueError("No processable documents found in the data directory.")

    return documents
