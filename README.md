
# Retrieval-Augmented Generation (RAG) System

## Overview

This Retrieval-Augmented Generation (RAG) system combines information retrieval with generative AI to answer user queries based on a provided document corpus. The system ingests and processes documents, creates an efficient search index, retrieves relevant context for a user's query, and then uses a large language model to generate an informative, context-aware response.

This project has been architected to be modular and scalable, with a focus on retrieval quality and user experience.

## Key Features

### Advanced Data Processing 📄

- Supports ingestion of .txt, .csv, .json, and .pdf files.
- Intelligently processes documents by splitting large texts into smaller, overlapping chunks and treating CSV rows as individual records for more precise context retrieval.

### Multi-Model Support 🧠

- Seamlessly integrates multiple Hugging Face models, including distilgpt2 and the flan-t5 series.
- The architecture is configured to easily add new models.

### Customizable Query Responses 🎛️

Offers flexibility for users to tailor the tone and style of the generated response by prioritizing:

- **Precision**: Accurate, fact-based answers.
- **Creativity**: Detailed, narrative-like responses.
- **Conciseness**: Brief, direct answers.

### Efficient Model & Data Handling ⚡

- Utilizes Docker volumes or local caching to persist models and data indices, ensuring fast and responsive performance after the initial setup.

### Optimized Retrieval 🔍

- Combines SentenceTransformer embeddings with a high-speed FAISS vector index for efficient and accurate document similarity searches.

### Interactive Streamlit Interface ✨

- An intuitive UI allows users to easily ingest data, select models, customize response preferences, and interact with the RAG system.

## System Requirements

⚠️ **Important: High Disk Space Requirement**

This is a heavyweight AI application that requires a significant amount of disk space, regardless of the installation method.

- **Initial Size:** Expect the application to consume over 7 GB of disk space after the initial setup.

**Why so large?** The storage is primarily used by:

- **Language Models (LLMs):** Models like flan-t5-large are over 3 GB each.
- **PyTorch Framework:** The deep learning library and its CUDA components are several gigabytes.
- **Model Caches:** Hugging Face and SentenceTransformers cache the downloaded models to avoid re-downloading them.
- **Future Growth:** The required space will grow as you ingest more documents or experiment with larger models.

## Setup and Installation

You can run this project using either Docker (recommended for a clean, isolated environment) or a local Python environment.

### Option 1: Using Docker (Recommended)

This method uses Docker and Docker Compose to ensure a consistent, reproducible environment.

#### Prerequisites

- Docker installed and running on your system.
- Git for cloning the repository.

#### 1. Clone the repository

```bash
git clone https://github.com/vbchivu/ragProject
cd ragProject
```

#### 2. Prepare Your Documents

Place your `.txt`, `.csv`, `.json`, or `.pdf` files into the `data/` directory.

#### 3. Build and Run the Container

From the root of the project directory, run:

```bash
docker-compose up --build
```

This command will build the Docker image, download all models (this will take a while on the first run), and launch the Streamlit application.

#### 4. Access the Application

Open your web browser and navigate to:  
[http://localhost:8501](http://localhost:8501)

### Option 2: Local Python Environment

This method involves setting up a local Python environment and installing dependencies directly.

#### Prerequisites

- Python 3.9+
- Git

#### 1. Clone the repository

```bash
git clone https://github.com/vbchivu/ragProject
cd ragProject
```

#### 2. Create a Virtual Environment (Recommended)

It's a best practice to use a virtual environment to manage project-specific dependencies.

```bash
# Create the virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Run the Application

Once the dependencies are installed, run the Streamlit application:

```bash
streamlit run app.py
```

## Usage

1. **Prepare Documents:** Place your `.txt`, `.csv`, `.json`, or `.pdf` files into a directory (default is `data/`).
2. **Start the Application:** Run using Docker or local environment.
3. **Ingest Documents:** Use the "Ingest Data" button to process documents and create a search index.
4. **Select Preferences:** Choose a model and a response preference.
5. **Ask Questions:** Enter a query and receive a tailored response from your documents.

## Next Steps and Enhancements

Our roadmap is focused on improving the core quality of the RAG pipeline and ensuring the system is robust and scalable.

### Priority 1: Enhance Core RAG Quality & Robustness

#### Upgrade to a Persistent Vector Database

- **What:** Replace the current in-memory FAISS index with a persistent, on-disk vector database like ChromaDB or LanceDB.
- **Why:** This will make the ingestion step a one-time process, allowing the app to start instantly with a pre-built index. It's a critical step for practical usability and scalability.

#### Implement a Re-ranking Step

- **What:** After retrieving an initial set of documents from the vector search (e.g., top 20), use a more sophisticated cross-encoder model to re-rank these results and pass only the most relevant ones (e.g., top 3-5) to the LLM.
- **Why:** This is a powerful technique to significantly boost retrieval accuracy and, consequently, the quality of the final answer.

#### Implement Query Caching

- **What:** Cache the final answers for previously asked questions.
- **Why:** To provide instant responses for repeated queries and improve the user's perception of the app's performance.

### Priority 2: Long-Term Growth and Specialization

#### Pursue Model Fine-Tuning

- **What:** Fine-tune an open-source model (like a FLAN-T5 or Llama variant) on a high-quality, domain-specific dataset.
- **Why:** To improve precision and response quality for specialized subject matter beyond what a general-purpose model can offer.

#### Expand Model and Embedding Options

- **What:** Integrate more powerful embedding models and LLMs as they become available.
- **Why:** To continuously leverage state-of-the-art technology for the best possible performance.

## Acknowledgments

This system leverages the following outstanding technologies and frameworks:

- **Docker**: For containerization and reproducible environments.
- **Hugging Face Transformers**: For state-of-the-art model loading and generation.
- **Sentence Transformers**: For high-quality document embeddings.
- **FAISS**: For fast and efficient vector similarity searches.
- **LangChain**: For robust text splitting and chunking.
- **Streamlit**: For creating the interactive user interface.
