# Retrieval-Augmented Generation (RAG) System

## Overview

This Retrieval-Augmented Generation (RAG) system combines information retrieval and generative AI capabilities to answer user queries based on provided documents. It incorporates various natural language processing techniques and deep learning models to efficiently retrieve relevant documents, process user queries, and generate informative and context-aware responses.

---

## Key Features

1. **Multi-Model Support**  
   - Seamlessly integrates multiple models (`distilgpt2`, `flan-t5-small`, `flan-t5-base`, `flan-t5-large`).  
   - Allows users to choose between speed, precision, and creativity based on their requirements.

2. **Customizable Query Responses**  
   - Offers flexibility to prioritize:
     - **Precision**: Focus on accurate, fact-based answers.
     - **Creativity**: Generate detailed, narrative-like responses.
     - **Conciseness**: Provide brief, direct answers.
     - **Default**: Balance between all traits.

3. **Memory-Efficient Model Loading**  
   - Utilizes the `Accelerate` library for optimized memory usage, enabling smooth operation on both GPUs and CPUs.

4. **Flexible Document Ingestion**  
   - Supports ingestion of `.txt`, `.csv`, `.json`, and `.pdf` files.  
   - Preprocesses and embeds documents using `SentenceTransformer` and FAISS for efficient retrieval.

5. **Interactive Streamlit Interface**  
   - Intuitive UI for easy interaction.  
   - Provides options to ingest data, select models, and customize query preferences.

6. **Accelerated Retrieval and Generation**  
   - Combines FAISS for fast document similarity searches with generative models for high-quality, context-aware answers.

---

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/vbchivu/ragProject
   cd ragProject

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Install Accelerate for memory-efficient model loading:
   ```bash
   pip install "accelerate>=0.26.0"

4. Run the system:
   ```bash
   streamlit run app.py

## Usage
1. Prepare Documents
    - Place your documents in a directory (e.g., data).

2. Start the Application
    ```bash
    streamlit run app.py

3. Ingest Documents
    - Provide the directory path in the app to ingest documents.
    - The system will preprocess the documents and create an index.

4. Select Preferences and Models
    - Choose a model based on speed and response quality.
    - Customize query preferences for precision, creativity, or conciseness.

5. Ask Questions
    - Enter your query in the app and receive tailored responses based on the selected settings.

## Next Steps and Enhancements
1. Model Expansion
    - Add support for more models like gpt-3, falcon, or other domain-specific models.

2. Advanced Query Preferences
    - Introduce additional query preferences like "explanatory," "summarized," or "comparative."

3. Fine-Tuning
    - Fine-tune models on domain-specific datasets for improved precision.

4. Batch Query Support
    - Allow batch processing of queries for large-scale use cases.

5. Caching
    - Implement caching mechanisms for repeated queries to reduce response time.

6. Scalability
    - Integrate distributed systems to support larger datasets and more concurrent users.

7. Enhanced Retrieval
    - Explore advanced embedding techniques (e.g., OpenAI Embeddings) for improved document similarity.

## Acknowledgments
This system leverages the following technologies and frameworks:
- Hugging Face Transformers for model loading and generation.
- Sentence Transformers for document embeddings.
- FAISS for fast similarity searches.
- Streamlit for the interactive user interface.

### Feel free to contribute to the project or provide feedback for improvements. 🚀
