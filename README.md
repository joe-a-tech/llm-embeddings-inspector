# llm-embeddings-explorer

CLI tool to explore how Large Language Models (LLMs) represent meaning using embeddings.

This project helps you understand the transition from:
- text → tokens → vectors → semantic similarity

---

## Features

- Generate embeddings locally using Ollama  
- Inspect tokenisation alongside embeddings  
- View vector dimensions and structure  
- Compare multiple inputs using cosine similarity  
- Understand how meaning is represented numerically  

---

## Requirements

- Python 3.10+  
- Ollama installed and running  

Install dependencies:

    pip install -r requirements.txt

Example `requirements.txt`:

    requests
    tiktoken

---

## Setup

Start Ollama:

    ollama serve

Pull the embedding model:

    ollama pull nomic-embed-text

---

## Usage

### Single input

    python embeddings.py "Example text here"

Shows:
- tokenisation  
- conceptual model pipeline  
- embedding vector (preview)  

---

### Compare inputs

    python embeddings.py "car" "automobile" "banana"

Outputs pairwise similarity scores.

---

## Example Output

    === TOKENISATION ===
    Token IDs: [42, 57796, 382, 11629]

    === FINAL EMBEDDING ===
    Dimensions: 768
    Vector norm: 22.63

    === PAIRWISE SIMILARITY ===
    'car' vs 'automobile' -> 0.91
    'car' vs 'banana' -> 0.32

---

## How it works (conceptual)

    text
    → tokenisation
    → token IDs
    → token vectors (internal)
    → transformer (context building)
    → final embedding vector

Each token can attend to every other token via self-attention.

The final embedding is a compressed representation of the entire input.

---

## Key Concepts

- **Tokens**: numeric representation of text  
- **Embeddings**: vectors representing meaning  
- **Transformer**: builds context via attention  
- **Similarity**: distance between vectors reflects semantic closeness  

---

## Notes

- Embeddings are not directly interpretable  
- Meaning is derived from comparing vectors, not inspecting them  
- Tokenisation shown uses `cl100k_base` for illustration (may differ from model internals)  

---

## License

MIT
