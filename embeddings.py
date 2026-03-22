#!/usr/bin/env python3

import sys
import requests
import math
import tiktoken

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "nomic-embed-text"
ENCODING = "cl100k_base"


def get_embedding(text):
    response = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt": text},
    )
    response.raise_for_status()
    return response.json()["embedding"]


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


def read_input():
    if len(sys.argv) > 1:
        return sys.argv[1:]
    data = sys.stdin.read().strip()
    return [data] if data else []


def tokenize_text(text):
    enc = tiktoken.get_encoding(ENCODING)
    token_ids = enc.encode(text)
    token_chunks = [enc.decode([t]) for t in token_ids]
    return token_ids, token_chunks


def vector_norm(vec):
    return math.sqrt(sum(x * x for x in vec))


def show_single(text, emb):
    token_ids, token_chunks = tokenize_text(text)
    norm = vector_norm(emb)

    print("\n=== INPUT ===")
    print(f"Text: {text}")

    print("\n=== TOKENISATION ===")
    print(f"Token count: {len(token_ids)}")
    print(f"Token IDs: {token_ids}")
    print("Token chunks:")
    for i, chunk in enumerate(token_chunks, start=1):
        print(f"  {i:02d}: {chunk!r}")

    print("\n=== INTERNAL MODEL STEP (conceptual) ===")
    print("Each token ID is looked up as an initial vector internally.")
    print(f"So this input becomes {len(token_ids)} initial token vectors.")
    print("In the transformer, each token can attend to every other token.")
    print("This is not a simple word-vs-word comparison.")
    print("It is weighted information-sharing across all token vectors.")
    print("Those updated, context-aware token vectors are then pooled")
    print("into one final embedding for the whole input.")

    print("\n=== FINAL EMBEDDING ===")
    print(f"Dimensions: {len(emb)}")
    print(f"Vector norm: {round(norm, 4)}")
    print("First 10 values:")
    for i, val in enumerate(emb[:10]):
        print(f"  {i:02d}: {round(val, 4)}")

    print("\n(remaining values omitted)")


def show_pairwise(embeddings):
    print("\n=== PAIRWISE SIMILARITY ===\n")
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            text1, emb1 = embeddings[i]
            text2, emb2 = embeddings[j]
            sim = cosine_similarity(emb1, emb2)
            print(f"{text1!r} vs {text2!r} -> {round(sim, 4)}")


def main():
    inputs = read_input()

    if not inputs:
        print("Provide text input")
        sys.exit(1)

    embeddings = [(text, get_embedding(text)) for text in inputs]

    if len(inputs) == 1:
        text, emb = embeddings[0]
        show_single(text, emb)
        return

    for text, emb in embeddings:
        print("\n" + "=" * 70)
        show_single(text, emb)

    show_pairwise(embeddings)


if __name__ == "__main__":
    main()
