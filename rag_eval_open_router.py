"""
rag_eval.py

Usage:
    python rag_eval.py --facts facts.txt --eval eval.jsonl --model qwen/qwen-2.5-7b-instruct --api-key YOUR_API_KEY
"""

import json
import argparse
import numpy as np
import faiss
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import requests
from dotenv import load_dotenv


def load_facts(path: str):
    with open(path, "r", encoding="utf-8") as f:
        facts = [line.strip() for line in f if line.strip()]
    return facts


def load_eval_data(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data


def build_index(facts, embedder, embeddings=None):
    if embeddings is None:
        print(f"Encoding {len(facts)} facts...")
        embeddings = embedder.encode(facts, normalize_embeddings=True, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, embeddings


def retrieve(query, embedder, index, facts, top_k=5):
    q_emb = embedder.encode([query], normalize_embeddings=True)
    scores, ids = index.search(np.array(q_emb), top_k)
    return [facts[i] for i in ids[0]]


def build_prompt(question, retrieved_facts):
    context = "\n".join(retrieved_facts)
    return f"""  
You are a database assistant capable of answering business questions in regard to TPC-H.
You are provided with the following facts.
You must answer the question based only on the provided facts.
You must follow the rules:
- If a numeric value or ID is requested, return only the number.
- Return no additional text beyond the answer.
- Dont return SQL queries. 
== Facts == 
{context}
== Question ==
{question}
"""


def call_openrouter(prompt, model_name, api_key, max_tokens=50):
    """Call OpenRouter API to generate a response."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0
    }
    
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    
    result = response.json()
    return result["choices"][0]["message"]["content"]


def run_eval(facts_file, eval_file, model_name, api_key, save_embeddings=None, load_embeddings=None):
    # --- Load data ---
    facts = load_facts(facts_file)
    eval_data = load_eval_data(eval_file)

    # --- Embedding model ---
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

    # --- Load or build embeddings ---
    if load_embeddings:
        print(f"Loading embeddings from {load_embeddings}")
        embeddings = np.load(load_embeddings)
        index, _ = build_index(facts, embedder, embeddings=embeddings)
    else:
        index, embeddings = build_index(facts, embedder)
        if save_embeddings:
            print(f"Saving embeddings to {save_embeddings}")
            np.save(save_embeddings, embeddings)

    # --- Evaluate ---
    correct = 0
    results = []

    print(f"\nUsing OpenRouter with model: {model_name}")
    print("Running evaluation...\n")

    for item in tqdm(eval_data):
        question = item["question"]
        gold = str(item["answer"]).strip()

        retrieved = retrieve(question, embedder, index, facts, len(facts))
        prompt = build_prompt(question, retrieved)

        try:
            response = call_openrouter(prompt, model_name, api_key)
            # Extract only the part after "Answer:" if present
            answer = response.split("Answer:")[-1].strip().split("\n")[0]
        except Exception as e:
            print(f"\nError processing question: {question}")
            print(f"Error: {e}")
            answer = ""

        results.append({
            "question": question,
            "gold": gold,
            "predicted": answer
        })

        if gold in answer:
            correct += 1

    acc = correct / len(eval_data)
    print(f"\n✅ Accuracy: {acc:.2%} ({correct}/{len(eval_data)})")

    # Optionally write results
    with open("rag_results.jsonl", "w", encoding="utf-8") as out:
        for r in results:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("Results written to rag_results.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--facts", type=str, required=True, help="Path to facts.txt")
    parser.add_argument("--eval", type=str, required=True, help="Path to eval.jsonl")
    parser.add_argument("--model", type=str, default="deepseek/deepseek-r1:free", help="OpenRouter model name")
    parser.add_argument("--api-key", type=str, default=None, help="OpenRouter API key (or set OPENROUTER_API_KEY env var)")
    parser.add_argument("--save-embeddings", type=str, default=None, help="Path to save embeddings (npy)")
    parser.add_argument("--load-embeddings", type=str, default=None, help="Path to load embeddings (npy)")
    args = parser.parse_args()

    # Get API key from args or environment variable
    load_dotenv()
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("API key required. Use --api-key or set OPENROUTER_API_KEY environment variable")

    run_eval(args.facts, args.eval, args.model, api_key, 
             save_embeddings=args.save_embeddings, load_embeddings=args.load_embeddings)