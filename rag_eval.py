"""
rag_eval.py

Usage:
    python rag_eval.py --facts facts.txt --eval eval.jsonl --model Qwen/Qwen2.5-1.5B-Instruct
"""

import json
import argparse
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


def load_facts(path: str):
    with open(path, "r", encoding="utf-8") as f:
        facts = [line.strip() for line in f if line.strip()]
    return facts


def load_eval_data(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data


def build_index(facts, embedder):
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
    return f"""You are a database assistant.
Use the following facts to answer the question concisely.
If a numeric value or ID is requested, return only the number.

Facts:
{context}

Question: {question}
Answer:"""


def run_eval(facts_file, eval_file, model_name):
    # --- Load data ---
    facts = load_facts(facts_file)
    eval_data = load_eval_data(eval_file)

    # --- Embedding model ---
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

    # --- Build FAISS index ---
    index, _ = build_index(facts, embedder)

    # --- Load LLM ---
    print(f"Loading LLM: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto"
    )
    llm = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)

    # --- Evaluate ---
    correct = 0
    results = []

    print("\nRunning evaluation...\n")

    for item in tqdm(eval_data):
        question = item["question"]
        gold = str(item["answer"]).strip()

        retrieved = retrieve(question, embedder, index, facts)
        prompt = build_prompt(question, retrieved)

        response = llm(prompt)[0]["generated_text"]
        # Extract only the part after "Answer:"
        answer = response.split("Answer:")[-1].strip().split("\n")[0]

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
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="LLM model name")
    args = parser.parse_args()

    run_eval(args.facts, args.eval, args.model)
