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


def run_eval(facts_file, eval_file, model_name, save_embeddings=None, load_embeddings=None):
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

    # --- Load LLM ---
    print(f"Loading LLM: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype="auto"
    )
    llm = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=50, return_full_text=False)

    # --- Evaluate ---
    correct = 0
    results = []

    print("\nRunning evaluation...\n")

    for item in tqdm(eval_data):
        question = item["question"]
        gold = str(item["answer"]).strip()

        retrieved = retrieve(question, embedder, index, facts, len(facts))
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
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="LLM model name")
    parser.add_argument("--save-embeddings", type=str, default=None, help="Path to save embeddings (npy)")
    parser.add_argument("--load-embeddings", type=str, default=None, help="Path to load embeddings (npy)")
    args = parser.parse_args()

    run_eval(args.facts, args.eval, args.model, save_embeddings=args.save_embeddings, load_embeddings=args.load_embeddings)
