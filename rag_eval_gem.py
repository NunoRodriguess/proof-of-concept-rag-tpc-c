import json
import argparse
import numpy as np
import faiss
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import requests
from dotenv import load_dotenv

# -----------------------------
# Data loading
# -----------------------------
def load_facts(path: str):
    with open(path, "r", encoding="utf-8") as f:
        facts = [line.strip() for line in f if line.strip()]
    return facts

def load_eval_data(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data

# -----------------------------
# Embeddings & Retrieval
# -----------------------------
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

# -----------------------------
# Prompt construction
# -----------------------------
def build_prompt(question, retrieved_facts):
    context = "\n".join(retrieved_facts)
    return f"""You are a database assistant capable of answering business questions in regard to TPC-H.
You are provided with the following facts.
You must answer the question based only on the provided facts.
You must follow the rules:
- If a numeric value or ID is requested, return only the number.
- Return no additional text beyond the answer.
- Don't return SQL queries.

== Facts == 
{context}

== Question ==
{question}

Answer:"""

# -----------------------------
# Gemini API call
# -----------------------------
def call_gemini(prompt, api_key, model_name="gemini-1.5-flash", max_tokens=300):
    """
    Calls the Google Gemini API using HTTP POST with improved error handling.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": max_tokens
        }
    }
    
    response = requests.post(url, headers=headers, params=params, json=data)
    
    if not response.ok:
        print("❌ Gemini API error:", response.text)
        response.raise_for_status()
    
    result = response.json()
    
    # Check for candidates
    if "candidates" not in result or len(result["candidates"]) == 0:
        print("⚠️ No candidates in response:", result)
        return ""
    
    candidate = result["candidates"][0]
    
    # Check finish reason
    finish_reason = candidate.get("finishReason", "")
    if finish_reason == "MAX_TOKENS":
        print(f"⚠️ Response truncated (MAX_TOKENS). Consider increasing maxOutputTokens.")
    elif finish_reason not in ["STOP", "MAX_TOKENS", ""]:
        print(f"⚠️ Unusual finish reason: {finish_reason}")
    
    # Extract text from content
    try:
        content = candidate.get("content", {})
        parts = content.get("parts", [])
        
        if not parts:
            print("⚠️ No parts in content:", result)
            return ""
        
        text = parts[0].get("text", "").strip()
        return text
    
    except Exception as e:
        print("⚠️ Failed to parse Gemini response:", result)
        print(f"Parse error: {e}")
        return ""

# -----------------------------
# Evaluation logic
# -----------------------------
def run_eval(facts_file, eval_file, api_key, save_embeddings=None, load_embeddings=None, 
             model_name="gemini-1.5-flash", max_tokens=300, retrieval_tokens=100):
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
    failed_count = 0
    
    print(f"\nUsing Google Gemini model: {model_name}")
    print("Running evaluation...\n")
    
    for item in tqdm(eval_data):
        question = item["question"]
        gold = str(item["answer"]).strip()

        retrieved = retrieve(question, embedder, index, facts, retrieval_tokens)
        # print(f"Retrieved facts for question '{question}': {retrieved}")
        prompt = build_prompt(question, retrieved)
        
        try:
            response = call_gemini(prompt, api_key, model_name=model_name, max_tokens=max_tokens)
            
            if not response:
                failed_count += 1
                answer = ""
            else:
                # Clean up the answer - remove "Answer:" prefix if present
                answer = response.split("Answer:")[-1].strip().split("\n")[0].strip()
        
        except Exception as e:
            print(f"\n❌ Error processing question: {question}")
            print(f"Error: {e}")
            answer = ""
            failed_count += 1
        
        results.append({
            "question": question,
            "gold": gold,
            "predicted": answer
        })
        
        # More flexible matching
        if gold.lower() in answer.lower() or answer.lower() in gold.lower():
            correct += 1
    
    acc = correct / len(eval_data) if len(eval_data) > 0 else 0
    print(f"\n✅ Accuracy: {acc:.2%} ({correct}/{len(eval_data)})")
    
    if failed_count > 0:
        print(f"⚠️  Failed responses: {failed_count}/{len(eval_data)}")
    
    # Write results
    output_file = "rag_results_gemini.jsonl"
    with open(output_file, "w", encoding="utf-8") as out:
        for r in results:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    print(f"Results written to {output_file}")

# -----------------------------
# Main entry
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--facts", type=str, required=True, help="Path to facts.txt")
    parser.add_argument("--eval", type=str, required=True, help="Path to eval.jsonl")
    parser.add_argument("--api-key", type=str, default=None, help="Google Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--save-embeddings", type=str, default=None, help="Path to save embeddings (npy)")
    parser.add_argument("--load-embeddings", type=str, default=None, help="Path to load embeddings (npy)")
    parser.add_argument("--model", type=str, default="gemini-1.5-flash", help="Gemini model name")
    parser.add_argument("--max-tokens", type=int, default=300, help="Maximum output tokens")
    parser.add_argument("--max-retrieval-tokens", type=int, default=100, help="Maximum retrieval tokens")
    
    args = parser.parse_args()
    
    # Get API key from args or environment variable
    load_dotenv()
    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("API key required. Use --api-key or set GEMINI_API_KEY environment variable")
    
    run_eval(
        args.facts,
        args.eval,
        api_key,
        save_embeddings=args.save_embeddings,
        load_embeddings=args.load_embeddings,
        model_name=args.model,
        max_tokens=args.max_tokens,
        retrieval_tokens=args.max_retrieval_tokens
    )