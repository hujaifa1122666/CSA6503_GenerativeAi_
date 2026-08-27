import argparse, csv, json, os, shutil, time
from pathlib import Path
import numpy as np
import pandas as pd
import faiss
import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "grievances.csv"
QUERIES = ROOT / "data" / "queries.json"
FAISS_DIR = ROOT / "artifacts" / "faiss"
CHROMA_DIR = ROOT / "artifacts" / "chroma"
RESULTS_DIR = ROOT / "results"

def size_bytes(path):
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())

def load_data():
    df = pd.read_csv(DATA)
    with open(QUERIES, encoding="utf-8") as f:
        queries = json.load(f)
    return df, queries

def relevant(dept_value, expected):
    expected_set = set(x.strip() for x in expected.split("|"))
    actual_set = set(x.strip() for x in str(dept_value).split("|"))
    return bool(expected_set & actual_set)

def relevance_at_5(result_rows, expected):
    return sum(relevant(x["department"], expected) for x in result_rows) / 5.0

def build_faiss(embeddings):
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    t0 = time.perf_counter()
    index.add(embeddings)
    indexing_time = time.perf_counter() - t0
    faiss.write_index(index, str(FAISS_DIR / "grievance.index"))
    return index, indexing_time

def build_chroma(embeddings, df):
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.create_collection(
        name="grievances",
        metadata={"hnsw:space": "cosine"}
    )
    t0 = time.perf_counter()
    collection.add(
        ids=df["id"].tolist(),
        embeddings=embeddings.tolist(),
        documents=df["grievance"].tolist(),
        metadatas=[
            {"department": d, "status": s}
            for d, s in zip(df["department"], df["status"])
        ]
    )
    indexing_time = time.perf_counter() - t0
    return client, collection, indexing_time

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    df, queries = load_data()

    print(f"Records: {len(df)}")
    print(f"Model: {args.model}")
    print("Loading embedding model...")
    model = SentenceTransformer(args.model)

    texts = df["grievance"].tolist()
    t0 = time.perf_counter()
    embeddings = model.encode(
        texts, normalize_embeddings=True, show_progress_bar=True
    ).astype("float32")
    embedding_time = time.perf_counter() - t0

    faiss_index, faiss_indexing = build_faiss(embeddings)
    chroma_client, chroma_collection, chroma_indexing = build_chroma(embeddings, df)

    faiss_rows = []
    chroma_rows = []
    comparison_rows = []

    for q in queries:
        qvec = model.encode([q["query"]], normalize_embeddings=True).astype("float32")

        t0 = time.perf_counter()
        scores, ids = faiss_index.search(qvec, args.k)
        faiss_latency = (time.perf_counter() - t0) * 1000
        fitems = []
        for rank, idx in enumerate(ids[0], 1):
            row = df.iloc[int(idx)].to_dict()
            row["rank"] = rank
            row["score"] = float(scores[0][rank-1])
            fitems.append(row)

        t0 = time.perf_counter()
        cres = chroma_collection.query(
            query_embeddings=qvec.tolist(),
            n_results=args.k,
            include=["documents", "metadatas", "distances"]
        )
        chroma_latency = (time.perf_counter() - t0) * 1000
        citems = []
        for rank, (doc, meta, dist) in enumerate(zip(
            cres["documents"][0], cres["metadatas"][0], cres["distances"][0]
        ), 1):
            citems.append({
                "rank": rank,
                "grievance": doc,
                "department": meta["department"],
                "status": meta["status"],
                "score": 1.0 - float(dist)
            })

        faiss_rel = relevance_at_5(fitems, q["expected"])
        chroma_rel = relevance_at_5(citems, q["expected"])
        fids = [x["id"] for x in fitems]
        # Chroma IDs are returned in the query result.
        cids = cres["ids"][0]
        overlap = len(set(fids) & set(cids))

        faiss_rows.append({
            "query_id": q["id"], "query": q["query"],
            "expected_department": q["expected"],
            "latency_ms": faiss_latency, "relevance_at_5": faiss_rel,
            "top5_ids": "|".join(fids)
        })
        chroma_rows.append({
            "query_id": q["id"], "query": q["query"],
            "expected_department": q["expected"],
            "latency_ms": chroma_latency, "relevance_at_5": chroma_rel,
            "top5_ids": "|".join(cids)
        })

        if q["id"] in {"Q01", "Q02", "Q09"}:
            comparison_rows.append({
                "query_id": q["id"],
                "query": q["query"],
                "faiss_top5": "|".join(fids),
                "chroma_top5": "|".join(cids),
                "overlap_count": overlap,
                "overlap_percent": overlap / args.k * 100
            })

        # Save readable per-query result details.
        with open(RESULTS_DIR / f"{q['id']}_details.txt", "w", encoding="utf-8") as out:
            out.write(f"Query: {q['query']}\nExpected department(s): {q['expected']}\n\n")
            out.write("FAISS TOP-5\n")
            for x in fitems:
                out.write(f"{x['rank']}. {x['id']} | {x['department']} | {x['score']:.4f} | {x['grievance']}\n")
            out.write("\nChromaDB TOP-5\n")
            for rank, (cid, x) in enumerate(zip(cids, citems), 1):
                out.write(f"{rank}. {cid} | {x['department']} | {x['score']:.4f} | {x['grievance']}\n")

    faiss_df = pd.DataFrame(faiss_rows)
    chroma_df = pd.DataFrame(chroma_rows)
    comp_df = pd.DataFrame(comparison_rows)

    summary = pd.DataFrame([
        {
            "database": "FAISS",
            "indexing_time_seconds": faiss_indexing,
            "average_query_latency_ms": faiss_df["latency_ms"].mean(),
            "average_relevance_at_5": faiss_df["relevance_at_5"].mean(),
            "storage_bytes": size_bytes(FAISS_DIR)
        },
        {
            "database": "ChromaDB",
            "indexing_time_seconds": chroma_indexing,
            "average_query_latency_ms": chroma_df["latency_ms"].mean(),
            "average_relevance_at_5": chroma_df["relevance_at_5"].mean(),
            "storage_bytes": size_bytes(CHROMA_DIR)
        }
    ])

    faiss_df.to_csv(RESULTS_DIR / "faiss_queries.csv", index=False)
    chroma_df.to_csv(RESULTS_DIR / "chroma_queries.csv", index=False)
    comp_df.to_csv(RESULTS_DIR / "top5_comparison.csv", index=False)
    summary.to_csv(RESULTS_DIR / "benchmark_summary.csv", index=False)

    with open(RESULTS_DIR / "benchmark_summary.txt", "w", encoding="utf-8") as f:
        f.write("GRIEVANCE REDRESSAL ROUTER - BENCHMARK SUMMARY\n")
        f.write("=" * 55 + "\n")
        f.write(f"Dataset records: {len(df)}\n")
        f.write(f"Embedding model: {args.model}\n")
        f.write(f"Embedding generation time: {embedding_time:.4f} seconds\n\n")
        f.write(summary.to_string(index=False))
        f.write("\n\nTop-5 comparison for Q01, Q02 and Q09:\n")
        f.write(comp_df.to_string(index=False))
        f.write("\n\nSpecial case Q09:\n")
        f.write("Expected departments: Electricity and Sanitation. Relevance@5 counts a result as relevant when it belongs to either department.\n")
        f.write("The query intentionally mentions two departments to test whether semantically similar results from both service areas appear in the top five.\n")

    print("\nBENCHMARK COMPLETE")
    print(summary.to_string(index=False))
    print("\nFiles written to:", RESULTS_DIR)

if __name__ == "__main__":
    run()
