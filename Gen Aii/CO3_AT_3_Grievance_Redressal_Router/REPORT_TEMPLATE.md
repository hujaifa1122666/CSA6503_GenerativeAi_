# Grievance Redressal Router — Benchmark Report

## 1. Objective
Benchmark FAISS and ChromaDB using the same 248 grievance texts, the same sentence-embedding model, and the same 10 semantic queries.

## 2. Dataset
The project uses 248 synthetically generated resolved grievance records. The records cover eight civic departments, with eight additional cross-department examples. Synthetic generation is used so the benchmark is reproducible and does not depend on an external dataset.

## 3. Embedding Model
Model: `sentence-transformers/all-MiniLM-L6-v2`

The same model is used for:
- all 248 grievance records
- all 10 search queries
- both FAISS and ChromaDB

Embeddings are normalized so cosine similarity can be used consistently.

## 4. Benchmark Method
1. Load the same CSV into memory.
2. Generate embeddings once.
3. Insert the exact same embeddings into FAISS and ChromaDB.
4. Execute the same 10 queries with k=5.
5. Record query latency and relevance@5.
6. Compare top-5 result IDs for Q01, Q02 and Q09.
7. Measure persisted storage footprint.

## 5. Benchmark Table
Run `python app.py` first, then copy the measured values from `results/benchmark_summary.csv`.

| Metric | FAISS | ChromaDB |
|---|---:|---:|
| Indexing time (seconds) | Fill from output | Fill from output |
| Average query latency (ms) | Fill from output | Fill from output |
| Average Relevance@5 | Fill from output | Fill from output |
| Storage footprint (bytes) | Fill from output | Fill from output |

## 6. Top-5 Comparison
Use `results/top5_comparison.csv`.

For each of Q01, Q02 and Q09, report:
- FAISS top-5 IDs
- ChromaDB top-5 IDs
- overlap count
- overlap percentage
- any department-level differences

Because both systems receive the same normalized embeddings and use cosine-equivalent similarity, high overlap is expected. Differences can occur because of database implementation details, index configuration, numerical effects, and ChromaDB's underlying approximate nearest-neighbor infrastructure.

## 7. Special Case
Query Q09:

> Street lights are not working and garbage has not been collected in my area.

Expected departments:
- Electricity
- Sanitation

The query contains two distinct civic issues. The top-5 results should ideally contain examples from both departments. In the benchmark, a result is considered relevant when it belongs to either expected department.

This case also shows a limitation of pure nearest-neighbor routing: retrieving similar records does not automatically guarantee that both departments are assigned. A production system should add multi-label classification or a department extraction step after semantic retrieval.

## 8. Conclusion
For a small civic grievance system, both FAISS and ChromaDB are appropriate.

**FAISS is recommended when:**
- the main requirement is fast vector similarity search,
- the dataset is relatively static,
- the application can manage metadata separately,
- minimal infrastructure is preferred.

**ChromaDB is recommended when:**
- persistent document storage is useful,
- metadata and filtering are important,
- the application will grow beyond a simple retrieval benchmark,
- a database-style vector store is preferred.

For this use case, choose the database according to whether the project prioritizes a lightweight retrieval engine (FAISS) or integrated persistent document/metadata management (ChromaDB).
