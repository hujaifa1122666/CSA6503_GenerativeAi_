# Grievance Redressal Router — FAISS vs ChromaDB

## Problem
A civic portal routes new grievances by semantic similarity to previously resolved grievances.

This project benchmarks **FAISS** and **ChromaDB** on the exact same 248 grievance texts and the exact same embeddings.

## Dataset
- **248 text records**
- 240 normal resolved grievances across 8 civic departments
- 8 cross-department records
- Dataset is **synthetically generated for academic benchmarking**; therefore no external dataset license/source is required.
- The synthetic data is designed to represent common civic-service complaints and to make the benchmark reproducible.

Departments:
1. Electricity
2. Water Supply
3. Sanitation
4. Roads
5. Public Transport
6. Health
7. Municipal Services
8. Public Safety

## Embeddings
The same `sentence-transformers/all-MiniLM-L6-v2` model is used for every record and every query.

Embeddings are normalized before indexing. FAISS uses inner-product similarity on normalized vectors, which is equivalent to cosine similarity. ChromaDB uses cosine distance.

## Databases
### FAISS
- In-memory vector index: `IndexFlatIP`
- Exact nearest-neighbor search
- Persisted to `artifacts/faiss/grievance.index`

### ChromaDB
- Persistent local collection
- Cosine similarity
- Persisted to `artifacts/chroma/`

## Queries
There are 10 semantic queries. Q09 is the special case:
> Street lights are not working and garbage has not been collected in my area.

This intentionally mentions **Electricity + Sanitation** at the same time.

## Metrics
For both databases:
- Indexing time
- Average query latency over 10 queries
- Relevance@5
- Storage footprint

For relevance@5, a top-5 result is considered relevant when its department matches one of the expected department labels. For Q09 and Q10, either listed department counts as relevant.

## Top-5 comparison
The program compares the top-5 IDs from FAISS and ChromaDB for Q01, Q02 and Q09 and calculates overlap.

## How to run on Windows PowerShell

### 1. Open this project folder
```powershell
cd CO3_AT_3_Grievance_Redressal_Router
```

### 2. Create a virtual environment
```powershell
py -m venv .venv
```

### 3. Activate it
```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 4. Install packages
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run the benchmark
```powershell
python app.py
```

The first run downloads `all-MiniLM-L6-v2` and then runs locally.

## Output
After the run, check:
- `results/benchmark_summary.csv`
- `results/benchmark_summary.txt`
- `results/faiss_queries.csv`
- `results/chroma_queries.csv`
- `results/top5_comparison.csv`
- `results/Q01_details.txt`
- `results/Q02_details.txt`
- `results/Q09_details.txt`

## Expected conclusion
FAISS is usually preferable when the main goal is a lightweight, fast, exact vector-search benchmark with a relatively static dataset and when you are comfortable managing persistence and metadata yourself.

ChromaDB is preferable when you want a more database-like developer experience with persistent collections, documents, metadata, filtering, and easier application-level management.

The benchmark's measured numbers in `results/benchmark_summary.csv` should be used in the final report rather than inventing fixed timings, because timings depend on the student's computer.

## Suggested report conclusion
For a civic grievance router with a few hundred records, both systems are suitable. FAISS is attractive for a simple high-speed retrieval layer and minimal overhead. ChromaDB is preferable when the project is expected to grow into a persistent application where documents and metadata need to be managed together. The special two-department query demonstrates that semantic retrieval can return evidence from multiple service categories; however, production routing should combine semantic similarity with an explicit multi-label department classifier or rule so that both departments are assigned reliably.
