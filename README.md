# 🔍 Production RAG: "Attention Is All You Need"

An end-to-end, citation-grounded Retrieval-Augmented Generation (RAG) system built over the seminal research paper [*Attention Is All You Need* (Vaswani et al.)](https://arxiv.org/abs/1706.03762).

This system implements strict evidence gating, dense vector indexing via FAISS, diversity-aware retrieval (MMR), and automated evaluation to minimize hallucinations and enforce exact page citations.

---

## 🚀 Key Architectural Features

- **Metadata-Preserving Document Processing**: Extracted with `pypdf` and chunked via `RecursiveCharacterTextSplitter` (chunk size: 900, overlap: 150) while retaining strict page-level tracking.
- **Dense Vector Embeddings**: Indexed into **FAISS** (squared Euclidean distance L2) using `sentence-transformers/all-MiniLM-L6-v2`.
- **Maximal Marginal Relevance (MMR)**: Balances relevance and query diversity ($\lambda = 0.65$) to avoid redundant context chunks.
- **Relevance Gating (Anti-Hallucination Guardrail)**: Automatically rejects queries with distance threshold `L2 > 1.25`, preventing out-of-domain prompts from triggering hallucinated answers.
- **Strict Evidence Prompting & Exact Page Attribution**: Powered by **Gemini 2.5 Flash** (`google-genai`), answering queries solely from retrieved context and tagging source claims with exact `[Page X]` markers.
- **Automated Evaluation Harness**: Built-in test cases validating technical explanations, architectural comparisons, and out-of-domain refusals.
- **Index Persistence**: Local disk serialization and deserialization for FAISS index reuse without re-embedding.

---

## 🛠️ Tech Stack

- **LLM**: Google Gemini 2.5 Flash via `google-genai`
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Database**: FAISS (`faiss-cpu`)
- **Orchestration / Text Processing**: LangChain Core, PyPDF
- **Environment**: Python 3.10+, Google Colab

---

## 📋 Pipeline Architecture

```text
User Question
      │
      ▼
FAISS Relevance Gate (L2 ≤ 1.25?)
      ├───[ > 1.25 ]──► Refuse Query ("No relevant passages")
      │
      └───[ ≤ 1.25 ]──► MMR Retrieval (k=4, λ=0.65)
                              │
                              ▼
                      Context + Prompt Assembly
                      (Strict Evidence & Page Rules)
                              │
                              ▼
                      Gemini 2.5 Flash
                              │
                              ▼
                Grounded Answer with [Page X] Citations


