# Learn to be AI Engineer
This repository documents what I have learned from various AI engineering tools. I use Claude.ai to generate code and learn by experimenting with different scenarios and analyzing the resulting behavior.

# What I Learned
## RAG Systems
Build complete RAG's pipeline: hybrid search -> evaluation -> graph retrieval. All integrated in one production-ready system.

### Code Explanation
#### Phase 1 - Hybrid Search
- Setup Qdrant + BGE-M3: Q-drant runs in-memory here, so no Docker needed for the lab. BGE-M3 is a special model because a single forward pass produces two outputs simultaneously: a 1024-dim dense vector for semantic search, and sparse lexical weights for keyword-based search.
- Indexing documents: model.encode(...) represents that each document gets encoded into two representations. Dense vectors are stored normally. Sparse vectors are stored as indices (XLM-RoBERTa token IDs) and values (weight per token) pairs. 
- Hybrid search with RRF: represented from prefetch to models.FusionQuery(...) where Qdrant pulls top-k candidated from both retrieval paths in parallel, then merges the rankings using Reciprocal Rank Fusion (RRF(doc) = Σ 1 / (60 + rank_i))
- Cross-encoder reranking: represented from reranker untill reranker.predict(pairs). Hybrid search retrieves the top-20 candidates, then the cross-encoder reevaluates each (query, document) pair jointly. far more accurate than a bi-encoder but O(n) in cost, which is why it;s only used for reranking, not the initial retrieval. 

#### Phase 2 - RAG Evaluation
- judge and emb: the LLM(Groq/Llama) acts as a "judge" for metrics that require reasoning. Embeddings are used for answer_relevancy, which is computed via cosine similarity.
- faithfulness: the LLM extracts atomic claims from the answer, then verifies each claim against the retrieved context. score = verified_claims/total_claims. 
- answer relevancy: cosine similarity between the embedding of the question and the embedding of the answer. no ground truth needed. 

#### Phase 3 - Graph RAG
Relational queries like "What did Apple acquire in 1997?" can fail because the chunk containing the acquisition fact may lose to Apple marketing content thath scores higher in semantic similarity. The solution is a knowledge graph. 

- TRIPLETS: relationships between entities are stored as triplets (subject, relation, object)
- graph traversal: starting from entities detected in the query, traversal expands 2 hops to neighboring nodes, the returns all chunks connected to reachable entities. This ensures relationally relevant documents are retrieved even if their semantic similarity score is low. 
- full pipeline: combining graph traversal + hybrid search + cross-encoder reranking is currently the most robust RAG architecture for queries that mix semantic understanding with relational reasoning.

### Case Studies
TBD

## Proper Fine-Tuning
TBD

## Agentic Systems
TBD

## MLOps - Production Grade
TBD

## Inference Optimization
TBD

## LLMOps
TBD

## Training at Scale
TBD

Credit for code and debugging: Claude.ai