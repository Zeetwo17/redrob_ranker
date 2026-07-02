"""
jd.py
=====
Turns the released job description into two things the ranker uses:

  1. IDEAL_QUERY  -- a dense, intent-level text used for TF-IDF semantic matching.
     This is deliberately NOT the raw JD. The raw JD is ~60% culture prose
     ("we disagree openly and decide quickly") that would add noise. Instead we
     distil the JD into the language a genuine fit would use to describe their own
     work, so semantic similarity rewards people who *describe building retrieval
     and ranking systems* — including the "plain-language Tier-5s" who never write
     the word "RAG".

  2. The structured requirement constants other modules consume (experience band,
     etc.) live in config.py; this file owns the text intent.

The distillation is hand-written from the JD, not generated, so we can defend
every term in it at review.
"""

# Intent-level query. Terms are weighted by repetition (TF-IDF rewards the terms
# that recur), mirroring the JD's actual emphasis: retrieval, ranking, embeddings,
# production, evaluation, product-company shipping.
IDEAL_QUERY = """
senior ai engineer machine learning engineer building production ranking and
retrieval systems. embeddings based retrieval, semantic search, vector search,
vector database, hybrid search, dense retrieval, bm25, learning to rank,
recommendation systems, information retrieval, candidate matching, search relevance.
shipped end to end search ranking and recommendation systems to real users at scale
at a product company. designed hybrid retrieval combining bm25 with dense vector
recall. sentence transformers, faiss, pinecone, weaviate, qdrant, elasticsearch,
opensearch, pgvector. large language models, llm re-ranking, retrieval augmented
generation, fine tuning, lora. strong python engineering and code quality.
evaluation of ranking systems, ndcg, mrr, map, offline online correlation, ab testing,
recruiter feedback loops, embedding drift, index refresh, retrieval quality regression.
applied machine learning in production, not pure research. nlp, deep learning, pytorch.
product engineering, ships fast, iterates with real users. located in india, pune,
noida, hyderabad, bangalore, mumbai, delhi, open to relocation.
"""

# Title text we append to each candidate's document so the candidate's own title
# participates in the semantic space (helps separate "Marketing Manager" docs from
# "Search Engineer" docs even when skill lists look similar).
def candidate_document(candidate) -> str:
    """Assemble the text blob used for semantic similarity, recruiter-style:
    title + headline + summary + every role description + skill names."""
    p = candidate.get("profile", {})
    parts = [
        p.get("current_title", ""),
        p.get("headline", ""),
        p.get("summary", ""),
        p.get("current_industry", ""),
    ]
    for h in candidate.get("career_history", []) or []:
        parts.append(h.get("title", ""))
        parts.append(h.get("description", ""))
    skills = candidate.get("skills", []) or []
    parts.append(" ".join(s.get("name", "") for s in skills))
    return " ".join(part for part in parts if part)
