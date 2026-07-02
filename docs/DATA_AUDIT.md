# Data Audit: what's in the 100K pool

Full streaming pass over `candidates.jsonl` (100,000 records, 487 MB). These numbers shaped the ranker's weights and taxonomy.

## Composition

- **100,000 candidates**, ~75 % India, then USA (~10 %), AU/CA/UK/DE/SG/UAE tails.
- **Experience**: median 6.8 yrs (p25 3.9, p75 9.9). The pool clusters right around the JD's 5-9 band, so experience alone barely separates anyone. More of a tie-breaker than a filter.
- **Activity**: 35 % `open_to_work`, mean recruiter response rate 0.44, 65 % have no GitHub linked (`github_activity_score = -1`).

## Title distribution

| Group | Count | Examples |
|---|---|---|
| Noise (non-tech) | ~68,000 | Business Analyst, HR Manager, Accountant, Mechanical Engineer, Content Writer … (~5.7K each) |
| General software | ~27,000 | Software Engineer, Full Stack, Java/.NET Dev, DevOps, QA … (~2.7K each) |
| Data/adjacent | ~3,600 | Data/Analytics/Backend Engineer, Data Analyst |
| **Genuine AI/ML/IR** | **~1,200** | ML Engineer (167), AI Research Engineer (153), Data Scientist (145), Rec-Systems Engineer (26), Search Engineer (23), Senior/Staff/Lead AI/ML (handful) |

Genuine fits are **~1 %** of the pool. Title + career trajectory is the highest-signal, lowest-noise discriminator, so it gets the largest weight.

## Skills come in three frequency bands

This is how you catch keyword stuffers.

| Band | Freq / skill | Contents | Treatment |
|---|---|---|---|
| **A — generic** | ~12,000 (≈12 %) | HTML, Redux, Terraform, Kafka, AWS, Spark, Airflow, dbt, Snowflake … | neutral / low weight |
| **B — buzzy AI** | ~5,000 (≈5 %) | RAG, Pinecone, FAISS, LangChain, Embeddings, Vector Search, Semantic Search, LLMs **and** YOLO, GANs, OpenCV, Diffusion, ASR | the cluster stuffers copy; retrieval terms weighted, CV/speech zeroed |
| **C — deep/rare** | ~1,400 (≈1.4 %) | pgvector, Weaviate, Milvus, Qdrant, OpenSearch, BM25, Learning-to-Rank, PyTorch, **Python**, scikit-learn | strongest authenticity signal; genuine retrieval engineers have these |

**Python and PyTorch are rare (~1.4 %)** while RAG/Pinecone are common (~5 %). Basically, the fundamentals are rare, so real engineers stand out from people who just pasted the hot-skill list. The ranker weights Band-C/retrieval skills highest and gives a small bonus for deep tooling.

## Companies split cleanly

- **AI-product** (~20-42 each): Sarvam AI, Krutrim, Haptik, Observe.AI, Yellow.ai, Mad Street Den, Niramai, Rephrase.ai, Genpact AI …
- **Indian unicorns / product** (~150-1,300): Swiggy, Zomato, CRED, Razorpay, Flipkart, Meesho, PhonePe, Paytm …
- **FAANG** (1-7 each): Google, Meta, Amazon, Microsoft, Apple, Netflix, LinkedIn …
- **Consulting / services** (~1,200-7,600): TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, HCL, Tech Mahindra, Mindtree, Mphasis. JD-penalized when the *whole* career is here.
- **Fictional placeholders** (~7,500 each): Wayne Enterprises, Initech, Pied Piper, Globex, Acme, Dunder Mifflin, Hooli, Stark Industries. Treated as neutral.

## Honeypots: structurally detectable subset

The spec seeds ~80 impossible profiles (forced to tier 0). Three structural checks fire on **65** of them with **zero** false positives against genuine candidates:

| Check | Count |
|---|---|
| `expert` proficiency, skill used **0 months** | 21 |
| role `duration_months` exceeds calendar span of its dates | 19 |
| stated `years_of_experience` exceeds summed career tenure by >4.5 yr | 25 |

The remaining ~15 rely on facts not in the data (e.g. company founding year). They're indistinguishable from strong profiles, but the role/behavioral gates keep them out of the top 100 anyway.

## Geography (Indian cities evenly spread)

The 18 Indian metros each hold ~4,000-4,300 candidates (Pune, Noida, Hyderabad, Bangalore, Mumbai, Delhi, Gurgaon present). Non-India cities ~2,400-2,600 each. No visa sponsorship, so the location gate strongly favors India and lightly rewards relocation willingness.
