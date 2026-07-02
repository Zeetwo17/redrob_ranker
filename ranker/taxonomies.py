"""
taxonomies.py
=============
Domain knowledge, encoded once. This is the "what a great recruiter already
knows" layer: which titles map to the role, which skills actually matter for a
retrieval/ranking engineer, which companies are product vs. services, and what
geography the JD targets.

Everything here is derived from (a) the released job description and (b) a full
pass over candidates.jsonl (see notebooks / data_audit). The frequency bands and
exact label strings are taken from the real data so matching is exact, not fuzzy.

Nothing in this file is candidate-specific or learned from labels — it is the
explicit prior we can defend in an interview.
"""

# ---------------------------------------------------------------------------
# 1. TITLES  --  role-fit prior
# ---------------------------------------------------------------------------
# The JD is "Senior AI Engineer — Founding Team", owning ranking / retrieval /
# matching. Titles are bucketed by how directly they map to that mandate.

BULLSEYE_TITLES = {
    "Senior AI Engineer", "Lead AI Engineer", "AI Engineer",
    "ML Engineer", "Machine Learning Engineer", "Senior Machine Learning Engineer",
    "Staff Machine Learning Engineer", "Applied ML Engineer",
    "Senior Software Engineer (ML)",
    "NLP Engineer", "Senior NLP Engineer",
    "Recommendation Systems Engineer", "Search Engineer",
    "Senior Applied Scientist",
}

STRONG_TITLES = {
    # Data science / applied AI — strong but slightly off the "engineer who ships
    # ranking systems" centre of mass.
    "Data Scientist", "Senior Data Scientist", "AI Specialist",
}

# Research-titled roles. The JD is explicit: pure-research backgrounds without
# production deployment are a disqualifier. We treat these as "strong only if the
# career history shows production work" (handled in features.py), so they start
# lower than BULLSEYE.
RESEARCH_TITLES = {
    "AI Research Engineer", "Senior Applied Scientist",
}

# CV / speech specialists. JD: "primary expertise is computer vision, speech, or
# robotics without significant NLP/IR exposure" -> not wanted. Penalised unless
# NLP/IR shows up elsewhere (handled in skill scoring).
CV_SPEECH_TITLES = {
    "Computer Vision Engineer",
}

JUNIOR_TITLES = {
    "Junior ML Engineer",
}

# Adjacent engineering — could pivot into the role; the JD's "Tier-5 who built a
# recommendation system at a product company" lives here when career history
# backs it up.
ADJACENT_DATA_TITLES = {
    "Data Engineer", "Senior Data Engineer", "Analytics Engineer",
    "Backend Engineer", "Data Analyst",
}
ADJACENT_SWE_TITLES = {
    "Software Engineer", "Senior Software Engineer", "Full Stack Developer",
}

# Tech, but not ML/AI.
OFFTARGET_TECH_TITLES = {
    "Cloud Engineer", "DevOps Engineer", "Frontend Engineer", "Mobile Developer",
    "QA Engineer", "Java Developer", ".NET Developer",
}

# The noise floor — ~68% of the pool. A "keyword stuffer" is one of these with an
# AI skill list bolted on. Their *career history* gives them away.
NOISE_TITLES = {
    "Business Analyst", "HR Manager", "Mechanical Engineer", "Accountant",
    "Project Manager", "Customer Support", "Operations Manager", "Content Writer",
    "Sales Executive", "Civil Engineer", "Graphic Designer", "Marketing Manager",
}

# Base role-fit score per bucket (0..1). Trajectory can lift a weak title
# (features.py), but a NOISE title with no ML career history stays near the floor.
TITLE_BASE = {}
for t in BULLSEYE_TITLES:        TITLE_BASE[t] = 1.00
for t in STRONG_TITLES:          TITLE_BASE[t] = 0.82
for t in RESEARCH_TITLES:        TITLE_BASE[t] = 0.78   # gated by production history
for t in ADJACENT_DATA_TITLES:   TITLE_BASE[t] = 0.55
for t in CV_SPEECH_TITLES:       TITLE_BASE[t] = 0.40   # gated by NLP/IR presence
for t in JUNIOR_TITLES:          TITLE_BASE[t] = 0.55   # gated by experience
for t in ADJACENT_SWE_TITLES:    TITLE_BASE[t] = 0.42
for t in OFFTARGET_TECH_TITLES:  TITLE_BASE[t] = 0.22
for t in NOISE_TITLES:           TITLE_BASE[t] = 0.06

# Keyword test for classifying *career-history* titles into "is this an AI/ML/IR
# role" without enumerating every variant.
ML_TITLE_KEYWORDS = (
    "ai engineer", "ml engineer", "machine learning", "applied scientist",
    "nlp", "recommendation", "search engineer", "data scientist",
    "ai specialist", "ai research", "relevance", "ranking",
)


def title_bucket_score(title: str) -> float:
    """Base role-fit for an exact title string, with a keyword fallback."""
    if title in TITLE_BASE:
        return TITLE_BASE[title]
    t = title.lower()
    if any(k in t for k in ML_TITLE_KEYWORDS):
        return 0.80
    if "data engineer" in t or "analytics engineer" in t or "backend" in t:
        return 0.55
    if "software engineer" in t or "developer" in t or "full stack" in t:
        return 0.42
    return 0.20


def is_ml_title(title: str) -> bool:
    t = title.lower()
    return any(k in t for k in ML_TITLE_KEYWORDS)


# ---------------------------------------------------------------------------
# 2. SKILLS  --  ontology with relevance weights for THIS jd
# ---------------------------------------------------------------------------
# Weight = how much this skill speaks to "production embeddings / retrieval /
# ranking", which is what the JD says it actually needs.

CORE_RETRIEVAL = {
    "Embeddings", "Vector Search", "Semantic Search", "Information Retrieval",
    "Information Retrieval Systems", "RAG", "Sentence Transformers",
    "Pinecone", "Weaviate", "Qdrant", "Milvus", "FAISS", "pgvector",
    "Elasticsearch", "OpenSearch", "Haystack", "BM25", "Learning to Rank",
    "Recommendation Systems",
}
LLM_SKILLS = {
    "LLMs", "LangChain", "LlamaIndex", "Fine-tuning LLMs", "LoRA", "QLoRA",
    "PEFT", "Prompt Engineering", "Hugging Face Transformers",
}
NLP_FOUNDATION = {"NLP"}
ML_FOUNDATION = {
    "Python", "PyTorch", "TensorFlow", "scikit-learn", "Machine Learning",
    "Deep Learning", "Feature Engineering", "MLOps", "MLflow", "Kubeflow",
    "Data Science", "Statistical Modeling", "BentoML", "Weights & Biases",
}
DATA_ENG = {
    "Spark", "Airflow", "dbt", "Snowflake", "ETL", "Data Pipelines", "Kafka",
    "BigQuery", "Databricks", "Hadoop", "Apache Beam", "Apache Flink",
}
# CV / speech — explicitly NOT what this JD wants. Zero positive weight; tracked
# separately to detect candidates whose AI identity is *dominantly* CV/speech.
CV_SPEECH_SKILLS = {
    "YOLO", "GANs", "OpenCV", "ASR", "Image Classification", "Computer Vision",
    "Speech Recognition", "CNN", "Object Detection", "Diffusion Models", "TTS",
}

SKILL_RELEVANCE = {}
for s in CORE_RETRIEVAL:  SKILL_RELEVANCE[s] = 1.00
for s in LLM_SKILLS:      SKILL_RELEVANCE[s] = 0.70
for s in NLP_FOUNDATION:  SKILL_RELEVANCE[s] = 0.80
for s in ML_FOUNDATION:   SKILL_RELEVANCE[s] = 0.50
for s in DATA_ENG:        SKILL_RELEVANCE[s] = 0.30
for s in CV_SPEECH_SKILLS: SKILL_RELEVANCE[s] = 0.00

# A short list of "deep" skills that stuffers tend NOT to have (they copy the
# buzzy cluster). Presence is a small positive authenticity signal.
DEEP_RETRIEVAL = {
    "pgvector", "Weaviate", "Milvus", "Qdrant", "OpenSearch", "Elasticsearch",
    "Haystack", "BM25", "Learning to Rank", "QLoRA", "PEFT",
}

PROFICIENCY_WEIGHT = {
    "beginner": 0.40, "intermediate": 0.70, "advanced": 1.00, "expert": 1.15,
}


# ---------------------------------------------------------------------------
# 3. COMPANIES  --  product vs. services prior
# ---------------------------------------------------------------------------
AI_PRODUCT_COMPANIES = {
    "Sarvam AI", "Aganitha", "Krutrim", "Haptik", "Wysa", "Observe.AI",
    "Yellow.ai", "Mad Street Den", "Niramai", "Rephrase.ai", "Saarthi.ai",
    "Verloop.io", "Glance", "Locobuzz", "Genpact AI",
}
PRODUCT_UNICORNS = {
    "Swiggy", "Zomato", "CRED", "Razorpay", "Flipkart", "Meesho", "InMobi",
    "Nykaa", "Zoho", "Freshworks", "Ola", "Paytm", "PhonePe", "Dream11",
    "PharmEasy", "PolicyBazaar", "Vedantu", "upGrad", "BYJU'S", "Unacademy",
}
FAANG = {
    "Google", "Meta", "Amazon", "Microsoft", "Apple", "Netflix", "LinkedIn",
    "Salesforce", "Adobe", "Uber",
}
# Career-long tenure at these is a JD disqualifier (services / staffing).
# NOTE: "Genpact AI" is the AI arm and is treated as AI_PRODUCT, not consulting.
CONSULTING = {
    "TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "Capgemini", "HCL",
    "Tech Mahindra", "Mindtree", "Mphasis",
}
# Fictional placeholders that fill the bulk of the pool — treated as neutral
# employers (no positive or negative product signal).
PLACEHOLDER_COMPANIES = {
    "Wayne Enterprises", "Initech", "Pied Piper", "Globex Inc", "Acme Corp",
    "Dunder Mifflin", "Hooli", "Stark Industries",
}

AI_INDUSTRIES = {"AI/ML", "Conversational AI", "Voice AI", "HealthTech AI", "AI Services"}
PRODUCT_INDUSTRIES = {
    "Fintech", "Food Delivery", "E-commerce", "SaaS", "Gaming", "EdTech",
    "HealthTech", "AdTech", "Insurance Tech", "Transportation", "Internet",
}


def company_score(company: str, industry: str) -> float:
    """Single-company product-vs-services prior (0..1)."""
    if company in AI_PRODUCT_COMPANIES or industry in AI_INDUSTRIES:
        return 1.00
    if company in FAANG:
        return 0.90
    if company in PRODUCT_UNICORNS or industry in PRODUCT_INDUSTRIES:
        return 0.85
    if company in CONSULTING:
        return 0.30
    if company in PLACEHOLDER_COMPANIES:
        return 0.50
    return 0.50  # unknown / generic software / IT services


def is_consulting(company: str) -> bool:
    return company in CONSULTING


# ---------------------------------------------------------------------------
# 4. GEOGRAPHY  --  the JD targets Pune / Noida, India; no visa sponsorship
# ---------------------------------------------------------------------------
TARGET_CITIES = ("pune", "noida")
# Welcomed in the JD ("Hyderabad, Pune, Mumbai, Delhi NCR welcome to apply") plus
# the major Indian tech hubs an AI hire realistically comes from.
GOOD_METROS = ("hyderabad", "mumbai", "delhi", "gurgaon", "gurugram", "bangalore",
               "bengaluru", "noida", "pune", "chennai")


def location_multiplier(location: str, country: str, willing_to_relocate: bool) -> float:
    """
    Multiplicative geo gate (0.35 .. 1.0).

    JD: Pune/Noida offices, open to relocation from Tier-1 Indian cities, and
    explicitly "Outside India: case-by-case, but we don't sponsor work visas."
    Non-India is therefore strongly suppressed, not zeroed.
    """
    loc = (location or "").lower()
    in_india = (country or "").strip().lower() == "india"

    if not in_india:
        return 0.45 if willing_to_relocate else 0.35

    if any(c in loc for c in TARGET_CITIES):
        return 1.00
    if any(c in loc for c in GOOD_METROS):
        return 1.00 if willing_to_relocate else 0.92
    # Other Indian city
    return 0.90 if willing_to_relocate else 0.78


# ---------------------------------------------------------------------------
# 5. EDUCATION  --  minor prior on institution tier
# ---------------------------------------------------------------------------
EDU_TIER_SCORE = {
    "tier_1": 1.00, "tier_2": 0.80, "tier_3": 0.60, "tier_4": 0.50,
    "unknown": 0.60,
}
