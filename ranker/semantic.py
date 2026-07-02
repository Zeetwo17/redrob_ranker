"""
semantic.py
===========
The "reading between the lines" layer. A character/word TF-IDF model over the
candidate corpus, scored by cosine similarity against the distilled ideal-candidate
query (jd.IDEAL_QUERY).

Why TF-IDF and not a neural embedding model? The competition's ranking step must
run in <=5 min on CPU with NO network. TF-IDF (scikit-learn) needs no model
download, fits + transforms 100K docs in seconds, and is fully reproducible in any
sandbox. The architecture is intentionally swappable: replace `SemanticMatcher`
with a precomputed sentence-transformer matrix and the rest of the pipeline is
unchanged. We keep the dependency-light path as default for reproducibility.

The semantic score complements the structured signals: it rewards candidates whose
*free text* (summary, role descriptions) describes building search / ranking /
retrieval systems, even when they don't list the buzzy skill keywords.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .jd import IDEAL_QUERY


class SemanticMatcher:
    def __init__(self, max_features=60000, ngram_range=(1, 2), min_df=3):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=True,
        )
        self._query_vec = None

    def fit_transform_scores(self, documents):
        """
        Fit TF-IDF on the candidate corpus + the query, then return an array of
        cosine similarities (candidate -> query), min-max scaled to [0,1].

        documents : list[str], one per candidate, in candidate order.
        """
        corpus = documents + [IDEAL_QUERY]
        matrix = self.vectorizer.fit_transform(corpus)
        cand_matrix = matrix[:-1]
        query_vec = matrix[-1]

        # Cosine == dot product on L2-normalised TF-IDF rows (sklearn normalises).
        sims = linear_kernel(query_vec, cand_matrix).ravel()  # shape (n_candidates,)

        # Scale to [0,1] for blending. Robust min-max (clip tiny negatives).
        lo, hi = float(sims.min()), float(sims.max())
        if hi - lo < 1e-9:
            return np.zeros_like(sims)
        scaled = (sims - lo) / (hi - lo)
        return scaled
