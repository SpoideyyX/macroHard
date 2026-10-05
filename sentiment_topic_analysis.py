"""
sentiment_topic_analysis.py
----------------------------
Sentiment scoring + topic modeling for scrubbed customer reviews, with
topics mapped to a fixed Product Area taxonomy so Power BI can slice by
both an ML-discovered topic AND a governed business dimension.

Two pluggable engines are provided for sentiment so the same pipeline runs
on a laptop (VADER, zero external calls) and in production (Azure AI
Language / HuggingFace transformer), without changing downstream code.

Topic modeling defaults to BERTopic (transformer embeddings + HDBSCAN)
when available, and falls back to scikit-learn LDA (bag-of-words) so the
pipeline still runs in constrained environments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Protocol

logger = logging.getLogger("reviews_pipeline.sentiment_topic")

# --------------------------------------------------------------------------
# Sentiment engines
# --------------------------------------------------------------------------
class SentimentEngine(Protocol):
    def score(self, text: str) -> "SentimentResult":
        ...


@dataclass
class SentimentResult:
    label: str          # "positive" | "neutral" | "negative"
    compound_score: float  # normalized [-1, 1]
    confidence: float      # [0, 1]


class VaderSentimentEngine:
    """Local, dependency-light baseline (NLTK's VADER). Good default for
    dev/test and for short, informal review text."""

    def __init__(self):
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            import nltk
            try:
                self._analyzer = SentimentIntensityAnalyzer()
            except LookupError:
                nltk.download("vader_lexicon", quiet=True)
                self._analyzer = SentimentIntensityAnalyzer()
            self._fallback = False
        except Exception:
            # Fallback simple sentiment analyzer if nltk is not installed
            self._fallback = True

    def score(self, text: str) -> SentimentResult:
        if not getattr(self, "_fallback", False):
            try:
                scores = self._analyzer.polarity_scores(text or "")
                compound = scores["compound"]
                if compound >= 0.05:
                    label = "positive"
                elif compound <= -0.05:
                    label = "negative"
                else:
                    label = "neutral"
                confidence = min(1.0, abs(compound) + 0.5)
                return SentimentResult(label, compound, round(confidence, 3))
            except Exception:
                pass
        
        # Rule-based fallback if NLTK is unavailable
        t = (text or "").lower()
        pos_words = {"great", "good", "love", "excellent", "fast", "awesome", "perfect", "easy", "satisfied", "helpful", "happy", "recommend"}
        neg_words = {"bad", "terrible", "late", "broken", "cheap", "refund", "horrible", "delayed", "damaged", "fail", "slow", "bug", "crash", "worst"}
        words = [w.strip(".,!?:;") for w in t.split()]
        pos_hits = sum(1 for w in words if w in pos_words)
        neg_hits = sum(1 for w in words if w in neg_words)
        
        if pos_hits > neg_hits:
            compound = min(1.0, 0.2 + 0.25 * pos_hits)
            label = "positive"
        elif neg_hits > pos_hits:
            compound = max(-1.0, -0.2 - 0.25 * neg_hits)
            label = "negative"
        else:
            compound = 0.0
            label = "neutral"
        confidence = min(1.0, abs(compound) + 0.45)
        return SentimentResult(label, round(compound, 3), round(confidence, 3))


class AzureLanguageSentimentEngine:
    """Production engine backed by Azure AI Language's sentiment API.
    Batches requests (max 10 docs/request per Azure limits) for throughput.
    """

    def __init__(self, endpoint: str, api_key: str):
        try:
            from azure.ai.textanalytics import TextAnalyticsClient
            from azure.core.credentials import AzureKeyCredential

            self._client = TextAnalyticsClient(
                endpoint=endpoint, credential=AzureKeyCredential(api_key)
            )
        except Exception as e:
            logger.warning("Could not initialize Azure client: %s", e)
            self._client = None

    def score_batch(self, texts: List[str]) -> List[SentimentResult]:
        if not self._client:
            vader = VaderSentimentEngine()
            return [vader.score(t) for t in texts]

        results = []
        for i in range(0, len(texts), 10):  # Azure batch limit
            batch = texts[i : i + 10]
            try:
                response = self._client.analyze_sentiment(batch)
                for doc in response:
                    if doc.is_error:
                        logger.error("Azure sentiment error: %s", doc.error)
                        results.append(SentimentResult("neutral", 0.0, 0.0))
                        continue
                    scores = doc.confidence_scores
                    compound = scores.positive - scores.negative
                    results.append(
                        SentimentResult(doc.sentiment, compound, max(scores.positive, scores.negative, scores.neutral))
                    )
            except Exception as ex:
                logger.error("Azure request failed: %s", ex)
                vader = VaderSentimentEngine()
                results.extend([vader.score(t) for t in batch])
        return results

    def score(self, text: str) -> SentimentResult:
        return self.score_batch([text])[0]


# --------------------------------------------------------------------------
# Product-area taxonomy for topic -> business dimension mapping
# --------------------------------------------------------------------------
PRODUCT_AREA_KEYWORDS = {
    "Shipping & Delivery": ["shipping", "delivery", "delayed", "late", "arrived", "package", "courier", "tracking"],
    "Product Quality": ["broken", "defective", "quality", "cheap", "material", "durable", "faulty", "damaged"],
    "Customer Support": ["support", "refund", "return", "agent", "response", "service", "representative", "help desk"],
    "Pricing & Value": ["price", "expensive", "value", "worth", "overpriced", "discount", "cost"],
    "App / Website UX": ["app", "website", "checkout", "login", "crash", "bug", "interface", "loading"],
    "Packaging": ["packaging", "box", "wrapping", "seal", "damaged box"],
}


def map_topic_to_product_area(topic_keywords: Iterable[str]) -> str:
    """Map an ML topic's top keywords to the closest governed Product Area
    using keyword overlap; falls back to 'Other / Uncategorized'."""
    best_area, best_score = "Other / Uncategorized", 0
    kw_set = {k.lower() for k in topic_keywords}
    for area, area_keywords in PRODUCT_AREA_KEYWORDS.items():
        overlap = len(kw_set.intersection(area_keywords))
        if overlap > best_score:
            best_area, best_score = area, overlap
    return best_area


# --------------------------------------------------------------------------
# Topic modeling
# --------------------------------------------------------------------------
@dataclass
class TopicAssignment:
    topic_id: int
    topic_keywords: List[str]
    product_area: str


class BERTopicModeler:
    def __init__(self, min_topic_size: int = 15):
        from bertopic import BERTopic

        self._model = BERTopic(min_topic_size=min_topic_size, verbose=False)

    def fit_transform(self, documents: List[str]) -> List[TopicAssignment]:
        topics, _ = self._model.fit_transform(documents)
        topic_info = self._model.get_topics()
        assignments = []
        for topic_id in topics:
            keywords = [kw for kw, _ in topic_info.get(topic_id, [])][:10]
            assignments.append(
                TopicAssignment(topic_id, keywords, map_topic_to_product_area(keywords))
            )
        return assignments


class LDATopicModeler:
    """Fallback topic modeler: scikit-learn LDA over TF-IDF/count vectors.
    No GPU/transformer dependency required."""

    def __init__(self, n_topics: int = 8, min_df: int = 3, random_state: int = 42):
        self._target_n_topics = n_topics
        self._target_min_df = min_df
        self._random_state = random_state
        self._vectorizer = None
        self._model = None

    def fit_transform(self, documents: List[str]) -> List[TopicAssignment]:
        try:
            from sklearn.decomposition import LatentDirichletAllocation
            from sklearn.feature_extraction.text import CountVectorizer

            n_docs = len(documents)
            effective_min_df = 1 if n_docs < 20 else min(self._target_min_df, max(1, n_docs // 10))
            effective_n_topics = max(1, min(self._target_n_topics, n_docs))

            self._vectorizer = CountVectorizer(
                stop_words="english", max_df=0.95, min_df=effective_min_df
            )
            self._model = LatentDirichletAllocation(
                n_components=effective_n_topics,
                random_state=self._random_state,
                learning_method="online",
            )

            doc_term_matrix = self._vectorizer.fit_transform(documents)
            if doc_term_matrix.shape[1] == 0:
                return [TopicAssignment(0, [], "Other / Uncategorized") for _ in documents]

            topic_distributions = self._model.fit_transform(doc_term_matrix)
            vocab = self._vectorizer.get_feature_names_out()

            topic_keywords_by_id = {}
            for topic_id, topic_weights in enumerate(self._model.components_):
                top_idx = topic_weights.argsort()[-10:][::-1]
                topic_keywords_by_id[topic_id] = [vocab[i] for i in top_idx]

            assignments = []
            for doc_dist in topic_distributions:
                topic_id = int(doc_dist.argmax())
                keywords = topic_keywords_by_id.get(topic_id, [])
                assignments.append(
                    TopicAssignment(topic_id, keywords, map_topic_to_product_area(keywords))
                )
            return assignments
        except Exception:
            # Fallback simple keyword extraction if scikit-learn is missing
            assignments = []
            for idx, doc in enumerate(documents):
                words = [w.lower().strip(".,!?:;") for w in doc.split() if len(w) > 3]
                matched_kws = []
                for kw_list in PRODUCT_AREA_KEYWORDS.values():
                    for k in kw_list:
                        if k in words and k not in matched_kws:
                            matched_kws.append(k)
                if not matched_kws:
                    matched_kws = words[:5]
                assignments.append(
                    TopicAssignment(idx % 5, matched_kws[:6], map_topic_to_product_area(matched_kws))
                )
            return assignments


def get_topic_modeler(prefer_bertopic: bool = True):
    if prefer_bertopic:
        try:
            return BERTopicModeler()
        except ImportError:
            logger.warning("BERTopic not installed; falling back to LDA topic modeler.")
    return LDATopicModeler()


def get_sentiment_engine(azure_endpoint: Optional[str] = None, azure_key: Optional[str] = None):
    if azure_endpoint and azure_key:
        try:
            return AzureLanguageSentimentEngine(azure_endpoint, azure_key)
        except ImportError:
            logger.warning("azure-ai-textanalytics not installed; falling back to VADER.")
    return VaderSentimentEngine()


def run_analysis(df, text_col: str = "review_text_scrubbed"):
    """
    Adds sentiment_label, sentiment_score, sentiment_confidence,
    topic_id, topic_keywords, product_area columns to df.
    """
    sentiment_engine = get_sentiment_engine()
    topic_modeler = get_topic_modeler()

    df = df.copy()
    texts = df[text_col].fillna("").astype(str).tolist()
    sentiment_results = [sentiment_engine.score(t) for t in texts]
    df["sentiment_label"] = [r.label for r in sentiment_results]
    df["sentiment_score"] = [r.compound_score for r in sentiment_results]
    df["sentiment_confidence"] = [r.confidence for r in sentiment_results]

    topic_assignments = topic_modeler.fit_transform(texts)
    df["topic_id"] = [t.topic_id for t in topic_assignments]
    df["topic_keywords"] = [", ".join(t.topic_keywords) for t in topic_assignments]
    df["product_area"] = [t.product_area for t in topic_assignments]

    return df


if __name__ == "__main__":
    engine = VaderSentimentEngine()
    for sample in [
        "The delivery was three weeks late and support never replied.",
        "Great quality for the price, would buy again.",
        "It's fine, does the job.",
    ]:
        print(sample, "->", engine.score(sample))
