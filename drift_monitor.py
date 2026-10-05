"""
drift_monitor.py
----------------
Model and Data Drift Monitoring Engine for Customer Review Intelligence.
Tracks concept drift, sentiment distribution shift (via Population Stability Index),
and vocabulary/topic divergence between reference baseline reviews and production batches.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SentimentDistribution:
    positive_pct: float
    neutral_pct: float
    negative_pct: float
    mean_compound: float


@dataclass
class DriftReport:
    psi_sentiment: float
    sentiment_drift_status: str  # "HEALTHY" | "MODERATE_DRIFT" | "SIGNIFICANT_DRIFT"
    reference_distribution: SentimentDistribution
    current_distribution: SentimentDistribution
    emerging_keywords: List[Dict[str, any]] = field(default_factory=list)
    area_distribution_shifts: Dict[str, float] = field(default_factory=dict)
    summary_message: str = ""


class DriftMonitor:
    """Monitors concept and distribution drift across review batches."""

    def __init__(self, psi_warning_threshold: float = 0.1, psi_critical_threshold: float = 0.25):
        self.psi_warning_threshold = psi_warning_threshold
        self.psi_critical_threshold = psi_critical_threshold

    @staticmethod
    def _compute_distribution(reviews: List[dict]) -> SentimentDistribution:
        if not reviews:
            return SentimentDistribution(0.0, 0.0, 0.0, 0.0)

        total = len(reviews)
        pos = sum(1 for r in reviews if r.get("sentiment_label") == "positive" or r.get("sentiment") == "positive")
        neu = sum(1 for r in reviews if r.get("sentiment_label") == "neutral" or r.get("sentiment") == "neutral")
        neg = sum(1 for r in reviews if r.get("sentiment_label") == "negative" or r.get("sentiment") == "negative")

        scores = [r.get("sentiment_score", 0.0) if r.get("sentiment_score") is not None else r.get("score", 0.0) for r in reviews]
        mean_score = sum(scores) / total if total > 0 else 0.0

        return SentimentDistribution(
            positive_pct=round((pos / total) * 100, 2),
            neutral_pct=round((neu / total) * 100, 2),
            negative_pct=round((neg / total) * 100, 2),
            mean_compound=round(mean_score, 3),
        )

    def calculate_psi(self, expected_proportions: List[float], actual_proportions: List[float], epsilon: float = 1e-4) -> float:
        """Computes Population Stability Index (PSI) between two discrete probability distributions."""
        psi = 0.0
        for exp, act in zip(expected_proportions, actual_proportions):
            # Clamp with epsilon to prevent division by zero or log(0)
            exp_safe = max(exp, epsilon)
            act_safe = max(act, epsilon)
            psi += (act_safe - exp_safe) * math.log(act_safe / exp_safe)
        return round(psi, 4)

    def compute_drift(
        self,
        reference_reviews: List[dict],
        current_reviews: List[dict],
    ) -> DriftReport:
        ref_dist = self._compute_distribution(reference_reviews)
        cur_dist = self._compute_distribution(current_reviews)

        # Proportions for positive, neutral, negative
        ref_props = [ref_dist.positive_pct / 100.0, ref_dist.neutral_pct / 100.0, ref_dist.negative_pct / 100.0]
        cur_props = [cur_dist.positive_pct / 100.0, cur_dist.neutral_pct / 100.0, cur_dist.negative_pct / 100.0]

        psi = self.calculate_psi(ref_props, cur_props)

        if psi >= self.psi_critical_threshold:
            status = "SIGNIFICANT_DRIFT"
            msg = f"Critical sentiment distribution drift detected (PSI = {psi:.4f} >= {self.psi_critical_threshold}). Model re-calibration recommended."
        elif psi >= self.psi_warning_threshold:
            status = "MODERATE_DRIFT"
            msg = f"Moderate sentiment shift detected (PSI = {psi:.4f}). Customer perception is shifting."
        else:
            status = "HEALTHY"
            msg = f"Sentiment distribution is stable (PSI = {psi:.4f} < {self.psi_warning_threshold}). No significant drift."

        # Detect emerging vocabulary / keywords in current reviews not prevalent in reference
        ref_words = self._extract_token_frequencies(reference_reviews)
        cur_words = self._extract_token_frequencies(current_reviews)

        emerging = []
        cur_total = sum(cur_words.values()) or 1
        ref_total = sum(ref_words.values()) or 1

        for word, count in cur_words.most_common(20):
            cur_freq = count / cur_total
            ref_freq = ref_words.get(word, 0) / ref_total
            growth_ratio = (cur_freq + 1e-4) / (ref_freq + 1e-4)
            if growth_ratio >= 1.5 and count >= 3:
                emerging.append({
                    "keyword": word,
                    "current_frequency": count,
                    "relative_growth": round(growth_ratio, 2)
                })

        # Calculate governed area percentage shifts
        ref_areas = Counter(r.get("product_area") or r.get("area") for r in reference_reviews if r.get("product_area") or r.get("area"))
        cur_areas = Counter(r.get("product_area") or r.get("area") for r in current_reviews if r.get("product_area") or r.get("area"))

        area_shifts = {}
        all_areas = set(ref_areas.keys()).union(set(cur_areas.keys()))
        for area in all_areas:
            ref_pct = (ref_areas.get(area, 0) / len(reference_reviews)) * 100 if reference_reviews else 0.0
            cur_pct = (cur_areas.get(area, 0) / len(current_reviews)) * 100 if current_reviews else 0.0
            area_shifts[area] = round(cur_pct - ref_pct, 2)

        return DriftReport(
            psi_sentiment=psi,
            sentiment_drift_status=status,
            reference_distribution=ref_dist,
            current_distribution=cur_dist,
            emerging_keywords=emerging[:8],
            area_distribution_shifts=area_shifts,
            summary_message=msg,
        )

    @staticmethod
    def _extract_token_frequencies(reviews: List[dict]) -> Counter:
        stopwords = {"the", "and", "was", "for", "with", "this", "that", "from", "were", "they", "have", "been", "about"}
        counts = Counter()
        for r in reviews:
            text = r.get("review_text") or r.get("review_text_scrubbed") or r.get("text") or ""
            words = [w.lower().strip(".,!?:;'\"()") for w in str(text).split() if len(w) > 3]
            counts.update(w for w in words if w not in stopwords)
        return counts


if __name__ == "__main__":
    monitor = DriftMonitor()
    ref = [{"text": "Great product, fast delivery", "sentiment": "positive", "score": 0.8, "area": "Product Quality"} for _ in range(50)]
    cur = [{"text": "Terrible quality, broken item, late shipping", "sentiment": "negative", "score": -0.7, "area": "Product Quality"} for _ in range(50)]
    report = monitor.compute_drift(ref, cur)
    print("PSI:", report.psi_sentiment)
    print("Status:", report.sentiment_drift_status)
    print("Message:", report.summary_message)
