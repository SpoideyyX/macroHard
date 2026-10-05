"""
pii_scrubber.py
---------------
High-performance PII (Personally Identifiable Information) scrubber for customer reviews.
Redacts sensitive identifiers before text is sent to sentiment scoring or topic modeling engines.
Outputs the governed column `review_text_scrubbed`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class ScrubResult:
    original_text: str
    scrubbed_text: str
    entities_found: Dict[str, int] = field(default_factory=dict)
    redaction_count: int = 0

    @property
    def has_pii(self) -> bool:
        return self.redaction_count > 0


class PIIScrubber:
    """Rule-based, high-performance PII detection and redaction engine.
    Supports email addresses, phone numbers, credit card numbers, national IDs,
    IP addresses, and sensitive customer identifiers.
    """

    # Compiled regex patterns for speed and efficiency
    PATTERNS: List[Tuple[str, re.Pattern, str]] = [
        # Email Addresses
        (
            "EMAIL",
            re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", re.IGNORECASE),
            "[EMAIL]",
        ),
        # Credit / Debit Card Numbers (13 to 19 digits, with or without spaces/dashes)
        (
            "CREDIT_CARD",
            re.compile(r"\b(?:\d{4}[-\s]?){3}\d{1,4}\b"),
            "[CARD_NUMBER]",
        ),
        # Phone Numbers (International & Domestic formats with optional extensions)
        (
            "PHONE",
            re.compile(
                r"(?:(?:\+?1\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?",
                re.IGNORECASE,
            ),
            "[PHONE]",
        ),
        # Alternate generic international phone regex (e.g. +91 9876543210, 09876543210)
        (
            "PHONE_INTL",
            re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
            "[PHONE]",
        ),
        # Social Security / National ID numbers (e.g., 000-00-0000)
        (
            "SSN_ID",
            re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            "[ID_NUMBER]",
        ),
        # IPv4 Addresses
        (
            "IP_ADDRESS",
            re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"),
            "[IP_ADDRESS]",
        ),
        # Order / Tracking Numbers (e.g., Order #12345678, Tracking: 1Z9999999999999999)
        (
            "ORDER_TRACKING",
            re.compile(r"\b(?:order|tracking|ticket|case|ref)\s*(?:#|no\.?|number|id)?\s*[:\-]?\s*([A-Za-z0-9\-_]{6,20})\b", re.IGNORECASE),
            "[ORDER_ID]",
        ),
        # Greeting names (e.g., "Hi John,", "Spoke with agent Sarah,")
        (
            "AGENT_CUSTOMER_NAME",
            re.compile(r"\b(?:rep|agent|representative|spoke with|talked to|named?)\s+([A-Z][a-z]{2,15})\b"),
            "[AGENT_NAME]",
        ),
    ]

    def scrub(self, text: str) -> ScrubResult:
        if not text:
            return ScrubResult(original_text="", scrubbed_text="", entities_found={}, redaction_count=0)

        scrubbed = str(text)
        entities_found: Dict[str, int] = {}
        total_redactions = 0

        for label, pattern, replacement in self.PATTERNS:
            matches = list(pattern.finditer(scrubbed))
            if matches:
                count = len(matches)
                entities_found[label] = entities_found.get(label, 0) + count
                total_redactions += count
                scrubbed = pattern.sub(replacement, scrubbed)

        return ScrubResult(
            original_text=text,
            scrubbed_text=scrubbed,
            entities_found=entities_found,
            redaction_count=total_redactions,
        )

    def scrub_batch(self, texts: List[str]) -> List[ScrubResult]:
        return [self.scrub(t) for t in texts]

    def scrub_dataframe(self, df, text_col: str = "review_text", output_col: str = "review_text_scrubbed"):
        """Utility for pandas DataFrames to add a scrubbed column."""
        df = df.copy()
        results = [self.scrub(t) for t in df[text_col].fillna("").astype(str)]
        df[output_col] = [r.scrubbed_text for r in results]
        df["pii_redaction_count"] = [r.redaction_count for r in results]
        df["pii_detected"] = [r.has_pii for r in results]
        return df


if __name__ == "__main__":
    scrubber = PIIScrubber()
    sample = "Hi, my name is John. Order #98214321 was lost! Contact me at john.doe@example.com or 415-555-2671. Spoke with agent Sarah."
    result = scrubber.scrub(sample)
    print("Original:", sample)
    print("Scrubbed:", result.scrubbed_text)
    print("Entities:", result.entities_found)
