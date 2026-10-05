#!/usr/bin/env python3
"""
server.py
---------
Lightweight REST API and static file server for the Sentiment & Topic Analysis Dashboard.
Runs without requiring external web frameworks (built on Python standard library `http.server`).
"""

import json
import os
import sys
from http import HTTPStatus
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

# Import analysis modules
from sentiment_topic_analysis import (
    PRODUCT_AREA_KEYWORDS,
    VaderSentimentEngine,
    AzureLanguageSentimentEngine,
    LDATopicModeler,
    map_topic_to_product_area,
)
from pii_scrubber import PIIScrubber
from drift_monitor import DriftMonitor

pii_scrubber = PIIScrubber()
drift_monitor = DriftMonitor()

PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# Company Product Catalog with Customer Reviews (Scraped/Ingested under each product)
PRODUCTS_CATALOG = {
    "prod_aurasound": {
        "id": "prod_aurasound",
        "name": "AuraSound Max ANC Wireless Headphones",
        "sku": "AU-9021-ANC",
        "category": "Consumer Electronics / Audio",
        "price": "$179.99",
        "rating": 3.9,
        "team_owner": "Audio Hardware & Logistics Squad",
        "image_icon": "headphones",
        "reviews": [
            "The delivery was three weeks late and customer support never replied to my refund request.",
            "Great quality for the price, durable material, would buy again.",
            "The packaging box was completely crushed and torn open when arrived.",
            "Defective item upon arrival, broken hinge on day one.",
            "Worth every penny, fantastic build quality and fast delivery.",
            "Terrible support, agent closed the ticket without responding.",
            "Packaging was neat with secure protective bubble wrap.",
            "Courier arrived 4 days later than promised with no tracking update.",
            "Package arrived safely on time with intact protective seal.",
            "Support agent resolved my issue within 10 minutes.",
            "Courier tracking never updated and package arrived damaged."
        ]
    },
    "prod_ergodesk": {
        "id": "prod_ergodesk",
        "name": "ErgoFlex Pro Motorized Standing Desk (Dual Motor)",
        "sku": "DK-4410-MOT",
        "category": "Office Furniture & Ergonomics",
        "price": "$389.00",
        "rating": 4.3,
        "team_owner": "Furniture QA & Heavy Logistics",
        "image_icon": "table",
        "reviews": [
            "Motor is super quiet, durable material and fantastic smooth lift.",
            "The heavy packaging box arrived completely ripped with screws missing.",
            "Delivery was on time and courier helped bring the heavy package inside.",
            "Assembly instructions website was confusing and login didn't work.",
            "Customer service representative sent replacement screws within 48 hours, excellent support.",
            "Way overpriced for basic particle board, expected solid wood at this discount cost.",
            "Clean modern interface on the digital height controller, works flawlessly.",
            "Motor stopped working after 2 weeks, broken control box."
        ]
    },
    "prod_purifier": {
        "id": "prod_purifier",
        "name": "BreezeGlow Smart HEPA Air Purifier (IoT Enabled)",
        "sku": "AP-2200-IOT",
        "category": "Home Appliances / Smart Home",
        "price": "$129.50",
        "rating": 3.6,
        "team_owner": "Smart Home App & Firmware Team",
        "image_icon": "wind",
        "reviews": [
            "Mobile app crashed during checkout and device pairing keeps failing with infinite spinner.",
            "Whisper quiet operation, air quality sensor works great in living room.",
            "Replacement filters are severely overpriced, not worth the ongoing maintenance cost.",
            "App login screen times out constantly on iOS.",
            "Neat packaging, arrived fast in double-layered bubble wrap.",
            "Customer support agent helped reset the Wi-Fi module in 5 minutes.",
            "Plastic smells cheap during first 3 days of use, quality could be better."
        ]
    }
}


def analyze_reviews_list(reviews_list):
    """Runs PII scrubbing, VADER sentiment, and LDA topic modeling, preserving metadata"""
    engine = VaderSentimentEngine()
    topic_modeler = LDATopicModeler()

    texts = []
    metadata = []
    pii_records = []
    for i, item in enumerate(reviews_list):
        if isinstance(item, dict):
            raw_text = item.get("review_text") or item.get("review_text_scrubbed") or item.get("text") or item.get("comment") or ""
            metadata.append(item)
        else:
            raw_text = str(item)
            metadata.append({})

        # Execute PII Scrubbing
        scrub_res = pii_scrubber.scrub(str(raw_text))
        texts.append(scrub_res.scrubbed_text)
        pii_records.append(scrub_res)

    topic_assignments = topic_modeler.fit_transform(texts)

    results = []
    for i, text in enumerate(texts):
        sentiment = engine.score(text)
        t_assign = topic_assignments[i] if i < len(topic_assignments) else None
        keywords = t_assign.topic_keywords if t_assign else []
        if not keywords:
            for kw_list in PRODUCT_AREA_KEYWORDS.values():
                for k in kw_list:
                    if k in text.lower() and k not in keywords:
                        keywords.append(k)

        meta = metadata[i]
        # Use existing product_area from CSV if given, otherwise ML-mapped area
        product_area = meta.get("product_area")
        if not product_area or product_area not in PRODUCT_AREA_KEYWORDS:
            product_area = t_assign.product_area if t_assign else map_topic_to_product_area(keywords)

        s_res = pii_records[i]
        results.append({
            "id": meta.get("review_id") or (i + 1),
            "review_text_scrubbed": text,
            "sentiment_label": sentiment.label,
            "sentiment_score": sentiment.compound_score,
            "sentiment_confidence": sentiment.confidence,
            "topic_id": t_assign.topic_id if t_assign else 0,
            "topic_keywords": keywords[:6],
            "product_area": product_area,
            "review_date": meta.get("review_date", ""),
            "true_label": meta.get("true_label", ""),
            "product_name": meta.get("product_name", ""),
            "pii_detected": s_res.has_pii,
            "pii_redactions": s_res.redaction_count,
            "pii_entities": s_res.entities_found
        })
    return results


class SentimentAPIHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def _set_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self._set_cors_headers()
        self.end_headers()

    def _send_json(self, data, status=HTTPStatus.OK):
        response_bytes = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self._set_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)

    def do_GET(self):
        parsed = urlparse(self.path)
        from urllib.parse import parse_qs
        query_params = parse_qs(parsed.query)

        if parsed.path == "/api/health":
            self._send_json({"status": "ok", "engine": "VaderSentimentEngine + LDA", "products_count": len(PRODUCTS_CATALOG)})
            return
        elif parsed.path == "/api/taxonomy":
            self._send_json({
                "taxonomy": PRODUCT_AREA_KEYWORDS,
                "product_areas": list(PRODUCT_AREA_KEYWORDS.keys()) + ["Other / Uncategorized"]
            })
            return
        elif parsed.path == "/api/products":
            # Return list of products for employee to choose from
            products_summary = []
            for pid, p in PRODUCTS_CATALOG.items():
                products_summary.append({
                    "id": p["id"],
                    "name": p["name"],
                    "sku": p["sku"],
                    "category": p["category"],
                    "price": p["price"],
                    "rating": p["rating"],
                    "team_owner": p["team_owner"],
                    "image_icon": p.get("image_icon", "package"),
                    "reviews_count": len(p["reviews"])
                })
            self._send_json({"products": products_summary})
            return
        elif parsed.path == "/api/product-reviews":
            # Analyze reviews for a specific product
            product_id = query_params.get("product_id", ["prod_aurasound"])[0]
            if product_id == "ALL":
                all_revs = []
                for p in PRODUCTS_CATALOG.values():
                    all_revs.extend(p["reviews"])
                analyzed = analyze_reviews_list(all_revs)
                self._send_json({
                    "product": {
                        "id": "ALL",
                        "name": "All Products (Catalog Portfolio)",
                        "sku": "PORTFOLIO-ALL",
                        "category": "Entire Storefront Catalog",
                        "price": "N/A",
                        "rating": 4.0,
                        "team_owner": "Cross-Functional Product Operations",
                        "reviews_count": len(all_revs)
                    },
                    "count": len(analyzed),
                    "data": analyzed
                })
                return

            product = PRODUCTS_CATALOG.get(product_id)
            if not product:
                self._send_json({"error": f"Product '{product_id}' not found"}, HTTPStatus.NOT_FOUND)
                return

            analyzed = analyze_reviews_list(product["reviews"])
            self._send_json({
                "product": {
                    "id": product["id"],
                    "name": product["name"],
                    "sku": product["sku"],
                    "category": product["category"],
                    "price": product["price"],
                    "rating": product["rating"],
                    "team_owner": product["team_owner"],
                    "reviews_count": len(product["reviews"])
                },
                "count": len(analyzed),
                "data": analyzed
            })
            return
        elif parsed.path == "/api/default-reviews":
            csv_path = os.path.join(DIRECTORY, "sample_reviews.csv")
            reviews = []
            if os.path.exists(csv_path):
                import csv
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        text = row.get("review_text_scrubbed") or row.get("text")
                        if text:
                            reviews.append(text)
            
            if not reviews:
                for p in PRODUCTS_CATALOG.values():
                    reviews.extend(p["reviews"])

            results = analyze_reviews_list(reviews)
            self._send_json({"count": len(results), "data": results})
            return
        
        # Fall back to serving static files (index.html, styles, etc.)
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except Exception as e:
            self._send_json({"error": f"Invalid JSON payload: {str(e)}"}, HTTPStatus.BAD_REQUEST)
            return

        if parsed.path == "/api/analyze":
            # Single review analysis
            text = payload.get("text", "")
            engine_name = payload.get("engine", "vader")
            azure_endpoint = payload.get("azure_endpoint")
            azure_key = payload.get("azure_key")

            if engine_name == "azure" and azure_endpoint and azure_key:
                engine = AzureLanguageSentimentEngine(azure_endpoint, azure_key)
            else:
                engine = VaderSentimentEngine()

            sentiment = engine.score(text)
            
            # Simple keyword extraction & product area mapping
            words = [w.lower().strip(".,!?:;'\"()") for w in text.split() if len(w) > 2]
            matched_kws = []
            for area, kw_list in PRODUCT_AREA_KEYWORDS.items():
                for k in kw_list:
                    if k in text.lower() and k not in matched_kws:
                        matched_kws.append(k)
            
            if not matched_kws:
                matched_kws = words[:5]

            product_area = map_topic_to_product_area(matched_kws)

            self._send_json({
                "text": text,
                "sentiment": {
                    "label": sentiment.label,
                    "compound_score": sentiment.compound_score,
                    "confidence": sentiment.confidence,
                },
                "topic": {
                    "keywords": matched_kws,
                    "product_area": product_area,
                }
            })
            return

        elif parsed.path == "/api/batch":
            reviews = payload.get("reviews", [])
            results = analyze_reviews_list(reviews)
            self._send_json({
                "count": len(results),
                "data": results
            })
            return

        elif parsed.path == "/api/ingest-product-reviews":
            # Ingest customer reviews scraped under a specific product
            product_id = payload.get("product_id") or f"custom_prod_{len(PRODUCTS_CATALOG)+1}"
            product_name = payload.get("product_name") or "Custom Ingested Product"
            reviews = payload.get("reviews", [])
            texts = [r.get("text", "") if isinstance(r, dict) else str(r) for r in reviews if (r.get("text") if isinstance(r, dict) else str(r)).strip()]
            
            if not texts:
                self._send_json({"error": "No reviews provided in payload"}, HTTPStatus.BAD_REQUEST)
                return

            PRODUCTS_CATALOG[product_id] = {
                "id": product_id,
                "name": product_name,
                "sku": payload.get("sku", f"INGEST-{len(PRODUCTS_CATALOG)+1:03d}"),
                "category": payload.get("category", "Custom Product Listing"),
                "price": payload.get("price", "$99.00"),
                "rating": payload.get("rating", 4.1),
                "team_owner": payload.get("team_owner", "Product Operations"),
                "reviews": texts
            }

            analyzed = analyze_reviews_list(texts)
            self._send_json({
                "product": PRODUCTS_CATALOG[product_id],
                "count": len(analyzed),
                "data": analyzed
            })
            return

        elif parsed.path == "/api/scrub":
            # Direct PII scrubbing endpoint
            text = payload.get("text", "")
            res = pii_scrubber.scrub(text)
            self._send_json({
                "original_text": res.original_text,
                "scrubbed_text": res.scrubbed_text,
                "entities_found": res.entities_found,
                "redaction_count": res.redaction_count,
                "has_pii": res.has_pii
            })
            return

        elif parsed.path == "/api/drift":
            # Concept & Sentiment Distribution Drift Monitoring
            ref = payload.get("reference_reviews", [])
            cur = payload.get("current_reviews", [])
            report = drift_monitor.compute_drift(ref, cur)
            self._send_json({
                "psi_sentiment": report.psi_sentiment,
                "sentiment_drift_status": report.sentiment_drift_status,
                "reference_distribution": {
                    "positive_pct": report.reference_distribution.positive_pct,
                    "neutral_pct": report.reference_distribution.neutral_pct,
                    "negative_pct": report.reference_distribution.negative_pct,
                    "mean_compound": report.reference_distribution.mean_compound
                },
                "current_distribution": {
                    "positive_pct": report.current_distribution.positive_pct,
                    "neutral_pct": report.current_distribution.neutral_pct,
                    "negative_pct": report.current_distribution.negative_pct,
                    "mean_compound": report.current_distribution.mean_compound
                },
                "emerging_keywords": report.emerging_keywords,
                "area_distribution_shifts": report.area_distribution_shifts,
                "summary_message": report.summary_message
            })
            return

        self._send_json({"error": "Endpoint not found"}, HTTPStatus.NOT_FOUND)


def run_server():
    server_address = ("", PORT)
    httpd = HTTPServer(server_address, SentimentAPIHandler)
    print(f"============================================================")
    print(f"🚀 Sentiment & Topic Analysis Frontend Server Running!")
    print(f"👉 Open in browser: http://localhost:{PORT}")
    print(f"============================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()


if __name__ == "__main__":
    run_server()
