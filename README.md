## 📂 Included Files & Project Structure

| File | Description |
| :--- | :--- |
| `executive_dashboard.html` | The primary Executive Dashboard with direct CSV drag-and-drop, product auto-discovery, pure-SVG Donut chart, governed stacked bars, and verbatim review cards. |
| `index.html` | Comprehensive 4-tab workbench (Executive Overview, Single Review Tester, Batch Explorer, Taxonomy Reference). |
| `server.py` | Lightweight REST API server built on Python standard library `http.server`. Supports `/api/batch`, `/api/products`, `/api/product-reviews`, `/api/ingest-product-reviews`, and `/api/taxonomy`. |
| `sentiment_topic_analysis.py` | Core NLP pipeline with VADER, LDA/BERTopic, and keyword-to-taxonomy mapping algorithms. |
| `reviews_dataset_products.csv` | Pre-loaded 566-review multi-product dataset (Nimbus Earbuds, Verve Standing Desk, Kestrel Shoes, Trailhead Backpack, Aurora Blender, Solace Blanket). |
| `reviews_dataset_labeled.csv` | Pre-loaded 66-review validation dataset with `true_label` ground truth. |
| `sample_reviews.csv` | Baseline 18-review test dataset. |
| `start.sh` | One-click launch script that boots the server and opens the browser. |
| `requirements.txt` | Optional package specifications for heavy transformer dependencies (optional, not required to run). |

---

## 🌟 Key Features

1. **Direct CSV File Upload & Drag-and-Drop**:
   - Upload any CSV or drop it into the upload zone.
   - Automatically detects columns (`review_text`, `review_date`, `product_area`, `true_label`, `review_id`).
   - Intelligently extracts mentioned products from review text and generates a dedicated product dropdown!

2. **Executive Product Operations Hub**:
   - Tailored specifically for internal company employees, product managers, and category leads.
   - Switch between individual products (e.g. *Nimbus Earbuds*, *Verve Desk*) or view the entire catalog portfolio at once.
   - Highlights 3 immediate takeaways:
     - 🚨 **Critical Operations Action** (e.g., Shipping delays / carrier complaints)
     - 💡 **Quality & Engineering Flag** (e.g., `#broken after days`, `#defective part`)
     - 🏆 **Customer Delight** (e.g., `#great quality`, `#quick support`)

3. **Governed Business Dimensions (6 Fixed Areas)**:
   - `Shipping & Delivery`
   - `Product Quality`
   - `Customer Support`
   - `Pricing & Value`
   - `App / Website UX`
   - `Packaging`
   - Shows stacked positive, neutral, and negative volume bars for each area.

4. **1-Click Escalation & Power BI Export**:
   - Hover over any review card and click **"Copy for Slack/Jira"** to immediately format and copy the quote for internal team tickets.
   - Export filtered dataset to CSV with the exact Power BI ingestion schema:
     `review_id, product_name, review_text_scrubbed, sentiment_label, sentiment_score, sentiment_confidence, product_area, topic_keywords, review_date`
