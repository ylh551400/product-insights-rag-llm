 

# 🚀 AI Product Insights System (RAG + LLM)

### *An interactive, data-driven insights engine powered by semantic retrieval & LLM reasoning.*

This project implements an **AI-powered Product Analyst** capable of answering natural-language questions about user complaints, feature requests, product issues, and emerging risks—using a **Retrieval-Augmented Generation (RAG)** pipeline and **LLM-based analytical reasoning**.

Instead of manual review scrolling or sentiment dashboards, this system generates **actionable product insights** grounded in real user evidence from the last 12 months.

Ask it questions like:

> “Why are users complaining about subscription pricing?”
> “What are the biggest pain points recently?”
> “What bugs most affect user experience?”
> “What new features do users want?”

The system retrieves → analyzes → and synthesizes insights—behaving much like a senior product analyst.

---

# ⭐ Key Capabilities

### 🔎 **1. Semantic Retrieval via RAG**

* Embeddings: **MiniLM-L6-v2**
* Vector database: **ChromaDB**
* Metadata-aware filtering (date, sentiment, rating, version, engagement)
* Optimized for recency: uses only **last 12 months** of reviews to maintain relevance.

---

### 🧠 **2. LLM-Based Analytical Reasoning**

Claude Sonnet 4 generates structured insights:

* complaint themes
* root causes
* trend shifts
* supporting user evidence (quotes)
* product recommendations with prioritization

Not just “summaries”—but *product insights*.

---

### 🗣️ **3. Natural-Language Q&A Interface**

You can ask:

```
"What are users' biggest complaints recently?"
"Why did sentiment decline this year?"
"What do users say about safety & fake profiles?"
"What features are users requesting?"
```

The system automatically retrieves and reasons over the most relevant reviews.

---

# 🧱 System Architecture

```
Natural-Language Question
        ↓
Semantic Retrieval (MiniLM + ChromaDB)
        ↓
Relevant Review Subset (filtered by recency, rating, metadata)
        ↓
LLM Analysis (Claude Sonnet 4)
        ↓
Structured Insights (themes, causes, recommendations)
```

---

# 📂 Project Structure

```
project/
│
├── README.md
├── data/
│     └── sample_reviews.csv         # 200-row sample dataset
│
├── src/
│     ├── build_rag_system_recent.py # builds the vector DB (last 12 months)
│     ├── rag_with_claude.py         # RAG + LLM analysis engine
│     └── __init__.py
│
├── examples/
│     └── demo_basic_usage.py        # simple usage example
│
├── .gitignore
└── requirements.txt
```

---

# 🧪 Example Q&A Showcase

*(Real outputs from the system)*

---

## **Q: “What are the biggest complaints in the last 12 months?”**

### **Main Themes**

* **Aggressive monetization** (weekly billing, hidden fees, unclear paywalls)
* **Poor customer support** (no escalation path, automated replies)
* **Core functionality issues** (filters, recycled profiles, broken payment flows)

### **Representative Evidence**

* “$44/week is absolutely insane.”
* “Customer service is non-existent.”
* “Keeps showing people I already declined.”

### **Recommended Actions**

* **High:** Improve pricing transparency; rebuild CS workflows
* **Medium:** Fix orientation filtering; remove profile recycling
* **Low:** Investigate multi-charge anomalies

---

## **Q: “Why are users canceling subscriptions or uninstalling?”**

### **Key Drivers**

#### 1. **Billing/Cancellation Failures (critical)**

* Charges continue after cancellation attempts
* Users billed despite deleting/banning accounts
* Cancellation paths missing or broken

#### 2. **Subscription-Triggered Account Bans**

* Users report getting banned immediately after subscribing
* Billing continues even after losing access

#### 3. **Deceptive Pricing Models**

* Weekly billing framed as monthly
* Paid features provide less functionality than free version
* Perception that matching is throttled unless paying

### **Highest-Priority Fixes**

1. Fix cancellation + billing systems
2. Automatically stop billing banned accounts
3. Improve clarity of recurring charges
4. Review ban algorithms for false positives

---

## **Q: “What new features are users requesting?”**

1. **Free “Undo” with cooldown** (accidental swipe reversal)
2. **Ad-supported free features** (users want “watch ads for more likes” back)
3. **Smarter notification controls** (messages yes, promotional spam no)
4. **Better filtering:**

   * Sexuality / intent filters
   * Filter out passport-mode users
   * More region-aware pricing

> These examples illustrate how the system performs *interactive product insight generation*
> with evidence grounding, theme extraction, and actionable recommendations.

---

# 🧩 Implementation Details

### 📌 Vector Store

* Embedding model: `all-MiniLM-L6-v2`
* Stores recency-filtered review dataset (12 months)
* Metadata schema includes:

  ```
  date, year, month, rating, version,
  thumbs_up, has_reply, is_negative, is_positive
  ```

### 📌 Retrieval Example

```python
results = collection.query(
    query_embeddings=embed_model.encode([question]).tolist(),
    n_results=20,
    where={"date": {"$gte": "2025-01-01"}, "is_negative": True}
)
```

### 📌 LLM-Oriented Analysis

Structured prompting instructs Claude to:

* extract themes
* identify user pain points
* detect trends
* infer root causes
* recommend prioritized actions

---

# 🛠 Setup

```
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key-here"
python src/build_rag_system_recent.py
```

Use in any script or notebook:

```python
from rag_with_claude import TinderRAGAnalyzer

analyzer = TinderRAGAnalyzer()

analyzer.ask(
    "What are users complaining about recently?",
    filters={"is_negative": True}
)
```

---

# 🧭 Applicability to Real-World Business (Generalization)

Even though this project uses Tinder reviews,
the architecture generalizes directly to:

## **E-commerce**

* Return/reason clustering
* Pricing sensitivity feedback
* Category-level complaint spikes
* Feature requests for search, checkout, delivery

## **SaaS / Subscription Products**

* Churn reasons
* Onboarding friction
* Paywall frustration
* Feature-gap analysis

## **Customer Support / CX**

* Daily ticket summarization
* Emerging bug identification
* Root cause analysis for escalations

---

## **How this becomes production-ready**

In an enterprise environment:

* Replace CSV ingestion with **API or warehouse pipelines** (Shopify, Amazon, Zendesk, BigQuery)
* Incrementally embed **only new data**
* Append new vectors to the existing DB (no rebuild required)
* Schedule daily/weekly automated analysis via **Airflow/Cron**
* Deliver insights via Slack/Email dashboards

This demonstrates end-to-end product thinking:
**how a prototype insight engine becomes a real operational analytics system.**

---

# 🎯 Skills Demonstrated

* RAG chain architecture
* Embedding-based semantic search
* Advanced LLM prompting for analytical reasoning
* Topic synthesis & labeling
* Trend detection & root cause analysis
* Product sense: prioritization, monetization insights, UX complaints
* Pipeline thinking (data ingestion → retrieval → LLM → insight delivery)
* Ability to generalize to enterprise analytics environments

---

# 🌟 Summary

This project delivers a fully functional **AI Product Insights Assistant** that:

✔ understands natural language
✔ retrieves the most relevant recent evidence
✔ synthesizes patterns and causes
✔ offers actionable recommendations
✔ generalizes to real-world analytics workflows

It demonstrates how **RAG + LLM** can transform user feedback into high-quality product intelligence—at scale and in real time.

 
