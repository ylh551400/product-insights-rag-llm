<p align="center">
  <img src="https://img.shields.io/badge/AI%20Product%20Insights-RAG%20%2B%20LLM-blueviolet?style=for-the-badge"/>
</p>

<p align="center">
  <b>Semantic Retrieval · LLM Reasoning · Product Intelligence</b><br>
  <sub>An interactive RAG-powered insight engine built on real user feedback</sub>
</p>

---
 
 

# 🚀 AI Product Insights System (RAG + LLM)

A fully interactive **AI Product Analyst** that extracts actionable insights from real user reviews using **Retrieval-Augmented Generation (RAG)** + **LLM analytical reasoning**.

Powered by:

* **ChromaDB** semantic search
* **MiniLM-L6-v2** embeddings
* **Claude Sonnet 4** for structured product analysis
* **Streamlit UI** for interactive exploration

This system behaves like a *senior product analyst*—retrieving relevant evidence, synthesizing themes, identifying root causes, and generating recommendations grounded in real user feedback.

---

# ⭐ Key Features

### 🔍 **Semantic Retrieval Engine**

* Uses MiniLM embeddings + ChromaDB
* Filters by recency, sentiment, engagement, metadata
* Optimized for **last 12 months** of reviews (keeps insights relevant)

### 🧠 **LLM-Based Product Analysis**

Claude generates structured insights:

* Complaint themes
* Root causes
* Trend shifts
* Product recommendations
* Evidence citations

### 🖥️ **Interactive Streamlit App**

A full UI that lets you:

* Choose analysis type (General / Root Cause / Feature Requests)
* Use built-in **Quick Questions** or type your own
* Inspect the **exact source reviews** used as evidence

👉 **Run the app:**

```bash
streamlit run src/app.py
```

---

# 📸 Streamlit Demo (UI Overview)

 <img width="1838" height="1145" alt="Screenshot 2025-12-07 143434" src="https://github.com/user-attachments/assets/f8c22b7e-e3e2-4e6f-8cd4-86ec979fb142" />


<img width="1859" height="1149" alt="Screenshot 2025-12-07 143446" src="https://github.com/user-attachments/assets/872ce2f1-662a-41e4-a550-89533d8157b2" />
 
<img width="1845" height="1142" alt="Screenshot 2025-12-07 143453" src="https://github.com/user-attachments/assets/832b8aa3-6b1d-406a-ba17-a08f49657801" />

The interface supports:

✔ Mode Selection

- General Analysis

- Root Cause Analysis

- Feature Requests

Each mode automatically loads different Quick Questions that match the analysis type.

✔ Ask Your Own Question: type any product question about complaints, matching algorithm issues, pricing, etc.


✔ Evidence Transparency: every insight comes with source review citations so conclusions are always grounded.

---

# 🧱 System Architecture

```
User Question
     ↓
Embedding → Semantic Retrieval (Chroma)
     ↓
Relevant Review Subset
     ↓
LLM Analysis (Claude Sonnet 4)
     ↓
Structured Insights + Evidence
```

---

# 📂 Project Structure

```
project/
│
├── data/
│     └── sample_reviews.csv              # Optional small demo dataset
│
├── src/
│     ├── app.py                          # Streamlit UI (main interface)
│     ├── build_rag_system_recent.py       # Builds 12-month vector DB
│     ├── rag_with_claude.py               # Core RAG + LLM analysis engine
│     └── __init__.py
│
├── examples/
│     └── demo_basic_usage.py
│
├── requirements.txt
└── README.md
```

Vector database directory (auto-generated):

```
tinder_rag_db_recent/
```

---

# 🛠 Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-api-key"
```

### Build the vector database (12-month window)

```bash
python src/build_rag_system_recent.py
```

### Run the interactive Streamlit app

```bash
streamlit run src/app.py
```

---

# 🧪 Programmatic Usage

```python
from rag_with_claude import TinderRAGAnalyzer

analyzer = TinderRAGAnalyzer()

analyzer.ask(
    "What are users complaining about recently?",
    filters={"is_negative": True}
)
```

---

# 🧾 Example Insights (Real Output)

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

* RAG system design
* Embedding-based semantic search
* Prompt engineering for analytical reasoning
* Trend detection & theme synthesis
* End-to-end pipeline thinking
* Streamlit UI development
* Evidence-grounded insight generation

---

# 🌟 Summary

This project delivers a complete **AI Product Insights Engine** that:

✔ retrieves the most relevant user evidence
✔ analyzes it with LLM reasoning
✔ synthesizes structured product insights
✔ visualizes everything in an interactive app
✔ generalizes directly to real-world product analytics workflows

It demonstrates how **RAG + LLM** can transform raw user feedback into *real, actionable product intelligence*.

 

 
