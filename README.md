# AI-Product-Insights-System-RAG-LLM-
Interactive, data-driven insight engine for user feedback analysis


# 🚀 AI Product Insights System (RAG + LLM)

### *Interactive, data-driven insight engine for user feedback analysis*

This project builds an **AI-powered Product Analyst** capable of answering natural-language questions about user complaints, product issues, feature requests, and emerging risks—powered by a **Retrieval-Augmented Generation (RAG)** pipeline and **LLM-generated insights**.

The system automatically retrieves the most relevant user reviews from a semantic vector store and asks an LLM (Claude Sonnet 4) to synthesize structured, actionable insights.

It works like asking a real product analyst:

> “Why are users complaining about subscription pricing?”
> “What are the biggest pain points in the last 12 months?”
> “What bugs or technical issues matter most?”
> “What new features do users want?”

All answers are **evidence-based**, **recent**, and **action-oriented**.

---

# ⭐ Key Capabilities

### 🔎 **1. RAG-based Semantic Retrieval**

* Uses **MiniLM-L6-v2** embeddings
* Stores 24k *recent* reviews in **ChromaDB**
* Supports metadata filters (date, rating, version, sentiment, engagement)

### 🧠 **2. LLM-Generated Product Insights**

Claude analyzes retrieved reviews to produce:

* complaint themes
* root causes
* trend summaries
* representative user quotes
* product recommendations with priority levels

### 🗣️ **3. Natural-Language Q&A**

Ask anything:

```
"What are users' biggest complaints recently?"
"Why did sentiment decline this year?"
"What do users say about safety & fake profiles?"
"Which features are most requested?"
```

The system retrieves → analyzes → answers.

### 🧩 **4. Recent-Only Knowledge Base**

Vector store is built with **last 12 months** of reviews to ensure analysis stays relevant.

---

# 🧱 System Architecture

```
User Question
     ↓
Semantic Retrieval (MiniLM + Chroma)
     ↓
Relevant Review Subset (filtered by time / score / metadata)
     ↓
LLM Analysis (Claude Sonnet 4)
     ↓
Structured Insights (themes, causes, actions, priorities)
```

---

# 📂 Project Structure

```
project/
│
├── README.md
├── data/
│     └── sample_reviews.csv (optional)
│
├── src/
│     ├── build_rag_system_recent.py      # builds vector DB (recent 12 months)
│     ├── rag_with_claude.py              # main RAG + analysis engine
│     └── utils.py (optional)
│
├── tinder_rag_db_recent/ (ignored)       # Chroma vector database
└── requirements.txt
```

---

# 🧪 Example Q&A Showcase

*(all outputs below are real excerpts from the system)*

---

### **Q: “What are the biggest complaints in the last 12 months?”**

**Main themes**

* Aggressive monetization (weekly pricing, hidden charges, unclear paywalls)
* Poor customer service (automated replies, no escalation)
* Core functionality issues (filtering, recycled profiles, payment bugs)

**Representative evidence**

* “$44/week is absolutely insane.”
* “Customer service is non-existent.”
* “Keeps showing people I already declined.”

**Recommendations**

* High: improve pricing transparency, overhaul CS workflows
* Medium: fix orientation filtering, eliminate profile recycling
* Low: investigate multi-charge anomalies

---

## Why Users Are Uninstalling/Canceling Subscriptions

**Direct Answer:** Users are primarily canceling due to **billing fraud concerns** and **account bans immediately after subscribing**. However, the critical issue is that **users cannot successfully cancel** - the app continues charging even after cancellation attempts, account deletions, and bans.

## Key Patterns & Trends

### 1. **Billing/Cancellation Crisis (Peak: Apr-Nov 2025)**
- **8+ reviews** report continued charges after cancellation
- Users charged even with **banned/deleted accounts** (Reviews 3, 7, 10, 18)
- **No accessible cancellation method** - not even in Play Store subscriptions (Review 8)
- Charges continue after **account deactivation** (Reviews 18, 19)

### 2. **Subscription-Triggered Account Bans (Consistent 2025)**
- **Immediate bans upon subscribing** (Reviews 10, 14: "as soon as I subscribed, I received a ban")
- Users lose access but **billing continues** (Reviews 3, 7)
- No refunds provided for banned accounts (Review 9)

### 3. **Deceptive Subscription Practices**
- **Hidden weekly billing** instead of expected monthly (Review 16)
- **Reduced functionality after subscribing** - fewer profiles shown (Review 5)
- Intentionally hiding matches to force continued payments (Review 11)

## Actionable Product Recommendations (Priority Order)

### **CRITICAL - Immediate Action Required**
1. **Fix cancellation system** - Ensure all cancellation methods work and stop billing immediately
2. **Stop billing banned accounts** - Automatic subscription cancellation when accounts are banned
3. **Implement clear checkout process** showing exact billing frequency and amounts

### **HIGH Priority**
4. **Review ban algorithms** - Investigate why subscriptions trigger immediate bans
5. **Add accessible customer support** - Phone/email support for billing issues
6. **Audit subscription functionality** - Ensure paid features work as advertised

---

### **Q: “What bugs affect user experience most?”**

* Login & session expiration failures
* Message sending failures
* Frozen UI during swipe
* Matches disappearing
* Payment flow breaks after upgrade

**Impact**: "Core usability blockers" → immediate engineering priority.

---

### **Q: “What new features are users requesting?”**

1. **Free "Undo" functionality with cooldown** - Users want ability to reverse accidental swipes without payment (Review #2, April 2025)

2. **Ad-supported free features** - Users specifically mention wanting the return of "watch ads to get another like" feature that was briefly available (Review #6, April 2025)

3. **Better notification controls** - Ability to receive match/message notifications WITHOUT spam promotional notifications (Review #14, April 2025)

4. **Enhanced filtering options** for premium users:
   - Filter out passport mode users
   - Sexuality-based filtering  
   - Regional pricing based on local statistics (Reviews #15, #6)

---

> These Q&A examples show how the system behaves like an *interactive product insights assistant*—able to pull evidence, summarize patterns, and propose actionable recommendations.

---

# 🧩 Implementation Details

### 📌 Vector Store

* Model: `all-MiniLM-L6-v2`
* DB: ChromaDB (persistent)
* Documents contain structured metadata:

  ```
  date, year, month, score, version,
  thumbs_up, has_reply, is_negative, is_positive
  ```

### 📌 Retrieval

```python
results = collection.query(
    query_embeddings=embed_model.encode([question]).tolist(),
    n_results=20,
    where={"date": {"$gte": "2025-01-01"}, "is_negative": True}
)
```

### 📌 LLM Analysis

A structured prompt guides Claude to:

* identify themes
* extract evidence
* reason about underlying causes
* prioritize actions

---

# 🛠 Setup

```
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key-here"
python src/build_rag_system_recent.py
```

Then in a notebook or script:

```python
from rag_with_claude import TinderRAGAnalyzer

analyzer = TinderRAGAnalyzer()
analyzer.ask(
    "What are users complaining about recently?",
    filters={"is_negative": True}
)
```

---

# 🧭 How This Generalizes to Real-World Business Scenarios

Although this project uses Tinder reviews,
the architecture is **industry-agnostic** and applies directly to:

### **E-commerce**

* Top reasons for returns
* Daily negative review spikes
* Emerging issues by category
* Feature requests for shopping experience
* Pricing sensitivity commentary

### **SaaS / Subscription Products**

* Onboarding friction
* Churn reasons
* Paywall frustration
* Feature gaps
* Ticket / support clustering

### **Customer Support / CX Analytics**

* Automated summarization of daily tickets
* Emerging bug detection
* Root cause analysis for escalations

### **How to adapt to enterprise settings**

(Conceptual, no code needed)

* Replace CSV with **API or warehouse ingestion** (Shopify, Amazon, Zendesk, BigQuery…)
* Incrementally embed **only new data**
* Append to the vector DB (no rebuild required)
* Schedule daily / weekly insight jobs (Airflow / Cron)
* Send insights to Slack / Email automatically
* Add dashboards on top of LLM insights

This shows you understand **how the project evolves into a real analytics platform**.

---

# 🎯 Skills Demonstrated (JD-aligned)

* RAG chain design
* Semantic retrieval & embedding architecture
* LLM-based analytical reasoning
* Topic synthesis & insight generation
* Product sense (pain points, opportunities, prioritization)
* End-to-end pipeline thinking
* Metadata schema design for retrieval
* Ability to generalize solution to real business environments

---

# 🌟 Summary

This system is a fully functional **AI Product Analyst** that:

✔ understands natural language
✔ retrieves the most relevant recent evidence
✔ synthesizes patterns, themes, and root causes
✔ produces actionable, product-ready insights

It demonstrates how **LLMs + RAG** can augment product teams and unlock fast, high-quality feedback analysis at scale.

 
