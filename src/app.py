#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# app.py
import streamlit as st
import os
from sentence_transformers import SentenceTransformer
import chromadb
import anthropic

# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Tinder Product Insights Engine",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Configuration
# ============================================================

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "tinder_rag_db_recent"


# ============================================================
# Initialize System (with caching)
# ============================================================

@st.cache_resource
def load_models():
    """Load models once and cache them"""
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    collection = chroma_client.get_collection("tinder_reviews_recent")
    return embed_model, collection

@st.cache_data
def get_collection_stats(_collection):
    """Get collection statistics"""
    metadata = _collection.metadata
    return {
        'total_reviews': _collection.count(),
        'date_range': f"{metadata.get('date_range_start', 'N/A')} to {metadata.get('date_range_end', 'N/A')}"
    }

# ============================================================
# Analysis Functions
# ============================================================

def retrieve_reviews(query, n_results, filters, embed_model, collection):
    """Retrieve relevant reviews"""
    query_embedding = embed_model.encode([query])
    
    where_filter = {}
    if filters['review_type'] == 'Negative only':
        where_filter['is_negative'] = True
    elif filters['review_type'] == 'Positive only':
        where_filter['is_positive'] = True
    
    if filters['min_thumbs'] > 0:
        where_filter['thumbs_up'] = {'$gte': filters['min_thumbs']}
    
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results,
        where=where_filter if where_filter else None
    )
    
    return results

def analyze_with_claude(query, context_reviews, api_key, analysis_type):
    """Generate analysis using Claude"""
    
    if not context_reviews['documents'][0]:
        return "No relevant reviews found with the selected filters."
    
    context_text = "\n\n".join([
        f"Review {i+1}:\n{doc}"
        for i, doc in enumerate(context_reviews['documents'][0])
    ])
    
    prompts = {
        "General Analysis": """You are a senior Product Analyst analyzing user feedback.

User Reviews:
{context}

Question: {query}

Provide:
1. Direct answer with specific evidence
2. Key patterns identified
3. Actionable recommendations with priority
4. Timeline context if relevant

Be concise and data-driven.""",

        "Root Cause Analysis": """You are a senior Product Analyst conducting root cause analysis.

User Reviews:
{context}

Question: {query}

Analyze:
1. Primary root causes (ranked by evidence strength)
2. Supporting evidence with dates
3. Timeline: When did this issue emerge?
4. Affected user segments
5. Quick wins vs long-term fixes

Focus on recent trends.""",

        "Feature Requests": """You are a senior Product Analyst evaluating feature requests.

User Reviews:
{context}

Question: {query}

Provide:
1. Most requested features (ranked by frequency)
2. User pain points driving each request
3. Expected impact on satisfaction
4. Implementation priority
5. Emerging trends

Focus on what users want NOW."""
    }
    
    prompt_template = prompts.get(analysis_type, prompts["General Analysis"])
    prompt = prompt_template.format(context=context_text, query=query)
    
    client = anthropic.Anthropic(api_key=api_key)
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text

# ============================================================
# Main App
# ============================================================

def main():
    
    # Header
    st.title("🤖 Tinder Product Insights Engine")
    st.markdown("**AI-Powered Analysis of User Feedback**")
    st.markdown("---")
    
    # Load models
    try:
        with st.spinner("Loading models..."):
            embed_model, collection = load_models()
            stats = get_collection_stats(collection)
        
        # Display stats in sidebar
        st.sidebar.success("✅ System Ready")
        st.sidebar.metric("Total Reviews", f"{stats['total_reviews']:,}")
        st.sidebar.info(f"📅 Date Range:\n{stats['date_range']}")
        
    except Exception as e:
        st.error(f"❌ Error loading system: {e}")
        st.stop()
    
    # Sidebar - Configuration
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Configuration")
    
    api_key = st.sidebar.text_input(
        "Claude API Key",
        type="password",
        help="Get your API key from https://console.anthropic.com"
    )
    
    if not api_key:
        st.warning("⚠️ Please enter your Claude API key in the sidebar to start analyzing.")
        st.stop()
    
    # Sidebar - Analysis Settings
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Analysis Settings")
    
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["General Analysis", "Root Cause Analysis", "Feature Requests"]
    )
    
    n_reviews = st.sidebar.slider(
        "Number of reviews to analyze",
        min_value=5,
        max_value=30,
        value=15,
        help="More reviews = more context but slower analysis"
    )
    
    # Filters
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Filters")
    
    review_type = st.sidebar.radio(
        "Review Type",
        ["All reviews", "Negative only", "Positive only"]
    )
    
    min_thumbs = st.sidebar.number_input(
        "Minimum thumbs up",
        min_value=0,
        max_value=100,
        value=0,
        help="Filter for high-quality feedback"
    )
    
    filters = {
        'review_type': review_type,
        'min_thumbs': min_thumbs
    }
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("💬 Ask a Question")
        
        # Quick question buttons
        st.markdown("**Quick Questions:**")
        quick_questions = [
            "What are the biggest complaints in the last 12 months?",
            "What features are users requesting most?",
            "Why has the rating declined recently?",
            "What do users think about pricing?",
            "What bugs are most commonly reported?"
        ]
        
        selected_quick = st.selectbox(
            "Select a quick question (or write your own below)",
            [""] + quick_questions,
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("")  # Spacing
    
    # Text input
    user_query = st.text_area(
        "Or ask your own question:",
        value=selected_quick if selected_quick else "",
        height=100,
        placeholder="Example: What are users saying about the matching algorithm?"
    )
    
    # Analyze button
    analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    # Analysis
    if analyze_button and user_query:
        
        with st.spinner("🔄 Retrieving relevant reviews..."):
            reviews = retrieve_reviews(
                user_query,
                n_reviews,
                filters,
                embed_model,
                collection
            )
        
        if not reviews['documents'][0]:
            st.warning("No reviews found matching your filters. Try adjusting the filters.")
            st.stop()
        
        st.success(f"✅ Retrieved {len(reviews['documents'][0])} relevant reviews")
        
        with st.spinner("🤖 Generating analysis with Claude..."):
            analysis = analyze_with_claude(
                user_query,
                reviews,
                api_key,
                analysis_type
            )
        
        # Display results
        st.markdown("---")
        st.header("📊 Analysis Results")
        
        # Analysis output
        st.markdown(analysis)
        
        # Show source reviews
        with st.expander("📄 View Source Reviews"):
            for i, (doc, meta) in enumerate(zip(reviews['documents'][0], reviews['metadatas'][0]), 1):
                st.markdown(f"**Review {i}** | ⭐ {meta['score']} stars | 📅 {meta['date']} | 👍 {meta['thumbs_up']}")
                
                content_start = doc.find("User Feedback:")
                if content_start != -1:
                    content = doc[content_start + 15:].strip()
                    st.text(content[:300] + "..." if len(content) > 300 else content)
                
                st.markdown("---")
    
    elif analyze_button and not user_query:
        st.warning("⚠️ Please enter a question to analyze.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray; font-size: 0.9em;'>
        Built with Streamlit & Claude API | Data: 24,643 Tinder reviews (Last 12 months)
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()


 




