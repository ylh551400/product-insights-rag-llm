import os
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import anthropic

class TinderRAGAnalyzer:
    """RAG-powered Product Analyst assistant - Last 12 months data"""
    
    def __init__(self, api_key=None):
        print("Loading embedding model...")
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("Connecting to vector database (recent data)...")
        chroma_client = chromadb.Client(Settings(
            persist_directory="./tinder_rag_db_recent",
            anonymized_telemetry=False
        ))
        self.collection = chroma_client.get_collection("tinder_reviews_recent")
        
        print("Initializing Claude API...")
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        
        print(f"\n✅ System ready! Knowledge base: {self.collection.count():,} reviews\n")
    
    def retrieve_reviews(self, query, n_results=10, filters=None):
        """Retrieve relevant reviews"""
        query_embedding = self.embed_model.encode([query])
        
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
            where=filters
        )
        
        return results
    
    def analyze_with_claude(self, query, context_reviews, analysis_type="general"):
        """Use Claude to analyze reviews"""
        context_text = "\n\n".join([
            f"Review {i+1}:\n{doc}"
            for i, doc in enumerate(context_reviews['documents'][0])
        ])
        
        prompt = f"""You are a senior Product Analyst analyzing RECENT user feedback (last 12 months).

User Reviews:
{context_text}

Question: {query}

Provide:
1. Direct answer with specific evidence from reviews
2. Key patterns and trends identified
3. Actionable product recommendations with priority
4. Timeline context (mention specific months if relevant)

Focus on RECENT issues and trends. Be concise and data-driven."""
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def ask(self, query, n_reviews=10, filters=None):
        """Ask a question and get AI-powered analysis"""
        print(f"🔍 Question: {query}")
        print(f"   Retrieving {n_reviews} relevant reviews...")
        
        reviews = self.retrieve_reviews(query, n_results=n_reviews, filters=filters)
        
        if not reviews['documents'][0]:
            return "No relevant reviews found."
        
        print(f"   ✅ Retrieved {len(reviews['documents'][0])} reviews")
        print(f"   🤖 Analyzing with Claude...\n")
        
        analysis = self.analyze_with_claude(query, reviews)
        
        return analysis
