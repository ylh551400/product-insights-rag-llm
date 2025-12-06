# build_rag_system_recent.py
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
from datetime import datetime, timedelta

print("=" * 60)
print("Building Tinder Reviews RAG System (Last 12 Months Only)")
print("=" * 60)

# Load cleaned data
print("\n[Step 1/5] Loading cleaned data...")
df = pd.read_csv('tinder_reviews_clean.csv')
df['at'] = pd.to_datetime(df['at'])

print(f"   Total reviews in dataset: {len(df):,}")

# Filter to last 12 months
cutoff_date = datetime.now() - timedelta(days=365)
df = df[df['at'] >= cutoff_date].copy()

print(f"   ✅ Filtered to last 12 months: {len(df):,} reviews")
print(f"   Date range: {df['at'].min().date()} to {df['at'].max().date()}")
print(f"   Avg rating: {df['score'].mean():.2f}⭐")

# Sample for faster testing (optional)
USE_SAMPLE = False  # Set to True if you want to test with smaller dataset
SAMPLE_SIZE = 10000

if USE_SAMPLE and len(df) > SAMPLE_SIZE:
    print(f"\n   ⚠️  Using sample of {SAMPLE_SIZE:,} reviews for faster testing")
    df = df.groupby('score', group_keys=False).apply(
        lambda x: x.sample(min(len(x), SAMPLE_SIZE // 5))
    ).sample(min(SAMPLE_SIZE, len(df)))
    print(f"   Sampled {len(df):,} reviews")

# Prepare documents
print("\n[Step 2/5] Preparing documents...")

documents = []
metadatas = []
ids = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing reviews"):
    doc_parts = [
        f"Review Date: {row['at'].strftime('%Y-%m-%d')}",
        f"Rating: {int(row['score'])} stars",
        f"App Version: {row['reviewCreatedVersion']}",
        f"Thumbs Up: {int(row['thumbsUpCount'])}",
    ]
    
    if pd.notna(row['replyContent']):
        doc_parts.append("Official Reply: Yes")
    
    doc_parts.append(f"\nUser Feedback:\n{row['content']}")
    
    doc_text = "\n".join(doc_parts)
    documents.append(doc_text)
    
    metadatas.append({
        'date': row['at'].strftime('%Y-%m-%d'),
        'year': int(row['at'].year),
        'month': row['at'].strftime('%Y-%m'),
        'score': int(row['score']),
        'version': str(row['reviewCreatedVersion']),
        'thumbs_up': int(row['thumbsUpCount']),
        'has_reply': bool(pd.notna(row['replyContent'])),
        'is_negative': bool(row['score'] <= 2),
        'is_positive': bool(row['score'] >= 4)
    })
    
    ids.append(f"review_{row['reviewId']}")

print(f"✅ Prepared {len(documents):,} documents")

# Generate embeddings
print("\n[Step 3/5] Generating embeddings...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = embed_model.encode(
    documents,
    show_progress_bar=True,
    batch_size=64,
    convert_to_numpy=True
)

print(f"✅ Generated embeddings: {embeddings.shape}")

# Create vector database
print("\n[Step 4/5] Creating Chroma vector database...")

chroma_client = chromadb.Client(Settings(
    persist_directory="./tinder_rag_db_recent",  # Different directory
    anonymized_telemetry=False
))

try:
    chroma_client.delete_collection("tinder_reviews_recent")
    print("   Deleted old database")
except:
    pass

collection = chroma_client.create_collection(
    name="tinder_reviews_recent",
    metadata={
        "description": "Tinder reviews - Last 12 months only",
        "total_reviews": len(documents),
        "date_created": pd.Timestamp.now().strftime('%Y-%m-%d'),
        "date_range_start": df['at'].min().strftime('%Y-%m-%d'),
        "date_range_end": df['at'].max().strftime('%Y-%m-%d')
    }
)

print("✅ Created new collection")

# Store in database
print("\n[Step 5/5] Storing vectors in database...")

batch_size = 1000
for i in tqdm(range(0, len(documents), batch_size), desc="Inserting batches"):
    end_idx = min(i + batch_size, len(documents))
    
    collection.add(
        embeddings=embeddings[i:end_idx].tolist(),
        documents=documents[i:end_idx],
        metadatas=metadatas[i:end_idx],
        ids=ids[i:end_idx]
    )

print("\n" + "=" * 60)
print("✅ RAG System Build Complete (Last 12 Months)")
print("=" * 60)
print(f"\n📚 Knowledge Base Stats:")
print(f"   Total reviews: {collection.count():,}")
print(f"   Date range: {df['at'].min().date()} to {df['at'].max().date()}")
print(f"   Time span: Last 12 months")
print(f"   Avg rating: {df['score'].mean():.2f}⭐")
print(f"   Versions covered: {df['reviewCreatedVersion'].nunique()}")
print(f"   Negative reviews: {(df['score'] <= 2).sum():,} ({(df['score'] <= 2).mean()*100:.1f}%)")
print(f"   Positive reviews: {(df['score'] >= 4).sum():,} ({(df['score'] >= 4).mean()*100:.1f}%)")

# Quick test
print("\n" + "=" * 60)
print("🔍 Testing Retrieval (Sample Queries)")
print("=" * 60)

test_queries = [
    "recent matching problems",
    "latest pricing complaints",
    "new feature requests"
]

for query in test_queries:
    print(f"\n📝 Query: '{query}'")
    print("-" * 40)
    
    query_embedding = embed_model.encode([query])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=2
    )
    
    for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
        print(f"\nResult {i}:")
        print(f"  Rating: {meta['score']}⭐ | Date: {meta['date']} | Version: {meta['version']}")
        
        content_start = doc.find("User Feedback:")
        if content_start != -1:
            content = doc[content_start + 15:content_start + 120]
            print(f"  Preview: {content.strip()}...")

print("\n" + "=" * 60)
print("🎉 System ready for recent data analysis!")
print("=" * 60)
