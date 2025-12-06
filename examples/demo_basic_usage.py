from src.rag_with_claude import TinderRAGAnalyzer

# Create analyzer (API key from environment variable)
analyzer = TinderRAGAnalyzer()

question = "What are the biggest complaints in the last 12 months?"

result = analyzer.ask(
    query=question,
    n_reviews=15,
    filters={'is_negative': True}
)

print("=" * 60)
print("Analysis Result:")
print("=" * 60)
print(result)
