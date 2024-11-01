from r2r import R2RClient
client = R2RClient("http://localhost:7272")
health_response = client.health()

hybrid_search_response = client.search(
    "What is a MXene?",
    vector_search_settings={
        "use_hybrid_search": True,
        "search_limit": 20,
        "hybrid_search_settings": {
            "full_text_weight": 1.0,
            "semantic_weight": 10.0,
            "full_text_limit": 200,
            "rrf_k": 25,
        },
    }
)

print(hybrid_search_response)

rag_response = client.rag("What is a MXene?")

print(rag_response)