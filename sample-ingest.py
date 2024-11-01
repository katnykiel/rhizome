from r2r import R2RClient
client = R2RClient("http://localhost:7272")
health_response = client.health()
# {"status":"ok"}
file_paths = ["/Users/kat/Library/Mobile Documents/com~apple~CloudDocs/grave/papers/nykielHighThroughputDensityFunctional2023.pdf"]

# Ingestion configuration for `R2R Full`
ingest_response = client.ingest_files(
    file_paths=file_paths,
    # Runtime chunking configuration
    ingestion_config={
        "provider": "unstructured_local",  # Local processing
        "strategy": "auto",                # Automatic processing strategy
        "chunking_strategy": "by_title",   # Split on title boundaries
        "new_after_n_chars": 256,          # Start new chunk (soft limit)
        "max_characters": 512,             # Maximum chunk size (hard limit)
        "combine_under_n_chars": 64,       # Minimum chunk size
        "overlap": 100,                    # Character overlap between chunks
        "chunk_enrichment_settings": {     # Document enrichment settings
            "enable_chunk_enrichment": False,
        }
    }
)
