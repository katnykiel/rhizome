from r2r import R2RClient

client = R2RClient("http://localhost:7272")

# Extract entities and relationships
document_id = ""
extract_response = client.documents.extract(document_id)

# View extracted knowledge
entities = client.documents.list_entities(document_id)
relationships = client.documents.list_relationships(document_id)

pass