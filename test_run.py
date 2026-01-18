#!/usr/bin/env python3
"""Quick test script to verify rhizome works."""

import sys
from pathlib import Path
from rhizome.chunker import NoteChunker
from rhizome.embedder import ChunkEmbedder

try:
    from langchain_community.llms import Ollama
    from langchain_community.embeddings import OllamaEmbeddings
except ImportError:
    print("Error: langchain-community not installed")
    print("Run: source .venv/bin/activate && uv pip install -e .")
    sys.exit(1)


def main():
    """Run a quick test."""
    print("🧪 Testing Rhizome\n")
    
    # Check if test notes exist
    test_dir = Path("test_notes")
    if not test_dir.exists():
        print(f"Error: {test_dir} not found")
        return 1
    
    output_dir = Path("output")
    chunks_dir = output_dir / "chunks"
    plateaus_dir = output_dir / "plateaus"
    
    # Initialize
    print("Initializing Ollama...")
    try:
        llm = Ollama(model="ministral-3:3b")
        embeddings = OllamaEmbeddings(model="embeddinggemma")
        print("✓ Ollama initialized\n")
    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nMake sure Ollama is running:")
        print("  1. ollama serve")
        print("  2. ollama pull ministral-3:3b")
        return 1
    
    # Chunk notes
    print("📝 Chunking notes...")
    chunker = NoteChunker(llm)
    chunk_files = chunker.process_folder(test_dir, chunks_dir)
    print(f"✓ Created {len(chunk_files)} chunks\n")
    
    # Create plateaus
    print("🔗 Creating plateaus...")
    embedder = ChunkEmbedder(embeddings)
    plateau_files = embedder.process_plateaus(chunks_dir, plateaus_dir, threshold=0.65, llm=llm)
    print(f"✓ Created {len(plateau_files)} plateaus\n")
    
    print("🎉 Test complete!")
    print(f"Results in: {output_dir}/")
    print(f"  - {len(chunk_files)} chunks")
    print(f"  - {len(plateau_files)} plateaus")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
