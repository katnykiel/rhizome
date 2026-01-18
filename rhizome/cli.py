"""Command-line interface for rhizome."""

import argparse
from pathlib import Path
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from rhizome.chunker import NoteChunker
from rhizome.embedder import ChunkEmbedder


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Rhizome: Atomize and connect your notes"
    )
    parser.add_argument(
        'input_dir',
        type=Path,
        help='Directory containing markdown notes'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('output'),
        help='Directory for output files (default: output)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.7,
        help='Similarity threshold for creating plateaus (default: 0.7)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='llama3.1',
        help='Ollama model to use (default: llama3.1)'
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.input_dir.exists():
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return 1
    
    # Setup directories
    chunks_dir = args.output_dir / 'chunks'
    plateaus_dir = args.output_dir / 'plateaus'
    
    print(f"🌱 Rhizome: Processing notes from {args.input_dir}")
    print(f"   Using model: {args.model}")
    print(f"   Output directory: {args.output_dir}")
    print()
    
    # Initialize LLM and embeddings
    try:
        llm = Ollama(model=args.model)
        embeddings = OllamaEmbeddings(model=args.model)
    except Exception as e:
        print(f"Error initializing Ollama: {e}")
        print("Make sure Ollama is running with: ollama serve")
        return 1
    
    # Step 1: Chunk notes
    print("📝 Step 1: Chunking notes...")
    chunker = NoteChunker(llm)
    chunk_files = chunker.process_folder(args.input_dir, chunks_dir)
    print(f"✓ Created {len(chunk_files)} chunks\n")
    
    # Step 2: Create embeddings and plateaus
    print("🔗 Step 2: Creating embeddings and finding connections...")
    embedder = ChunkEmbedder(embeddings)
    plateau_files = embedder.process_plateaus(chunks_dir, plateaus_dir, args.threshold, llm=llm)
    print(f"✓ Created {len(plateau_files)} plateaus\n")
    
    print(f"🎉 Done! Check {args.output_dir} for results")
    print(f"   Chunks: {chunks_dir}")
    print(f"   Plateaus: {plateaus_dir}")
    
    return 0


if __name__ == '__main__':
    exit(main())
