"""Command-line interface for rhizome."""

import sys
import argparse
from pathlib import Path

try:
    from langchain_ollama import OllamaLLM, OllamaEmbeddings
except ImportError:
    print("Error: langchain-ollama not installed")
    print("Run: pip install -U langchain-ollama")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
except ImportError:
    print("Error: rich not installed")
    print("Run: pip install rich")
    sys.exit(1)

from rhizome.chunker import NoteChunker
from rhizome.embedder import ChunkEmbedder

console = Console()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Rhizome: Atomize and connect your notes",
        prog="rhizome"
    )
    parser.add_argument(
        '-i', '--input',
        type=Path,
        required=True,
        help='Directory containing markdown notes'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=None,
        help='Output directory for chunks and plateaus (default: input directory)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.65,
        help='Similarity threshold for creating plateaus (default: 0.65)'
    )
    parser.add_argument(
        '--min-plateau-distance',
        type=float,
        default=0.3,
        help='Minimum distance between plateau centroids for diversity (default: 0.3, range: 0.2-0.5)'
    )
    parser.add_argument(
        '--llm-model',
        type=str,
        default='deepseek-r1:8b',
        help='Ollama model to use for text generation (default: deepseek-r1:8b)'
    )
    parser.add_argument(
        '--embedding-model',
        type=str,
        default='embeddinggemma',
        help='Ollama model to use for embeddings (default: embeddinggemma)'
    )
    
    args = parser.parse_args()
    
    # Resolve input directory
    input_dir = args.input.resolve()
    
    # Validate input
    if not input_dir.exists():
        console.print(f"[red]Error: Input directory '{input_dir}' does not exist[/red]")
        return 1
    
    if not input_dir.is_dir():
        console.print(f"[red]Error: '{input_dir}' is not a directory[/red]")
        return 1
    
    # Determine output directory
    if args.output:
        output_dir = args.output.resolve()
    else:
        output_dir = input_dir
    
    # Setup chunk and plateau directories
    chunks_dir = output_dir / 'chunks'
    plateaus_dir = output_dir / 'plateaus'
    
    console.print("[bold]Rhizome[/bold]", justify="center")
    console.print()
    
    # Initialize LLM and embeddings
    with console.status("[cyan]Initializing Ollama...[/cyan]"):
        try:
            llm = OllamaLLM(model=args.llm_model)
            embeddings = OllamaEmbeddings(model=args.embedding_model)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            console.print("[yellow]Make sure Ollama is running:[/yellow]")
            console.print("  ollama serve")
            return 1
    
    # Chunk notes (skip if chunks directory already exists with files)
    chunk_files = []
    if chunks_dir.exists() and list(chunks_dir.glob('*.md')):
        chunk_files = list(chunks_dir.glob('*.md'))
        console.print(f"[yellow]Using existing {len(chunk_files)} chunks[/yellow]")
    else:
        chunker = NoteChunker(llm)
        with Progress(
            SpinnerColumn(),
            TextColumn("[cyan]Chunking notes..."),
            console=console,
        ) as progress:
            progress.add_task("chunk", total=None)
            chunk_files = chunker.process_folder(input_dir, chunks_dir)
        
        console.print(f"[green]Created {len(chunk_files)} chunks[/green]")
    
    # Create plateaus (skip if plateaus directory already exists with files)
    plateau_files = []
    if plateaus_dir.exists() and list(plateaus_dir.glob('*.md')):
        plateau_files = list(plateaus_dir.glob('*.md'))
        console.print(f"[yellow]Using existing {len(plateau_files)} plateaus[/yellow]")
    else:
        embedder = ChunkEmbedder(embeddings)
        with Progress(
            SpinnerColumn(),
            TextColumn("[cyan]Creating plateaus..."),
            console=console,
        ) as progress:
            progress.add_task("plateau", total=None)
            # Cache embeddings in the output directory
            plateau_files = embedder.process_plateaus(
                chunks_dir, 
                plateaus_dir, 
                threshold=args.threshold,
                min_plateau_distance=args.min_plateau_distance,
                llm=llm, 
                cache_dir=output_dir
            )
        
        console.print(f"[green]Created {len(plateau_files)} plateaus[/green]")
    
    console.print()
    console.print("[bold cyan]Done![/bold cyan]")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
