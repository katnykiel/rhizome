"""Module for creating embeddings and finding connections between chunks."""

import pickle
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import yaml
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class ChunkEmbedder:
    """Creates embeddings for chunks and finds related content."""
    
    def __init__(self, embeddings):
        """Initialize with langchain embeddings model."""
        self.embeddings = embeddings
        self.chunks = []
        self.chunk_embeddings = []
    
    def load_chunks(self, chunks_dir: Path) -> List[Dict]:
        """Load all chunk files from a directory."""
        self.chunks = []
        
        for chunk_file in chunks_dir.glob('*.md'):
            with open(chunk_file, 'r') as f:
                content = f.read()
            
            if content.startswith('---'):
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    metadata = yaml.safe_load(parts[1])
                    if metadata.get('type') == 'chunk':
                        self.chunks.append({
                            'file': str(chunk_file),
                            'metadata': metadata,
                            'content': parts[2].strip()
                        })
        
        return self.chunks
    
    def _load_cache(self, cache_file: Path) -> bool:
        """Try to load embeddings from cache. Returns True if successful."""
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Validate cache: same files and content
            cached_files = {c['file']: c['content'] for c in cached_data['chunks']}
            current_files = {c['file']: c['content'] for c in self.chunks}
            
            if cached_files == current_files:
                self.chunk_embeddings = cached_data['embeddings']
                return True
        except Exception:
            pass
        return False
    
    def _save_cache(self, cache_file: Path):
        """Save embeddings to cache."""
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'chunks': self.chunks,
                    'embeddings': self.chunk_embeddings
                }, f)
        except Exception:
            pass
    
    def create_embeddings(self, cache_dir: Optional[Path] = None):
        """Create embeddings for all loaded chunks with optional caching."""
        if cache_dir:
            cache_file = cache_dir / '.embeddings_cache.pkl'
            if cache_file.exists() and self._load_cache(cache_file):
                return
        
        # Generate embeddings
        texts = [chunk['content'] for chunk in self.chunks]
        self.chunk_embeddings = self.embeddings.embed_documents(texts)
        
        if cache_dir:
            self._save_cache(cache_dir / '.embeddings_cache.pkl')
    
    def find_similar_chunks(self, threshold: float = 0.7) -> List[List[int]]:
        """Find groups of semantically similar chunks. Chunks can belong to multiple groups.
        
        Creates overlapping groups where each group contains chunks that are all 
        similar to each other (forming a clique). This allows chunks to appear in 
        multiple plateaus if they connect different concepts.
        """
        if not self.chunk_embeddings:
            raise ValueError("No embeddings found. Call create_embeddings() first.")
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(np.array(self.chunk_embeddings))
        np.fill_diagonal(similarity_matrix, 0)
        
        n = len(self.chunks)
        groups = []
        
        # For each chunk, find all chunks similar to it and create a group
        for i in range(n):
            similar = [i]  # Start with the chunk itself
            
            # Find all chunks similar to chunk i
            for j in range(n):
                if i != j and similarity_matrix[i][j] > threshold:
                    similar.append(j)
            
            # Only create a group if there are at least 2 chunks
            if len(similar) >= 2:
                # Sort to make groups comparable
                similar.sort()
                # Avoid duplicate groups
                if similar not in groups:
                    groups.append(similar)
        
        return groups
    
    def _extract_dates(self, chunks: List[Dict]) -> List[Tuple[Optional[datetime], Dict]]:
        """Extract dates from chunk filenames and sort chronologically."""
        dated = []
        for chunk in chunks:
            filename = Path(chunk['file']).stem
            try:
                date = datetime.strptime(filename[:10], '%Y-%m-%d').date()
                dated.append((date, chunk))
            except (ValueError, IndexError):
                dated.append((None, chunk))
        
        dated.sort(key=lambda x: (x[0] is None, x[0]))
        return dated
    
    def _generate_llm_content(self, dated_chunks: List[Tuple], llm) -> Tuple[str, str, str, str]:
        """Generate all LLM-based content for a plateau."""
        # Prepare chunk text with backlinks
        chunks_text = "\n\n---\n\n".join([
            f"[[{Path(chunk['file']).stem}]]{f' ({date})' if date else ''}: {chunk['content'][:300]}"
            for date, chunk in dated_chunks
        ])
        
        # Generate synthesis
        summary = llm.invoke(f"""Read these related ideas and create a bulleted synthesis that shows how they connect. Each bullet should capture one key connection or concept.

{chunks_text}

Write only 3-5 bullet points using "- " format (dash space). No preamble:""").strip()
        
        # Generate title
        title = llm.invoke(f"""Read this synthesis and generate a 2-4 word title capturing the main theme.

{summary[:300]}

Write only the title. No other text, no punctuation, no explanation:""").strip()
        
        # Clean title
        title = re.sub(r'[*_`#"\',:;!?.]', '', title.split('\n')[0])
        title = re.sub(r'^(okay|here|well|the|a|an|sure)\s+', '', title, flags=re.IGNORECASE).strip()[:50]
        
        # Generate differences
        differences = llm.invoke(f"""Read these chunks and create 2-3 bullet points describing key differences in perspective or approach.

{chunks_text}

Write only bullet points using "- " format (dash space). No preamble:""").strip()
        
        # Generate temporal trends if we have dates
        trends = ""
        if any(date for date, _ in dated_chunks):
            trends = llm.invoke(f"""These chunks are dated and explore related ideas over time. Create 2-3 bullet points about how the thinking or emphasis changed:

{chunks_text}

Write only bullet points using "- " format (dash space). No preamble:""").strip()
        
        return title, summary, differences, trends
    
    def create_plateau(self, chunk_indices: List[int], plateau_id: int, output_dir: Path, llm=None) -> str:
        """Create a plateau file from a group of related chunks."""
        related_chunks = [self.chunks[i] for i in chunk_indices]
        dated_chunks = self._extract_dates(related_chunks)
        
        # Default values
        plateau_title = f"Plateau {plateau_id}"
        filename_base = f"plateau-{plateau_id:03d}"
        summary = differences = trends = ""
        
        # Generate LLM content if available
        if llm:
            try:
                plateau_title, summary, differences, trends = self._generate_llm_content(dated_chunks, llm)
                if plateau_title:
                    safe_title = re.sub(r'[^\w\s-]', '', plateau_title.lower())
                    safe_title = re.sub(r'[-\s]+', '-', safe_title)
                    filename_base = safe_title[:40] or f"plateau-{plateau_id:03d}"
            except Exception as e:
                print(f"Warning: LLM generation failed: {e}")
        
        # Write plateau file
        filepath = output_dir / f"{filename_base}.md"
        with open(filepath, 'w') as f:
            f.write(f"""---
type: plateau
plateau_id: {plateau_id}
---

# {plateau_title}

**Related chunks:** {", ".join(
    f"[[{Path(chunk['file']).stem}]]" + (f" ({date})" if date else "")
    for date, chunk in dated_chunks
)}

## Synthesis

{summary}

## Differences

{differences}
""")
            if trends:
                f.write(f"\n## How Ideas Evolved\n\n{trends}\n")
        
        return str(filepath)
    
    def process_plateaus(self, chunks_dir: Path, output_dir: Path, threshold: float = 0.7, llm=None, cache_dir: Path = None) -> List[str]:
        """Complete pipeline: load chunks, create embeddings, find connections, create plateaus.
        
        Args:
            chunks_dir: Directory containing chunk files
            output_dir: Directory to save plateau files
            threshold: Similarity threshold for grouping
            llm: Optional LLM for generating plateau summaries
            cache_dir: Directory to store embedding cache (default: same as chunks_dir parent)
            
        Returns:
            List of paths to created plateau files
        """
        # Load chunks
        self.load_chunks(chunks_dir)
        
        if not self.chunks:
            print("No chunks found!")
            return []
        
        # Use chunks_dir parent as cache directory by default
        if cache_dir is None:
            cache_dir = chunks_dir.parent
        
        # Create embeddings with caching
        self.create_embeddings(cache_dir=cache_dir)
        
        # Find similar chunks
        groups = self.find_similar_chunks(threshold)
        
        # Create plateau files
        output_dir.mkdir(parents=True, exist_ok=True)
        plateau_files = []
        
        for i, group in enumerate(groups):
            plateau_file = self.create_plateau(group, i, output_dir, llm=llm)
            plateau_files.append(plateau_file)
        
        return plateau_files