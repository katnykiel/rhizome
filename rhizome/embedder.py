"""Module for creating embeddings and finding connections between chunks."""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
import yaml
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class ChunkEmbedder:
    """Creates embeddings for chunks and finds related content."""
    
    def __init__(self, embeddings):
        """Initialize with langchain embeddings model.
        
        Args:
            embeddings: A langchain embeddings instance
        """
        self.embeddings = embeddings
        self.chunks = []
        self.chunk_embeddings = []
    
    def load_chunks(self, chunks_dir: Path) -> List[Dict]:
        """Load all chunk files from a directory.
        
        Args:
            chunks_dir: Directory containing chunk markdown files
            
        Returns:
            List of chunk data dicts
        """
        self.chunks = []
        
        for chunk_file in chunks_dir.glob('*.md'):
            with open(chunk_file, 'r') as f:
                content = f.read()
            
            # Extract YAML frontmatter
            if content.startswith('---'):
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    metadata = yaml.safe_load(parts[1])
                    text_content = parts[2].strip()
                    
                    # Only load chunks, not plateaus
                    if metadata.get('type') == 'chunk':
                        self.chunks.append({
                            'file': str(chunk_file),
                            'metadata': metadata,
                            'content': text_content
                        })
        
        return self.chunks
    
    def create_embeddings(self):
        """Create embeddings for all loaded chunks."""
        texts = [chunk['content'] for chunk in self.chunks]
        self.chunk_embeddings = self.embeddings.embed_documents(texts)
    
    def find_similar_chunks(self, threshold: float = 0.7) -> List[List[int]]:
        """Find groups of semantically similar chunks using cosine similarity.
        
        Uses connected components algorithm to group chunks where each chunk
        in a group is similar to at least one other chunk in the group.
        Only returns groups with 2 or more chunks.
        
        Args:
            threshold: Similarity threshold (0-1) for considering chunks as similar
            
        Returns:
            List of lists, where each inner list contains indices of similar chunks (min 2 per group)
        """
        if not self.chunk_embeddings:
            raise ValueError("No embeddings found. Call create_embeddings() first.")
        
        # Calculate similarity matrix using cosine similarity
        embeddings_array = np.array(self.chunk_embeddings)
        similarity_matrix = cosine_similarity(embeddings_array)
        
        # Set diagonal to 0 to avoid self-loops
        np.fill_diagonal(similarity_matrix, 0)
        
        # Build adjacency list for connected components
        # Two chunks are connected if their similarity exceeds the threshold
        n = len(self.chunks)
        adj = [[] for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i][j] > threshold:
                    adj[i].append(j)
                    adj[j].append(i)
        
        # Find connected components using DFS
        visited = [False] * n
        groups = []
        
        def dfs(node, component):
            visited[node] = True
            component.append(node)
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    dfs(neighbor, component)
        
        # Process all nodes and collect groups with 2+ chunks
        for i in range(n):
            if not visited[i] and adj[i]:  # Only process unvisited nodes with connections
                component = []
                dfs(i, component)
                if len(component) >= 2:  # Only keep groups with 2 or more chunks
                    groups.append(component)
        
        return groups
    
    def create_plateau(self, chunk_indices: List[int], plateau_id: int, output_dir: Path, llm=None) -> str:
        """Create a plateau file from a group of related chunks.
        
        Args:
            chunk_indices: Indices of chunks to combine
            plateau_id: Unique ID for this plateau
            output_dir: Directory to save the plateau
            llm: Optional LLM for generating summaries
            
        Returns:
            Path to the created plateau file
        """
        # Gather content from all chunks in the group
        related_chunks = [self.chunks[i] for i in chunk_indices]
        
        # Create a title for the plateau (will be updated after summary generation)
        plateau_title = f"Plateau {plateau_id}"
        filename_base = f"plateau-{plateau_id:03d}"
        
        # Create backlinks list using file names instead of titles
        related_chunks_backlinks = [Path(chunk['file']).stem for chunk in related_chunks]
        
        # Generate combined summary using LLM if available
        combined_summary = ""
        agreements_differences = ""
        
        if llm:
            try:
                # Prepare chunks content for LLM
                chunks_text = "\n\n---\n\n".join([f"Chunk {i+1} ({chunk['metadata'].get('title', 'Untitled')}): {chunk['content'][:300]}" 
                                                    for i, chunk in enumerate(related_chunks)])
                
                # Generate combined summary
                summary_prompt = f"""Synthesize these related ideas into a single, coherent summary that reduces redundancy:

{chunks_text}

Provide a concise combined summary:"""
                combined_summary = llm.invoke(summary_prompt).strip()
                
                # Generate title from summary
                title_prompt = f"""Generate a 2-4 word title that captures the main theme. Output ONLY the title, nothing else.

Synthesis:
{combined_summary[:200]}

Title:"""
                plateau_title = llm.invoke(title_prompt).strip()
                # Clean title - remove markdown, punctuation, and extra phrases
                plateau_title = plateau_title.split('\n')[0]  # Take only first line
                plateau_title = re.sub(r'[*_`#"\',:;!?.]', '', plateau_title)  # Remove punctuation
                plateau_title = plateau_title.strip()
                # Remove common extra phrases
                plateau_title = re.sub(r'^(okay|here|well|the|a|an)\s+', '', plateau_title, flags=re.IGNORECASE)
                plateau_title = plateau_title[:50].strip()
                # Ensure title is not empty
                if not plateau_title:
                    plateau_title = f"plateau-{plateau_id:03d}"
                
                # Update filename
                safe_title = re.sub(r'[^\w\s-]', '', plateau_title.lower())
                safe_title = re.sub(r'[-\s]+', '-', safe_title)
                filename_base = safe_title[:40] if safe_title else f"plateau-{plateau_id:03d}"
                
                # Generate comparison
                comparison_prompt = f"""Analyze these related chunks and describe:
1. What they agree on (common themes)
2. How they differ (unique perspectives or details)

{chunks_text}

Provide your analysis:"""
                agreements_differences = llm.invoke(comparison_prompt).strip()
            except Exception as e:
                print(f"Warning: LLM generation failed: {e}")
                combined_summary = "Failed to generate summary."
                agreements_differences = "Failed to generate comparison."
                filename_base = f"plateau-{plateau_id:03d}"
        
        # Create final filepath
        filename = f"{filename_base}.md"
        filepath = output_dir / filename
        
        # Write plateau file
        with open(filepath, 'w') as f:
            f.write('---\n')
            f.write('type: plateau\n')
            f.write(f'plateau_id: {plateau_id}\n')
            f.write('---\n\n')
            f.write(f"# {plateau_title}\n\n")
            
            # Backlinks section
            f.write("**Related chunks:** ")
            f.write(", ".join([f"[[{name}]]" for name in related_chunks_backlinks]))
            f.write("\n\n")
            
            # Combined summary
            f.write("## Synthesis\n\n")
            if combined_summary:
                f.write(combined_summary)
            f.write("\n\n")
            
            # Agreements and differences
            f.write("## Differences\n\n")
            if agreements_differences:
                f.write(agreements_differences)
            f.write("\n")
        
        return str(filepath)
    
    def process_plateaus(self, chunks_dir: Path, output_dir: Path, threshold: float = 0.7, llm=None) -> List[str]:
        """Complete pipeline: load chunks, create embeddings, find connections, create plateaus.
        
        Args:
            chunks_dir: Directory containing chunk files
            output_dir: Directory to save plateau files
            threshold: Similarity threshold for grouping
            llm: Optional LLM for generating plateau summaries
            
        Returns:
            List of paths to created plateau files
        """
        # Load chunks
        self.load_chunks(chunks_dir)
        
        if not self.chunks:
            print("No chunks found!")
            return []
        
        # Create embeddings
        self.create_embeddings()
        
        # Find similar chunks
        groups = self.find_similar_chunks(threshold)
        
        # Create plateau files
        output_dir.mkdir(parents=True, exist_ok=True)
        plateau_files = []
        
        for i, group in enumerate(groups):
            plateau_file = self.create_plateau(group, i, output_dir, llm=llm)
            plateau_files.append(plateau_file)
        
        return plateau_files
