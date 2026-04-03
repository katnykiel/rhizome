"""Module for creating embeddings and finding connections between chunks."""

import pickle
import re
import hashlib
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
    
    def _compute_chunk_signature(self, chunk: Dict) -> str:
        """Compute a stable signature for chunk content to detect additions/updates."""
        content_bytes = (chunk.get('content') or '').encode('utf-8')
        return hashlib.md5(content_bytes).hexdigest()

    def _load_cache(self, cache_file: Path) -> Optional[Dict]:
        """Load cache with embeddings and chunk metadata. Returns None if invalid."""
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)

            if not isinstance(cached_data, dict):
                return None

            chunks = cached_data.get('chunks')
            embeddings = cached_data.get('embeddings')
            if chunks is None or embeddings is None:
                return None

            signatures = {}
            for c, emb in zip(chunks, embeddings):
                file_ = c.get('file')
                if not file_ or 'content' not in c:
                    continue
                signatures[file_] = c.get('signature') or hashlib.md5(c['content'].encode('utf-8')).hexdigest()

            return {
                'chunks': chunks,
                'embeddings': embeddings,
                'signatures': signatures
            }
        except Exception:
            return None

    def _save_cache(self, cache_file: Path):
        """Save embeddings and chunk data with signatures."""
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            serializable_chunks = []
            for chunk in self.chunks:
                chunk_copy = chunk.copy()
                chunk_copy['signature'] = self._compute_chunk_signature(chunk)
                serializable_chunks.append(chunk_copy)

            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'chunks': serializable_chunks,
                    'embeddings': self.chunk_embeddings
                }, f)
        except Exception:
            pass

    def create_embeddings(self, cache_dir: Optional[Path] = None):
        """Create or incrementally update embeddings for loaded chunks."""
        if cache_dir is not None:
            cache_file = cache_dir / '.embeddings_cache.pkl'
            cached = self._load_cache(cache_file) if cache_file.exists() else None

            if cached:
                cached_signatures = cached['signatures']
                cached_embeddings_by_file = {
                    c['file']: emb for c, emb in zip(cached['chunks'], cached['embeddings'])
                    if c.get('file') in cached_signatures
                }

                # Determine chunk embedding state by current signatures
                current_signatures = {
                    chunk['file']: self._compute_chunk_signature(chunk)
                    for chunk in self.chunks
                }

                merged_embeddings = []
                new_texts = []
                new_indexes = []

                for idx, chunk in enumerate(self.chunks):
                    path = chunk['file']
                    sig = current_signatures[path]
                    if path in cached_signatures and cached_signatures[path] == sig:
                        merged_embeddings.append(cached_embeddings_by_file[path])
                    else:
                        merged_embeddings.append(None)
                        new_texts.append(chunk['content'])
                        new_indexes.append(idx)

                # Generate only missing or changed embeddings
                if new_texts:
                    new_embs = self.embeddings.embed_documents(new_texts)
                    for idx, emb in zip(new_indexes, new_embs):
                        merged_embeddings[idx] = emb

                # If any cached chunk was removed, we keep only existing ones by current chunks order
                self.chunk_embeddings = merged_embeddings
                self._save_cache(cache_file)
                return

        # Fallback: full embedding refresh
        texts = [chunk['content'] for chunk in self.chunks]
        self.chunk_embeddings = self.embeddings.embed_documents(texts)

        if cache_dir is not None:
            self._save_cache(cache_dir / '.embeddings_cache.pkl')
    
    def find_similar_chunks(self, threshold: float = 0.7, min_plateau_distance: float = 0.3) -> List[List[int]]:
        """Find groups of semantically similar chunks with diversity enforcement.
        
        Creates overlapping groups where each group contains chunks that are all 
        similar to each other. Ensures plateaus are sufficiently different from 
        each other by enforcing minimum centroid distance in embedding space.
        
        Args:
            threshold: Similarity threshold for grouping chunks (0-1)
            min_plateau_distance: Minimum distance between plateau centroids (0-1).
                                 Higher = more diverse plateaus. Recommended: 0.2-0.4
        
        Returns:
            List of chunk index lists, where plateaus are semantically diverse
        """
        if not self.chunk_embeddings:
            raise ValueError("No embeddings found. Call create_embeddings() first.")
        
        # Calculate similarity matrix
        embeddings_array = np.array(self.chunk_embeddings)
        similarity_matrix = cosine_similarity(embeddings_array)
        np.fill_diagonal(similarity_matrix, 0)
        
        n = len(self.chunks)
        candidate_groups = []
        
        # Generate candidate groups
        for i in range(n):
            similar = [i]
            
            for j in range(n):
                if i != j and similarity_matrix[i][j] > threshold:
                    similar.append(j)
            
            if len(similar) >= 2:
                similar.sort()
                if similar not in candidate_groups:
                    candidate_groups.append(similar)
        
        # Filter groups by centroid distance to ensure plateau diversity
        diverse_groups = []
        plateau_centroids = []
        
        for group in candidate_groups:
            # Calculate centroid of this plateau
            group_embeddings = embeddings_array[group]
            centroid = np.mean(group_embeddings, axis=0)
            
            # Check if this plateau is sufficiently different from existing ones
            is_diverse = True
            for existing_centroid in plateau_centroids:
                # Calculate cosine distance (1 - similarity)
                distance = 1 - cosine_similarity([centroid], [existing_centroid])[0][0]
                
                if distance < min_plateau_distance:
                    is_diverse = False
                    break
            
            # Only add if sufficiently different from all existing plateaus
            if is_diverse:
                diverse_groups.append(group)
                plateau_centroids.append(centroid)
        
        return diverse_groups

    def load_plateaus(self, plateaus_dir: Path) -> List[Dict]:
        """Load existing plateau metadata and chunk associations from disk."""
        plateaus = []
        for plateau_file in sorted(plateaus_dir.glob('*.md')):
            try:
                text = plateau_file.read_text(encoding='utf-8')
            except Exception:
                continue

            metadata = {}
            if text.startswith('---'):
                parts = text.split('---', 3)
                if len(parts) >= 3:
                    metadata = yaml.safe_load(parts[1]) or {}

            related_chunks = []
            match = re.search(r'Related chunks:\s*(.+)', text)
            if match:
                related_chunks = re.findall(r'\[\[([^\]]+)\]\]', match.group(1))

            plateaus.append({
                'path': plateau_file,
                'plateau_id': metadata.get('plateau_id'),
                'title': metadata.get('title'),
                'chunk_files': related_chunks
            })
        return plateaus

    def _find_similar_groups_for_indices(self, indices: List[int], threshold: float, min_plateau_distance: float) -> List[List[int]]:
        """Find related groups among a subset of chunk indices."""
        if not indices:
            return []

        embeddings_array = np.array([self.chunk_embeddings[i] for i in indices])
        sim_matrix = cosine_similarity(embeddings_array)
        np.fill_diagonal(sim_matrix, 0)

        candidate_groups = []
        for local_i, global_i in enumerate(indices):
            group = [global_i]
            for local_j, global_j in enumerate(indices):
                if local_i != local_j and sim_matrix[local_i][local_j] > threshold:
                    group.append(global_j)
            if len(group) >= 2:
                group.sort()
                if group not in candidate_groups:
                    candidate_groups.append(group)

        diverse_groups = []
        plateau_centroids = []

        for group in candidate_groups:
            group_embeddings = np.array([self.chunk_embeddings[i] for i in group])
            centroid = np.mean(group_embeddings, axis=0)
            is_diverse = True

            for existing_centroid in plateau_centroids:
                distance = 1 - cosine_similarity([centroid], [existing_centroid])[0][0]
                if distance < min_plateau_distance:
                    is_diverse = False
                    break

            if is_diverse:
                diverse_groups.append(group)
                plateau_centroids.append(centroid)

        return diverse_groups

    def update_plateaus(self, plateaus_dir: Path, threshold: float = 0.7, min_plateau_distance: float = 0.3, llm=None) -> List[str]:
        """Incrementally update or create plateaus with newly added chunks."""
        plateaus_dir.mkdir(parents=True, exist_ok=True)

        existing_plateaus = self.load_plateaus(plateaus_dir)
        existing_groups = []
        existing_files = set()

        # Map file stems to chunk indices
        stem_to_index = {Path(chunk['file']).stem: i for i, chunk in enumerate(self.chunks)}

        for plateau in existing_plateaus:
            group_indices = [stem_to_index.get(stem) for stem in plateau['chunk_files'] if stem_to_index.get(stem) is not None]
            if group_indices:
                existing_groups.append(sorted(set(group_indices)))
                existing_files.update(group_indices)

        # Identify newly added chunks (or changed ones will be included here)
        new_indices = [i for i in range(len(self.chunks)) if i not in existing_files]

        # Connect new chunks to existing plateaus
        for idx in new_indices:
            best_plateau = None
            best_similarity = threshold
            for p_idx, group in enumerate(existing_groups):
                sim = max(cosine_similarity([self.chunk_embeddings[idx]], [self.chunk_embeddings[j]])[0][0] for j in group)
                if sim > best_similarity:
                    best_similarity = sim
                    best_plateau = p_idx

            if best_plateau is not None:
                existing_groups[best_plateau].append(idx)
                existing_groups[best_plateau] = sorted(set(existing_groups[best_plateau]))

        # New-to-new plateau grouping for any still-unassigned chunks
        assigned = {i for group in existing_groups for i in group}
        remaining = [i for i in new_indices if i not in assigned]
        new_group_sets = self._find_similar_groups_for_indices(remaining, threshold, min_plateau_distance)

        if new_group_sets:
            existing_groups.extend(new_group_sets)

        # Create or rewrite plateau files
        for file in plateaus_dir.glob('*.md'):
            try:
                file.unlink()
            except Exception:
                pass

        plateau_paths = []
        existing_plateau_ids = [p.get('plateau_id', -1) for p in existing_plateaus if isinstance(p.get('plateau_id'), int)]
        next_plateau_id = max(existing_plateau_ids, default=-1) + 1

        for group in existing_groups:
            plateau_paths.append(self.create_plateau(group, next_plateau_id, plateaus_dir, llm=llm))
            next_plateau_id += 1

        return plateau_paths

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
    
    def process_plateaus(self, chunks_dir: Path, output_dir: Path, threshold: float = 0.7, 
                        min_plateau_distance: float = 0.3, llm=None, cache_dir: Path = None) -> List[str]:
        """Complete pipeline: load chunks, create embeddings, find connections, create plateaus.
        
        Args:
            chunks_dir: Directory containing chunk files
            output_dir: Directory to save plateau files
            threshold: Similarity threshold for grouping
            min_plateau_distance: Minimum distance between plateau centroids (0.2-0.5 recommended)
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
        
        # Create or update plateaus
        output_dir.mkdir(parents=True, exist_ok=True)
        existing_plateaus = list(output_dir.glob('*.md'))

        if existing_plateaus:
            plateau_files = self.update_plateaus(output_dir, threshold=threshold,
                                                 min_plateau_distance=min_plateau_distance,
                                                 llm=llm)
        else:
            groups = self.find_similar_chunks(threshold, min_plateau_distance)
            plateau_files = []
            for i, group in enumerate(groups):
                plateau_file = self.create_plateau(group, i, output_dir, llm=llm)
                plateau_files.append(plateau_file)

        return plateau_files