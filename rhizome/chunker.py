"""Module for breaking down markdown notes into atomic chunks."""

import os
import re
from pathlib import Path
from typing import List, Dict
import yaml


class NoteChunker:
    """Breaks down markdown notes into atomic chunks."""
    
    def __init__(self, llm):
        """Initialize the chunker with an LLM for intelligent chunking.
        
        Args:
            llm: A langchain LLM instance for content analysis
        """
        self.llm = llm
    
    def chunk_note(self, content: str, source_file: str, start_chunk_id: int = 0) -> List[Dict[str, str]]:
        """Break a note into atomic chunks.
        
        Args:
            content: The markdown content to chunk
            source_file: The original filename for reference
            start_chunk_id: The starting chunk ID for cumulative indexing
            
        Returns:
            List of dicts with 'content' and 'metadata' keys
        """
        # Split by blank lines (paragraph breaks)
        chunks = []
        
        # Split by one or more blank lines
        paragraphs = re.split(r'\n\n+', content.strip())
        
        chunk_id = start_chunk_id
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Skip very short chunks
            if len(paragraph) < 50:
                continue
            
            # Extract content
            lines = paragraph.split('\n')
            if lines[0].startswith('#'):
                chunk_content = '\n'.join(lines[1:]).strip()
            else:
                chunk_content = paragraph
            
            # Generate concise title using LLM
            try:
                prompt = f"""Generate a 1-3 word title for this text. Output ONLY the title, nothing else.

Text:
{paragraph[:200]}

Title:"""
                title = self.llm.invoke(prompt).strip()
                # Clean up the title - remove markdown, quotes, colons, and extra text
                title = title.split('\n')[0]  # Take only first line
                title = re.sub(r'[*_`#"\',:;!?]', '', title)  # Remove markdown, quotes, and punctuation
                title = title.strip()
                # Remove common extra phrases the LLM might add
                title = re.sub(r'^(okay|here|well|the|a|an)\s+', '', title, flags=re.IGNORECASE)
                if len(title) > 30:
                    title = ' '.join(title.split()[:3])
                # Ensure title is not empty
                if not title:
                    words = paragraph.split()[:2]
                    title = ' '.join(words)
            except Exception as e:
                # Fallback to first two words
                words = paragraph.split()[:2]
                title = ' '.join(words)
            
            chunks.append({
                'title': title,
                'content': chunk_content,
                'source': source_file,
                'chunk_id': chunk_id
            })
            chunk_id += 1
        
        return chunks
    
    def save_chunk(self, chunk: Dict[str, str], output_dir: Path) -> str:
        """Save a chunk as a markdown file with YAML frontmatter.
        
        Args:
            chunk: Dict containing chunk data
            output_dir: Directory to save the chunk
            
        Returns:
            Path to the saved file
        """
        # Create a filename using the chunk_id to ensure uniqueness
        filename = f"chunk-{chunk['chunk_id']:05d}.md"
        
        filepath = output_dir / filename
        
        # Write file with frontmatter and backlinks in content
        with open(filepath, 'w') as f:
            f.write('---\n')
            f.write(f"type: chunk\n")
            f.write(f"chunk_id: {chunk['chunk_id']}\n")
            # Quote title to handle special characters in YAML
            f.write(f'title: "{chunk["title"]}"\n')
            f.write('---\n\n')
            f.write(f"# {chunk['title']}\n\n")
            f.write(f"Source: [[{chunk['source']}]]\n\n")
            f.write(chunk['content'])
        
        return str(filepath)
    
    def process_folder(self, input_dir: Path, output_dir: Path) -> List[str]:
        """Process all markdown files in a folder.
        
        Args:
            input_dir: Directory containing markdown notes
            output_dir: Directory to save chunks
            
        Returns:
            List of paths to created chunk files
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        chunk_files = []
        cumulative_chunk_id = 0
        
        for md_file in sorted(input_dir.glob('*.md')):
            with open(md_file, 'r') as f:
                content = f.read()
            
            chunks = self.chunk_note(content, md_file.name, start_chunk_id=cumulative_chunk_id)
            
            for chunk in chunks:
                chunk_file = self.save_chunk(chunk, output_dir)
                chunk_files.append(chunk_file)
                cumulative_chunk_id = chunk['chunk_id'] + 1
        
        return chunk_files
