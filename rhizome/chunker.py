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
    
    def chunk_note(self, content: str, source_file: str) -> List[Dict[str, str]]:
        """Break a note into atomic chunks.
        
        Args:
            content: The markdown content to chunk
            source_file: The original filename for reference
            
        Returns:
            List of dicts with 'content' and 'metadata' keys
        """
        # Split by blank lines (paragraph breaks)
        chunks = []
        
        # Split by one or more blank lines
        paragraphs = re.split(r'\n\n+', content.strip())
        
        chunk_id = 0
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
                prompt = f"Generate a concise 1-3 word title for this text:\n\n{paragraph[:200]}\n\nTitle:"
                title = self.llm.invoke(prompt).strip()
                # Clean up the title - remove markdown, quotes, and formatting
                title = title.split('\n')[0]  # Take only first line
                title = re.sub(r'[*_`#"\']', '', title)  # Remove markdown and quotes
                title = title.strip()
                if len(title) > 30:
                    title = ' '.join(title.split()[:3])
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
        # Create a filename from the title
        filename = re.sub(r'[^\w\s-]', '', chunk['title'].lower())
        filename = re.sub(r'[-\s]+', '-', filename)
        filename = f"{filename}-{chunk['chunk_id']}.md"
        
        filepath = output_dir / filename
        
        # Write file with frontmatter and backlinks in content
        with open(filepath, 'w') as f:
            f.write('---\n')
            f.write(f"type: chunk\n")
            f.write(f"chunk_id: {chunk['chunk_id']}\n")
            f.write(f"title: {chunk['title']}\n")
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
        
        for md_file in input_dir.glob('*.md'):
            print(f"Processing {md_file.name}...")
            
            with open(md_file, 'r') as f:
                content = f.read()
            
            chunks = self.chunk_note(content, md_file.name)
            
            for chunk in chunks:
                chunk_file = self.save_chunk(chunk, output_dir)
                chunk_files.append(chunk_file)
                print(f"  Created chunk: {Path(chunk_file).name}")
        
        return chunk_files
