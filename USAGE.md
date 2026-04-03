# Rhizome Usage Guide

## Installation

Install from PyPI:

```bash
pip install rhizome-cli
```

The `rhizome` command will be available in your environment.

## Prerequisites

Rhizome uses two distinct Ollama models:

- **LLM Model** (for synthesizing and generating insights): `deepseek-r1:8b`
- **Embedding Model** (for semantic similarity): `embeddinggemma`

Setup steps:

1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai)
2. **Pull both models**:
   ```bash
   ollama pull deepseek-r1:8b
   ollama pull embeddinggemma
   ```
3. **Start Ollama**: Run `ollama serve` in a terminal (keep it running while using rhizome)

## Basic Usage

Process a folder of markdown notes:

```bash
rhizome -i your_notes_folder
```

This will:
1. Read all markdown files in the folder
2. Break them into atomic chunks (ideas)
3. Generate embeddings for similarity analysis
4. Find and synthesize related chunks into "plateaus" (connected groups)
5. Save results to `your_notes_folder/` directory (chunks and plateaus subdirectories)

## Options

```bash
rhizome -i your_notes_folder -o my_output --threshold 0.6 --min-plateau-distance 0.3 --llm-model deepseek-r1:8b --embedding-model embeddinggemma
```

- `-i, --input`: Directory containing markdown notes (required)
- `-o, --output`: Output directory for chunks and plateaus (default: input directory)
- `--threshold`: Similarity threshold for creating plateaus (0-1, default: 0.65)
- `--min-plateau-distance`: Minimum distance between plateau centroids for diversity (default: 0.3, range: 0.2-0.5)
- `--llm-model`: LLM model to use for synthesis (default: `deepseek-r1:8b`)
- `--embedding-model`: Embedding model for similarity (default: `embeddinggemma`)

## Understanding the Output

### Chunks
Each chunk file has YAML frontmatter:
```yaml
---
type: chunk
source: original-file.md
chunk_id: 0
title: Section Title
---
```

### Plateaus
Plateau files connect related chunks:
```yaml
---
type: plateau
plateau_id: 0
---
```

## Example Workflow

1. Put your markdown notes in a folder (e.g., `notes/`)
2. Run: `rhizome -i notes`
3. Explore the results:
   - `notes/chunks/` - Atomic ideas from your notes
   - `notes/plateaus/` - Connected ideas across notes

## Tips

- Lower threshold (e.g., 0.5) creates more, looser connections
- Higher threshold (e.g., 0.8) creates fewer, tighter connections
- Different Ollama models may produce different embeddings
- The system works best with notes that have clear sections
