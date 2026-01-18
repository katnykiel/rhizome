# Rhizome Usage Guide

## Prerequisites

1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai)
2. **Pull a model**: Run `ollama pull llama3.2` (or another model)
3. **Start Ollama**: Run `ollama serve` in a terminal

## Installation

The package is already installed in the virtual environment. To activate it:

```bash
source .venv/bin/activate
```

## Basic Usage

Process a folder of markdown notes:

```bash
rhizome test_notes
```

This will:
1. Create atomic chunks from your notes in `output/chunks/`
2. Generate embeddings for each chunk
3. Find similar chunks and create plateau files in `output/plateaus/`

## Options

```bash
rhizome test_notes --output-dir my_output --threshold 0.6 --model llama3.2
```

- `--output-dir`: Where to save results (default: `output`)
- `--threshold`: Similarity threshold for grouping chunks (0-1, default: 0.7)
- `--model`: Ollama model to use (default: `llama3.2`)

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
related_chunks:
  - Chunk Title 1
  - Chunk Title 2
source_files:
  - file1.md
  - file2.md
---
```

## Example Workflow

1. Put your markdown notes in a folder (e.g., `notes/`)
2. Run: `rhizome notes`
3. Explore the results:
   - `output/chunks/` - Atomic ideas from your notes
   - `output/plateaus/` - Connected ideas across notes

## Tips

- Lower threshold (e.g., 0.5) creates more, looser connections
- Higher threshold (e.g., 0.8) creates fewer, tighter connections
- Different Ollama models may produce different embeddings
- The system works best with notes that have clear sections
