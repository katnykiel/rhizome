#!/usr/bin/env python3
"""Quick test script to verify rhizome works."""

from pathlib import Path
from rhizome.cli import main

if __name__ == "__main__":
    # Test the CLI with test_notes
    import sys
    sys.argv = ["rhizome", "-i", "test_notes", "--threshold", "0.4"]
    exit(main())