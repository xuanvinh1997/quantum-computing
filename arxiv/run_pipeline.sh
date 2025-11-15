#!/bin/bash
# ArXiv Paper Crawler - Full Pipeline Runner
# This script runs the complete pipeline: search -> process -> export

set -e  # Exit on error

echo "======================================================================="
echo "ARXIV PAPER CRAWLER - FULL PIPELINE"
echo "======================================================================="
echo ""

# Configuration
SEARCH_KEYWORDS="${1:-quantum computing}"
MAX_RESULTS="${2:-10}"
BATCH_SIZE="${3:-5}"
MAX_PAGES="${4:-15}"

echo "Configuration:"
echo "  Search keywords: $SEARCH_KEYWORDS"
echo "  Max results:     $MAX_RESULTS"
echo "  Batch size:      $BATCH_SIZE"
echo "  Max pages:       $MAX_PAGES"
echo ""

# Step 1: Search for papers
echo "======================================================================="
echo "STEP 1: SEARCHING ARXIV"
echo "======================================================================="
python main.py search --keywords "$SEARCH_KEYWORDS" --max-results "$MAX_RESULTS"
echo ""

# Step 2: Process papers
echo "======================================================================="
echo "STEP 2: PROCESSING PAPERS (OCR + SUMMARIZE)"
echo "======================================================================="
python main.py process --batch-size "$BATCH_SIZE" --max-pages "$MAX_PAGES"
echo ""

# Step 3: Export to markdown
echo "======================================================================="
echo "STEP 3: EXPORTING TO MARKDOWN"
echo "======================================================================="
python main.py export --processed-only --summary
echo ""

# Step 4: Show final statistics
echo "======================================================================="
echo "STEP 4: FINAL STATISTICS"
echo "======================================================================="
python main.py stats --recent 5
echo ""

echo "======================================================================="
echo "PIPELINE COMPLETE!"
echo "======================================================================="
echo "Check the 'papers_output' directory for markdown files"
echo ""
