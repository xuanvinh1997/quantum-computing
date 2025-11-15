#!/usr/bin/env python3
"""
Example Usage - ArXiv Paper Tool

This file demonstrates various ways to use the tool programmatically.
"""

import os
from pathlib import Path

# Make sure to set environment variables first
# Either in .env file or export them:
# export OCR_API_KEY="your-key"
# export SUMMARY_API_KEY="your-key"

# Example 1: Simple Search
def example_simple_search():
    """Search for papers and display results"""
    print("Example 1: Simple Search")
    print("=" * 60)

    from arxiv_search import ArxivSearcher

    searcher = ArxivSearcher()

    # Search for quantum machine learning papers
    papers = searcher.search(
        keywords="quantum machine learning",
        max_results=5,
        filter_quantum=True
    )

    print(f"Found {len(papers)} papers:\n")

    for i, paper in enumerate(papers, 1):
        print(f"{i}. {paper['title']}")
        print(f"   ArXiv ID: {paper['arxiv_id']}")
        print(f"   Categories: {', '.join(paper['categories'])}")
        print()


# Example 2: Search and Save to Database
def example_search_and_save():
    """Search and save papers to database"""
    print("\nExample 2: Search and Save to Database")
    print("=" * 60)

    from main import ArxivPaperTool

    tool = ArxivPaperTool()

    # Search for VQE papers
    papers = tool.search_and_save(
        keywords="variational quantum eigensolver",
        max_results=10
    )

    print(f"\nSaved {len(papers)} papers to database")

    # Show statistics
    tool.print_statistics()


# Example 3: Process a Single Paper
def example_process_single_paper():
    """Process a specific paper by ArXiv ID"""
    print("\nExample 3: Process Single Paper")
    print("=" * 60)

    from arxiv_search import ArxivSearcher
    from main import ArxivPaperTool

    # Get a specific paper
    searcher = ArxivSearcher()
    paper = searcher.get_paper_by_id("2401.00001")  # Replace with actual ID

    if not paper:
        print("Paper not found!")
        return

    # Process it
    tool = ArxivPaperTool()
    result = tool.process_paper(paper, max_pages=10)

    if result:
        print("\n✓ Paper processed successfully!")
        print(f"\nMethodology Summary Preview:")
        print(result['methodology_summary'][:500] + "...")
    else:
        print("\n✗ Processing failed or paper not relevant")


# Example 4: Database Queries
def example_database_queries():
    """Query the database"""
    print("\nExample 4: Database Queries")
    print("=" * 60)

    from database import PaperDatabase

    db = PaperDatabase("arxiv_papers.db")

    # Get statistics
    stats = db.get_statistics()
    print(f"\nDatabase Statistics:")
    print(f"  Total papers: {stats['total_papers']}")
    print(f"  Processed: {stats['processed_papers']}")
    print(f"  Unprocessed: {stats['unprocessed_papers']}")

    # Search for quantum computing papers
    papers = db.search_papers(
        query="quantum computing",
        limit=5
    )

    print(f"\nFound {len(papers)} papers matching 'quantum computing':")
    for paper in papers:
        print(f"  - {paper['title'][:60]}...")

    db.close()


# Example 5: Export to Markdown
def example_export_markdown():
    """Export papers to markdown"""
    print("\nExample 5: Export to Markdown")
    print("=" * 60)

    from database import PaperDatabase
    from markdown_exporter import MarkdownExporter

    db = PaperDatabase("arxiv_papers.db")
    exporter = MarkdownExporter("example_output")

    # Get processed papers
    papers = db.search_papers(processed_only=True, limit=5)

    if not papers:
        print("No processed papers to export")
        db.close()
        return

    # Get full data with summaries
    papers_with_summaries = []
    for paper in papers:
        full_data = db.get_paper_with_summary(paper['id'])
        if full_data:
            papers_with_summaries.append(full_data)

    # Export
    created_files = exporter.export_multiple_papers(
        papers_with_summaries,
        create_index=True
    )

    print(f"Exported {len(created_files)} papers to example_output/")

    db.close()


# Example 6: OCR Only (No Database)
def example_ocr_only():
    """Extract text from a PDF without using database"""
    print("\nExample 6: OCR Only")
    print("=" * 60)

    from pdf_ocr import PDFOCRProcessor
    import os

    # Check if API key is set
    if not os.getenv("OCR_API_KEY"):
        print("OCR_API_KEY not set. Skipping this example.")
        return

    processor = PDFOCRProcessor(
        api_key=os.getenv("OCR_API_KEY"),
        base_url=os.getenv("OCR_BASE_URL"),
        model_name="nanonets/nanonets-ocr2-3b"
    )

    # Example: Extract from a real arXiv paper
    # Replace with an actual paper URL
    pdf_url = "https://arxiv.org/pdf/2401.00001.pdf"

    print(f"Downloading and processing: {pdf_url}")
    print("This may take a minute...")

    try:
        extracted = processor.extract_text_from_url(
            pdf_url,
            max_pages=3  # Just first 3 pages for demo
        )

        full_text = processor.get_full_text(extracted)

        print(f"\n✓ Extracted {len(full_text)} characters")
        print(f"\nFirst 500 characters:")
        print(full_text[:500])

    except Exception as e:
        print(f"✗ Error: {e}")


# Example 7: Batch Processing
def example_batch_processing():
    """Process multiple unprocessed papers"""
    print("\nExample 7: Batch Processing")
    print("=" * 60)

    from main import ArxivPaperTool

    tool = ArxivPaperTool()

    # Process up to 3 papers (for demo)
    tool.process_unprocessed_papers(limit=3)

    print("\nBatch processing complete!")


# Example 8: Custom Workflow
def example_custom_workflow():
    """A complete custom workflow"""
    print("\nExample 8: Custom Workflow")
    print("=" * 60)

    from main import ArxivPaperTool

    tool = ArxivPaperTool()

    # Step 1: Search for multiple topics
    topics = [
        "quantum error correction",
        "quantum neural networks"
    ]

    for topic in topics:
        print(f"\nSearching for: {topic}")
        papers = tool.search_and_save(
            keywords=topic,
            max_results=5
        )

    # Step 2: Process some papers
    print("\nProcessing papers...")
    tool.process_unprocessed_papers(limit=2)

    # Step 3: Export
    print("\nExporting processed papers...")
    tool.export_all_processed()

    print("\n✓ Workflow complete!")
