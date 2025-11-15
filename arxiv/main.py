#!/usr/bin/env python3
"""
ArXiv Paper Crawler - Main Entry Point

This script provides a complete pipeline for:
1. Searching ArXiv for quantum computing papers
2. Extracting content from PDFs using OCR
3. Summarizing papers using LLM
4. Storing metadata to SQLite database
5. Exporting content to markdown files

Usage:
    python main.py search --keywords "quantum computing" --max-results 10
    python main.py process --batch-size 5
    python main.py crawl --interval 6
    python main.py stats
"""

import argparse
import sys
from pathlib import Path

from config import Config
from arxiv_search import ArxivSearcher
from pdf_ocr import PDFOCRProcessor
from summarizer import PaperSummarizer
from database import PaperDatabase
from markdown_exporter import MarkdownExporter
from crawler import ArxivCrawler


def setup_components():
    """Initialize all components"""
    print("Initializing components...")

    # Validate configuration
    errors = Config.validate()
    if errors:
        print("� Configuration warnings:")
        for error in errors:
            print(f"  - {error}")
        print()

    # Initialize components
    searcher = ArxivSearcher()
    database = PaperDatabase(Config.DATABASE_PATH)
    exporter = MarkdownExporter(Config.MARKDOWN_OUTPUT_DIR)

    ocr_processor = None
    summarizer = None

    if Config.OCR_API_KEY and Config.SUMMARY_API_KEY:
        ocr_processor = PDFOCRProcessor(
            api_key=Config.OCR_API_KEY,
            base_url=Config.OCR_BASE_URL,
            model_name=Config.OCR_MODEL
        )
        summarizer = PaperSummarizer(
            api_key=Config.SUMMARY_API_KEY,
            base_url=Config.SUMMARY_BASE_URL,
            model_name=Config.SUMMARY_MODEL
        )
        print(" All components initialized\n")
    else:
        print("� OCR/Summarizer not initialized - only search functionality available\n")

    return searcher, database, exporter, ocr_processor, summarizer


def cmd_search(args):
    """Search for papers and add to database"""
    print("=" * 70)
    print("SEARCH ARXIV PAPERS")
    print("=" * 70)

    searcher, database, _, _, _ = setup_components()

    try:
        # Perform search
        if args.keywords and args.category:
            print(f"Searching for: '{args.keywords}' in category '{args.category}'")
            papers = searcher.search(
                keywords=args.keywords,
                category=args.category,
                max_results=args.max_results,
                filter_quantum=args.filter_quantum
            )
        elif args.keywords:
            print(f"Searching for: '{args.keywords}'")
            papers = searcher.search(
                keywords=args.keywords,
                max_results=args.max_results,
                filter_quantum=args.filter_quantum
            )
        elif args.category:
            print(f"Searching category: '{args.category}'")
            papers = searcher.search(
                category=args.category,
                max_results=args.max_results,
                filter_quantum=args.filter_quantum
            )
        else:
            print("Searching default: quantum computing papers (quant-ph)")
            papers = searcher.search(
                category='quant-ph',
                max_results=args.max_results,
                filter_quantum=args.filter_quantum
            )

        print(f"\nFound {len(papers)} papers")

        # Save to database
        new_papers = 0
        duplicate_papers = 0

        for paper in papers:
            paper_id = database.insert_paper(paper)
            if paper_id:
                new_papers += 1
                print(f"   Added: {paper['title'][:60]}... (ID: {paper_id})")
            else:
                duplicate_papers += 1
                print(f"  - Duplicate: {paper['arxiv_id']}")

        print(f"\nResults:")
        print(f"  New papers: {new_papers}")
        print(f"  Duplicates: {duplicate_papers}")
        print(f"  Total in database: {database.get_statistics()['total_papers']}")

        # Log search
        database.log_search(
            args.keywords or args.category or 'quant-ph',
            args.category,
            len(papers)
        )

    finally:
        database.close()

    print("=" * 70)


def cmd_process(args):
    """Process unprocessed papers (OCR + Summarize)"""
    print("=" * 70)
    print("PROCESS PAPERS")
    print("=" * 70)

    searcher, database, exporter, ocr_processor, summarizer = setup_components()

    if not ocr_processor or not summarizer:
        print("ERROR: OCR_API_KEY and SUMMARY_API_KEY must be set to process papers")
        sys.exit(1)

    try:
        # Get unprocessed papers
        papers = database.get_unprocessed_papers(limit=args.batch_size)

        if not papers:
            print("No unprocessed papers found")
            return

        print(f"Processing {len(papers)} papers...\n")

        processed = 0
        skipped = 0
        errors = 0

        for i, paper in enumerate(papers, 1):
            try:
                print(f"[{i}/{len(papers)}] {paper['title'][:60]}...")
                print(f"  ArXiv ID: {paper['arxiv_id']}")

                # Check quantum relevance
                print("  Checking relevance...")
                relevance = summarizer.check_quantum_relevance(paper)
                print(f"  Relevance score: {relevance['relevance_score']:.2f}")

                if not relevance['is_relevant']:
                    print(f"  �  Skipping (not relevant)")
                    # Mark as processed even if not relevant
                    database.insert_summary(
                        paper['id'],
                        "Not relevant to quantum computing",
                        "N/A",
                        None
                    )
                    skipped += 1
                    continue

                # OCR PDF
                print(f"  Extracting text from PDF (max {args.max_pages} pages)...")
                extracted_text_dict = ocr_processor.extract_text_from_url(
                    paper['pdf_link'],
                    max_pages=args.max_pages
                )
                full_text = ocr_processor.get_full_text(extracted_text_dict)
                print(f"  Extracted {len(full_text)} characters")

                # Summarize
                print("  Generating methodology summary...")
                methodology_summary = summarizer.summarize_methodology(
                    full_text,
                    paper
                )

                print("  Extracting key contributions...")
                key_contributions = summarizer.extract_key_contributions(
                    full_text,
                    paper
                )

                # Save to database
                summary_id = database.insert_summary(
                    paper['id'],
                    methodology_summary,
                    key_contributions,
                    full_text[:10000]  # Limit stored text size
                )

                if summary_id:
                    # Export to markdown
                    paper_with_summary = paper.copy()
                    paper_with_summary.update({
                        'methodology_summary': methodology_summary,
                        'key_contributions': key_contributions,
                        'relevance_score': relevance['relevance_score']
                    })

                    filepath = exporter.export_paper(
                        paper_with_summary,
                        methodology_summary=methodology_summary,
                        key_contributions=key_contributions
                    )

                    print(f"   Processed and exported to: {Path(filepath).name}")
                    processed += 1
                else:
                    print(f"   Failed to save summary")
                    errors += 1

                print()

            except Exception as e:
                print(f"   Error: {e}")
                errors += 1
                print()

        # Summary
        print("=" * 70)
        print("PROCESSING SUMMARY")
        print("=" * 70)
        print(f"Processed:  {processed}")
        print(f"Skipped:    {skipped}")
        print(f"Errors:     {errors}")
        print(f"Total:      {len(papers)}")

        # Update collection summary if papers were processed
        if processed > 0:
            print("\nUpdating collection summary...")
            papers_all = database.search_papers(processed_only=True, limit=1000)
            if papers_all:
                papers_with_summaries = []
                for p in papers_all:
                    full_data = database.get_paper_with_summary(p['id'])
                    if full_data and full_data.get('methodology_summary') != "Not relevant to quantum computing":
                        papers_with_summaries.append(full_data)

                if papers_with_summaries:
                    exporter.create_collection_summary(papers_with_summaries)
                    exporter._create_index(papers_with_summaries)
                    print(f" Collection summary created ({len(papers_with_summaries)} papers)")

    finally:
        database.close()

    print("=" * 70)


def cmd_crawl(args):
    """Start continuous crawler"""
    print("=" * 70)
    print("START CONTINUOUS CRAWLER")
    print("=" * 70)

    crawler = ArxivCrawler()

    crawler.start(
        interval_hours=args.interval,
        run_immediately=True,
        continuous=not args.once
    )


def cmd_stats(args):
    """Show database statistics"""
    print("=" * 70)
    print("DATABASE STATISTICS")
    print("=" * 70)

    database = PaperDatabase(Config.DATABASE_PATH)

    try:
        stats = database.get_statistics()

        print(f"\nTotal papers:        {stats['total_papers']}")
        print(f"Processed:           {stats['processed_papers']}")
        print(f"Unprocessed:         {stats['unprocessed_papers']}")
        print(f"Last 7 days:         {stats['papers_last_7_days']}")

        print(f"\nDatabase path:       {Config.DATABASE_PATH}")
        print(f"Markdown output:     {Config.MARKDOWN_OUTPUT_DIR}")

        # Show recent papers
        if args.recent:
            print(f"\nRecent papers:")
            recent = database.search_papers(limit=args.recent)
            for paper in recent:
                status = "" if paper['processed'] else "�"
                print(f"  {status} [{paper['arxiv_id']}] {paper['title'][:60]}...")
                print(f"      Published: {paper['published']}")

    finally:
        database.close()

    print("=" * 70)


def cmd_config(args):
    """Show configuration"""
    Config.print_config()


def cmd_export(args):
    """Export papers to markdown"""
    print("=" * 70)
    print("EXPORT TO MARKDOWN")
    print("=" * 70)

    database = PaperDatabase(Config.DATABASE_PATH)
    exporter = MarkdownExporter(Config.MARKDOWN_OUTPUT_DIR)

    try:
        # Get papers
        if args.processed_only:
            papers = database.search_papers(processed_only=True, limit=args.limit)
        else:
            papers = database.search_papers(limit=args.limit)

        if not papers:
            print("No papers found to export")
            return

        print(f"Exporting {len(papers)} papers...\n")

        # Get papers with summaries
        papers_with_summaries = []
        for paper in papers:
            full_data = database.get_paper_with_summary(paper['id'])
            if full_data:
                papers_with_summaries.append(full_data)

        # Export
        created_files = exporter.export_multiple_papers(
            papers_with_summaries,
            create_index=True
        )

        print(f"\n Exported {len(created_files)} papers")
        print(f"  Output directory: {Config.MARKDOWN_OUTPUT_DIR}")

        # Create collection summary
        if args.summary:
            exporter.create_collection_summary(papers_with_summaries)
            print(f" Collection summary created")

    finally:
        database.close()

    print("=" * 70)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="ArXiv Paper Crawler - Search, Process, and Export quantum computing papers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search for papers
  python main.py search --keywords "quantum machine learning" --max-results 20

  # Process papers (OCR + Summarize)
  python main.py process --batch-size 5 --max-pages 15

  # Start continuous crawler
  python main.py crawl --interval 6

  # Run once and exit
  python main.py crawl --once

  # Show statistics
  python main.py stats --recent 10

  # Export to markdown
  python main.py export --processed-only --summary

  # Show configuration
  python main.py config
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Search command
    search_parser = subparsers.add_parser('search', help='Search ArXiv and add papers to database')
    search_parser.add_argument('--keywords', '-k', help='Search keywords')
    search_parser.add_argument('--category', '-c', help='ArXiv category (e.g., quant-ph)')
    search_parser.add_argument('--max-results', '-m', type=int, default=50, help='Maximum results (default: 50)')
    search_parser.add_argument('--no-filter', dest='filter_quantum', action='store_false',
                              help='Disable quantum-related filtering')
    search_parser.set_defaults(func=cmd_search, filter_quantum=True)

    # Process command
    process_parser = subparsers.add_parser('process', help='Process unprocessed papers (OCR + Summarize)')
    process_parser.add_argument('--batch-size', '-b', type=int, default=5,
                               help='Number of papers to process (default: 5)')
    process_parser.add_argument('--max-pages', '-p', type=int, default=15,
                               help='Max pages to OCR per paper (default: 15)')
    process_parser.set_defaults(func=cmd_process)

    # Crawl command
    crawl_parser = subparsers.add_parser('crawl', help='Start continuous crawler')
    crawl_parser.add_argument('--interval', '-i', type=int, default=6,
                             help='Hours between crawl cycles (default: 6)')
    crawl_parser.add_argument('--once', action='store_true',
                             help='Run once and exit')
    crawl_parser.set_defaults(func=cmd_crawl)

    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show database statistics')
    stats_parser.add_argument('--recent', '-r', type=int, help='Show N recent papers')
    stats_parser.set_defaults(func=cmd_stats)

    # Config command
    config_parser = subparsers.add_parser('config', help='Show configuration')
    config_parser.set_defaults(func=cmd_config)

    # Export command
    export_parser = subparsers.add_parser('export', help='Export papers to markdown')
    export_parser.add_argument('--processed-only', action='store_true',
                              help='Only export processed papers')
    export_parser.add_argument('--limit', '-l', type=int, default=1000,
                              help='Maximum papers to export (default: 1000)')
    export_parser.add_argument('--summary', '-s', action='store_true',
                              help='Create collection summary')
    export_parser.set_defaults(func=cmd_export)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()
