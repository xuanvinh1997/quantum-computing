#!/usr/bin/env python3
"""
ArXiv Crawler - Continuous paper collection and processing

This crawler runs continuously to:
1. Search for new quantum computing papers
2. Process them automatically
3. Export results to markdown
"""

import time
import schedule
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import signal
import sys
from pathlib import Path

from config import Config
from arxiv_search import ArxivSearcher
from pdf_ocr import PDFOCRProcessor
from summarizer import PaperSummarizer
from database import PaperDatabase
from markdown_exporter import MarkdownExporter


class ArxivCrawler:
    """Continuous crawler for ArXiv quantum computing papers"""

    def __init__(self, config: Config = None):
        """
        Initialize crawler

        Args:
            config: Configuration object
        """
        self.config = config or Config
        self.running = False
        self.stats = {
            'total_searches': 0,
            'papers_found': 0,
            'papers_processed': 0,
            'errors': 0,
            'start_time': None
        }

        # Initialize components
        print("Initializing crawler components...")
        self.searcher = ArxivSearcher()
        self.database = PaperDatabase(self.config.DATABASE_PATH)
        self.exporter = MarkdownExporter(self.config.MARKDOWN_OUTPUT_DIR)

        # Initialize OCR and summarizer if keys are available
        self.ocr_processor = None
        self.summarizer = None

        if self.config.OCR_API_KEY and self.config.SUMMARY_API_KEY:
            self.ocr_processor = PDFOCRProcessor(
                api_key=self.config.OCR_API_KEY,
                base_url=self.config.OCR_BASE_URL,
                model_name=self.config.OCR_MODEL
            )
            self.summarizer = PaperSummarizer(
                api_key=self.config.SUMMARY_API_KEY,
                base_url=self.config.SUMMARY_BASE_URL,
                model_name=self.config.SUMMARY_MODEL
            )
            print("✓ All components initialized")
        else:
            print("⚠ OCR/Summarizer not initialized - crawler will only collect papers")

    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")

    def search_new_papers(self, categories: Optional[List[str]] = None,
                         keywords: Optional[List[str]] = None,
                         max_results_per_query: int = 30) -> int:
        """
        Search for new papers across categories and keywords

        Args:
            categories: List of ArXiv categories to search
            keywords: List of keyword strings to search
            max_results_per_query: Max results per search query

        Returns:
            Number of new papers found
        """
        if categories is None:
            categories = ['quant-ph', 'cs.ET']

        if keywords is None:
            keywords = [
                'quantum computing',
                'quantum machine learning',
                'variational quantum',
                'quantum algorithm',
                'quantum circuit'
            ]

        self.log(f"Starting search: {len(categories)} categories, {len(keywords)} keywords")
        total_new_papers = 0

        # Search by categories
        for category in categories:
            try:
                self.log(f"Searching category: {category}")
                papers = self.searcher.search(
                    category=category,
                    max_results=max_results_per_query,
                    filter_quantum=True
                )

                # Save to database
                new_papers = 0
                for paper in papers:
                    paper_id = self.database.insert_paper(paper)
                    if paper_id:
                        new_papers += 1

                self.log(f"  Found {len(papers)} papers, {new_papers} new")
                total_new_papers += new_papers
                self.stats['total_searches'] += 1

            except Exception as e:
                self.log(f"  Error searching category {category}: {e}", "ERROR")
                self.stats['errors'] += 1

        # Search by keywords
        for keyword in keywords:
            try:
                self.log(f"Searching keyword: {keyword}")
                papers = self.searcher.search(
                    keywords=keyword,
                    max_results=max_results_per_query,
                    filter_quantum=True
                )

                # Save to database
                new_papers = 0
                for paper in papers:
                    paper_id = self.database.insert_paper(paper)
                    if paper_id:
                        new_papers += 1

                self.log(f"  Found {len(papers)} papers, {new_papers} new")
                total_new_papers += new_papers
                self.stats['total_searches'] += 1

            except Exception as e:
                self.log(f"  Error searching keyword '{keyword}': {e}", "ERROR")
                self.stats['errors'] += 1

        self.stats['papers_found'] += total_new_papers
        self.log(f"Search complete: {total_new_papers} new papers added to database")
        return total_new_papers

    def process_papers(self, batch_size: int = 5, max_pages: int = 15) -> int:
        """
        Process unprocessed papers

        Args:
            batch_size: Number of papers to process in this batch
            max_pages: Maximum pages to OCR per paper

        Returns:
            Number of papers successfully processed
        """
        if not self.ocr_processor or not self.summarizer:
            self.log("OCR/Summarizer not initialized - skipping processing", "WARNING")
            return 0

        self.log(f"Processing up to {batch_size} papers...")
        papers = self.database.get_unprocessed_papers(limit=batch_size)

        if not papers:
            self.log("No unprocessed papers found")
            return 0

        processed = 0

        for i, paper in enumerate(papers, 1):
            try:
                self.log(f"[{i}/{len(papers)}] Processing: {paper['title'][:60]}...")

                # Check relevance
                relevance = self.summarizer.check_quantum_relevance(paper)
                if not relevance['is_relevant']:
                    self.log(f"  Not relevant (score: {relevance['relevance_score']:.2f})")
                    # Mark as processed even if not relevant
                    self.database.insert_summary(
                        paper['id'],
                        "Not relevant to quantum computing",
                        "N/A",
                        None
                    )
                    continue

                # OCR PDF
                self.log(f"  OCRing PDF (max {max_pages} pages)...")
                extracted_text_dict = self.ocr_processor.extract_text_from_url(
                    paper['pdf_link'],
                    max_pages=max_pages
                )
                full_text = self.ocr_processor.get_full_text(extracted_text_dict)

                # Summarize
                self.log("  Generating summary...")
                methodology_summary = self.summarizer.summarize_methodology(
                    full_text,
                    paper
                )

                key_contributions = self.summarizer.extract_key_contributions(
                    full_text,
                    paper
                )

                # Save to database
                summary_id = self.database.insert_summary(
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

                    filepath = self.exporter.export_paper(
                        paper_with_summary,
                        methodology_summary=methodology_summary,
                        key_contributions=key_contributions
                    )

                    self.log(f"  ✓ Processed and exported: {Path(filepath).name}")
                    processed += 1
                    self.stats['papers_processed'] += 1

            except Exception as e:
                self.log(f"  ✗ Error processing paper: {e}", "ERROR")
                self.stats['errors'] += 1

        self.log(f"Processing batch complete: {processed}/{len(papers)} successful")
        return processed

    def export_collection_summary(self):
        """Export collection summary of all processed papers"""
        try:
            papers = self.database.search_papers(processed_only=True, limit=1000)
            if papers:
                papers_with_summaries = []
                for paper in papers:
                    full_data = self.database.get_paper_with_summary(paper['id'])
                    if full_data and full_data.get('methodology_summary') != "Not relevant to quantum computing":
                        papers_with_summaries.append(full_data)

                if papers_with_summaries:
                    self.exporter.create_collection_summary(papers_with_summaries)
                    self.log(f"Collection summary updated ({len(papers_with_summaries)} papers)")

        except Exception as e:
            self.log(f"Error creating collection summary: {e}", "ERROR")

    def print_stats(self):
        """Print crawler statistics"""
        if self.stats['start_time']:
            runtime = datetime.now() - self.stats['start_time']
            runtime_str = str(runtime).split('.')[0]  # Remove microseconds
        else:
            runtime_str = "N/A"

        db_stats = self.database.get_statistics()

        print("\n" + "=" * 70)
        print("CRAWLER STATISTICS")
        print("=" * 70)
        print(f"Runtime:             {runtime_str}")
        print(f"Total searches:      {self.stats['total_searches']}")
        print(f"Papers found:        {self.stats['papers_found']}")
        print(f"Papers processed:    {self.stats['papers_processed']}")
        print(f"Errors:              {self.stats['errors']}")
        print()
        print("DATABASE STATISTICS")
        print("-" * 70)
        print(f"Total papers:        {db_stats['total_papers']}")
        print(f"Processed:           {db_stats['processed_papers']}")
        print(f"Unprocessed:         {db_stats['unprocessed_papers']}")
        print(f"Last 7 days:         {db_stats['papers_last_7_days']}")
        print("=" * 70 + "\n")

    def run_cycle(self):
        """Run one complete crawl cycle"""
        self.log("=" * 70)
        self.log("Starting crawl cycle")
        self.log("=" * 70)

        try:
            # Step 1: Search for new papers
            new_papers = self.search_new_papers()

            # Step 2: Process papers (small batch)
            if new_papers > 0 or self.database.get_statistics()['unprocessed_papers'] > 0:
                processed = self.process_papers(batch_size=3)

                # Step 3: Update collection summary if papers were processed
                if processed > 0:
                    self.export_collection_summary()

            # Step 4: Print stats
            self.print_stats()

        except Exception as e:
            self.log(f"Error in crawl cycle: {e}", "ERROR")
            self.stats['errors'] += 1

        self.log("Crawl cycle complete")
        self.log("=" * 70 + "\n")

    def start(self,
              interval_hours: int = 6,
              run_immediately: bool = True,
              continuous: bool = True):
        """
        Start the crawler

        Args:
            interval_hours: Hours between crawl cycles
            run_immediately: Run first cycle immediately
            continuous: Keep running continuously
        """
        self.running = True
        self.stats['start_time'] = datetime.now()

        self.log(f"Starting ArXiv Crawler")
        self.log(f"Interval: Every {interval_hours} hours")
        self.log(f"Database: {self.config.DATABASE_PATH}")
        self.log(f"Output: {self.config.MARKDOWN_OUTPUT_DIR}")

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Schedule the crawl cycle
        schedule.every(interval_hours).hours.do(self.run_cycle)

        # Run immediately if requested
        if run_immediately:
            self.run_cycle()

        if continuous:
            self.log(f"Crawler running. Next cycle in {interval_hours} hours.")
            self.log("Press Ctrl+C to stop gracefully\n")

            # Keep running
            while self.running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute

        self.log("Crawler stopped")

    def stop(self):
        """Stop the crawler gracefully"""
        self.log("Stopping crawler...")
        self.running = False
        self.print_stats()
        self.database.close()

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print()  # New line after ^C
        self.log("Received shutdown signal")
        self.stop()
        sys.exit(0)
