#!/usr/bin/env python3
"""
Export paper summaries to markdown files
"""

import os
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime


class MarkdownExporter:
    """Export papers and summaries to markdown format"""

    def __init__(self, output_dir: str = "papers_output"):
        """
        Initialize exporter

        Args:
            output_dir: Directory to save markdown files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _sanitize_filename(self, title: str, arxiv_id: str) -> str:
        """
        Create safe filename from paper title

        Args:
            title: Paper title
            arxiv_id: ArXiv ID

        Returns:
            Safe filename
        """
        # Remove special characters and limit length
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_'))
        safe_title = safe_title.replace(' ', '_')
        safe_title = safe_title[:50]  # Limit length

        return f"{arxiv_id}_{safe_title}.md"

    def export_paper(self,
                    paper: Dict,
                    methodology_summary: Optional[str] = None,
                    key_contributions: Optional[str] = None) -> str:
        """
        Export a single paper to markdown

        Args:
            paper: Paper dictionary
            methodology_summary: Optional methodology summary
            key_contributions: Optional key contributions

        Returns:
            Path to created markdown file
        """
        filename = self._sanitize_filename(paper['title'], paper['arxiv_id'])
        filepath = self.output_dir / filename

        # Parse authors if stored as JSON string
        authors = paper.get('authors', [])
        if isinstance(authors, str):
            import json
            try:
                authors = json.loads(authors)
            except:
                authors = [authors]

        # Parse categories similarly
        categories = paper.get('categories', [])
        if isinstance(categories, str):
            import json
            try:
                categories = json.loads(categories)
            except:
                categories = [categories]

        # Build markdown content
        md_content = f"""# {paper['title']}

## Metadata

- **ArXiv ID**: [{paper['arxiv_id']}]({paper.get('abstract_link', '#')})
- **PDF Link**: [{paper.get('pdf_link', '#')}]({paper.get('pdf_link', '#')})
- **Authors**: {', '.join(authors)}
- **Categories**: {', '.join(categories)}
- **Published**: {paper.get('published', 'N/A')}
- **Updated**: {paper.get('updated', 'N/A')}

---

## Abstract

{paper.get('abstract', 'No abstract available.')}

---
"""

        # Add key contributions if available
        if key_contributions:
            md_content += f"""
## Key Contributions

{key_contributions}

---
"""

        # Add methodology summary if available
        if methodology_summary:
            md_content += f"""
## Methodology Summary

{methodology_summary}

---
"""

        # Add footer
        md_content += f"""
## Export Information

- **Exported**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Quantum Relevant**: {paper.get('is_quantum_relevant', 'Unknown')}
- **Relevance Score**: {paper.get('relevance_score', 'N/A')}
"""

        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md_content)

        return str(filepath)

    def export_multiple_papers(self,
                              papers_with_summaries: List[Dict],
                              create_index: bool = True) -> List[str]:
        """
        Export multiple papers to markdown

        Args:
            papers_with_summaries: List of paper dictionaries with summaries
            create_index: Create an index file

        Returns:
            List of created file paths
        """
        created_files = []

        for paper in papers_with_summaries:
            try:
                filepath = self.export_paper(
                    paper,
                    methodology_summary=paper.get('methodology_summary'),
                    key_contributions=paper.get('key_contributions')
                )
                created_files.append(filepath)
            except Exception as e:
                print(f"Error exporting paper {paper.get('arxiv_id', 'unknown')}: {e}")

        # Create index file
        if create_index and created_files:
            self._create_index(papers_with_summaries)

        return created_files

    def _create_index(self, papers: List[Dict]):
        """
        Create an index markdown file listing all papers

        Args:
            papers: List of paper dictionaries
        """
        index_path = self.output_dir / "INDEX.md"

        content = f"""# ArXiv Papers Index

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Total Papers**: {len(papers)}

---

## Papers

"""

        # Group by category
        from collections import defaultdict
        by_category = defaultdict(list)

        for paper in papers:
            categories = paper.get('categories', [])
            if isinstance(categories, str):
                import json
                try:
                    categories = json.loads(categories)
                except:
                    categories = [categories]

            primary_category = categories[0] if categories else 'Unknown'
            by_category[primary_category].append(paper)

        # Write papers by category
        for category, cat_papers in sorted(by_category.items()):
            content += f"\n### {category}\n\n"

            for paper in cat_papers:
                filename = self._sanitize_filename(paper['title'], paper['arxiv_id'])
                content += f"- [{paper['title']}]({filename})\n"
                content += f"  - ArXiv: [{paper['arxiv_id']}]({paper.get('abstract_link', '#')})\n"
                content += f"  - Published: {paper.get('published', 'N/A')}\n"

                # Add summary status
                if paper.get('methodology_summary'):
                    content += f"  - Status: ✅ Summarized\n"
                else:
                    content += f"  - Status: ⏳ Not summarized\n"

                content += "\n"

        # Write index file
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"Index created at: {index_path}")

    def create_collection_summary(self, papers: List[Dict], output_name: str = "COLLECTION_SUMMARY.md"):
        """
        Create a summary of all papers in the collection

        Args:
            papers: List of papers
            output_name: Output filename
        """
        filepath = self.output_dir / output_name

        content = f"""# Quantum Computing Papers Collection Summary

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Total Papers**: {len(papers)}

---

## Statistics

"""

        # Calculate statistics
        from collections import Counter

        # By category
        all_categories = []
        for paper in papers:
            categories = paper.get('categories', [])
            if isinstance(categories, str):
                import json
                try:
                    categories = json.loads(categories)
                except:
                    categories = [categories]
            all_categories.extend(categories)

        category_counts = Counter(all_categories)

        content += "### Papers by Category\n\n"
        for cat, count in category_counts.most_common():
            content += f"- **{cat}**: {count} papers\n"

        # Processed vs unprocessed
        processed = sum(1 for p in papers if p.get('processed', False))
        content += f"\n### Processing Status\n\n"
        content += f"- Processed: {processed}\n"
        content += f"- Unprocessed: {len(papers) - processed}\n"

        # By year
        years = []
        for paper in papers:
            published = paper.get('published', '')
            if published:
                year = published[:4]
                years.append(year)

        year_counts = Counter(years)
        content += f"\n### Papers by Year\n\n"
        for year, count in sorted(year_counts.items(), reverse=True):
            content += f"- **{year}**: {count} papers\n"

        content += "\n---\n\n"
        content += "## Recent Papers\n\n"

        # List most recent papers
        sorted_papers = sorted(
            papers,
            key=lambda p: p.get('published', ''),
            reverse=True
        )[:10]

        for i, paper in enumerate(sorted_papers, 1):
            content += f"{i}. **{paper['title']}**\n"
            content += f"   - ArXiv: [{paper['arxiv_id']}]({paper.get('abstract_link', '#')})\n"
            content += f"   - Published: {paper.get('published', 'N/A')}\n\n"

        # Write file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"Collection summary created at: {filepath}")

