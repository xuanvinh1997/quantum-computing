#!/usr/bin/env python3
"""
Enhanced ArXiv Search with deduplication and category filtering
Focuses on quantum physics and quantum computing papers
"""

import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Set
from datetime import datetime
import hashlib
import json


class ArxivSearcher:
    """ArXiv search with built-in deduplication for quantum-related papers"""

    # Quantum-related categories
    QUANTUM_CATEGORIES = [
        'quant-ph',      # Quantum Physics
        'cond-mat.mes-hall',  # Mesoscale and Nanoscale Physics
        'cond-mat.str-el',    # Strongly Correlated Electrons
        'cs.ET',         # Emerging Technologies (includes quantum computing)
        'math-ph',       # Mathematical Physics
        'physics.atom-ph',    # Atomic Physics
    ]

    def __init__(self):
        self.seen_papers: Set[str] = set()  # Track arxiv_ids
        self.base_url = "http://export.arxiv.org/api/query?"

    def _generate_paper_hash(self, paper: Dict) -> str:
        """Generate unique hash for a paper based on arxiv_id"""
        return paper['arxiv_id']

    def _is_quantum_related(self, paper: Dict) -> bool:
        """
        Check if paper is quantum-related based on:
        - Categories
        - Keywords in title/abstract
        """
        # Check categories
        categories = paper.get('categories', [])
        for cat in categories:
            if any(qc in cat for qc in self.QUANTUM_CATEGORIES):
                return True

        # Check keywords in title and abstract
        quantum_keywords = [
            'quantum', 'qubit', 'entanglement', 'superposition',
            'quantum computing', 'quantum algorithm', 'quantum circuit',
            'quantum machine learning', 'qml', 'variational quantum',
            'qaoa', 'vqe', 'quantum annealing', 'quantum cryptography',
            'quantum information', 'quantum error correction'
        ]

        text = (paper.get('title', '') + ' ' + paper.get('abstract', '')).lower()
        return any(keyword in text for keyword in quantum_keywords)

    def search(self,
               keywords: Optional[str] = None,
               category: Optional[str] = None,
               max_results: int = 50,
               filter_quantum: bool = True) -> List[Dict]:
        """
        Search ArXiv with deduplication

        Args:
            keywords: Search terms (optional if category is specified)
            category: ArXiv category (e.g., 'quant-ph')
            max_results: Maximum number of results
            filter_quantum: Only return quantum-related papers

        Returns:
            List of unique paper dictionaries
        """
        # Build search query
        if category and keywords:
            search_query = f'cat:{category} AND all:{keywords}'
        elif category:
            search_query = f'cat:{category}'
        elif keywords:
            search_query = f'all:{keywords}'
        else:
            search_query = 'cat:quant-ph'  # Default to quantum physics

        params = {
            'search_query': search_query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }

        url = self.base_url + urllib.parse.urlencode(params)

        try:
            with urllib.request.urlopen(url) as response:
                xml_data = response.read()
        except Exception as e:
            print(f"Error fetching from ArXiv: {e}")
            return []

        # Parse XML
        papers = self._parse_arxiv_response(xml_data)

        # Filter and deduplicate
        unique_papers = []
        for paper in papers:
            paper_id = self._generate_paper_hash(paper)

            # Skip if already seen
            if paper_id in self.seen_papers:
                continue

            # Skip if not quantum-related (when filter is enabled)
            if filter_quantum and not self._is_quantum_related(paper):
                continue

            self.seen_papers.add(paper_id)
            unique_papers.append(paper)

        return unique_papers

    def _parse_arxiv_response(self, xml_data: bytes) -> List[Dict]:
        """Parse ArXiv XML response"""
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        root = ET.fromstring(xml_data)

        papers = []
        for entry in root.findall('atom:entry', ns):
            try:
                # Extract basic information
                title = entry.find('atom:title', ns)
                summary = entry.find('atom:summary', ns)
                id_elem = entry.find('atom:id', ns)
                published = entry.find('atom:published', ns)
                updated = entry.find('atom:updated', ns)

                # Get authors
                authors = []
                for author in entry.findall('atom:author', ns):
                    name = author.find('atom:name', ns)
                    if name is not None:
                        authors.append(name.text)

                # Get categories
                categories = []
                for category in entry.findall('atom:category', ns):
                    term = category.get('term')
                    if term:
                        categories.append(term)

                # Get arXiv ID
                arxiv_id = id_elem.text.split('/abs/')[-1] if id_elem is not None else 'N/A'

                # Create paper dictionary
                paper = {
                    'arxiv_id': arxiv_id,
                    'title': title.text.strip().replace('\n', ' ') if title is not None else 'N/A',
                    'abstract': summary.text.strip().replace('\n', ' ') if summary is not None else 'N/A',
                    'authors': authors,
                    'categories': categories,
                    'published': published.text if published is not None else 'N/A',
                    'updated': updated.text if updated is not None else 'N/A',
                    'pdf_link': f'https://arxiv.org/pdf/{arxiv_id}.pdf',
                    'abstract_link': f'https://arxiv.org/abs/{arxiv_id}',
                    'fetched_at': datetime.now().isoformat()
                }
                papers.append(paper)
            except Exception as e:
                print(f"Error parsing entry: {e}")
                continue

        return papers

    def search_multiple_categories(self,
                                   keywords: Optional[str] = None,
                                   max_results_per_category: int = 20) -> List[Dict]:
        """
        Search across all quantum-related categories

        Args:
            keywords: Optional search keywords
            max_results_per_category: Max results per category

        Returns:
            Combined list of unique papers across all categories
        """
        all_papers = []

        for category in self.QUANTUM_CATEGORIES:
            papers = self.search(
                keywords=keywords,
                category=category,
                max_results=max_results_per_category,
                filter_quantum=True
            )
            all_papers.extend(papers)

        return all_papers

    def get_paper_by_id(self, arxiv_id: str) -> Optional[Dict]:
        """Get a specific paper by its arXiv ID"""
        params = {
            'id_list': arxiv_id,
            'max_results': 1
        }
        url = self.base_url + urllib.parse.urlencode(params)

        try:
            with urllib.request.urlopen(url) as response:
                xml_data = response.read()
        except Exception as e:
            print(f"Error fetching paper {arxiv_id}: {e}")
            return None

        papers = self._parse_arxiv_response(xml_data)
        return papers[0] if papers else None

    def reset_seen_papers(self):
        """Clear the deduplication cache"""
        self.seen_papers.clear()
