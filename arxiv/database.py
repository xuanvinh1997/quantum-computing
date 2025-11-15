#!/usr/bin/env python3
"""
SQLite database for storing ArXiv papers and summaries
"""

import sqlite3
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path


class PaperDatabase:
    """SQLite database manager for ArXiv papers"""

    def __init__(self, db_path: str = "arxiv_papers.db"):
        """
        Initialize database connection

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self._connect()
        self._create_tables()

    def _connect(self):
        """Establish database connection"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name

    def _create_tables(self):
        """Create database tables if they don't exist"""
        cursor = self.conn.cursor()

        # Papers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                arxiv_id TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                abstract TEXT,
                authors TEXT,  -- JSON array
                categories TEXT,  -- JSON array
                published DATE,
                updated DATE,
                pdf_link TEXT,
                abstract_link TEXT,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed BOOLEAN DEFAULT 0,
                is_quantum_relevant BOOLEAN DEFAULT 1,
                relevance_score REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Summaries table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id INTEGER NOT NULL,
                methodology_summary TEXT,
                key_contributions TEXT,
                extracted_text TEXT,  -- Full OCR text
                summary_created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (paper_id) REFERENCES papers (id) ON DELETE CASCADE
            )
        """)

        # Search history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                category TEXT,
                num_results INTEGER,
                search_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_arxiv_id ON papers(arxiv_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_published ON papers(published)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_processed ON papers(processed)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_paper_id ON summaries(paper_id)
        """)

        self.conn.commit()

    def insert_paper(self, paper: Dict) -> Optional[int]:
        """
        Insert a paper into database

        Args:
            paper: Paper dictionary from ArXiv search

        Returns:
            Paper ID if successful, None if duplicate
        """
        cursor = self.conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO papers (
                    arxiv_id, title, abstract, authors, categories,
                    published, updated, pdf_link, abstract_link,
                    is_quantum_relevant, relevance_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                paper['arxiv_id'],
                paper['title'],
                paper['abstract'],
                json.dumps(paper.get('authors', [])),
                json.dumps(paper.get('categories', [])),
                paper.get('published', ''),
                paper.get('updated', ''),
                paper['pdf_link'],
                paper['abstract_link'],
                paper.get('is_quantum_relevant', True),
                paper.get('relevance_score', 0.0)
            ))

            self.conn.commit()
            return cursor.lastrowid

        except sqlite3.IntegrityError:
            # Paper already exists
            return None
        except Exception as e:
            print(f"Error inserting paper: {e}")
            self.conn.rollback()
            return None

    def insert_summary(self,
                      paper_id: int,
                      methodology_summary: str,
                      key_contributions: str,
                      extracted_text: Optional[str] = None) -> Optional[int]:
        """
        Insert summary for a paper

        Args:
            paper_id: ID of the paper
            methodology_summary: Methodology summary text
            key_contributions: Key contributions text
            extracted_text: Full extracted OCR text

        Returns:
            Summary ID if successful, None otherwise
        """
        cursor = self.conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO summaries (
                    paper_id, methodology_summary, key_contributions, extracted_text
                ) VALUES (?, ?, ?, ?)
            """, (paper_id, methodology_summary, key_contributions, extracted_text))

            # Mark paper as processed
            cursor.execute("""
                UPDATE papers SET processed = 1 WHERE id = ?
            """, (paper_id,))

            self.conn.commit()
            return cursor.lastrowid

        except Exception as e:
            print(f"Error inserting summary: {e}")
            self.conn.rollback()
            return None

    def get_paper_by_arxiv_id(self, arxiv_id: str) -> Optional[Dict]:
        """Get paper by ArXiv ID"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,))
        row = cursor.fetchone()

        if row:
            return dict(row)
        return None

    def get_paper_with_summary(self, paper_id: int) -> Optional[Dict]:
        """
        Get paper with its summary

        Args:
            paper_id: Paper ID

        Returns:
            Dictionary with paper and summary data
        """
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT
                p.*,
                s.methodology_summary,
                s.key_contributions,
                s.summary_created_at
            FROM papers p
            LEFT JOIN summaries s ON p.id = s.paper_id
            WHERE p.id = ?
        """, (paper_id,))

        row = cursor.fetchone()
        if row:
            result = dict(row)
            # Parse JSON fields
            result['authors'] = json.loads(result['authors']) if result['authors'] else []
            result['categories'] = json.loads(result['categories']) if result['categories'] else []
            return result
        return None

    def get_unprocessed_papers(self, limit: int = 10) -> List[Dict]:
        """
        Get papers that haven't been processed yet

        Args:
            limit: Maximum number of papers to return

        Returns:
            List of paper dictionaries
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM papers
            WHERE processed = 0
            ORDER BY published DESC
            LIMIT ?
        """, (limit,))

        papers = []
        for row in cursor.fetchall():
            paper = dict(row)
            paper['authors'] = json.loads(paper['authors']) if paper['authors'] else []
            paper['categories'] = json.loads(paper['categories']) if paper['categories'] else []
            papers.append(paper)

        return papers

    def search_papers(self,
                     query: Optional[str] = None,
                     category: Optional[str] = None,
                     processed_only: bool = False,
                     limit: int = 50) -> List[Dict]:
        """
        Search papers in database

        Args:
            query: Text search in title/abstract
            category: Filter by category
            processed_only: Only return processed papers
            limit: Maximum results

        Returns:
            List of papers
        """
        cursor = self.conn.cursor()

        sql = "SELECT * FROM papers WHERE 1=1"
        params = []

        if query:
            sql += " AND (title LIKE ? OR abstract LIKE ?)"
            params.extend([f"%{query}%", f"%{query}%"])

        if category:
            sql += " AND categories LIKE ?"
            params.append(f"%{category}%")

        if processed_only:
            sql += " AND processed = 1"

        sql += " ORDER BY published DESC LIMIT ?"
        params.append(limit)

        cursor.execute(sql, params)

        papers = []
        for row in cursor.fetchall():
            paper = dict(row)
            paper['authors'] = json.loads(paper['authors']) if paper['authors'] else []
            paper['categories'] = json.loads(paper['categories']) if paper['categories'] else []
            papers.append(paper)

        return papers

    def get_statistics(self) -> Dict:
        """Get database statistics"""
        cursor = self.conn.cursor()

        stats = {}

        # Total papers
        cursor.execute("SELECT COUNT(*) FROM papers")
        stats['total_papers'] = cursor.fetchone()[0]

        # Processed papers
        cursor.execute("SELECT COUNT(*) FROM papers WHERE processed = 1")
        stats['processed_papers'] = cursor.fetchone()[0]

        # Unprocessed papers
        stats['unprocessed_papers'] = stats['total_papers'] - stats['processed_papers']

        # Papers by category
        cursor.execute("SELECT categories, COUNT(*) FROM papers GROUP BY categories")
        stats['by_category'] = dict(cursor.fetchall())

        # Recent papers
        cursor.execute("""
            SELECT COUNT(*) FROM papers
            WHERE DATE(published) >= DATE('now', '-7 days')
        """)
        stats['papers_last_7_days'] = cursor.fetchone()[0]

        return stats

    def log_search(self, query: str, category: Optional[str], num_results: int):
        """Log a search query"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO search_history (query, category, num_results)
            VALUES (?, ?, ?)
        """, (query, category, num_results))
        self.conn.commit()

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
