#!/usr/bin/env python3
"""
Configuration management for ArXiv Paper Tool
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration settings"""

    # API Settings - OCR
    OCR_API_KEY = os.getenv("OCR_API_KEY", "")
    OCR_BASE_URL = os.getenv("OCR_BASE_URL", "https://api.example.com/v1")
    OCR_MODEL = os.getenv("OCR_MODEL", "nanonets/nanonets-ocr2-3b")

    # API Settings - Summarization
    SUMMARY_API_KEY = os.getenv("SUMMARY_API_KEY", "")
    SUMMARY_BASE_URL = os.getenv("SUMMARY_BASE_URL", "https://api.example.com/v1")
    SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "openai/gpt-oss-20b")

    # Database Settings
    DATABASE_PATH = os.getenv("DATABASE_PATH", "arxiv_papers.db")

    # Output Settings
    MARKDOWN_OUTPUT_DIR = os.getenv("MARKDOWN_OUTPUT_DIR", "papers_output")

    # Processing Settings
    DEFAULT_MAX_PAGES = int(os.getenv("DEFAULT_MAX_PAGES", "20"))
    DEFAULT_OCR_DPI = int(os.getenv("DEFAULT_OCR_DPI", "200"))

    # Search Settings
    DEFAULT_MAX_RESULTS = int(os.getenv("DEFAULT_MAX_RESULTS", "50"))
    FILTER_QUANTUM_ONLY = os.getenv("FILTER_QUANTUM_ONLY", "True").lower() == "true"

    @classmethod
    def validate(cls) -> list:
        """
        Validate configuration

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if not cls.OCR_API_KEY:
            errors.append("OCR_API_KEY not set")

        if not cls.SUMMARY_API_KEY:
            errors.append("SUMMARY_API_KEY not set")

        if not cls.OCR_BASE_URL:
            errors.append("OCR_BASE_URL not set")

        if not cls.SUMMARY_BASE_URL:
            errors.append("SUMMARY_BASE_URL not set")

        return errors

    @classmethod
    def print_config(cls):
        """Print current configuration (hiding sensitive data)"""
        print("=" * 60)
        print("ArXiv Paper Tool Configuration")
        print("=" * 60)
        print(f"OCR Model: {cls.OCR_MODEL}")
        print(f"OCR Base URL: {cls.OCR_BASE_URL}")
        print(f"OCR API Key: {'*' * 10 if cls.OCR_API_KEY else 'NOT SET'}")
        print()
        print(f"Summary Model: {cls.SUMMARY_MODEL}")
        print(f"Summary Base URL: {cls.SUMMARY_BASE_URL}")
        print(f"Summary API Key: {'*' * 10 if cls.SUMMARY_API_KEY else 'NOT SET'}")
        print()
        print(f"Database Path: {cls.DATABASE_PATH}")
        print(f"Markdown Output Dir: {cls.MARKDOWN_OUTPUT_DIR}")
        print()
        print(f"Default Max Pages: {cls.DEFAULT_MAX_PAGES}")
        print(f"Default OCR DPI: {cls.DEFAULT_OCR_DPI}")
        print(f"Default Max Results: {cls.DEFAULT_MAX_RESULTS}")
        print(f"Filter Quantum Only: {cls.FILTER_QUANTUM_ONLY}")
        print("=" * 60)
