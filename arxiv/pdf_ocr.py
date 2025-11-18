#!/usr/bin/env python3
"""
PDF OCR using nanonets-ocr2-3b model via OpenAI-compatible endpoint
"""

import os
import io
import base64
import tempfile
from typing import List, Optional, Dict
from pathlib import Path
import requests
from PIL import Image
from pdf2image import convert_from_path, convert_from_bytes
from openai import OpenAI


class PDFOCRProcessor:
    """PDF OCR using nanonets-ocr2-3b model"""

    def __init__(
        self, api_key: str, base_url: str, model_name: str = "nanonets/nanonets-ocr2-3b"
    ):
        """
        Initialize OCR processor

        Args:
            api_key: API key for the service
            base_url: Base URL for the OpenAI-compatible endpoint
            model_name: Model name to use for OCR
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def _extract_text_from_image(self, image: Image.Image) -> str:
        """
        Extract text from a single image using OCR model

        Args:
            image: PIL Image object

        Returns:
            Extracted text
        """
        try:
            # Convert image to base64
            image_base64 = self._image_to_base64(image)

            # Call vision model
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful OCR assistant."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                },
                            },
                        ],
                    },
                ],
                max_tokens=2048,
                temperature=0.0,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error extracting text from image: {e}")
            return ""

    def download_pdf(self, pdf_url: str, output_path: Optional[str] = None) -> str:
        """
        Download PDF from URL

        Args:
            pdf_url: URL to PDF file
            output_path: Optional path to save PDF

        Returns:
            Path to downloaded PDF
        """
        try:
            response = requests.get(pdf_url, timeout=60)
            response.raise_for_status()

            if output_path is None:
                # Create temp file
                fd, output_path = tempfile.mkstemp(suffix=".pdf")
                os.close(fd)

            with open(output_path, "wb") as f:
                f.write(response.content)

            return output_path

        except Exception as e:
            print(f"Error downloading PDF from {pdf_url}: {e}")
            raise

    def pdf_to_images(
        self, pdf_path: str, dpi: int = 200, max_pages: Optional[int] = None
    ) -> List[Image.Image]:
        """
        Convert PDF to images

        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for conversion
            max_pages: Maximum number of pages to process

        Returns:
            List of PIL Image objects
        """
        try:
            images = convert_from_path(
                pdf_path, dpi=dpi, first_page=1, last_page=max_pages
            )
            return images
        except Exception as e:
            print(f"Error converting PDF to images: {e}")
            return []

    def extract_text_from_pdf(
        self, pdf_path: str, max_pages: Optional[int] = 20, dpi: int = 200
    ) -> Dict[str, str]:
        """
        Extract text from PDF using OCR

        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum number of pages to process
            dpi: Resolution for image conversion

        Returns:
            Dictionary with page numbers and extracted text
        """
        print(f"Converting PDF to images (max {max_pages} pages)...")
        images = self.pdf_to_images(pdf_path, dpi=dpi, max_pages=max_pages)

        if not images:
            return {}

        extracted_text = {}
        total_pages = len(images)

        print(f"Processing {total_pages} pages with OCR...")
        for i, image in enumerate(images, 1):
            print(f"  Processing page {i}/{total_pages}...")
            text = self._extract_text_from_image(image)
            extracted_text[f"page_{i}"] = text

        return extracted_text

    def extract_text_from_url(
        self, pdf_url: str, max_pages: Optional[int] = 20, cleanup: bool = True
    ) -> Dict[str, str]:
        """
        Download PDF from URL and extract text

        Args:
            pdf_url: URL to PDF file
            max_pages: Maximum number of pages to process
            cleanup: Delete downloaded PDF after processing

        Returns:
            Dictionary with page numbers and extracted text
        """
        print(f"Downloading PDF from {pdf_url}...")
        pdf_path = self.download_pdf(pdf_url)

        try:
            extracted_text = self.extract_text_from_pdf(pdf_path, max_pages=max_pages)
            return extracted_text
        finally:
            if cleanup and os.path.exists(pdf_path):
                os.remove(pdf_path)

    def get_full_text(self, extracted_text: Dict[str, str]) -> str:
        """
        Combine all pages into a single text string

        Args:
            extracted_text: Dictionary from extract_text_from_pdf

        Returns:
            Combined text from all pages
        """
        pages = sorted(extracted_text.keys(), key=lambda x: int(x.split("_")[1]))
        return "\n\n--- Page Break ---\n\n".join(extracted_text[page] for page in pages)
