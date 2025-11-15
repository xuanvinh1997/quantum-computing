#!/usr/bin/env python3
"""
Paper summarization using openai/gpt-oss-20b model
Focuses on extracting methodology from quantum computing papers
"""

import os
from typing import Dict, Optional
from openai import OpenAI


class PaperSummarizer:
    """Summarize research papers focusing on methodology"""

    def __init__(self,
                 api_key: str,
                 base_url: str,
                 model_name: str = "openai/gpt-oss-20b"):
        """
        Initialize summarizer

        Args:
            api_key: API key for the service
            base_url: Base URL for the OpenAI-compatible endpoint
            model_name: Model name to use for summarization
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model_name = model_name

    def _create_methodology_prompt(self, paper_text: str, paper_metadata: Dict) -> str:
        """
        Create prompt for methodology extraction

        Args:
            paper_text: Full text of the paper
            paper_metadata: Paper metadata (title, abstract, etc.)

        Returns:
            Formatted prompt
        """
        prompt = f"""You are an expert in quantum computing and quantum physics research.
Your task is to analyze the following research paper and extract a detailed summary of its methodology.

**Paper Title:** {paper_metadata.get('title', 'N/A')}

**Abstract:**
{paper_metadata.get('abstract', 'N/A')}

**Full Paper Text:**
{paper_text[:15000]}  # Limit text to avoid token limits

---

Please provide a comprehensive summary of the paper's methodology in the following structure:

1. **Research Objective**: What problem is the paper trying to solve?

2. **Methodology Overview**: What approach does the paper take? (e.g., theoretical analysis, experimental setup, algorithmic approach, simulation)

3. **Key Techniques**: What specific techniques, algorithms, or methods are used?
   - Mathematical frameworks
   - Quantum circuits or algorithms
   - Classical preprocessing/postprocessing
   - Optimization methods

4. **Implementation Details**:
   - Hardware/software platforms used
   - Quantum gate decompositions
   - Parameter settings
   - Computational resources

5. **Evaluation Approach**: How do they validate their results?
   - Benchmarks
   - Baselines
   - Metrics used
   - Experimental setup

6. **Key Results**: What are the main quantitative findings?

7. **Limitations**: What limitations do the authors acknowledge?

8. **Reproducibility**: Based on the paper, how reproducible is this work?
   - Are code/data available?
   - Are parameters clearly specified?
   - Are there ambiguities?

Please be specific and technical. Extract concrete details like parameter values, equations, algorithm steps, etc.
"""
        return prompt

    def summarize_methodology(self,
                            paper_text: str,
                            paper_metadata: Dict,
                            max_tokens: int = 2048,
                            temperature: float = 0.3) -> str:
        """
        Extract methodology summary from paper

        Args:
            paper_text: Full text extracted from PDF
            paper_metadata: Paper metadata dictionary
            max_tokens: Maximum tokens for response
            temperature: Sampling temperature

        Returns:
            Methodology summary
        """
        try:
            prompt = self._create_methodology_prompt(paper_text, paper_metadata)

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert research analyst specializing in quantum computing and quantum physics. You extract detailed, technical methodology summaries from research papers."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error generating summary: {e}")
            return f"Error: Unable to generate summary - {str(e)}"

    def extract_key_contributions(self,
                                  paper_text: str,
                                  paper_metadata: Dict) -> str:
        """
        Extract key contributions from paper

        Args:
            paper_text: Full text extracted from PDF
            paper_metadata: Paper metadata dictionary

        Returns:
            Key contributions summary
        """
        try:
            prompt = f"""Based on the following quantum computing research paper, extract the 3-5 key contributions in bullet points.

**Title:** {paper_metadata.get('title', 'N/A')}

**Abstract:**
{paper_metadata.get('abstract', 'N/A')}

**Paper Text:**
{paper_text[:10000]}

Provide a concise list of key contributions:"""

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=512,
                temperature=0.2
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error extracting contributions: {e}")
            return "Error: Unable to extract contributions"

    def check_quantum_relevance(self,
                               paper_metadata: Dict,
                               threshold: float = 0.6) -> Dict[str, any]:
        """
        Check if paper is relevant to quantum computing/physics

        Args:
            paper_metadata: Paper metadata dictionary
            threshold: Relevance threshold (0-1)

        Returns:
            Dictionary with relevance score and reasoning
        """
        try:
            prompt = f"""Analyze the following research paper metadata and determine its relevance to quantum computing or quantum physics.

**Title:** {paper_metadata.get('title', 'N/A')}

**Abstract:**
{paper_metadata.get('abstract', 'N/A')}

**Categories:** {', '.join(paper_metadata.get('categories', []))}

Provide:
1. A relevance score from 0.0 to 1.0 (where 1.0 is highly relevant to quantum computing/physics)
2. A brief explanation of why this score was assigned
3. Key quantum-related topics covered (if any)

Format your response as:
SCORE: [0.0-1.0]
EXPLANATION: [Your explanation]
TOPICS: [Comma-separated list of quantum topics, or "None"]
"""

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=256,
                temperature=0.1
            )

            content = response.choices[0].message.content.strip()

            # Parse response
            lines = content.split('\n')
            score = 0.0
            explanation = ""
            topics = []

            for line in lines:
                if line.startswith('SCORE:'):
                    try:
                        score = float(line.split(':')[1].strip())
                    except:
                        score = 0.0
                elif line.startswith('EXPLANATION:'):
                    explanation = line.split(':', 1)[1].strip()
                elif line.startswith('TOPICS:'):
                    topics_str = line.split(':', 1)[1].strip()
                    if topics_str.lower() != 'none':
                        topics = [t.strip() for t in topics_str.split(',')]

            return {
                'is_relevant': score >= threshold,
                'relevance_score': score,
                'explanation': explanation,
                'topics': topics,
                'raw_response': content
            }

        except Exception as e:
            print(f"Error checking relevance: {e}")
            return {
                'is_relevant': False,
                'relevance_score': 0.0,
                'explanation': f"Error: {str(e)}",
                'topics': [],
                'raw_response': ''
            }
