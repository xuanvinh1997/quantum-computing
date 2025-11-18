#!/usr/bin/env python3
"""
Deep Research Tool - Prompt-based research engine for quantum computing papers
Enables complex research queries, synthesis, and analysis across paper collections
"""

from typing import Dict, List, Optional, Tuple
from openai import OpenAI
from database import PaperDatabase
import json


class DeepResearchEngine:
    """
    Advanced research tool that uses LLM to perform deep analysis on paper collections
    """

    def __init__(self,
                 api_key: str,
                 base_url: str,
                 database: PaperDatabase,
                 model_name: str = "openai/gpt-oss-20b"):
        """
        Initialize deep research engine

        Args:
            api_key: API key for the LLM service
            base_url: Base URL for OpenAI-compatible endpoint
            database: PaperDatabase instance
            model_name: Model name for research queries
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model_name = model_name
        self.database = database

    def _gather_papers_context(self,
                               query: Optional[str] = None,
                               category: Optional[str] = None,
                               processed_only: bool = True,
                               limit: int = 50) -> List[Dict]:
        """
        Gather relevant papers from database

        Args:
            query: Optional search query
            category: Optional category filter
            processed_only: Only include processed papers
            limit: Maximum papers to retrieve

        Returns:
            List of paper dictionaries with summaries
        """
        papers = self.database.search_papers(
            query=query,
            category=category,
            processed_only=processed_only,
            limit=limit
        )

        # Enrich with summaries
        enriched_papers = []
        for paper in papers:
            full_data = self.database.get_paper_with_summary(paper['id'])
            if full_data and full_data.get('methodology_summary'):
                enriched_papers.append(full_data)

        return enriched_papers

    def _format_paper_for_prompt(self, paper: Dict) -> str:
        """
        Format a single paper for inclusion in research prompts

        Args:
            paper: Paper dictionary with metadata and summary

        Returns:
            Formatted paper text
        """
        authors = ", ".join(paper.get('authors', [])[:3])
        if len(paper.get('authors', [])) > 3:
            authors += " et al."

        formatted = f"""
---
**Paper ID:** {paper['arxiv_id']}
**Title:** {paper['title']}
**Authors:** {authors}
**Published:** {paper.get('published', 'N/A')[:10]}
**Categories:** {', '.join(paper.get('categories', []))}

**Abstract:**
{paper.get('abstract', 'N/A')}

**Methodology Summary:**
{paper.get('methodology_summary', 'Not available')}

**Key Contributions:**
{paper.get('key_contributions', 'Not available')}
---
"""
        return formatted

    def research_query(self,
                      research_question: str,
                      search_query: Optional[str] = None,
                      category: Optional[str] = None,
                      max_papers: int = 20,
                      temperature: float = 0.4,
                      max_tokens: int = 4096) -> Dict:
        """
        Perform a deep research query on the paper collection

        Args:
            research_question: The research question to answer
            search_query: Optional filter for relevant papers
            category: Optional category filter
            max_papers: Maximum papers to analyze
            temperature: LLM temperature
            max_tokens: Maximum tokens for response

        Returns:
            Dictionary with research findings
        """
        print(f"\n{'='*70}")
        print("DEEP RESEARCH QUERY")
        print(f"{'='*70}")
        print(f"Question: {research_question}")
        print(f"\nGathering relevant papers...")

        # Gather papers
        papers = self._gather_papers_context(
            query=search_query,
            category=category,
            processed_only=True,
            limit=max_papers
        )

        if not papers:
            return {
                'success': False,
                'error': 'No processed papers found matching criteria',
                'answer': None,
                'papers_analyzed': 0
            }

        print(f"Found {len(papers)} relevant papers")
        print(f"Analyzing papers to answer research question...\n")

        # Format papers for context
        papers_context = "\n".join([
            self._format_paper_for_prompt(p) for p in papers[:max_papers]
        ])

        # Create research prompt
        prompt = f"""You are an expert quantum computing researcher conducting a comprehensive literature review.

You have access to {len(papers)} research papers on quantum computing and related topics.

**Research Question:**
{research_question}

**Available Papers:**
{papers_context}

**Your Task:**
Based on the papers provided, please provide a comprehensive answer to the research question. Your answer should:

1. **Direct Answer**: Provide a clear, direct answer to the research question
2. **Evidence from Literature**: Cite specific papers and their findings that support your answer
3. **Synthesis**: Synthesize insights across multiple papers
4. **Methodologies**: Discuss relevant methodologies used in the papers
5. **Key Findings**: Highlight key quantitative results and achievements
6. **Gaps & Opportunities**: Identify any gaps in the current research or future research opportunities
7. **Contradictions**: Note any contradictions or debates in the literature
8. **Citations**: Reference papers by their ArXiv ID (e.g., [2511.10646v1])

Please be thorough, technical, and cite specific papers to support your claims.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert quantum computing researcher with deep knowledge of quantum algorithms, quantum hardware, quantum machine learning, and quantum information theory. You provide comprehensive, well-cited research analysis."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )

            answer = response.choices[0].message.content.strip()

            return {
                'success': True,
                'research_question': research_question,
                'answer': answer,
                'papers_analyzed': len(papers),
                'paper_ids': [p['arxiv_id'] for p in papers],
                'model_used': self.model_name,
                'error': None
            }

        except Exception as e:
            return {
                'success': False,
                'error': f"Error during research query: {str(e)}",
                'answer': None,
                'papers_analyzed': len(papers)
            }

    def comparative_analysis(self,
                           topic: str,
                           aspects: List[str],
                           search_query: Optional[str] = None,
                           max_papers: int = 15,
                           temperature: float = 0.3,
                           max_tokens: int = 3072) -> Dict:
        """
        Perform comparative analysis across papers on a specific topic

        Args:
            topic: The topic to analyze
            aspects: List of aspects to compare (e.g., ["methodology", "performance", "scalability"])
            search_query: Optional search filter
            max_papers: Maximum papers to analyze
            temperature: LLM temperature
            max_tokens: Maximum response tokens

        Returns:
            Dictionary with comparative analysis
        """
        print(f"\n{'='*70}")
        print("COMPARATIVE ANALYSIS")
        print(f"{'='*70}")
        print(f"Topic: {topic}")
        print(f"Comparing: {', '.join(aspects)}")
        print(f"\nGathering papers...")

        # Gather papers
        papers = self._gather_papers_context(
            query=search_query or topic,
            processed_only=True,
            limit=max_papers
        )

        if not papers:
            return {
                'success': False,
                'error': 'No papers found',
                'analysis': None
            }

        print(f"Found {len(papers)} papers")
        print(f"Performing comparative analysis...\n")

        # Format papers
        papers_context = "\n".join([
            self._format_paper_for_prompt(p) for p in papers
        ])

        aspects_list = "\n".join([f"- {aspect}" for aspect in aspects])

        prompt = f"""You are conducting a comparative analysis of quantum computing research papers.

**Topic:** {topic}

**Aspects to Compare:**
{aspects_list}

**Papers for Analysis:**
{papers_context}

**Your Task:**
Create a comprehensive comparative analysis table and discussion comparing these papers across the specified aspects.

1. **Comparison Table**: Create a structured comparison showing how each paper approaches the topic
2. **Key Differences**: Highlight the main differences between approaches
3. **Key Similarities**: Identify common themes and techniques
4. **Performance Comparison**: If applicable, compare quantitative results
5. **Methodology Comparison**: Compare the methodologies used
6. **Strengths & Weaknesses**: Discuss the strengths and weaknesses of each approach
7. **Recommendations**: Which papers/approaches are most promising for different use cases?

Cite papers by ArXiv ID throughout your analysis.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert research analyst specializing in quantum computing. You excel at comparative analysis and identifying patterns across research papers."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )

            analysis = response.choices[0].message.content.strip()

            return {
                'success': True,
                'topic': topic,
                'aspects': aspects,
                'analysis': analysis,
                'papers_analyzed': len(papers),
                'paper_ids': [p['arxiv_id'] for p in papers]
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'analysis': None
            }

    def trend_analysis(self,
                      time_period: str = "recent",
                      focus_area: Optional[str] = None,
                      max_papers: int = 30,
                      temperature: float = 0.4,
                      max_tokens: int = 3072) -> Dict:
        """
        Analyze research trends across papers

        Args:
            time_period: Time period to analyze (e.g., "recent", "last_year")
            focus_area: Optional focus area (e.g., "quantum machine learning")
            max_papers: Maximum papers to analyze
            temperature: LLM temperature
            max_tokens: Maximum response tokens

        Returns:
            Dictionary with trend analysis
        """
        print(f"\n{'='*70}")
        print("TREND ANALYSIS")
        print(f"{'='*70}")
        print(f"Time Period: {time_period}")
        if focus_area:
            print(f"Focus Area: {focus_area}")
        print(f"\nGathering papers...")

        # Gather papers
        papers = self._gather_papers_context(
            query=focus_area,
            processed_only=True,
            limit=max_papers
        )

        if not papers:
            return {
                'success': False,
                'error': 'No papers found',
                'analysis': None
            }

        # Sort by publication date
        papers.sort(key=lambda x: x.get('published', ''), reverse=True)

        print(f"Found {len(papers)} papers")
        print(f"Analyzing trends...\n")

        # Format papers
        papers_context = "\n".join([
            self._format_paper_for_prompt(p) for p in papers
        ])

        prompt = f"""You are analyzing research trends in quantum computing.

**Time Period:** {time_period}
{f"**Focus Area:** {focus_area}" if focus_area else ""}

**Papers (sorted by publication date):**
{papers_context}

**Your Task:**
Analyze the research trends across these papers and provide insights on:

1. **Emerging Topics**: What new topics or approaches are emerging?
2. **Evolving Methodologies**: How are methodologies evolving over time?
3. **Performance Improvements**: What quantitative improvements are being achieved?
4. **Popular Research Directions**: What areas are getting the most attention?
5. **Gaps & Opportunities**: What gaps exist in current research?
6. **Future Predictions**: Based on these trends, what do you predict for future research?
7. **Key Breakthroughs**: What are the most significant breakthroughs?
8. **Convergence**: Are there areas where different approaches are converging?

Provide specific examples and cite papers by ArXiv ID.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert research analyst with deep knowledge of quantum computing trends, capable of identifying patterns and predicting future directions."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )

            analysis = response.choices[0].message.content.strip()

            return {
                'success': True,
                'time_period': time_period,
                'focus_area': focus_area,
                'analysis': analysis,
                'papers_analyzed': len(papers),
                'date_range': {
                    'earliest': papers[-1].get('published', 'N/A')[:10] if papers else None,
                    'latest': papers[0].get('published', 'N/A')[:10] if papers else None
                }
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'analysis': None
            }

    def find_paper_connections(self,
                              arxiv_id: str,
                              max_related: int = 10,
                              temperature: float = 0.3,
                              max_tokens: int = 2048) -> Dict:
        """
        Find connections between a specific paper and other papers in the collection

        Args:
            arxiv_id: ArXiv ID of the source paper
            max_related: Maximum related papers to find
            temperature: LLM temperature
            max_tokens: Maximum response tokens

        Returns:
            Dictionary with paper connections
        """
        print(f"\n{'='*70}")
        print("FIND PAPER CONNECTIONS")
        print(f"{'='*70}")
        print(f"Source Paper: {arxiv_id}")
        print(f"\nRetrieving paper data...")

        # Get source paper
        source_paper = self.database.get_paper_by_arxiv_id(arxiv_id)
        if not source_paper:
            return {
                'success': False,
                'error': f'Paper {arxiv_id} not found in database'
            }

        source_full = self.database.get_paper_with_summary(source_paper['id'])
        if not source_full or not source_full.get('methodology_summary'):
            return {
                'success': False,
                'error': f'Paper {arxiv_id} has not been processed yet'
            }

        # Get all other processed papers
        all_papers = self._gather_papers_context(processed_only=True, limit=100)
        # Remove source paper
        other_papers = [p for p in all_papers if p['arxiv_id'] != arxiv_id][:max_related]

        print(f"Analyzing connections with {len(other_papers)} other papers...\n")

        # Format papers
        source_context = self._format_paper_for_prompt(source_full)
        others_context = "\n".join([
            self._format_paper_for_prompt(p) for p in other_papers
        ])

        prompt = f"""You are analyzing connections between quantum computing research papers.

**Source Paper:**
{source_context}

**Other Papers in Collection:**
{others_context}

**Your Task:**
Identify and explain connections between the source paper and other papers. For each connection, provide:

1. **Related Papers**: List papers that relate to the source paper
2. **Connection Type**:
   - Builds upon / extends
   - Uses similar methodology
   - Addresses related problem
   - Provides complementary approach
   - Contradicts or challenges
   - Cites or is cited by (if evident)
3. **Strength of Connection**: (Strong / Moderate / Weak)
4. **Explanation**: Explain the specific connection
5. **Potential Synergies**: How could combining insights from these papers be valuable?
6. **Research Gaps**: Are there gaps between these papers that represent opportunities?

Format as a structured analysis with clear connections.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at identifying connections between research papers, understanding how papers build upon each other, and finding synergies across research."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )

            analysis = response.choices[0].message.content.strip()

            return {
                'success': True,
                'source_paper': arxiv_id,
                'source_title': source_full['title'],
                'analysis': analysis,
                'papers_analyzed': len(other_papers),
                'related_paper_ids': [p['arxiv_id'] for p in other_papers]
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'analysis': None
            }

    def custom_prompt_research(self,
                             custom_prompt: str,
                             search_query: Optional[str] = None,
                             max_papers: int = 20,
                             temperature: float = 0.5,
                             max_tokens: int = 4096) -> Dict:
        """
        Execute a custom research prompt on the paper collection

        Args:
            custom_prompt: Custom research prompt/question
            search_query: Optional filter for papers
            max_papers: Maximum papers to include
            temperature: LLM temperature
            max_tokens: Maximum response tokens

        Returns:
            Dictionary with research results
        """
        print(f"\n{'='*70}")
        print("CUSTOM RESEARCH PROMPT")
        print(f"{'='*70}")
        print(f"Prompt: {custom_prompt[:100]}...")
        print(f"\nGathering papers...")

        # Gather papers
        papers = self._gather_papers_context(
            query=search_query,
            processed_only=True,
            limit=max_papers
        )

        if not papers:
            return {
                'success': False,
                'error': 'No papers found',
                'result': None
            }

        print(f"Found {len(papers)} papers")
        print(f"Executing custom research prompt...\n")

        # Format papers
        papers_context = "\n".join([
            self._format_paper_for_prompt(p) for p in papers
        ])

        full_prompt = f"""You are a quantum computing research expert.

**Available Research Papers:**
{papers_context}

**Research Task:**
{custom_prompt}

Please provide a comprehensive, well-researched response based on the papers provided. Cite specific papers by ArXiv ID to support your analysis.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert quantum computing researcher capable of performing various research tasks and analysis on scientific papers."
                    },
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )

            result = response.choices[0].message.content.strip()

            return {
                'success': True,
                'custom_prompt': custom_prompt,
                'result': result,
                'papers_analyzed': len(papers),
                'paper_ids': [p['arxiv_id'] for p in papers]
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'result': None
            }

    def recursive_research(self,
                          initial_question: str,
                          max_depth: int = 3,
                          max_papers_per_iteration: int = 15,
                          temperature: float = 0.4,
                          max_tokens: int = 4096) -> Dict:
        """
        Perform recursive deep research with automatic follow-up questions

        Args:
            initial_question: The starting research question
            max_depth: Maximum recursion depth (number of follow-up iterations)
            max_papers_per_iteration: Papers to analyze per iteration
            temperature: LLM temperature
            max_tokens: Maximum tokens per response

        Returns:
            Dictionary with comprehensive recursive research findings
        """
        print(f"\n{'='*70}")
        print("RECURSIVE DEEP RESEARCH")
        print(f"{'='*70}")
        print(f"Initial Question: {initial_question}")
        print(f"Max Depth: {max_depth}")

        # Store all research iterations
        research_iterations = []
        all_papers_analyzed = set()

        # Current question to investigate
        current_question = initial_question

        for depth in range(max_depth):
            print(f"\n--- Iteration {depth + 1}/{max_depth} ---")
            print(f"Question: {current_question}")

            # Perform research query
            result = self.research_query(
                research_question=current_question,
                max_papers=max_papers_per_iteration,
                temperature=temperature,
                max_tokens=max_tokens
            )

            if not result['success']:
                print(f"Failed at depth {depth + 1}: {result.get('error')}")
                break

            # Track papers
            all_papers_analyzed.update(result.get('paper_ids', []))

            # Store iteration
            research_iterations.append({
                'depth': depth + 1,
                'question': current_question,
                'answer': result['answer'],
                'papers_analyzed': result['papers_analyzed'],
                'paper_ids': result.get('paper_ids', [])
            })

            # Generate follow-up question based on current findings
            if depth < max_depth - 1:  # Don't generate follow-up on last iteration
                follow_up = self._generate_follow_up_question(
                    original_question=initial_question,
                    current_findings=result['answer'],
                    depth=depth + 1
                )

                if follow_up:
                    current_question = follow_up
                    print(f"\nFollow-up question generated: {current_question[:100]}...")
                else:
                    print("No meaningful follow-up question generated. Stopping.")
                    break

        # Synthesize all findings
        print(f"\nSynthesizing findings from {len(research_iterations)} iterations...")
        synthesis = self._synthesize_recursive_findings(
            initial_question=initial_question,
            iterations=research_iterations
        )

        return {
            'success': True,
            'initial_question': initial_question,
            'total_iterations': len(research_iterations),
            'total_papers_analyzed': len(all_papers_analyzed),
            'iterations': research_iterations,
            'synthesis': synthesis,
            'paper_ids': list(all_papers_analyzed)
        }

    def _generate_follow_up_question(self,
                                    original_question: str,
                                    current_findings: str,
                                    depth: int) -> Optional[str]:
        """
        Generate intelligent follow-up question based on current findings

        Args:
            original_question: The original research question
            current_findings: Current iteration's findings
            depth: Current recursion depth

        Returns:
            Follow-up question string or None
        """
        prompt = f"""Based on the research conducted so far, generate ONE specific follow-up question
that would deepen our understanding.

**Original Research Question:**
{original_question}

**Current Findings (Iteration {depth}):**
{current_findings[:2000]}

**Your Task:**
Identify the most important gap, ambiguity, or area that needs deeper investigation.
Generate ONE specific, focused follow-up question that would:
1. Address a knowledge gap revealed by current findings
2. Drill deeper into a promising area
3. Clarify contradictions or ambiguities
4. Explore a specific methodology or approach in more detail

Return ONLY the follow-up question, nothing else. If no meaningful follow-up is needed, return "NONE".
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a research strategist who identifies knowledge gaps and generates targeted follow-up questions."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=200,
                temperature=0.6
            )

            follow_up = response.choices[0].message.content.strip()

            if follow_up.upper() == "NONE" or len(follow_up) < 10:
                return None

            return follow_up

        except Exception as e:
            print(f"Error generating follow-up: {e}")
            return None

    def _synthesize_recursive_findings(self,
                                       initial_question: str,
                                       iterations: List[Dict]) -> str:
        """
        Synthesize findings from all recursive iterations

        Args:
            initial_question: The original research question
            iterations: List of iteration dictionaries

        Returns:
            Synthesized findings as a string
        """
        # Combine all findings
        all_findings = "\n\n".join([
            f"**Iteration {it['depth']} - {it['question']}**\n{it['answer']}"
            for it in iterations
        ])

        prompt = f"""Synthesize the findings from a multi-iteration recursive research process.

**Original Research Question:**
{initial_question}

**Findings from {len(iterations)} Research Iterations:**
{all_findings}

**Your Task:**
Create a comprehensive synthesis that:
1. **Main Answer**: Provide a complete answer to the original question
2. **Key Insights**: Highlight the most important insights discovered
3. **Depth Progression**: Show how understanding deepened through iterations
4. **Comprehensive Evidence**: Cite papers across all iterations
5. **Contradictions Resolved**: Address any contradictions found
6. **Future Directions**: Suggest areas for continued research
7. **Executive Summary**: 2-3 paragraph summary of key findings

Make this a cohesive narrative, not just a list of iteration summaries.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at synthesizing complex research findings into coherent narratives."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=3072,
                temperature=0.4
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Error during synthesis: {e}\n\nRaw iterations available in result."


def format_research_output(result: Dict, output_file: Optional[str] = None) -> str:
    """
    Format research results for display or saving

    Args:
        result: Research result dictionary
        output_file: Optional file to save output to

    Returns:
        Formatted output string
    """
    if not result.get('success'):
        output = f"❌ Research Failed\nError: {result.get('error', 'Unknown error')}\n"
    else:
        output = f"""
{'='*70}
RESEARCH RESULTS
{'='*70}

"""
        # Handle recursive research results
        if 'total_iterations' in result:
            output += f"Research Type: Recursive Deep Research\n"
            output += f"Initial Question: {result['initial_question']}\n"
            output += f"Total Iterations: {result['total_iterations']}\n"
            output += f"Total Papers Analyzed: {result['total_papers_analyzed']}\n\n"

            # Show each iteration
            output += f"{'='*70}\n"
            output += "ITERATION DETAILS\n"
            output += f"{'='*70}\n\n"

            for iteration in result.get('iterations', []):
                output += f"--- Iteration {iteration['depth']} ---\n"
                output += f"Question: {iteration['question']}\n"
                output += f"Papers: {iteration['papers_analyzed']}\n\n"
                output += f"{iteration['answer']}\n\n"
                output += f"{'-'*70}\n\n"

            # Show synthesis
            output += f"{'='*70}\n"
            output += "COMPREHENSIVE SYNTHESIS\n"
            output += f"{'='*70}\n\n"
            output += f"{result.get('synthesis', 'No synthesis available')}\n"

        else:
            # Regular research results
            output += f"Papers Analyzed: {result.get('papers_analyzed', 0)}\n"
            output += f"Model Used: {result.get('model_used', 'N/A')}\n\n"

            if 'research_question' in result:
                output += f"Research Question: {result['research_question']}\n\n"

            if 'topic' in result:
                output += f"Topic: {result['topic']}\n"
                output += f"Aspects: {', '.join(result.get('aspects', []))}\n\n"

            if 'answer' in result and result['answer']:
                output += f"{result['answer']}\n"
            elif 'analysis' in result and result['analysis']:
                output += f"{result['analysis']}\n"
            elif 'result' in result and result['result']:
                output += f"{result['result']}\n"

        output += f"\n{'='*70}\n"

    # Save to file if specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"\n✅ Results saved to: {output_file}")

    return output
