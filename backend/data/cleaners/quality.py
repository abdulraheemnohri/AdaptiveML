"""
Quality Cleaner - Filters and scores content quality
"""

import re
from typing import Any, Dict, List, Optional

from .base import BaseCleaner, CleanerConfig, CleaningResult


class QualityCleaner(BaseCleaner):
    """Filters and scores content quality"""
    
    def __init__(self, config: CleanerConfig):
        super().__init__(config)
    
    async def clean(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> CleaningResult:
        """Evaluate and filter content based on quality"""
        original_content = content
        
        # Calculate quality metrics
        metrics = await self._calculate_quality_metrics(content)
        
        # Calculate overall quality score
        quality_score = self._calculate_overall_score(metrics)
        
        # Check if content meets minimum quality threshold
        if quality_score < self.config.min_quality_score:
            return CleaningResult(
                content="",
                original_content=original_content,
                cleaner_name="quality",
                success=True,
                error=f"Quality score {quality_score:.2f} below threshold {self.config.min_quality_score}",
                metadata={
                    'metrics': metrics,
                    'quality_score': quality_score,
                    'reason': 'low_quality',
                },
                removed_count=1,
            )
        
        # Apply quality-based modifications
        modified_content = content
        modified_count = 0
        
        # Fix common quality issues
        if self.config.options.get('fix_common_issues', True):
            modified_content, count = self._fix_common_issues(modified_content)
            modified_count += count
        
        # Ensure proper sentence endings
        if self.config.options.get('fix_sentences', True):
            modified_content, count = self._fix_sentence_endings(modified_content)
            modified_count += count
        
        return CleaningResult(
            content=modified_content,
            original_content=original_content,
            cleaner_name="quality",
            success=True,
            metadata={
                'metrics': metrics,
                'quality_score': quality_score,
                'passed_threshold': True,
            },
            modified_count=modified_count,
            quality_score=quality_score,
        )
    
    async def clean_batch(self, items: List[Dict[str, Any]]) -> List[CleaningResult]:
        """Clean multiple items and filter by quality"""
        results = []
        passed_count = 0
        filtered_count = 0
        total_quality = 0.0
        
        for item in items:
            content = item.get('content', '')
            metadata = item.get('metadata')
            
            result = await self.clean(content, metadata)
            results.append(result)
            
            if result.success and result.content:
                passed_count += 1
                total_quality += result.quality_score
            else:
                filtered_count += 1
        
        # Add batch summary
        if results:
            results[-1].metadata['batch_summary'] = {
                'total': len(items),
                'passed': passed_count,
                'filtered': filtered_count,
                'average_quality': total_quality / max(passed_count, 1),
                'pass_rate': passed_count / len(items),
            }
        
        return results
    
    async def _calculate_quality_metrics(self, content: str) -> Dict[str, Any]:
        """Calculate various quality metrics for content"""
        metrics = {
            'length': len(content),
            'word_count': len(content.split()),
            'sentence_count': 0,
            'paragraph_count': 0,
            'avg_sentence_length': 0,
            'avg_word_length': 0,
            'vocabulary_richness': 0,
            'has_structure': False,
            'has_numbers': False,
            'has_special_chars_ratio': 0,
            'repetition_ratio': 0,
        }
        
        # Sentence count (rough estimate)
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        metrics['sentence_count'] = len(sentences)
        
        # Paragraph count
        paragraphs = content.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        metrics['paragraph_count'] = len(paragraphs)
        
        # Average sentence length
        if metrics['sentence_count'] > 0:
            metrics['avg_sentence_length'] = metrics['word_count'] / metrics['sentence_count']
        
        # Average word length
        words = content.split()
        if words:
            metrics['avg_word_length'] = sum(len(w) for w in words) / len(words)
        
        # Vocabulary richness (type-token ratio)
        unique_words = set(w.lower() for w in words)
        if words:
            metrics['vocabulary_richness'] = len(unique_words) / len(words)
        
        # Has structure (headings, lists, etc.)
        has_headings = bool(re.search(r'^#+\s|^\d+\.\s|^[•\-*]\s', content, re.MULTILINE))
        has_paragraphs = metrics['paragraph_count'] > 1
        metrics['has_structure'] = has_headings or has_paragraphs
        
        # Has numbers
        metrics['has_numbers'] = bool(re.search(r'\d+', content))
        
 # Special characters ratio
        special_chars = sum(1 for c in content if not c.isalnum() and not c.isspace())
        metrics['has_special_chars_ratio'] = special_chars / max(len(content), 1)
        
        # Repetition ratio (how much content is repeated)
        word_counts = {}
        for word in words:
            word_lower = word.lower()
            word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
        
        repeated_words = sum(count - 1 for count in word_counts.values() if count > 1)
        metrics['repetition_ratio'] = repeated_words / max(len(words), 1)
        
        return metrics
    
    def _calculate_overall_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall quality score from metrics"""
        score = 0.5  # Base score
        
        # Length scoring (prefer content between 100-5000 words)
        word_count = metrics['word_count']
        if 100 <= word_count <= 5000:
            score += 0.2
        elif 50 <= word_count <= 10000:
            score += 0.1
        elif word_count < 10:
            score -= 0.3
        elif word_count > 20000:
            score -= 0.1
        
        # Sentence structure scoring
        avg_sentence_length = metrics['avg_sentence_length']
        if 10 <= avg_sentence_length <= 25:
            score += 0.15
        elif 5 <= avg_sentence_length <= 40:
            score += 0.05
        else:
            score -= 0.1
        
        # Vocabulary richness scoring
        vocab_richness = metrics['vocabulary_richness']
        if vocab_richness > 0.6:
            score += 0.15
        elif vocab_richness > 0.4:
            score += 0.1
        elif vocab_richness < 0.2:
            score -= 0.1
        
        # Structure scoring
        if metrics['has_structure']:
            score += 0.1
        
        # Penalize high repetition
        if metrics['repetition_ratio'] > 0.5:
            score -= 0.15
        elif metrics['repetition_ratio'] > 0.3:
            score -= 0.05
        
        # Penalize too many special characters
        if metrics['has_special_chars_ratio'] > 0.3:
            score -= 0.1
        
        # Bonus for having numbers (often indicates factual content)
        if metrics['has_numbers']:
            score += 0.05
        
        return max(0.0, min(1.0, score))
    
    def _fix_common_issues(self, text: str) -> tuple:
        """Fix common quality issues in text"""
        modifications = 0
        
        # Fix multiple spaces
        text, count = re.subn(r'  +', ' ', text)
        modifications += count
        
        # Fix space before punctuation
        text, count = re.subn(r' +([.,!?;:])', r'\1', text)
        modifications += count
        
        # Fix missing space after punctuation
        text, count = re.subn(r'([.,!?;:])([A-Z])', r'\1 \2', text)
        modifications += count
        
        # Fix uppercase i as pronoun
        text, count = re.subn(r'\bi\b', 'I', text)
        modifications += count
        
        # Common typo fixes
        typos = {
            'teh': 'the',
            'adn': 'and',
            'waht': 'what',
            'hte': 'the',
            'wih': 'with',
            'thsi': 'this',
            'taht': 'that',
        }
        
        for typo, correct in typos.items():
            text, count = re.subn(r'\b' + typo + r'\b', correct, text, flags=re.IGNORECASE)
            modifications += count
        
        return text, modifications
    
    def _fix_sentence_endings(self, text: str) -> tuple:
        """Ensure proper sentence endings"""
        modifications = 0
        
        # Split into lines and process
        lines = text.split('\n')
        fixed_lines = []
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                fixed_lines.append(line)
                continue
            
            # Skip lines that are headings or list items
            if re.match(r'^#+\s|^\d+\.\s|^[•\-*]\s', stripped):
                fixed_lines.append(line)
                continue
            
            # Skip very short lines
            if len(stripped) < 10:
                fixed_lines.append(line)
                continue
            
            # Add period if line looks like a sentence but has no ending punctuation
            if not re.search(r'[.!?]"?$', stripped):
                # Check if it starts like a sentence
                if stripped[0].isupper() or len(stripped.split()) > 5:
                    stripped = stripped.rstrip('"\'') + '.'
                    modifications += 1
            
            fixed_lines.append(stripped)
        
        return '\n'.join(fixed_lines), modifications
    
    async def score_content(self, content: str) -> Dict[str, Any]:
        """Get detailed quality score for content"""
        metrics = await self._calculate_quality_metrics(content)
        overall_score = self._calculate_overall_score(metrics)
        
        return {
            'overall_score': overall_score,
            'metrics': metrics,
            'recommendation': self._get_recommendation(overall_score, metrics),
        }
    
    def _get_recommendation(self, score: float, metrics: Dict[str, Any]) -> str:
        """Get recommendation based on quality score"""
        if score >= 0.8:
            return "Excellent quality - suitable for training"
        elif score >= 0.6:
            return "Good quality - minor improvements may help"
        elif score >= 0.4:
            return "Moderate quality - consider improvements"
        elif score >= 0.2:
            return "Low quality - significant improvements needed"
        else:
            return "Very low quality - consider excluding from training"
