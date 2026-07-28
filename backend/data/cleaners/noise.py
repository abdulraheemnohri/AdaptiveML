"""
Noise Cleaner - Removes noise and irrelevant content
"""

import re
from typing import Any, Dict, List, Optional

from .base import BaseCleaner, CleanerConfig, CleaningResult


class NoiseCleaner(BaseCleaner):
    """Removes noise and irrelevant content from text"""
    
    def __init__(self, config: CleanerConfig):
        super().__init__(config)
    
    async def clean(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> CleaningResult:
        """Remove noise from content"""
        original_content = content
        modified_count = 0
        
        # Remove URLs if configured
        if self.config.options.get('remove_urls', False):
            content, count = self._remove_urls(content)
            modified_count += count
        
        # Remove email addresses
        if self.config.options.get('remove_emails', False):
            content, count = self._remove_emails(content)
            modified_count += count
        
        # Remove HTML tags
        if self.config.options.get('remove_html', True):
            content, count = self._remove_html_tags(content)
            modified_count += count
        
        # Remove special characters if configured
        if self.config.options.get('remove_special_chars', False):
            content, count = self._remove_special_characters(content)
            modified_count += count
        
        # Normalize whitespace
        if self.config.options.get('normalize_whitespace', True):
            content, count = self._normalize_whitespace(content)
            modified_count += count
        
        # Remove boilerplate patterns
        if self.config.options.get('remove_boilerplate', True):
            content, count = self._remove_boilerplate(content)
            modified_count += count
        
        # Remove repeated characters (e.g., "!!!!!" -> "!")
        if self.config.options.get('remove_repeated_chars', True):
            content, count = self._remove_repeated_characters(content)
            modified_count += count
        
        # Strip leading/trailing whitespace
        content = content.strip()
        
        return CleaningResult(
            content=content,
            original_content=original_content,
            cleaner_name="noise",
            success=True,
            metadata={
                'modifications': modified_count,
                'original_length': len(original_content),
                'cleaned_length': len(content),
                'reduction_ratio': 1 - (len(content) / max(len(original_content), 1)),
            },
            modified_count=modified_count,
        )
    
    async def clean_batch(self, items: List[Dict[str, Any]]) -> List[CleaningResult]:
        """Clean multiple items"""
        results = []
        total_modified = 0
        total_removed = 0
        
        for item in items:
            content = item.get('content', '')
            metadata = item.get('metadata')
            
            result = await self.clean(content, metadata)
            results.append(result)
            
            total_modified += result.modified_count
            if not result.content and result.original_content:
                total_removed += 1
        
        # Add batch summary
        if results:
            results[-1].metadata['batch_summary'] = {
                'total': len(items),
                'total_modifications': total_modified,
                'removed_count': total_removed,
            }
        
        return results
    
    def _remove_urls(self, text: str) -> tuple:
        """Remove URLs from text"""
        url_pattern = r'https?://\S+|www\.\S+'
        matches = re.findall(url_pattern, text)
        cleaned = re.sub(url_pattern, '', text)
        return cleaned, len(matches)
    
    def _remove_emails(self, text: str) -> tuple:
        """Remove email addresses from text"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        matches = re.findall(email_pattern, text)
        cleaned = re.sub(email_pattern, '', text)
        return cleaned, len(matches)
    
    def _remove_html_tags(self, text: str) -> tuple:
        """Remove HTML tags from text"""
        html_pattern = r'<[^>]+>'
        matches = re.findall(html_pattern, text)
        cleaned = re.sub(html_pattern, '', text)
        return cleaned, len(matches)
    
    def _remove_special_characters(self, text: str) -> tuple:
        """Remove special characters while keeping alphanumeric and basic punctuation"""
        # Keep letters, numbers, spaces, and basic punctuation
        pattern = r'[^\w\s.,!?;:\'"()\-\n]'
        matches = re.findall(pattern, text)
        cleaned = re.sub(pattern, '', text)
        return cleaned, len(matches)
    
    def _normalize_whitespace(self, text: str) -> tuple:
        """Normalize whitespace"""
        # Replace tabs with spaces
        text = text.replace('\t', ' ')
        
        # Replace multiple spaces with single space
        text, count1 = re.subn(r' +', ' ', text)
        
        # Replace multiple newlines with double newline
        text, count2 = re.subn(r'\n\s*\n', '\n\n', text)
        
        # Strip whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text, count1 + count2
    
    def _remove_boilerplate(self, text: str) -> tuple:
        """Remove common boilerplate patterns"""
        removed_count = 0
        modifications = 0
        
        # Common boilerplate patterns
        patterns = [
            r'(?:copyright|©)\s*[\d\-]{4,}',  # Copyright notices
            r'all\s+rights\s+reserved',  # Rights reserved
            r'powered\s+by\s+\w+',  # Powered by
            r'designed\s+by\s+\w+',  # Designed by
            r'subscribe\s+(?:to\s+)?(?:our\s+)?(?:newsletter|updates)',  # Subscribe CTAs
            r'follow\s+us\s+(?:on\s+)?',  # Follow us
            r'share\s+(?:this\s+)?(?:article|post)',  # Share prompts
            r'related\s+(?:articles?|posts?|content)',  # Related content headers
            r'leave\s+a\s+(?:comment|reply)',  # Comment prompts
            r'click\s+here\s+(?:to\s+)?(?:subscribe|learn\s+more|read\s+more)',  # Click here
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                text = re.sub(pattern, '', text, flags=re.IGNORECASE)
                modifications += 1
                removed_count += len(matches)
        
        return text, removed_count
    
    def _remove_repeated_characters(self, text: str) -> tuple:
        """Remove repeated characters (3+ repetitions)"""
        # Match any character repeated 3 or more times
        pattern = r'(.)\1{2,}'
        
        def replace_repeated(match):
            char = match.group(1)
            # Keep up to 2 repetitions for punctuation, 1 for others
            if char in '!?.':
                return char * min(2, len(match.group(0)))
            return char
        
        matches = re.findall(pattern, text)
        cleaned = re.sub(pattern, replace_repeated, text)
        
        return cleaned, len(matches)
