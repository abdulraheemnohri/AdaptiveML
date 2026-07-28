"""
Text Processor - Handles plain text processing
"""

from typing import Any, Dict, Optional

from .base import BaseProcessor, ProcessorConfig, ProcessingResult, ProcessorType


class TextProcessor(BaseProcessor):
    """Processes plain text content"""
    
    def __init__(self, config: ProcessorConfig):
        super().__init__(config)
    
    async def process(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """Process text content"""
        original_content = content
        
        # Validate first
        errors = await self.validate(content)
        if errors:
            return ProcessingResult(
                content="",
                original_content=original_content,
                processor_type=ProcessorType.TEXT,
                success=False,
                error="; ".join(errors),
            )
        
        # Normalize whitespace if configured
        if self.config.normalize_whitespace:
            content = self._normalize_whitespace(content)
        
        # Remove special characters if configured
        if self.config.remove_special_chars:
            content = self._remove_special_chars(content)
        
        # Detect language if configured
        language = None
        if self.config.detect_language:
            language = self._detect_language(content)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(content, metadata)
        
        # Check minimum quality threshold
        if quality_score < self.config.min_quality_score:
            return ProcessingResult(
                content=content,
                original_content=original_content,
                processor_type=ProcessorType.TEXT,
                success=False,
                error=f"Quality score {quality_score:.2f} below threshold {self.config.min_quality_score}",
                quality_score=quality_score,
            )
        
        # Create chunks if configured
        chunks = self._chunk_text(content)
        
        # Count tokens
        tokens_count = self._count_tokens(content)
        
        return ProcessingResult(
            content=content,
            original_content=original_content,
            processor_type=ProcessorType.TEXT,
            success=True,
            metadata=metadata or {},
            chunks=chunks,
            language=language,
            quality_score=quality_score,
            tokens_count=tokens_count,
        )
    
    async def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract named entities from text"""
        try:
            import spacy
            
            # Load model (user needs to install: python -m spacy download en_core_web_sm)
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                return {"error": "SpaCy model not installed"}
            
            doc = nlp(text[:10000])  # Limit for performance
            
            entities = {
                "persons": [ent.text for ent in doc.ents if ent.label_ == "PERSON"],
                "organizations": [ent.text for ent in doc.ents if ent.label_ == "ORG"],
                "locations": [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC")],
                "dates": [ent.text for ent in doc.ents if ent.label_ in ("DATE", "TIME")],
                "money": [ent.text for ent in doc.ents if ent.label_ == "MONEY"],
                "percentages": [ent.text for ent in doc.ents if ent.label_ == "PERCENT"],
            }
            
            return entities
        
        except ImportError:
            return {"error": "spacy not installed"}
    
    async def summarize(self, text: str, max_length: int = 200) -> str:
        """Generate a simple extractive summary"""
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 3:
            return text
        
        # Score sentences by various features
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            score = 0
            
            # Position score (first and last sentences often important)
            if i == 0:
                score += 2
            elif i == len(sentences) - 1:
                score += 1
            
            # Length score (prefer medium-length sentences)
            word_count = len(sentence.split())
            if 10 <= word_count <= 30:
                score += 1
            
            # Keyword score
            keywords = ['important', 'significant', 'key', 'main', 'conclusion', 
                       'result', 'finding', 'demonstrate', 'show', 'prove']
            for keyword in keywords:
                if keyword in sentence.lower():
                    score += 1
            
            scored_sentences.append((score, sentence))
        
        # Sort by score and take top sentences
        scored_sentences.sort(reverse=True)
        top_sentences = [s[1] for s in scored_sentences[:3]]
        
        # Reorder by original position
        summary = ' '.join(sorted(top_sentences, key=lambda s: sentences.index(s)))
        
        # Truncate if needed
        if len(summary) > max_length:
            summary = summary[:max_length-3] + '...'
        
        return summary
