"""
Deduplication Cleaner - Removes duplicate and near-duplicate content
"""

import hashlib
from typing import Any, Dict, List, Optional, Set, Tuple

from .base import BaseCleaner, CleanerConfig, CleaningResult


class DeduplicationCleaner(BaseCleaner):
    """Removes duplicate and near-duplicate content"""
    
    def __init__(self, config: CleanerConfig):
        super().__init__(config)
        self._seen_hashes: Set[str] = set()
        self._content_cache: List[Tuple[str, str]] = []  # (hash, content)
    
    async def clean(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> CleaningResult:
        """Remove duplicates from content"""
        original_content = content
        
        # Check for exact duplicate using hash
        content_hash = self._hash_content(content)
        
        if content_hash in self._seen_hashes:
            return CleaningResult(
                content="",
                original_content=original_content,
                cleaner_name="deduplication",
                success=True,
                metadata={"reason": "exact_duplicate", "hash": content_hash},
                removed_count=1,
            )
        
        # Check for near-duplicates
        is_near_duplicate, similar_content = await self._check_near_duplicate(content)
        
        if is_near_duplicate:
            similarity = self._calculate_similarity(content, similar_content)
            return CleaningResult(
                content="",
                original_content=original_content,
                cleaner_name="deduplication",
                success=True,
                metadata={
                    "reason": "near_duplicate",
                    "similarity": similarity,
                    "similar_to": self._hash_content(similar_content),
                },
                removed_count=1,
            )
        
        # Add to seen content
        self._seen_hashes.add(content_hash)
        self._content_cache.append((content_hash, content))
        
        # Limit cache size
        max_cache_size = self.config.options.get('max_cache_size', 10000)
        if len(self._content_cache) > max_cache_size:
            # Remove oldest entries
            remove_count = len(self._content_cache) - max_cache_size
            self._content_cache = self._content_cache[remove_count:]
            # Rebuild hash set
            self._seen_hashes = {h for h, _ in self._content_cache}
        
        return CleaningResult(
            content=content,
            original_content=original_content,
            cleaner_name="deduplication",
            success=True,
            metadata={"hash": content_hash, "is_new": True},
        )
    
    async def clean_batch(self, items: List[Dict[str, Any]]) -> List[CleaningResult]:
        """Clean multiple items and remove duplicates"""
        results = []
        unique_items = []
        duplicate_count = 0
        near_duplicate_count = 0
        
        for item in items:
            content = item.get('content', '')
            metadata = item.get('metadata')
            
            result = await self.clean(content, metadata)
            results.append(result)
            
            if result.success and result.content:
                unique_items.append({
                    'content': result.content,
                    'metadata': result.metadata,
                })
            elif result.metadata.get('reason') == 'exact_duplicate':
                duplicate_count += 1
            elif result.metadata.get('reason') == 'near_duplicate':
                near_duplicate_count += 1
        
        # Add summary to last result
        if results:
            results[-1].metadata['batch_summary'] = {
                'total': len(items),
                'unique': len(unique_items),
                'duplicates': duplicate_count,
                'near_duplicates': near_duplicate_count,
            }
        
        return results
    
    def _hash_content(self, content: str) -> str:
        """Generate hash for content"""
        # Normalize before hashing
        normalized = content.lower().strip()
        normalized = ' '.join(normalized.split())  # Normalize whitespace
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    async def _check_near_duplicate(self, content: str) -> Tuple[bool, Optional[str]]:
        """Check if content is near-duplicate of existing content"""
        threshold = self.config.similarity_threshold
        
        # For large caches, use sampling for performance
        cache_size = len(self._content_cache)
        if cache_size > 1000:
            # Sample recent entries
            import random
            sample_size = min(100, cache_size // 10)
            sampled = random.sample(self._content_cache, sample_size)
        else:
            sampled = self._content_cache
        
        for _, cached_content in sampled:
            similarity = self._calculate_similarity(content, cached_content)
            if similarity >= threshold:
                return True, cached_content
        
        return False, None
    
    async def find_duplicates(self, contents: List[str]) -> Dict[str, List[int]]:
        """Find all duplicates in a list of contents"""
        hash_to_indices: Dict[str, List[int]] = {}
        
        for i, content in enumerate(contents):
            content_hash = self._hash_content(content)
            if content_hash not in hash_to_indices:
                hash_to_indices[content_hash] = []
            hash_to_indices[content_hash].append(i)
        
        # Return only groups with duplicates
        return {h: indices for h, indices in hash_to_indices.items() if len(indices) > 1}
    
    async def find_near_duplicates(self, contents: List[str]) -> List[Tuple[int, int, float]]:
        """Find near-duplicate pairs in a list of contents"""
        near_duplicates = []
        threshold = self.config.similarity_threshold
        
        n = len(contents)
        for i in range(n):
            for j in range(i + 1, n):
                similarity = self._calculate_similarity(contents[i], contents[j])
                if similarity >= threshold:
                    near_duplicates.append((i, j, similarity))
        
        return near_duplicates
    
    def reset(self):
        """Reset the deduplication state"""
        self._seen_hashes.clear()
        self._content_cache.clear()
