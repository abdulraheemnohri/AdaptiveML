"""
HTML Processor - Handles HTML content processing and extraction
"""

from typing import Any, Dict, Optional

from .base import BaseProcessor, ProcessorConfig, ProcessingResult, ProcessorType


class HTMLProcessor(BaseProcessor):
    """Processes HTML content"""
    
    def __init__(self, config: ProcessorConfig):
        super().__init__(config)
    
    async def process(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """Process HTML content"""
        original_content = content
        
        # Validate first
        errors = await self.validate(content)
        if errors:
            return ProcessingResult(
                content="",
                original_content=original_content,
                processor_type=ProcessorType.HTML,
                success=False,
                error="; ".join(errors),
            )
        
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract title
            title = None
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text(strip=True)
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'noscript', 'iframe', 
                                  'nav', 'footer', 'header', 'aside', 'menu']):
                element.decompose()
            
            # Try to find main content area
            main_content = None
            for selector in ['article', 'main', '[role="main"]', '.content', '.post', '.article']:
                if '[' in selector:
                    element = soup.select_one(selector)
                else:
                    element = soup.find(selector)
                if element:
                    main_content = element
                    break
            
            if not main_content:
                main_content = soup.body or soup
            
            # Extract text
            text = main_content.get_text(separator='\n', strip=True)
            
            # Normalize whitespace
            if self.config.normalize_whitespace:
                text = self._normalize_whitespace(text)
            
            # Detect language
            language = None
            if self.config.detect_language:
                language = self._detect_language(text)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(text, metadata)
            
            # Create chunks
            chunks = self._chunk_text(text)
            
            # Count tokens
            tokens_count = self._count_tokens(text)
            
            # Extract additional metadata
            extracted_metadata = await self._extract_metadata(soup)
            if metadata:
                extracted_metadata.update(metadata)
            
            return ProcessingResult(
                content=text,
                original_content=original_content,
                processor_type=ProcessorType.HTML,
                success=True,
                metadata=extracted_metadata,
                chunks=chunks,
                language=language,
                quality_score=quality_score,
                tokens_count=tokens_count,
            )
        
        except ImportError:
            return ProcessingResult(
                content="",
                original_content=original_content,
                processor_type=ProcessorType.HTML,
                success=False,
                error="BeautifulSoup not installed (install: pip install beautifulsoup4)",
            )
        except Exception as e:
            return ProcessingResult(
                content="",
                original_content=original_content,
                processor_type=ProcessorType.HTML,
                success=False,
                error=str(e),
            )
    
    async def _extract_metadata(self, soup) -> Dict[str, Any]:
        """Extract metadata from HTML"""
        metadata = {}
        
        # Meta description
        desc = soup.find('meta', attrs={'name': 'description'})
        if desc and desc.get('content'):
            metadata['description'] = desc['content']
        
        # Meta keywords
        keywords = soup.find('meta', attrs={'name': 'keywords'})
        if keywords and keywords.get('content'):
            metadata['keywords'] = [k.strip() for k in keywords['content'].split(',')]
        
        # Meta author
        author = soup.find('meta', attrs={'name': 'author'})
        if author and author.get('content'):
            metadata['author'] = author['content']
        
        # Open Graph tags
        og_tags = ['og:title', 'og:description', 'og:image', 'og:url', 'og:type']
        for tag in og_tags:
            og = soup.find('meta', property=tag)
            if og and og.get('content'):
                metadata[tag.replace('og:', 'og_')] = og['content']
        
        # Article published time
        published = soup.find('meta', property='article:published_time')
        if published and published.get('content'):
            metadata['published_time'] = published['content']
        
        # Language from html tag
        html_tag = soup.find('html')
        if html_tag and html_tag.get('lang'):
            metadata['html_lang'] = html_tag['lang']
        
        return metadata
    
    async def extract_links(self, html: str, internal_only: bool = False, base_url: str = None) -> list:
        """Extract links from HTML"""
        try:
            from bs4 import BeautifulSoup
            from urllib.parse import urljoin, urlparse
            
            soup = BeautifulSoup(html, 'html.parser')
            links = []
            seen = set()
            
            for a in soup.find_all('a', href=True):
                href = a['href'].strip()
                
                # Convert relative to absolute URL
                if base_url:
                    href = urljoin(base_url, href)
                
                # Skip javascript and mailto links
                if href.startswith(('javascript:', 'mailto:', '#')):
                    continue
                
                # Filter internal links only
                if internal_only and base_url:
                    base_domain = urlparse(base_url).netloc
                    link_domain = urlparse(href).netloc
                    if base_domain != link_domain:
                        continue
                
                if href not in seen:
                    seen.add(href)
                    links.append({
                        'url': href,
                        'text': a.get_text(strip=True),
                        'title': a.get('title', ''),
                    })
            
            return links
        
        except Exception as e:
            return [{'error': str(e)}]
    
    async def extract_images(self, html: str, base_url: str = None) -> list:
        """Extract image sources from HTML"""
        try:
            from bs4 import BeautifulSoup
            from urllib.parse import urljoin
            
            soup = BeautifulSoup(html, 'html.parser')
            images = []
            seen = set()
            
            for img in soup.find_all('img'):
                src = img.get('src') or img.get('data-src')
                if not src:
                    continue
                
                if base_url and not src.startswith(('http://', 'https://')):
                    src = urljoin(base_url, src)
                
                if src not in seen:
                    seen.add(src)
                    images.append({
                        'src': src,
                        'alt': img.get('alt', ''),
                        'title': img.get('title', ''),
                        'width': img.get('width'),
                        'height': img.get('height'),
                    })
            
            return images
        
        except Exception as e:
            return [{'error': str(e)}]
