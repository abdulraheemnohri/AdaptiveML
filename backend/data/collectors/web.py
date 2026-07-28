"""
Web Collector - Scrapes websites for content
"""

import asyncio
from typing import List, Optional
from urllib.parse import urljoin, urlparse

from .base import BaseCollector, CollectorConfig, CollectedData, SourceType


class WebCollector(BaseCollector):
    """Collects data from websites via scraping"""
    
    def __init__(self, config: CollectorConfig):
        super().__init__(config)
        self._session = None
        self._visited_urls = set()
    
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self._session is None:
            import aiohttp
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def collect(self) -> List[CollectedData]:
        """Collect data from website(s)"""
        results = []
        
        if not self.config.url:
            raise ValueError("URL is required for web collection")
        
        session = await self._get_session()
        
        # Handle sitemap or single URL
        if "sitemap" in self.config.metadata.get("mode", ""):
            urls = await self._extract_sitemap_urls(session, self.config.url)
        else:
            urls = [self.config.url]
        
        for url in urls[:self.config.max_items]:
            if url in self._visited_urls:
                continue
            self._visited_urls.add(url)
            
            try:
                data = await self._scrape_url(session, url)
                if data:
                    results.append(data)
            except Exception as e:
                print(f"Error scraping {url}: {e}")
                continue
        
        return results
    
    async def _scrape_url(self, session, url: str) -> Optional[CollectedData]:
        """Scrape a single URL"""
        import aiohttp
        
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; AdaptiveOmniML/1.0; +https://adaptive-omni.ml)"
        }
        
        async with session.get(url, headers=headers, ssl=False) as response:
            if response.status != 200:
                return None
            
            html = await response.text()
            
            # Parse HTML
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract title
                title_tag = soup.find('title')
                title = title_tag.string.strip() if title_tag else None
                
                # Remove script and style elements
                for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                    element.decompose()
                
                # Extract main content
                main = soup.find('main') or soup.find('article') or soup.body
                if main:
                    text = main.get_text(separator='\n', strip=True)
                else:
                    text = soup.get_text(separator='\n', strip=True)
                
                # Extract metadata
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                meta_author = soup.find('meta', attrs={'name': 'author'})
                published = soup.find('meta', property='article:published_time')
                
                from datetime import datetime
                pub_date = None
                if published and published.get('content'):
                    try:
                        pub_date = datetime.fromisoformat(published['content'].replace('Z', '+00:00'))
                    except:
                        pass
                
                return CollectedData(
                    content=text,
                    source_url=url,
                    source_type=SourceType.WEBSITE,
                    title=title,
                    author=meta_author['content'] if meta_author else None,
                    published_date=pub_date,
                    metadata={
                        'description': meta_desc['content'] if meta_desc else None,
                        'links': len(soup.find_all('a')),
                        'images': len(soup.find_all('img')),
                    },
                    mime_type='text/html'
                )
            except Exception as e:
                print(f"Parse error for {url}: {e}")
                return None
    
    async def _extract_sitemap_urls(self, session, sitemap_url: str) -> List[str]:
        """Extract URLs from sitemap"""
        import aiohttp
        urls = []
        
        try:
            async with session.get(sitemap_url, ssl=False) as response:
                if response.status == 200:
                    xml = await response.text()
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(xml, 'xml')
                    
                    for loc in soup.find_all('loc'):
                        if loc.parent.name == 'url':
                            urls.append(loc.text.strip())
        except Exception as e:
            print(f"Error parsing sitemap: {e}")
        
        return urls
    
    async def test_connection(self) -> bool:
        """Test connection to the website"""
        if not self.config.url:
            return False
        
        try:
            session = await self._get_session()
            async with session.head(self.config.url, ssl=False) as response:
                return response.status == 200
        except:
            return False
    
    async def close(self):
        """Close the session"""
        if self._session:
            await self._session.close()
            self._session = None
