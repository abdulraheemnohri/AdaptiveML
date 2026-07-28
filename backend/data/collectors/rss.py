"""
RSS Collector - Fetches and parses RSS/Atom feeds
"""

from datetime import datetime
from typing import List, Optional

from .base import BaseCollector, CollectorConfig, CollectedData, SourceType


class RSSCollector(BaseCollector):
    """Collects data from RSS and Atom feeds"""
    
    def __init__(self, config: CollectorConfig):
        super().__init__(config)
        self._session = None
    
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self._session is None:
            import aiohttp
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def collect(self) -> List[CollectedData]:
        """Collect data from RSS/Atom feed"""
        results = []
        
        if not self.config.url:
            raise ValueError("URL is required for RSS collection")
        
        session = await self._get_session()
        
        try:
            async with session.get(self.config.url) as response:
                if response.status != 200:
                    print(f"Failed to fetch RSS feed: {response.status}")
                    return results
                
                xml_content = await response.text()
                
                # Parse feed
                try:
                    import feedparser
                    feed = feedparser.parse(xml_content)
                    
                    entries = feed.entries[:self.config.max_items]
                    
                    for entry in entries:
                        published_date = None
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            try:
                                published_date = datetime(*entry.published_parsed[:6])
                            except:
                                pass
                        
                        content = ""
                        if hasattr(entry, 'content') and entry.content:
                            content = entry.content[0].value
                        elif hasattr(entry, 'summary'):
                            content = entry.summary
                        elif hasattr(entry, 'description'):
                            content = entry.description
                        
                        # Extract full content if link available
                        if hasattr(entry, 'link') and self.config.metadata.get('fetch_full_content', False):
                            full_content = await self._fetch_full_content(session, entry.link)
                            if full_content:
                                content = full_content
                        
                        data = CollectedData(
                            content=content,
                            source_url=entry.get('link', ''),
                            source_type=SourceType.RSS,
                            title=entry.get('title'),
                            author=entry.get('author'),
                            published_date=published_date,
                            metadata={
                                'feed_title': feed.feed.get('title'),
                                'feed_link': feed.feed.get('link'),
                                'entry_id': entry.get('id'),
                                'tags': [tag.term for tag in entry.get('tags', [])] if hasattr(entry, 'tags') else [],
                                'original_feed': self.config.url,
                            },
                            mime_type='text/html' if '<' in content else 'text/plain'
                        )
                        results.append(data)
                
                except ImportError:
                    print("feedparser not installed, install with: pip install feedparser")
                    # Fallback: basic XML parsing
                    results = await self._parse_xml_basic(xml_content)
                except Exception as e:
                    print(f"Error parsing RSS feed: {e}")
        
        except Exception as e:
            print(f"Error fetching RSS feed: {e}")
        
        return results
    
    async def _fetch_full_content(self, session, url: str) -> Optional[str]:
        """Fetch full article content from URL"""
        try:
            import aiohttp
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; AdaptiveOmniML/1.0)"
            }
            
            async with session.get(url, headers=headers, ssl=False) as response:
                if response.status == 200:
                    html = await response.text()
                    
                    # Simple extraction - could use readability-lxml for better results
                    try:
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Remove unwanted elements
                        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                            element.decompose()
                        
                        # Try to find main content
                        article = soup.find('article') or soup.find('main')
                        if article:
                            return article.get_text(separator='\n', strip=True)
                        
                        # Fallback to body
                        body = soup.find('body')
                        if body:
                            return body.get_text(separator='\n', strip=True)
                    except:
                        pass
        except Exception as e:
            print(f"Error fetching full content: {e}")
        
        return None
    
    async def _parse_xml_basic(self, xml_content: str) -> List[CollectedData]:
        """Basic XML parsing fallback"""
        import re
        results = []
        
        # Very basic item extraction
        items = re.findall(r'<item>(.*?)</item>', xml_content, re.DOTALL)
        
        for item in items[:self.config.max_items]:
            title_match = re.search(r'<title>(.*?)</title>', item)
            link_match = re.search(r'<link>(.*?)</link>', item)
            desc_match = re.search(r'<description>(.*?)</description>', item, re.DOTALL)
            
            if title_match and link_match:
                results.append(CollectedData(
                    content=desc_match.group(1) if desc_match else '',
                    source_url=link_match.group(1),
                    source_type=SourceType.RSS,
                    title=title_match.group(1),
                    metadata={'parsed': 'basic'},
                    mime_type='text/plain'
                ))
        
        return results
    
    async def test_connection(self) -> bool:
        """Test connection to the RSS feed"""
        if not self.config.url:
            return False
        
        try:
            session = await self._get_session()
            async with session.head(self.config.url, timeout=10) as response:
                return response.status == 200
        except:
            return False
    
    async def close(self):
        """Close the session"""
        if self._session:
            await self._session.close()
            self._session = None
