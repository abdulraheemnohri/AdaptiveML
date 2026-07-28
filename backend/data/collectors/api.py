"""
API Collector - Fetches data from REST APIs
"""

from typing import List, Optional, Dict, Any
from urllib.parse import urlparse, parse_qs, urlencode

from .base import BaseCollector, CollectorConfig, CollectedData, SourceType


class APICollector(BaseCollector):
    """Collects data from REST APIs"""
    
    def __init__(self, config: CollectorConfig):
        super().__init__(config)
        self._session = None
    
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self._session is None:
            import aiohttp
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            
            # Build headers from config
            headers = {
                "User-Agent": "AdaptiveOmniML/1.0",
                "Accept": "application/json",
            }
            
            # Add custom headers
            custom_headers = self.config.metadata.get('headers', {})
            headers.update(custom_headers)
            
            # Add auth header if provided
            api_key = self.config.credentials.get('api_key') if self.config.credentials else None
            auth_type = self.config.metadata.get('auth_type', 'bearer')
            
            if api_key:
                if auth_type == 'bearer':
                    headers["Authorization"] = f"Bearer {api_key}"
                elif auth_type == 'basic':
                    import base64
                    credentials = base64.b64encode(api_key.encode()).decode()
                    headers["Authorization"] = f"Basic {credentials}"
                elif auth_type == 'header':
                    header_name = self.config.metadata.get('auth_header', 'X-API-Key')
                    headers[header_name] = api_key
            
            self._session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        
        return self._session
    
    async def collect(self) -> List[CollectedData]:
        """Collect data from API"""
        results = []
        
        if not self.config.url:
            raise ValueError("URL is required for API collection")
        
        session = await self._get_session()
        
        # Get pagination config
        pagination = self.config.metadata.get('pagination', {})
        pagination_type = pagination.get('type', 'offset')
        
        # Collect with pagination
        page = 0
        offset = 0
        has_more = True
        
        while has_more and len(results) < self.config.max_items:
            url = self._build_url(page, offset)
            
            try:
                async with session.get(url) as response:
                    if response.status != 200:
                        print(f"API error: {response.status}")
                        break
                    
                    data = await response.json()
                    
                    # Extract items based on config
                    items_path = self.config.metadata.get('items_path', '')
                    items = self._extract_path(data, items_path) if items_path else (data if isinstance(data, list) else [data])
                    
                    for item in items:
                        if len(results) >= self.config.max_items:
                            has_more = False
                            break
                        
                        collected = self._process_item(item)
                        if collected:
                            results.append(collected)
                    
                    # Check for more pages
                    if pagination_type == 'offset':
                        offset += pagination.get('step', 10)
                        has_more = len(items) > 0 and len(items) >= pagination.get('step', 10)
                    elif pagination_type == 'page':
                        page += 1
                        has_more = len(items) > 0
                    elif pagination_type == 'cursor':
                        next_cursor = self._extract_path(data, pagination.get('cursor_path', 'next_cursor'))
                        has_more = next_cursor is not None
                    else:
                        has_more = False
            
            except Exception as e:
                print(f"API collection error: {e}")
                break
        
        return results
    
    def _build_url(self, page: int = 0, offset: int = 0) -> str:
        """Build URL with pagination parameters"""
        from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
        
        parsed = urlparse(self.config.url)
        query_params = parse_qs(parsed.query)
        
        pagination = self.config.metadata.get('pagination', {})
        pagination_type = pagination.get('type', 'offset')
        
        if pagination_type == 'offset':
            query_params[pagination.get('offset_param', 'offset')] = [str(offset)]
            query_params[pagination.get('limit_param', 'limit')] = [str(pagination.get('step', 10))]
        elif pagination_type == 'page':
            query_params[pagination.get('page_param', 'page')] = [str(page + 1)]
            query_params[pagination.get('per_page_param', 'per_page')] = [str(pagination.get('per_page', 10))]
        elif pagination_type == 'cursor' and pagination.get('cursor'):
            query_params[pagination.get('cursor_param', 'cursor')] = [pagination['cursor']]
        
        # Add any additional query params from config
        extra_params = self.config.metadata.get('query_params', {})
        for key, value in extra_params.items():
            query_params[key] = [str(value)]
        
        # Rebuild URL
        new_query = urlencode(query_params, doseq=True)
        new_parsed = parsed._replace(query=new_query)
        
        return urlunparse(new_parsed)
    
    def _extract_path(self, data: Any, path: str) -> Any:
        """Extract value from nested dict/list using dot notation path"""
        if not path:
            return data
        
        keys = path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict):
                current = current.get(key)
            elif isinstance(current, list):
                try:
                    current = current[int(key)]
                except (ValueError, IndexError):
                    return None
            else:
                return None
            
            if current is None:
                return None
        
        return current
    
    def _process_item(self, item: dict) -> Optional[CollectedData]:
        """Process a single API response item"""
        content_path = self.config.metadata.get('content_path', 'content')
        title_path = self.config.metadata.get('title_path', 'title')
        id_path = self.config.metadata.get('id_path', 'id')
        url_path = self.config.metadata.get('url_path')
        author_path = self.config.metadata.get('author_path')
        date_path = self.config.metadata.get('date_path')
        
        content = self._extract_path(item, content_path)
        if content is None:
            content = str(item)
        else:
            content = str(content)
        
        title = self._extract_path(item, title_path)
        item_id = self._extract_path(item, id_path)
        item_url = self._extract_path(item, url_path)
        author = self._extract_path(item, author_path)
        
        # Parse date if provided
        published_date = None
        date_value = self._extract_path(item, date_path)
        if date_value:
            try:
                from datetime import datetime
                if isinstance(date_value, str):
                    published_date = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
            except:
                pass
        
        # Build source URL
        source_url = item_url or f"{self.config.url}/{item_id}" if item_id else self.config.url
        
        return CollectedData(
            content=content,
            source_url=source_url,
            source_type=SourceType.API,
            title=str(title) if title else None,
            author=str(author) if author else None,
            published_date=published_date,
            metadata={
                'api_endpoint': self.config.url,
                'item_id': item_id,
                'raw_data': item if self.config.metadata.get('store_raw', False) else None,
            },
            mime_type='application/json'
        )
    
    async def test_connection(self) -> bool:
        """Test connection to the API"""
        if not self.config.url:
            return False
        
        try:
            session = await self._get_session()
            
            # Make a simple request to test
            async with session.get(self.config.url, timeout=10) as response:
                # Accept 2xx and 4xx (might need auth for actual data)
                return response.status < 500
        except Exception as e:
            print(f"API connection test failed: {e}")
            return False
    
    async def close(self):
        """Close the session"""
        if self._session:
            await self._session.close()
            self._session = None
