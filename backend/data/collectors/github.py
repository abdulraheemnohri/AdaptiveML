"""
GitHub Collector - Fetches code, issues, PRs, and documentation from GitHub repositories
"""

from datetime import datetime
from typing import List, Optional

from .base import BaseCollector, CollectorConfig, CollectedData, SourceType


class GitHubCollector(BaseCollector):
    """Collects data from GitHub repositories"""
    
    def __init__(self, config: CollectorConfig):
        super().__init__(config)
        self._token = config.credentials.get("github_token") if config.credentials else None
        self._base_url = "https://api.github.com"
    
    async def collect(self) -> List[CollectedData]:
        """Collect data from GitHub"""
        results = []
        
        url = self.config.url
        if not url:
            raise ValueError("URL is required for GitHub collection")
        
        # Parse GitHub URL
        repo_info = self._parse_github_url(url)
        if not repo_info:
            print(f"Invalid GitHub URL: {url}")
            return results
        
        owner, repo = repo_info['owner'], repo_info['repo']
        content_type = self.config.metadata.get('content_type', 'readme')
        
        # Collect based on content type
        if content_type == 'readme':
            data = await self._collect_readme(owner, repo)
            if data:
                results.append(data)
        elif content_type == 'code':
            code_results = await self._collect_code(owner, repo)
            results.extend(code_results)
        elif content_type == 'issues':
            issue_results = await self._collect_issues(owner, repo)
            results.extend(issue_results)
        elif content_type == 'prs':
            pr_results = await self._collect_prs(owner, repo)
            results.extend(pr_results)
        elif content_type == 'wiki':
            wiki_results = await self._collect_wiki(owner, repo)
            results.extend(wiki_results)
        elif content_type == 'all':
            # Collect everything
            readme_data = await self._collect_readme(owner, repo)
            if readme_data:
                results.append(readme_data)
            
            code_results = await self._collect_code(owner, repo)
            results.extend(code_results[:50])  # Limit code files
            
            issue_results = await self._collect_issues(owner, repo)
            results.extend(issue_results[:20])
        
        return results
    
    def _parse_github_url(self, url: str) -> Optional[dict]:
        """Parse GitHub repository URL"""
        import re
        
        patterns = [
            r'github\.com/([^/]+)/([^/]+?)(?:\.git)?(?:/|$)',
            r'github\.com/([^/]+)/([^/]+?)/(?:tree|blob)/[^/]+/(.*)?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                groups = match.groups()
                result = {'owner': groups[0], 'repo': groups[1]}
                if len(groups) > 2 and groups[2]:
                    result['path'] = groups[2]
                return result
        
        return None
    
    async def _make_request(self, endpoint: str) -> Optional[dict]:
        """Make authenticated request to GitHub API"""
        import aiohttp
        
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "AdaptiveOmniML/1.0"
        }
        
        if self._token:
            headers["Authorization"] = f"token {self._token}"
        
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = f"{self._base_url}{endpoint}"
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 403:
                        print(f"Rate limited by GitHub: {endpoint}")
                    else:
                        print(f"GitHub API error {response.status}: {endpoint}")
        except Exception as e:
            print(f"GitHub request error: {e}")
        
        return None
    
    async def _collect_readme(self, owner: str, repo: str) -> Optional[CollectedData]:
        """Collect README file"""
        data = await self._make_request(f"/repos/{owner}/{repo}/readme")
        
        if data:
            import base64
            content = base64.b64decode(data['content']).decode('utf-8')
            
            return CollectedData(
                content=content,
                source_url=f"https://github.com/{owner}/{repo}",
                source_type=SourceType.GITHUB,
                title=f"README - {repo}",
                published_date=datetime.utcnow(),
                metadata={
                    'type': 'readme',
                    'encoding': data.get('encoding'),
                    'path': data.get('path'),
                    'owner': owner,
                    'repo': repo,
                },
                mime_type='text/markdown'
            )
        
        return None
    
    async def _collect_code(self, owner: str, repo: str) -> List[CollectedData]:
        """Collect source code files"""
        results = []
        
        # Get repository tree
        tree_data = await self._make_request(f"/repos/{owner}/{repo}/git/trees/main?recursive=1")
        if not tree_data:
            # Try master branch
            tree_data = await self._make_request(f"/repos/{owner}/{repo}/git/trees/master?recursive=1")
        
        if not tree_data or 'tree' not in tree_data:
            return results
        
        # Filter for code files
        code_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', 
                          '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala', '.cs',
                          '.html', '.css', '.scss', '.vue', '.svelte'}
        
        count = 0
        for item in tree_data['tree']:
            if count >= self.config.max_items:
                break
            
            if item['type'] == 'blob':
                path = item['path']
                ext = '.' + path.split('.')[-1] if '.' in path else ''
                
                if ext in code_extensions or path.endswith(('.md', '.txt', '.json', '.yaml', '.yml')):
                    # Skip large files
                    if item.get('size', 0) > 100000:  # 100KB limit
                        continue
                    
                    file_data = await self._make_request(f"/repos/{owner}/{repo}/contents/{path}")
                    if file_data:
                        import base64
                        content = base64.b64decode(file_data['content']).decode('utf-8', errors='ignore')
                        
                        results.append(CollectedData(
                            content=content,
                            source_url=f"https://github.com/{owner}/{repo}/blob/main/{path}",
                            source_type=SourceType.GITHUB,
                            title=path,
                            published_date=datetime.utcnow(),
                            metadata={
                                'type': 'code',
                                'path': path,
                                'extension': ext,
                                'size': item.get('size'),
                                'owner': owner,
                                'repo': repo,
                            },
                            mime_type=self._get_mime_type(ext)
                        ))
                        count += 1
        
        return results
    
    async def _collect_issues(self, owner: str, repo: str) -> List[CollectedData]:
        """Collect issues"""
        results = []
        
        data = await self._make_request(f"/repos/{owner}/{repo}/issues?state=all&per_page={min(self.config.max_items, 100)}")
        
        if data:
            for issue in data:
                # Skip pull requests (they appear in issues endpoint)
                if 'pull_request' in issue:
                    continue
                
                content = f"# {issue['title']}\n\n{issue.get('body', '')}"
                
                results.append(CollectedData(
                    content=content,
                    source_url=issue['html_url'],
                    source_type=SourceType.GITHUB,
                    title=issue['title'],
                    author=issue['user']['login'],
                    published_date=datetime.fromisoformat(issue['created_at'].replace('Z', '+00:00')),
                    metadata={
                        'type': 'issue',
                        'number': issue['number'],
                        'state': issue['state'],
                        'labels': [label['name'] for label in issue.get('labels', [])],
                        'comments': issue.get('comments', 0),
                        'owner': owner,
                        'repo': repo,
                    },
                    mime_type='text/markdown'
                ))
        
        return results
    
    async def _collect_prs(self, owner: str, repo: str) -> List[CollectedData]:
        """Collect pull requests"""
        results = []
        
        data = await self._make_request(f"/repos/{owner}/{repo}/pulls?state=all&per_page={min(self.config.max_items, 100)}")
        
        if data:
            for pr in data:
                content = f"# {pr['title']}\n\n{pr.get('body', '')}"
                
                results.append(CollectedData(
                    content=content,
                    source_url=pr['html_url'],
                    source_type=SourceType.GITHUB,
                    title=pr['title'],
                    author=pr['user']['login'],
                    published_date=datetime.fromisoformat(pr['created_at'].replace('Z', '+00:00')),
                    metadata={
                        'type': 'pull_request',
                        'number': pr['number'],
                        'state': pr['state'],
                        'merged': pr.get('merged', False),
                        'additions': pr.get('additions', 0),
                        'deletions': pr.get('deletions', 0),
                        'owner': owner,
                        'repo': repo,
                    },
                    mime_type='text/markdown'
                ))
        
        return results
    
    async def _collect_wiki(self, owner: str, repo: str) -> List[CollectedData]:
        """Collect wiki pages"""
        results = []
        
        # Wiki is accessed via git trees in special refs
        data = await self._make_request(f"/repos/{owner}/{repo}/git/trees/wiki?recursive=1")
        
        if data and 'tree' in data:
            for item in data['tree']:
                if item['type'] == 'blob' and item['path'].endswith('.md'):
                    page_data = await self._make_request(f"/repos/{owner}/{repo}/contents/wiki/{item['path']}")
                    if page_data:
                        import base64
                        content = base64.b64decode(page_data['content']).decode('utf-8')
                        
                        results.append(CollectedData(
                            content=content,
                            source_url=f"https://github.com/{owner}/{repo}/wiki/{page_data['path'][:-3]}",
                            source_type=SourceType.GITHUB,
                            title=page_data['path'][:-3],
                            published_date=datetime.utcnow(),
                            metadata={
                                'type': 'wiki',
                                'path': page_data['path'],
                                'owner': owner,
                                'repo': repo,
                            },
                            mime_type='text/markdown'
                        ))
        
        return results
    
    def _get_mime_type(self, ext: str) -> str:
        """Get MIME type for file extension"""
        mime_types = {
            '.py': 'text/x-python',
            '.js': 'text/javascript',
            '.ts': 'text/typescript',
            '.jsx': 'text/jsx',
            '.tsx': 'text/tsx',
            '.java': 'text/x-java',
            '.cpp': 'text/x-c++src',
            '.c': 'text/x-csrc',
            '.h': 'text/x-chdr',
            '.go': 'text/x-go',
            '.rs': 'text/x-rustsrc',
            '.rb': 'text/x-ruby',
            '.php': 'text/x-php',
            '.swift': 'text/x-swift',
            '.kt': 'text/x-kotlin',
            '.html': 'text/html',
            '.css': 'text/css',
            '.md': 'text/markdown',
            '.json': 'application/json',
            '.yaml': 'text/yaml',
            '.yml': 'text/yaml',
        }
        return mime_types.get(ext, 'text/plain')
    
    async def test_connection(self) -> bool:
        """Test connection to GitHub API"""
        try:
            data = await self._make_request("/")
            return data is not None
        except:
            return False
