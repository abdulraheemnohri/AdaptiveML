"""
YouTube Collector - Downloads transcripts and metadata from YouTube videos
"""

from datetime import datetime
from typing import List, Optional

from .base import BaseCollector, CollectorConfig, CollectedData, SourceType


class YouTubeCollector(BaseCollector):
    """Collects transcripts and metadata from YouTube videos"""
    
    def __init__(self, config: CollectorConfig):
        super().__init__(config)
        self._api_key = config.credentials.get("youtube_api_key") if config.credentials else None
    
    async def collect(self) -> List[CollectedData]:
        """Collect data from YouTube"""
        results = []
        
        urls = self._parse_urls()
        
        for url in urls[:self.config.max_items]:
            video_id = self._extract_video_id(url)
            if not video_id:
                continue
            
            try:
                data = await self._collect_video(video_id)
                if data:
                    results.append(data)
            except Exception as e:
                print(f"Error collecting YouTube video {video_id}: {e}")
                continue
        
        return results
    
    def _parse_urls(self) -> List[str]:
        """Parse URLs from config"""
        urls = []
        url = self.config.url
        
        if not url:
            return urls
        
        # Check if it's a playlist
        if "playlist" in url:
            # Would need to fetch playlist videos via API
            urls.append(url)
        else:
            urls.append(url)
        
        return urls
    
    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL"""
        import re
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n]+)',
            r'youtube\.com\/embed\/([^&\n]+)',
            r'youtube\.com\/v\/([^&\n]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    async def _collect_video(self, video_id: str) -> Optional[CollectedData]:
        """Collect transcript and metadata for a video"""
        # Try using youtube-transcript-api first (no API key needed)
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Prefer English transcript
            transcript = None
            try:
                transcript = transcript_list.find_manually_created_transcript(['en'])
            except:
                try:
                    transcript = transcript_list.find_generated_transcript(['en'])
                except:
                    # Fall back to any available transcript
                    transcript = transcript_list.find_transcript(['en'])
            
            if transcript:
                transcript_data = transcript.fetch()
                content = ' '.join([entry['text'] for entry in transcript_data])
                
                # Get video metadata
                metadata = await self._get_video_metadata(video_id)
                
                return CollectedData(
                    content=content,
                    source_url=f"https://www.youtube.com/watch?v={video_id}",
                    source_type=SourceType.YOUTUBE,
                    title=metadata.get('title'),
                    author=metadata.get('channel'),
                    published_date=metadata.get('published_at'),
                    metadata={
                        'video_id': video_id,
                        'duration': metadata.get('duration'),
                        'view_count': metadata.get('view_count'),
                        'language': transcript.language_code if hasattr(transcript, 'language_code') else 'en',
                        'transcript_type': 'manual' if transcript.is_manual else 'auto-generated',
                    },
                    mime_type='video/mp4'
                )
        except ImportError:
            print("youtube-transcript-api not installed, skipping transcript")
        except Exception as e:
            print(f"Transcript error: {e}")
        
        # Fallback: just return metadata if transcript unavailable
        metadata = await self._get_video_metadata(video_id)
        if metadata:
            return CollectedData(
                content=metadata.get('description', ''),
                source_url=f"https://www.youtube.com/watch?v={video_id}",
                source_type=SourceType.YOUTUBE,
                title=metadata.get('title'),
                author=metadata.get('channel'),
                published_date=metadata.get('published_at'),
                metadata={
                    'video_id': video_id,
                    'duration': metadata.get('duration'),
                    'view_count': metadata.get('view_count'),
                    'transcript_available': False,
                },
                mime_type='video/mp4'
            )
        
        return None
    
    async def _get_video_metadata(self, video_id: str) -> dict:
        """Get video metadata from YouTube Data API or oEmbed"""
        metadata = {
            'title': None,
            'channel': None,
            'description': None,
            'published_at': None,
            'duration': None,
            'view_count': None,
        }
        
        # Try oEmbed endpoint (no API key required)
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
                async with session.get(oembed_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        metadata['title'] = data.get('title')
                        metadata['channel'] = data.get('author_name')
        except Exception as e:
            print(f"oEmbed error: {e}")
        
        # If API key available, get full metadata
        if self._api_key:
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    api_url = f"https://www.googleapis.com/youtube/v3/videos?id={video_id}&part=snippet,contentDetails,statistics&key={self._api_key}"
                    async with session.get(api_url) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get('items'):
                                item = data['items'][0]
                                snippet = item['snippet']
                                content_details = item['contentDetails']
                                statistics = item['statistics']
                                
                                metadata.update({
                                    'title': snippet['title'],
                                    'channel': snippet['channelTitle'],
                                    'description': snippet['description'],
                                    'published_at': datetime.fromisoformat(snippet['publishedAt'].replace('Z', '+00:00')),
                                    'duration': content_details['duration'],
                                    'view_count': int(statistics.get('viewCount', 0)),
                                })
            except Exception as e:
                print(f"YouTube API error: {e}")
        
        return metadata
    
    async def test_connection(self) -> bool:
        """Test connection to YouTube"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get("https://www.youtube.com", timeout=10) as response:
                    return response.status == 200
        except:
            return False
