"""
Multimodal Processor - Handles image, audio, and video processing
"""

import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base import BaseProcessor, ProcessorConfig, ProcessingResult, ProcessorType


class MultimodalProcessor(BaseProcessor):
    """Processes multimodal content (images, audio, video)"""
    
    def __init__(self, config: ProcessorConfig):
        super().__init__(config)
    
    async def process(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """Process multimodal content"""
        original_content = content
        
        # Determine media type from metadata
        media_type = None
        if metadata:
            media_type = metadata.get('media_type') or metadata.get('mime_type')
        
        if not media_type:
            # Try to detect from content (if it's a file path or base64)
            if content.startswith('data:'):
                media_type = content.split(';')[0].split(':')[1]
            elif content.endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
                media_type = 'image'
            elif content.endswith(('.mp3', '.wav', '.ogg', '.flac')):
                media_type = 'audio'
            elif content.endswith(('.mp4', '.avi', '.mkv', '.webm')):
                media_type = 'video'
        
        if not media_type:
            media_type = 'unknown'
        
        # Route to appropriate processor
        if media_type.startswith('image') or media_type == 'image':
            return await self._process_image(content, metadata)
        elif media_type.startswith('audio') or media_type == 'audio':
            return await self._process_audio(content, metadata)
        elif media_type.startswith('video') or media_type == 'video':
            return await self._process_video(content, metadata)
        else:
            return ProcessingResult(
                content=content,
                original_content=original_content,
                processor_type=ProcessorType.MULTIMODAL,
                success=False,
                error=f"Unknown media type: {media_type}",
            )
    
    async def _process_image(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """Process image content"""
        result_metadata = {'media_type': 'image'}
        
        try:
            # If content is a file path, load the image
            image_path = None
            if content.startswith(('http://', 'https://')):
                # URL - download first
                image_path = await self._download_file(content)
            elif Path(content).exists():
                image_path = content
            
            if image_path:
                # Extract EXIF and other metadata
                exif_data = await self._extract_image_metadata(image_path)
                result_metadata.update(exif_data)
                
                # Generate caption/description using vision model if available
                caption = await self._generate_image_caption(image_path)
                if caption:
                    result_metadata['caption'] = caption
                
                # Extract colors
                colors = await self._extract_dominant_colors(image_path)
                result_metadata['dominant_colors'] = colors
                
                # Detect objects/scenes if models available
                detections = await self._detect_objects(image_path)
                if detections:
                    result_metadata['detected_objects'] = detections
                
                # Create text representation for training
                text_content = self._build_image_description(result_metadata)
                
            elif content.startswith('data:image'):
                # Base64 encoded image
                result_metadata['encoded'] = True
                text_content = "[Base64 encoded image]"
            else:
                text_content = content
            
            return ProcessingResult(
                content=text_content,
                original_content=content,
                processor_type=ProcessorType.IMAGE,
                success=True,
                metadata=result_metadata,
                language='en',
                quality_score=0.9,
                tokens_count=self._count_tokens(text_content),
            )
        
        except Exception as e:
            return ProcessingResult(
                content="",
                original_content=content,
                processor_type=ProcessorType.IMAGE,
                success=False,
                error=str(e),
                metadata=result_metadata,
            )
    
    async def _process_audio(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """Process audio content"""
        result_metadata = {'media_type': 'audio'}
        
        try:
            audio_path = None
            if content.startswith(('http://', 'https://')):
                audio_path = await self._download_file(content)
            elif Path(content).exists():
                audio_path = content
            
            if audio_path:
                # Extract audio metadata
                audio_meta = await self._extract_audio_metadata(audio_path)
                result_metadata.update(audio_meta)
                
                # Transcribe speech to text if available
                transcription = await self._transcribe_audio(audio_path)
                if transcription:
                    result_metadata['transcription'] = transcription
                    
                    # Also create chunks from transcription
                    chunks = self._chunk_text(transcription)
                    
                    return ProcessingResult(
                        content=transcription,
                        original_content=content,
                        processor_type=ProcessorType.AUDIO,
                        success=True,
                        metadata=result_metadata,
                        chunks=chunks,
                        language=result_metadata.get('language'),
                        quality_score=0.9,
                        tokens_count=self._count_tokens(transcription),
                    )
                
                text_content = f"[Audio file: {result_metadata.get('filename', 'unknown')}]"
            else:
                text_content = content
                chunks = []
            
            return ProcessingResult(
                content=text_content,
                original_content=content,
                processor_type=ProcessorType.AUDIO,
                success=True,
                metadata=result_metadata,
                chunks=chunks if 'chunks' in dir() else [],
                language=None,
                quality_score=0.8,
                tokens_count=self._count_tokens(text_content),
            )
        
        except Exception as e:
            return ProcessingResult(
                content="",
                original_content=content,
                processor_type=ProcessorType.AUDIO,
                success=False,
                error=str(e),
                metadata=result_metadata,
            )
    
    async def _process_video(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """Process video content"""
        result_metadata = {'media_type': 'video'}
        
        try:
            video_path = None
            if content.startswith(('http://', 'https://')):
                video_path = await self._download_file(content)
            elif Path(content).exists():
                video_path = content
            
            if video_path:
                # Extract video metadata
                video_meta = await self._extract_video_metadata(video_path)
                result_metadata.update(video_meta)
                
                # Extract frames for analysis
                frames = await self._extract_key_frames(video_path)
                if frames:
                    result_metadata['key_frames'] = len(frames)
                    
                    # Analyze frames
                    frame_descriptions = []
                    for frame in frames[:5]:  # Limit to 5 frames
                        desc = await self._generate_image_caption(frame)
                        if desc:
                            frame_descriptions.append(desc)
                    
                    if frame_descriptions:
                        result_metadata['scene_descriptions'] = frame_descriptions
                
                # Extract and transcribe audio track
                audio_transcript = await self._extract_audio_from_video(video_path)
                if audio_transcript:
                    result_metadata['audio_transcript'] = audio_transcript
                
                # Build comprehensive description
                text_content = self._build_video_description(result_metadata)
                
                # Create chunks from transcript if available
                chunks = []
                if audio_transcript:
                    chunks = self._chunk_text(audio_transcript)
            else:
                text_content = content
                chunks = []
            
            return ProcessingResult(
                content=text_content,
                original_content=content,
                processor_type=ProcessorType.VIDEO,
                success=True,
                metadata=result_metadata,
                chunks=chunks,
                language=result_metadata.get('language'),
                quality_score=0.85,
                tokens_count=self._count_tokens(text_content),
            )
        
        except Exception as e:
            return ProcessingResult(
                content="",
                original_content=content,
                processor_type=ProcessorType.VIDEO,
                success=False,
                error=str(e),
                metadata=result_metadata,
            )
    
    async def _download_file(self, url: str) -> str:
        """Download file from URL to temp location"""
        import aiohttp
        import tempfile
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.read()
                    
                    # Save to temp file
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=self._get_extension_from_url(url))
                    temp_file.write(content)
                    temp_file.close()
                    
                    return temp_file.name
        
        raise Exception(f"Failed to download file from {url}")
    
    def _get_extension_from_url(self, url: str) -> str:
        """Get file extension from URL"""
        ext_map = {
            'image/jpeg': '.jpg',
            'image/png': '.png',
            'image/gif': '.gif',
            'image/webp': '.webp',
            'audio/mpeg': '.mp3',
            'audio/wav': '.wav',
            'video/mp4': '.mp4',
            'video/webm': '.webm',
        }
        # Simple heuristic based on URL
        for mime, ext in ext_map.items():
            if mime.split('/')[1] in url.lower():
                return ext
        return ''
    
    async def _extract_image_metadata(self, image_path: str) -> Dict[str, Any]:
        """Extract EXIF and other metadata from image"""
        metadata = {'filename': Path(image_path).name}
        
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS
            
            img = Image.open(image_path)
            
            # Basic info
            metadata['width'] = img.width
            metadata['height'] = img.height
            metadata['format'] = img.format
            metadata['mode'] = img.mode
            
            # EXIF data
            exif_data = img._getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if isinstance(value, (str, int, float)):
                        metadata[f'exif_{tag.lower()}'] = value
            
            img.close()
        
        except ImportError:
            pass
        except Exception as e:
            metadata['error'] = str(e)
        
        return metadata
    
    async def _extract_audio_metadata(self, audio_path: str) -> Dict[str, Any]:
        """Extract metadata from audio file"""
        metadata = {'filename': Path(audio_path).name}
        
        try:
            # Try using mutagen for metadata
            from mutagen import File
            audio = File(audio_path)
            
            if audio:
                metadata['duration'] = audio.info.length
                metadata['sample_rate'] = getattr(audio.info, 'sample_rate', None)
                metadata['channels'] = getattr(audio.info, 'channels', None)
                metadata['bits_per_sample'] = getattr(audio.info, 'bits_per_sample', None)
                
                # ID3 tags
                if hasattr(audio, 'tags') and audio.tags:
                    metadata['title'] = str(audio.tags.get('TIT2', ''))
                    metadata['artist'] = str(audio.tags.get('TPE1', ''))
                    metadata['album'] = str(audio.tags.get('TALB', ''))
        
        except ImportError:
            pass
        except Exception as e:
            metadata['error'] = str(e)
        
        return metadata
    
    async def _extract_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Extract metadata from video file"""
        metadata = {'filename': Path(video_path).name}
        
        try:
            # Try using mutagen or moviepy
            from mutagen import File
            video = File(video_path)
            
            if video:
                metadata['duration'] = video.info.length
                
            # Try moviepy for more detailed info
            try:
                from moviepy.editor import VideoFileClip
                clip = VideoFileClip(video_path)
                metadata['fps'] = clip.fps
                metadata['resolution'] = f"{clip.size[0]}x{clip.size[1]}"
                clip.close()
            except:
                pass
        
        except ImportError:
            pass
        except Exception as e:
            metadata['error'] = str(e)
        
        return metadata
    
    async def _generate_image_caption(self, image_path: str) -> Optional[str]:
        """Generate caption for image using vision model"""
        # This would integrate with a vision model like BLIP, LLaVA, etc.
        # For now, return None - can be extended later
        return None
    
    async def _extract_dominant_colors(self, image_path: str) -> List[str]:
        """Extract dominant colors from image"""
        try:
            from PIL import Image
            import colorsys
            
            img = Image.open(image_path)
            img.thumbnail((100, 100))  # Resize for faster processing
            pixels = list(img.getdata())
            
            # Count colors
            color_count = {}
            for pixel in pixels:
                if len(pixel) >= 3:
                    r, g, b = pixel[:3]
                    # Quantize colors
                    r, g, b = r // 32 * 32, g // 32 * 32, b // 32 * 32
                    color = (r, g, b)
                    color_count[color] = color_count.get(color, 0) + 1
            
            # Get top 5 colors
            top_colors = sorted(color_count.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Convert to hex
            hex_colors = ['#%02x%02x%02x' % c[0] for c, _ in top_colors]
            
            img.close()
            return hex_colors
        
        except:
            return []
    
    async def _detect_objects(self, image_path: str) -> Optional[List[Dict]]:
        """Detect objects in image"""
        # Would integrate with detection models like YOLO, DETR, etc.
        return None
    
    async def _transcribe_audio(self, audio_path: str) -> Optional[str]:
        """Transcribe audio to text"""
        # Would integrate with Whisper or similar
        return None
    
    async def _extract_key_frames(self, video_path: str) -> Optional[List[str]]:
        """Extract key frames from video"""
        # Would use OpenCV or moviepy
        return None
    
    async def _extract_audio_from_video(self, video_path: str) -> Optional[str]:
        """Extract and transcribe audio from video"""
        # Would extract audio track and transcribe
        return None
    
    def _build_image_description(self, metadata: Dict) -> str:
        """Build text description of image"""
        parts = [f"Image: {metadata.get('filename', 'unknown')}"]
        
        if metadata.get('width') and metadata.get('height'):
            parts.append(f"Size: {metadata['width']}x{metadata['height']}")
        
        if metadata.get('format'):
            parts.append(f"Format: {metadata['format']}")
        
        if metadata.get('caption'):
            parts.append(f"Description: {metadata['caption']}")
        
        if metadata.get('dominant_colors'):
            parts.append(f"Colors: {', '.join(metadata['dominant_colors'])}")
        
        if metadata.get('detected_objects'):
            objects = metadata['detected_objects']
            parts.append(f"Objects: {', '.join(objects)}")
        
        return '. '.join(parts)
    
    def _build_video_description(self, metadata: Dict) -> str:
        """Build text description of video"""
        parts = [f"Video: {metadata.get('filename', 'unknown')}"]
        
        if metadata.get('duration'):
            duration = metadata['duration']
            parts.append(f"Duration: {duration:.1f}s")
        
        if metadata.get('resolution'):
            parts.append(f"Resolution: {metadata['resolution']}")
        
        if metadata.get('fps'):
            parts.append(f"FPS: {metadata['fps']}")
        
        if metadata.get('audio_transcript'):
            parts.append(f"Audio: Transcript available ({len(metadata['audio_transcript'])} chars)")
        
        if metadata.get('scene_descriptions'):
            parts.append(f"Scenes: {len(metadata['scene_descriptions'])} key scenes described")
        
        return '. '.join(parts)
