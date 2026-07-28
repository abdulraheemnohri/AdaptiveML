"""
Code Processor - Handles source code processing and analysis
"""

from typing import Any, Dict, Optional, List

from .base import BaseProcessor, ProcessorConfig, ProcessingResult, ProcessorType


class CodeProcessor(BaseProcessor):
    """Processes source code content"""
    
    # Language extensions mapping
    LANGUAGE_EXTENSIONS = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'javascript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        '.c': 'c',
        '.h': 'c',
        '.hpp': 'cpp',
        '.go': 'go',
        '.rs': 'rust',
        '.rb': 'ruby',
        '.php': 'php',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.cs': 'csharp',
        '.fs': 'fsharp',
        '.ex': 'elixir',
        '.exs': 'elixir',
        '.erl': 'erlang',
        '.hs': 'haskell',
        '.clj': 'clojure',
        '.lua': 'lua',
        '.r': 'r',
        '.R': 'r',
        '.sql': 'sql',
        '.sh': 'shell',
        '.bash': 'shell',
        '.zsh': 'shell',
        '.ps1': 'powershell',
        '.html': 'html',
        '.css': 'css',
        '.scss': 'scss',
        '.sass': 'sass',
        '.less': 'less',
        '.vue': 'vue',
        '.svelte': 'svelte',
        '.md': 'markdown',
        '.json': 'json',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.toml': 'toml',
        '.xml': 'xml',
    }
    
    def __init__(self, config: ProcessorConfig):
        super().__init__(config)
        self.language = config.options.get('language')
    
    async def process(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """Process code content"""
        original_content = content
        
        # Validate first
        errors = await self.validate(content)
        if errors:
            return ProcessingResult(
                content="",
                original_content=original_content,
                processor_type=ProcessorType.CODE,
                success=False,
                error="; ".join(errors),
            )
        
        # Detect language from metadata or file extension
        language = self.language
        if not language and metadata:
            ext = metadata.get('extension', '')
            language = self.LANGUAGE_EXTENSIONS.get(ext, 'unknown')
        
        # Normalize whitespace (preserve code structure but clean up)
        if self.config.normalize_whitespace:
            content = self._normalize_code_whitespace(content)
        
        # Calculate quality score specific to code
        quality_score = self._calculate_code_quality(content, language)
        
        # Extract code metrics
        metrics = self._extract_code_metrics(content, language)
        
        # Create chunks (by functions/classes for code)
        chunks = self._chunk_code(content, language)
        
        # Count tokens
        tokens_count = self._count_tokens(content)
        
        # Build metadata
        code_metadata = {
            'language': language,
            'metrics': metrics,
        }
        if metadata:
            code_metadata.update(metadata)
        
        return ProcessingResult(
            content=content,
            original_content=original_content,
            processor_type=ProcessorType.CODE,
            success=True,
            metadata=code_metadata,
            chunks=chunks,
            language=language,
            quality_score=quality_score,
            tokens_count=tokens_count,
        )
    
    def _normalize_code_whitespace(self, code: str) -> str:
        """Normalize code whitespace while preserving structure"""
        lines = code.split('\n')
        normalized = []
        
        for line in lines:
            # Remove trailing whitespace but preserve indentation
            normalized.append(line.rstrip())
        
        # Remove excessive blank lines (more than 2 consecutive)
        result = []
        blank_count = 0
        for line in normalized:
            if line.strip() == '':
                blank_count += 1
                if blank_count <= 2:
                    result.append(line)
            else:
                blank_count = 0
                result.append(line)
        
        return '\n'.join(result)
    
    def _calculate_code_quality(self, code: str, language: str) -> float:
        """Calculate quality score for code"""
        score = 1.0
        lines = code.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        
        # Penalize very short files
        if len(non_empty_lines) < 3:
            score -= 0.3
        
        # Penalize very long lines
        long_lines = sum(1 for l in non_empty_lines if len(l) > 120)
        if long_lines / max(len(non_empty_lines), 1) > 0.3:
            score -= 0.2
        
        # Bonus for having comments
        comment_ratio = self._estimate_comment_ratio(code, language)
        if 0.05 <= comment_ratio <= 0.3:
            score += 0.1
        
        # Penalize code with no structure (single long line)
        if len(lines) == 1 and len(code) > 500:
            score -= 0.4
        
        return max(0.0, min(1.0, score))
    
    def _estimate_comment_ratio(self, code: str, language: str) -> float:
        """Estimate ratio of comments in code"""
        comment_patterns = {
            'python': ['#'],
            'javascript': ['//', '/*'],
            'typescript': ['//', '/*'],
            'java': ['//', '/*'],
            'cpp': ['//', '/*'],
            'c': ['//', '/*'],
            'go': ['//'],
            'rust': ['//', '/*'],
            'ruby': ['#'],
            'php': ['//', '#', '/*'],
            'shell': ['#'],
        }
        
        patterns = comment_patterns.get(language, ['#'])
        lines = code.split('\n')
        comment_lines = 0
        
        for line in lines:
            stripped = line.strip()
            for pattern in patterns:
                if stripped.startswith(pattern):
                    comment_lines += 1
                    break
        
        return comment_lines / max(len(lines), 1)
    
    def _extract_code_metrics(self, code: str, language: str) -> Dict[str, Any]:
        """Extract metrics from code"""
        lines = code.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        
        metrics = {
            'total_lines': len(lines),
            'code_lines': len(non_empty_lines),
            'comment_lines': 0,
            'blank_lines': len(lines) - len(non_empty_lines),
            'average_line_length': sum(len(l) for l in non_empty_lines) / max(len(non_empty_lines), 1),
            'max_line_length': max((len(l) for l in non_empty_lines), default=0),
            'character_count': len(code),
        }
        
        # Count comment lines
        comment_patterns = {
            'python': ['#'],
            'javascript': ['//'],
            'typescript': ['//'],
            'java': ['//'],
            'cpp': ['//'],
            'go': ['//'],
            'rust': ['//'],
            'ruby': ['#'],
            'shell': ['#'],
        }
        
        patterns = comment_patterns.get(language, [])
        for line in non_empty_lines:
            stripped = line.strip()
            for pattern in patterns:
                if stripped.startswith(pattern):
                    metrics['comment_lines'] += 1
                    break
        
        # Estimate complexity (very rough)
        complexity_keywords = ['if', 'else', 'elif', 'for', 'while', 'switch', 'case', 
                               'try', 'catch', 'except', 'finally', 'with']
        complexity = sum(code.count(kw) for kw in complexity_keywords)
        metrics['estimated_complexity'] = complexity
        
        return metrics
    
    def _chunk_code(self, code: str, language: str) -> List[str]:
        """Split code into logical chunks (functions, classes)"""
        if not self.config.chunk_enabled:
            return [code]
        
        chunks = []
        
        # Try to split by function/class definitions
        if language == 'python':
            chunks = self._split_by_pattern(code, r'^(def |class |async def )', include_match=True)
        elif language in ('javascript', 'typescript'):
            chunks = self._split_by_pattern(code, r'^(function |const \w+ = \(|class )', include_match=True)
        elif language == 'java':
            chunks = self._split_by_pattern(code, r'^(public |private |protected |class )', include_match=True)
        
        # If no good splits found or chunks too large, use size-based chunking
        if not chunks or all(len(c) > self.config.chunk_size * 2 for c in chunks):
            chunks = self._chunk_text(code)
        
        return chunks
    
    def _split_by_pattern(self, text: str, pattern: str, include_match: bool = False) -> List[str]:
        """Split text by regex pattern"""
        import re
        
        matches = list(re.finditer(pattern, text, re.MULTILINE))
        if not matches:
            return [text]
        
        chunks = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            
            chunk = text[start:end]
            if not include_match:
                chunk = text[match.end():end]
            
            chunks.append(chunk.strip())
        
        return chunks
    
    async def extract_functions(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Extract function/method definitions from code"""
        functions = []
        
        if language == 'python':
            import re
            pattern = r'^(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*([^\n:]+))?:'
            
            for match in re.finditer(pattern, code, re.MULTILINE):
                functions.append({
                    'name': match.group(1),
                    'parameters': match.group(2),
                    'return_type': match.group(3),
                    'start': match.start(),
                })
        
        return functions
    
    async def extract_imports(self, code: str, language: str) -> List[str]:
        """Extract import statements from code"""
        imports = []
        
        if language == 'python':
            import re
            # Match: import x, from x import y
            pattern = r'^(?:import\s+[\w.,\s]+|from\s+\S+\s+import\s+(?:\S+|\([^)]+\)))'
            
            for match in re.finditer(pattern, code, re.MULTILINE):
                imports.append(match.group(0).strip())
        
        elif language in ('javascript', 'typescript'):
            import re
            pattern = r'^\s*(?:import\s+.*?from\s+[\'"].*?[\'"]|export\s+.*)'
            
            for match in re.finditer(pattern, code, re.MULTILINE):
                imports.append(match.group(0).strip())
        
        return imports
