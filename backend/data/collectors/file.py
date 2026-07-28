"""
File Collector - Reads local files (PDF, DOCX, Markdown, TXT, CSV, JSON, Excel)
"""

import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from .base import BaseCollector, CollectorConfig, CollectedData, SourceType


class FileCollector(BaseCollector):
    """Collects data from local files"""
    
    def __init__(self, config: CollectorConfig):
        super().__init__(config)
        self.supported_extensions = {
            '.pdf': self._read_pdf,
            '.docx': self._read_docx,
            '.md': self._read_markdown,
            '.markdown': self._read_markdown,
            '.txt': self._read_text,
            '.csv': self._read_csv,
            '.json': self._read_json,
            '.xlsx': self._read_excel,
            '.xls': self._read_excel,
        }
    
    async def collect(self) -> List[CollectedData]:
        """Collect data from files"""
        results = []
        
        path = self.config.path or self.config.url
        if not path:
            raise ValueError("Path is required for file collection")
        
        path_obj = Path(path)
        
        if path_obj.is_file():
            data = await self._collect_file(path_obj)
            if data:
                results.append(data)
        elif path_obj.is_dir():
            # Collect from directory
            extensions = self.config.metadata.get('extensions', list(self.supported_extensions.keys()))
            recursive = self.config.metadata.get('recursive', True)
            
            if recursive:
                files = path_obj.rglob('*')
            else:
                files = path_obj.glob('*')
            
            count = 0
            for file_path in files:
                if count >= self.config.max_items:
                    break
                if file_path.is_file() and file_path.suffix.lower() in extensions:
                    data = await self._collect_file(file_path)
                    if data:
                        results.append(data)
                        count += 1
        
        return results
    
    async def _collect_file(self, file_path: Path) -> Optional[CollectedData]:
        """Collect data from a single file"""
        ext = file_path.suffix.lower()
        
        if ext not in self.supported_extensions:
            print(f"Unsupported file type: {ext}")
            return None
        
        try:
            read_func = self.supported_extensions[ext]
            content, metadata = await read_func(file_path)
            
            # Determine source type
            source_type_map = {
                '.pdf': SourceType.PDF,
                '.docx': SourceType.DOCX,
                '.md': SourceType.MARKDOWN,
                '.markdown': SourceType.MARKDOWN,
                '.txt': SourceType.TXT,
                '.csv': SourceType.CSV,
                '.json': SourceType.JSON,
                '.xlsx': SourceType.EXCEL,
                '.xls': SourceType.EXCEL,
            }
            
            stat = file_path.stat()
            
            return CollectedData(
                content=content,
                source_url=f"file://{file_path.absolute()}",
                source_type=source_type_map.get(ext, SourceType.LOCAL_FOLDER),
                title=file_path.stem,
                published_date=datetime.fromtimestamp(stat.st_mtime),
                metadata={
                    'file_path': str(file_path.absolute()),
                    'file_size': stat.st_size,
                    'extension': ext,
                    **metadata
                },
                mime_type=self._get_mime_type(ext)
            )
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None
    
    async def _read_pdf(self, file_path: Path) -> tuple:
        """Read PDF file"""
        try:
            import pypdf
            reader = pypdf.PdfReader(str(file_path))
            text_parts = []
            for page in reader.pages:
                text_parts.append(page.extract_text())
            content = '\n'.join(text_parts)
            
            metadata = {
                'pages': len(reader.pages),
                'pdf_metadata': reader.metadata,
            }
            return content, metadata
        except ImportError:
            print("pypdf not installed, install with: pip install pypdf")
            return "", {}
    
    async def _read_docx(self, file_path: Path) -> tuple:
        """Read DOCX file"""
        try:
            from docx import Document
            doc = Document(str(file_path))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            content = '\n'.join(paragraphs)
            return content, {}
        except ImportError:
            print("python-docx not installed, install with: pip install python-docx")
            return "", {}
    
    async def _read_markdown(self, file_path: Path) -> tuple:
        """Read Markdown file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content, {}
    
    async def _read_text(self, file_path: Path) -> tuple:
        """Read plain text file"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return content, {}
    
    async def _read_csv(self, file_path: Path) -> tuple:
        """Read CSV file"""
        import csv
        rows = []
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append(','.join(row))
        content = '\n'.join(rows)
        return content, {'rows': len(rows)}
    
    async def _read_json(self, file_path: Path) -> tuple:
        """Read JSON file"""
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Convert to string representation
        content = json.dumps(data, indent=2)
        return content, {'keys': list(data.keys()) if isinstance(data, dict) else 'array'}
    
    async def _read_excel(self, file_path: Path) -> tuple:
        """Read Excel file"""
        try:
            import pandas as pd
            df = pd.read_excel(str(file_path))
            content = df.to_string()
            return content, {'sheets': 1, 'rows': len(df), 'columns': len(df.columns)}
        except ImportError:
            print("pandas not installed, install with: pip install pandas openpyxl")
            return "", {}
    
    def _get_mime_type(self, ext: str) -> str:
        """Get MIME type for file extension"""
        mime_types = {
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.md': 'text/markdown',
            '.markdown': 'text/markdown',
            '.txt': 'text/plain',
            '.csv': 'text/csv',
            '.json': 'application/json',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.xls': 'application/vnd.ms-excel',
        }
        return mime_types.get(ext, 'application/octet-stream')
    
    async def test_connection(self) -> bool:
        """Test if the file/directory exists and is accessible"""
        path = self.config.path or self.config.url
        if not path:
            return False
        
        path_obj = Path(path)
        return path_obj.exists() and (path_obj.is_file() or path_obj.is_dir())
