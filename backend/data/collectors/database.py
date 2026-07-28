"""
Database Collector - Fetches data from SQL and NoSQL databases
"""

from typing import List, Optional, Any, Dict

from .base import BaseCollector, CollectorConfig, CollectedData, SourceType


class DatabaseCollector(BaseCollector):
    """Collects data from SQL and NoSQL databases"""
    
    def __init__(self, config: CollectorConfig):
        super().__init__(config)
        self.db_type = config.metadata.get('db_type', 'postgresql')
        self.connection_string = config.credentials.get('connection_string') if config.credentials else None
    
    async def collect(self) -> List[CollectedData]:
        """Collect data from database"""
        results = []
        
        queries = self._get_queries()
        
        for query_config in queries:
            try:
                if self.db_type in ('postgresql', 'mysql', 'sqlite', 'mssql'):
                    data = await self._collect_sql(query_config)
                elif self.db_type == 'mongodb':
                    data = await self._collect_mongodb(query_config)
                else:
                    print(f"Unsupported database type: {self.db_type}")
                    continue
                
                results.extend(data)
            except Exception as e:
                print(f"Error collecting from database: {e}")
                continue
        
        return results
    
    def _get_queries(self) -> List[dict]:
        """Get queries from config"""
        queries = self.config.metadata.get('queries', [])
        
        if not queries and self.config.metadata.get('table'):
            # Default query for a table
            queries = [{
                'table': self.config.metadata['table'],
                'columns': self.config.metadata.get('columns', '*'),
                'where': self.config.metadata.get('where'),
                'order_by': self.config.metadata.get('order_by'),
                'limit': self.config.max_items,
            }]
        
        return queries
    
    async def _collect_sql(self, query_config: dict) -> List[CollectedData]:
        """Collect from SQL database"""
        results = []
        
        try:
            import asyncpg
            import sqlalchemy
            from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
            from sqlalchemy import text
        except ImportError:
            print("SQLAlchemy/asyncpg not installed for database collection")
            return results
        
        if not self.connection_string:
            print("No database connection string provided")
            return results
        
        # Build query
        table = query_config.get('table', '')
        columns = query_config.get('columns', '*')
        where = query_config.get('where')
        order_by = query_config.get('order_by')
        limit = query_config.get('limit', self.config.max_items)
        
        query = f"SELECT {columns} FROM {table}"
        params = {}
        
        if where:
            query += f" WHERE {where}"
        if order_by:
            query += f" ORDER BY {order_by}"
        query += f" LIMIT {limit}"
        
        try:
            engine = create_async_engine(self.connection_string, echo=False)
            
            async with AsyncSession(engine) as session:
                result = await session.execute(text(query), params)
                rows = result.fetchall()
                
                # Get column names
                column_names = list(result.keys()) if hasattr(result, 'keys') else []
                
                content_column = query_config.get('content_column', column_names[0] if column_names else None)
                title_column = query_config.get('title_column')
                id_column = query_config.get('id_column', column_names[0] if column_names else None)
                
                for row in rows:
                    row_dict = dict(zip(column_names, row)) if column_names else {}
                    
                    content = str(row_dict.get(content_column, '')) if content_column else str(row)
                    title = str(row_dict.get(title_column, '')) if title_column else None
                    
                    results.append(CollectedData(
                        content=content,
                        source_url=f"{self.db_type}://{table}/{row_dict.get(id_column, 'unknown')}",
                        source_type=SourceType.DATABASE,
                        title=title,
                        metadata={
                            'db_type': self.db_type,
                            'table': table,
                            'row_data': row_dict,
                            'columns': column_names,
                        },
                        mime_type='text/plain'
                    ))
            
            await engine.dispose()
        
        except Exception as e:
            print(f"SQL collection error: {e}")
        
        return results
    
    async def _collect_mongodb(self, query_config: dict) -> List[CollectedData]:
        """Collect from MongoDB"""
        results = []
        
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
        except ImportError:
            print("motor not installed for MongoDB collection")
            return results
        
        if not self.connection_string:
            print("No MongoDB connection string provided")
            return results
        
        collection_name = query_config.get('collection', '')
        filter_query = query_config.get('filter', {})
        projection = query_config.get('projection')
        limit = query_config.get('limit', self.config.max_items)
        sort = query_config.get('sort')
        
        try:
            client = AsyncIOMotorClient(self.connection_string)
            db_name = query_config.get('database', 'default')
            db = client[db_name]
            collection = db[collection_name]
            
            cursor = collection.find(filter_query, projection).limit(limit)
            
            if sort:
                cursor = cursor.sort(sort)
            
            content_field = query_config.get('content_field', 'content')
            title_field = query_config.get('title_field', 'title')
            id_field = query_config.get('id_field', '_id')
            
            async for doc in cursor:
                # Convert ObjectId to string
                doc['_id'] = str(doc['_id'])
                
                content = str(doc.get(content_field, ''))
                title = str(doc.get(title_field, '')) if title_field else None
                
                results.append(CollectedData(
                    content=content,
                    source_url=f"mongodb://{collection_name}/{doc['_id']}",
                    source_type=SourceType.DATABASE,
                    title=title,
                    metadata={
                        'db_type': 'mongodb',
                        'collection': collection_name,
                        'document': doc,
                    },
                    mime_type='text/plain'
                ))
            
            client.close()
        
        except Exception as e:
            print(f"MongoDB collection error: {e}")
        
        return results
    
    async def test_connection(self) -> bool:
        """Test database connection"""
        if not self.connection_string:
            return False
        
        try:
            if self.db_type in ('postgresql', 'mysql', 'sqlite', 'mssql'):
                from sqlalchemy.ext.asyncio import create_async_engine
                from sqlalchemy import text
                
                engine = create_async_engine(self.connection_string, echo=False)
                async with engine.connect() as conn:
                    await conn.execute(text("SELECT 1"))
                await engine.dispose()
                return True
            
            elif self.db_type == 'mongodb':
                from motor.motor_asyncio import AsyncIOMotorClient
                client = AsyncIOMotorClient(self.connection_string, serverSelectionTimeoutMS=5000)
                await client.admin.command('ping')
                client.close()
                return True
        
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False
        
        return False
