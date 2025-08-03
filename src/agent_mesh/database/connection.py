"""Database connection and session management.

Provides database connectivity using SQLAlchemy with support for
SQLite (development) and PostgreSQL (production).
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import structlog
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import (
    AsyncSession, 
    create_async_engine, 
    async_sessionmaker
)
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from .models import Base


logger = structlog.get_logger("database")


class DatabaseManager:
    """
    Database connection and session manager.
    
    Handles both synchronous and asynchronous database operations
    with connection pooling, health monitoring, and migration support.
    """
    
    def __init__(
        self,
        database_url: str,
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_timeout: int = 30,
        auto_migrate: bool = True,
        echo: bool = False
    ):
        """
        Initialize database manager.
        
        Args:
            database_url: Database connection URL
            pool_size: Connection pool size
            max_overflow: Maximum overflow connections
            pool_timeout: Pool timeout in seconds
            auto_migrate: Whether to auto-run migrations
            echo: Whether to echo SQL statements
        """
        self.database_url = database_url
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.auto_migrate = auto_migrate
        self.echo = echo
        
        # Engine and session factories
        self.sync_engine: Optional[Engine] = None
        self.async_engine = None
        self.sync_session_factory: Optional[sessionmaker] = None
        self.async_session_factory = None
        
        # Health monitoring
        self._health_check_interval = 60  # seconds
        self._health_check_task: Optional[asyncio.Task] = None
        self._is_healthy = False
    
    def initialize_sync(self) -> None:
        """Initialize synchronous database connection."""
        logger.info("Initializing synchronous database connection", 
                   url=self._mask_password(self.database_url))
        
        # Configure engine based on database type
        if self.database_url.startswith("sqlite"):
            # SQLite configuration
            connect_args = {
                "check_same_thread": False,
                "timeout": self.pool_timeout
            }
            
            self.sync_engine = create_engine(
                self.database_url,
                echo=self.echo,
                connect_args=connect_args
            )
        else:
            # PostgreSQL configuration
            self.sync_engine = create_engine(
                self.database_url,
                echo=self.echo,
                poolclass=QueuePool,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_timeout=self.pool_timeout,
                pool_pre_ping=True,
                pool_recycle=3600  # 1 hour
            )
        
        # Configure SQLite for better concurrency
        if self.database_url.startswith("sqlite"):
            @event.listens_for(self.sync_engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA cache_size=10000")
                cursor.execute("PRAGMA temp_store=MEMORY")
                cursor.close()
        
        # Create session factory
        self.sync_session_factory = sessionmaker(
            bind=self.sync_engine,
            autocommit=False,
            autoflush=False
        )
        
        # Auto-migrate if enabled
        if self.auto_migrate:
            self.create_tables()
        
        logger.info("Synchronous database connection initialized")
    
    async def initialize_async(self) -> None:
        """Initialize asynchronous database connection."""
        logger.info("Initializing asynchronous database connection",
                   url=self._mask_password(self.database_url))
        
        # Convert sync URL to async URL if needed
        async_url = self.database_url
        if async_url.startswith("sqlite:///"):
            async_url = async_url.replace("sqlite:///", "sqlite+aiosqlite:///")
        elif async_url.startswith("postgresql://"):
            async_url = async_url.replace("postgresql://", "postgresql+asyncpg://")
        
        # Configure async engine
        if async_url.startswith("sqlite"):
            # SQLite async configuration
            self.async_engine = create_async_engine(
                async_url,
                echo=self.echo,
                connect_args={"check_same_thread": False}
            )
        else:
            # PostgreSQL async configuration
            self.async_engine = create_async_engine(
                async_url,
                echo=self.echo,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_timeout=self.pool_timeout,
                pool_pre_ping=True,
                pool_recycle=3600
            )
        
        # Create async session factory
        self.async_session_factory = async_sessionmaker(
            bind=self.async_engine,
            class_=AsyncSession,
            autocommit=False,
            autoflush=False
        )
        
        # Auto-migrate if enabled
        if self.auto_migrate:
            await self.create_tables_async()
        
        # Start health monitoring
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info("Asynchronous database connection initialized")
    
    def create_tables(self) -> None:
        """Create database tables synchronously."""
        logger.info("Creating database tables")
        
        if not self.sync_engine:
            raise RuntimeError("Sync engine not initialized")
        
        Base.metadata.create_all(bind=self.sync_engine)
        logger.info("Database tables created successfully")
    
    async def create_tables_async(self) -> None:
        """Create database tables asynchronously."""
        logger.info("Creating database tables (async)")
        
        if not self.async_engine:
            raise RuntimeError("Async engine not initialized")
        
        async with self.async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Database tables created successfully (async)")
    
    def get_sync_session(self) -> Session:
        """Get synchronous database session."""
        if not self.sync_session_factory:
            raise RuntimeError("Sync session factory not initialized")
        
        return self.sync_session_factory()
    
    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get asynchronous database session context manager."""
        if not self.async_session_factory:
            raise RuntimeError("Async session factory not initialized")
        
        async with self.async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def health_check(self) -> bool:
        """Perform database health check."""
        try:
            if self.async_engine:
                async with self.async_engine.begin() as conn:
                    await conn.execute("SELECT 1")
                return True
            elif self.sync_engine:
                with self.sync_engine.connect() as conn:
                    conn.execute("SELECT 1")
                return True
            else:
                return False
        except Exception as e:
            logger.error("Database health check failed", error=str(e))
            return False
    
    async def close(self) -> None:
        """Close database connections."""
        logger.info("Closing database connections")
        
        # Stop health monitoring
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close async engine
        if self.async_engine:
            await self.async_engine.dispose()
        
        # Close sync engine
        if self.sync_engine:
            self.sync_engine.dispose()
        
        logger.info("Database connections closed")
    
    def get_connection_info(self) -> dict:
        """Get database connection information."""
        return {
            "database_url": self._mask_password(self.database_url),
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "pool_timeout": self.pool_timeout,
            "is_healthy": self._is_healthy,
            "auto_migrate": self.auto_migrate
        }
    
    # Private methods
    
    def _mask_password(self, url: str) -> str:
        """Mask password in database URL for logging."""
        # Simple password masking for security
        if "://" in url and "@" in url:
            parts = url.split("://")
            if len(parts) == 2:
                scheme = parts[0]
                rest = parts[1]
                if "@" in rest:
                    auth_part, host_part = rest.split("@", 1)
                    if ":" in auth_part:
                        user, _ = auth_part.split(":", 1)
                        return f"{scheme}://{user}:***@{host_part}"
        return url
    
    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while True:
            try:
                self._is_healthy = await self.health_check()
                
                if not self._is_healthy:
                    logger.warning("Database health check failed")
                
                await asyncio.sleep(self._health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health check loop error", error=str(e))
                await asyncio.sleep(self._health_check_interval)


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def initialize_database(
    database_url: Optional[str] = None,
    **kwargs
) -> DatabaseManager:
    """Initialize global database manager."""
    global _db_manager
    
    if not database_url:
        database_url = os.getenv(
            "DATABASE_URL", 
            "sqlite:///data/agent_mesh.db"
        )
    
    _db_manager = DatabaseManager(database_url, **kwargs)
    return _db_manager


def get_database_manager() -> DatabaseManager:
    """Get global database manager instance."""
    global _db_manager
    
    if _db_manager is None:
        _db_manager = initialize_database()
    
    return _db_manager


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session context manager."""
    db_manager = get_database_manager()
    
    if not db_manager.async_session_factory:
        await db_manager.initialize_async()
    
    async with db_manager.get_async_session() as session:
        yield session


def get_sync_db_session() -> Session:
    """Get synchronous database session."""
    db_manager = get_database_manager()
    
    if not db_manager.sync_session_factory:
        db_manager.initialize_sync()
    
    return db_manager.get_sync_session()