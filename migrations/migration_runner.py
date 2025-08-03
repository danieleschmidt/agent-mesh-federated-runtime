"""Database migration runner and management utility.

Provides tools for applying, rolling back, and managing database schema migrations
for the Agent Mesh system.
"""

import asyncio
import importlib
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.agent_mesh.database.connection import DatabaseManager

logger = structlog.get_logger("migration_runner")


class MigrationRunner:
    """Database migration runner and manager."""
    
    def __init__(self, migrations_dir: Optional[Path] = None):
        """Initialize migration runner.
        
        Args:
            migrations_dir: Directory containing migration files
        """
        self.migrations_dir = migrations_dir or Path(__file__).parent
        self.db_manager = DatabaseManager()
        
    async def initialize(self) -> None:
        """Initialize database connection."""
        await self.db_manager.initialize()
    
    async def close(self) -> None:
        """Close database connection."""
        await self.db_manager.close()
    
    async def get_applied_migrations(self) -> List[str]:
        """Get list of applied migration versions."""
        async with self.db_manager.async_engine.begin() as conn:
            # Ensure migration history table exists
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS migration_history (
                    id SERIAL PRIMARY KEY,
                    version VARCHAR(50) NOT NULL UNIQUE,
                    name VARCHAR(255) NOT NULL,
                    description TEXT,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    applied_by VARCHAR(255),
                    execution_time_ms INTEGER
                )
            """))
            
            result = await conn.execute(text("""
                SELECT version FROM migration_history ORDER BY version
            """))
            
            return [row[0] for row in result.fetchall()]
    
    async def get_available_migrations(self) -> List[str]:
        """Get list of available migration files."""
        migration_files = []
        
        for file_path in self.migrations_dir.glob("*.py"):
            if file_path.name.startswith("0") and file_path.name != "__init__.py":
                # Extract version from filename (e.g., "001_initial_schema.py" -> "001")
                version = file_path.stem.split("_")[0]
                migration_files.append(version)
        
        return sorted(migration_files)
    
    async def get_pending_migrations(self) -> List[str]:
        """Get list of migrations that need to be applied."""
        applied = set(await self.get_applied_migrations())
        available = set(await self.get_available_migrations())
        
        pending = available - applied
        return sorted(list(pending))
    
    async def load_migration_module(self, version: str):
        """Load migration module by version."""
        # Find migration file
        migration_file = None
        for file_path in self.migrations_dir.glob(f"{version}_*.py"):
            migration_file = file_path
            break
        
        if not migration_file:
            raise ValueError(f"Migration file for version {version} not found")
        
        # Import module dynamically
        spec = importlib.util.spec_from_file_location(
            f"migration_{version}",
            migration_file
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        return module
    
    async def apply_migration(self, version: str) -> None:
        """Apply a specific migration."""
        logger.info("Applying migration", version=version)
        
        # Check if already applied
        applied_migrations = await self.get_applied_migrations()
        if version in applied_migrations:
            logger.warning("Migration already applied", version=version)
            return
        
        start_time = time.time()
        
        try:
            # Load migration module
            migration_module = await self.load_migration_module(version)
            
            # Apply upgrade
            await migration_module.upgrade(self.db_manager.async_engine)
            
            # Verify migration
            if hasattr(migration_module, 'verify_migration'):
                success = await migration_module.verify_migration(self.db_manager.async_engine)
                if not success:
                    raise RuntimeError(f"Migration {version} verification failed")
            
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            # Record migration in history
            async with self.db_manager.async_engine.begin() as conn:
                await conn.execute(text("""
                    UPDATE migration_history 
                    SET execution_time_ms = :execution_time_ms
                    WHERE version = :version
                """), {
                    "version": version,
                    "execution_time_ms": execution_time_ms
                })
            
            logger.info("Migration applied successfully", 
                       version=version, 
                       execution_time_ms=execution_time_ms)
            
        except Exception as e:
            logger.error("Migration failed", version=version, error=str(e))
            raise
    
    async def rollback_migration(self, version: str) -> None:
        """Rollback a specific migration."""
        logger.info("Rolling back migration", version=version)
        
        # Check if migration is applied
        applied_migrations = await self.get_applied_migrations()
        if version not in applied_migrations:
            logger.warning("Migration not applied, cannot rollback", version=version)
            return
        
        try:
            # Load migration module
            migration_module = await self.load_migration_module(version)
            
            # Apply downgrade
            await migration_module.downgrade(self.db_manager.async_engine)
            
            logger.info("Migration rolled back successfully", version=version)
            
        except Exception as e:
            logger.error("Migration rollback failed", version=version, error=str(e))
            raise
    
    async def migrate_to_latest(self) -> None:
        """Apply all pending migrations."""
        logger.info("Migrating to latest version")
        
        pending_migrations = await self.get_pending_migrations()
        
        if not pending_migrations:
            logger.info("No pending migrations")
            return
        
        logger.info("Applying pending migrations", count=len(pending_migrations))
        
        for version in pending_migrations:
            await self.apply_migration(version)
        
        logger.info("Migration to latest version completed")
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """Get current migration status."""
        applied = await self.get_applied_migrations()
        available = await self.get_available_migrations()
        pending = await self.get_pending_migrations()
        
        # Get migration history
        async with self.db_manager.async_engine.begin() as conn:
            result = await conn.execute(text("""
                SELECT version, name, description, applied_at, execution_time_ms
                FROM migration_history
                ORDER BY applied_at DESC
                LIMIT 10
            """))
            
            history = [
                {
                    "version": row[0],
                    "name": row[1],
                    "description": row[2],
                    "applied_at": row[3],
                    "execution_time_ms": row[4]
                }
                for row in result.fetchall()
            ]
        
        return {
            "current_version": applied[-1] if applied else None,
            "applied_migrations": applied,
            "available_migrations": available,
            "pending_migrations": pending,
            "migration_history": history
        }
    
    async def verify_all_migrations(self) -> bool:
        """Verify all applied migrations."""
        logger.info("Verifying all applied migrations")
        
        applied_migrations = await self.get_applied_migrations()
        
        for version in applied_migrations:
            try:
                migration_module = await self.load_migration_module(version)
                
                if hasattr(migration_module, 'verify_migration'):
                    success = await migration_module.verify_migration(self.db_manager.async_engine)
                    if not success:
                        logger.error("Migration verification failed", version=version)
                        return False
                    else:
                        logger.info("Migration verification passed", version=version)
                
            except Exception as e:
                logger.error("Migration verification error", version=version, error=str(e))
                return False
        
        logger.info("All migration verifications passed")
        return True


async def main():
    """CLI interface for migration runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Database migration runner")
    parser.add_argument("command", choices=[
        "status", "migrate", "apply", "rollback", "verify", "history"
    ], help="Migration command")
    parser.add_argument("--version", help="Specific migration version")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        structlog.configure(level="DEBUG")
    
    runner = MigrationRunner()
    
    try:
        await runner.initialize()
        
        if args.command == "status":
            status = await runner.get_migration_status()
            print(f"Current version: {status['current_version']}")
            print(f"Applied migrations: {len(status['applied_migrations'])}")
            print(f"Pending migrations: {len(status['pending_migrations'])}")
            
            if status['pending_migrations']:
                print(f"Pending: {', '.join(status['pending_migrations'])}")
        
        elif args.command == "migrate":
            await runner.migrate_to_latest()
        
        elif args.command == "apply":
            if not args.version:
                print("Version required for apply command")
                sys.exit(1)
            await runner.apply_migration(args.version)
        
        elif args.command == "rollback":
            if not args.version:
                print("Version required for rollback command")
                sys.exit(1)
            await runner.rollback_migration(args.version)
        
        elif args.command == "verify":
            success = await runner.verify_all_migrations()
            sys.exit(0 if success else 1)
        
        elif args.command == "history":
            status = await runner.get_migration_status()
            print("Migration History:")
            for migration in status['migration_history']:
                print(f"  {migration['version']}: {migration['name']} "
                     f"({migration['applied_at']}, {migration['execution_time_ms']}ms)")
    
    finally:
        await runner.close()


if __name__ == "__main__":
    asyncio.run(main())