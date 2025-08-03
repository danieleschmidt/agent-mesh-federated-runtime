"""Initial database schema migration.

Creates the core tables for Agent Mesh:
- Node: P2P network nodes
- Task: Distributed task management
- TrainingRound: Federated learning rounds
- MetricEntry: Performance and monitoring metrics
- ConsensusRound: Consensus protocol rounds

Revision ID: 001
Create Date: 2024-01-01 00:00:00.000000
"""

import asyncio
from typing import Dict, Any

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from src.agent_mesh.database.connection import DatabaseManager
from src.agent_mesh.database.models import Base

logger = structlog.get_logger("migration_001")

MIGRATION_VERSION = "001"
MIGRATION_NAME = "initial_schema"
MIGRATION_DESCRIPTION = "Create initial database schema for Agent Mesh"


async def upgrade(engine: AsyncEngine) -> None:
    """Apply the migration - create all tables."""
    logger.info("Applying migration 001: Creating initial schema")
    
    async with engine.begin() as conn:
        # Create all tables defined in models
        await conn.run_sync(Base.metadata.create_all)
        
        # Create migration tracking table if it doesn't exist
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
        
        # Record this migration
        await conn.execute(text("""
            INSERT INTO migration_history (version, name, description, applied_by)
            VALUES (:version, :name, :description, 'system')
            ON CONFLICT (version) DO NOTHING
        """), {
            "version": MIGRATION_VERSION,
            "name": MIGRATION_NAME,
            "description": MIGRATION_DESCRIPTION
        })
        
        # Create indexes for performance
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_nodes_status ON nodes(status);
        """))
        
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_nodes_last_seen ON nodes(last_seen);
        """))
        
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
        """))
        
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_tasks_assigned_node ON tasks(assigned_node_id);
        """))
        
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_training_rounds_session ON training_rounds(session_id);
        """))
        
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metric_entries(timestamp);
        """))
        
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_metrics_component ON metric_entries(component);
        """))
    
    logger.info("Migration 001 applied successfully")


async def downgrade(engine: AsyncEngine) -> None:
    """Rollback the migration - drop all tables."""
    logger.info("Rolling back migration 001: Dropping initial schema")
    
    async with engine.begin() as conn:
        # Drop all tables in reverse order to handle foreign keys
        await conn.execute(text("DROP TABLE IF EXISTS metric_entries CASCADE"))
        await conn.execute(text("DROP TABLE IF EXISTS consensus_rounds CASCADE"))
        await conn.execute(text("DROP TABLE IF EXISTS training_rounds CASCADE"))
        await conn.execute(text("DROP TABLE IF EXISTS tasks CASCADE"))
        await conn.execute(text("DROP TABLE IF EXISTS nodes CASCADE"))
        
        # Remove migration record
        await conn.execute(text("""
            DELETE FROM migration_history WHERE version = :version
        """), {"version": MIGRATION_VERSION})
    
    logger.info("Migration 001 rolled back successfully")


async def verify_migration(engine: AsyncEngine) -> bool:
    """Verify that the migration was applied correctly."""
    logger.info("Verifying migration 001")
    
    try:
        async with engine.begin() as conn:
            # Check that all expected tables exist
            result = await conn.execute(text("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('nodes', 'tasks', 'training_rounds', 'metric_entries', 'consensus_rounds')
            """))
            
            tables = [row[0] for row in result.fetchall()]
            expected_tables = ['nodes', 'tasks', 'training_rounds', 'metric_entries', 'consensus_rounds']
            
            if set(tables) >= set(expected_tables):
                logger.info("Migration 001 verification passed")
                return True
            else:
                missing_tables = set(expected_tables) - set(tables)
                logger.error("Migration 001 verification failed", missing_tables=missing_tables)
                return False
                
    except Exception as e:
        logger.error("Migration 001 verification failed", error=str(e))
        return False


async def get_migration_info() -> Dict[str, Any]:
    """Get information about this migration."""
    return {
        "version": MIGRATION_VERSION,
        "name": MIGRATION_NAME,
        "description": MIGRATION_DESCRIPTION,
        "tables_created": [
            "nodes",
            "tasks", 
            "training_rounds",
            "metric_entries",
            "consensus_rounds",
            "migration_history"
        ],
        "indexes_created": [
            "idx_nodes_status",
            "idx_nodes_last_seen",
            "idx_tasks_status",
            "idx_tasks_assigned_node",
            "idx_training_rounds_session",
            "idx_metrics_timestamp",
            "idx_metrics_component"
        ]
    }


# CLI interface for running migrations
if __name__ == "__main__":
    import sys
    import time
    from pathlib import Path
    
    # Add src to path
    sys.path.append(str(Path(__file__).parent.parent / "src"))
    
    async def main():
        """Run migration from command line."""
        db_manager = DatabaseManager()
        await db_manager.initialize()
        
        command = sys.argv[1] if len(sys.argv) > 1 else "upgrade"
        
        start_time = time.time()
        
        try:
            if command == "upgrade":
                await upgrade(db_manager.async_engine)
            elif command == "downgrade":
                await downgrade(db_manager.async_engine)
            elif command == "verify":
                success = await verify_migration(db_manager.async_engine)
                sys.exit(0 if success else 1)
            elif command == "info":
                info = await get_migration_info()
                print(f"Migration {info['version']}: {info['name']}")
                print(f"Description: {info['description']}")
                print(f"Tables: {', '.join(info['tables_created'])}")
                print(f"Indexes: {', '.join(info['indexes_created'])}")
            else:
                print(f"Unknown command: {command}")
                print("Available commands: upgrade, downgrade, verify, info")
                sys.exit(1)
                
            execution_time = int((time.time() - start_time) * 1000)
            logger.info(f"Migration command '{command}' completed", 
                       execution_time_ms=execution_time)
                       
        finally:
            await db_manager.close()
    
    asyncio.run(main())