"""
OCCUR-CAM Database Table Creation
Creates all database tables and initializes the schema.
"""

import logging
from pathlib import Path
from sqlalchemy import text
from config.database import (
    auth_engine, main_engine, 
    auth_metadata, main_metadata,
    test_connections, create_tables as create_tables_func
)

def create_all_tables():
    """Create all database tables for both auth and main databases."""
    try:
        logging.info("Starting database table creation...")
        
        # Test connections first
        if not test_connections():
            logging.error("Database connection test failed")
            return False
        
        # Import schemas to register them with metadata
        from database.schemas import auth_schemas, main_schemas
        
        # Create auth database tables using the base classes
        logging.info("Creating authentication database tables...")
        auth_schemas.AuthBase.metadata.create_all(bind=auth_engine)
        
        # Create main database tables using the base classes
        logging.info("Creating main database tables...")
        main_schemas.MainBase.metadata.create_all(bind=main_engine)
        
        # Verify table creation
        if verify_tables_created():
            logging.info("All database tables created successfully")
            return True
        else:
            logging.error("Table creation verification failed")
            return False
            
    except Exception as e:
        logging.error(f"Error creating database tables: {e}")
        return False

def drop_all_tables():
    """Drop all database tables (use with caution)."""
    try:
        logging.warning("Dropping all database tables...")
        
        # Import schemas to register them with metadata
        from database.schemas import auth_schemas, main_schemas
        
        # Drop auth database tables
        auth_metadata.drop_all(bind=auth_engine)
        logging.info("Authentication database tables dropped")
        
        # Drop main database tables
        main_metadata.drop_all(bind=main_engine)
        logging.info("Main database tables dropped")
        
        logging.warning("All database tables have been dropped")
        return True
        
    except Exception as e:
        logging.error(f"Error dropping database tables: {e}")
        return False

def verify_tables_created():
    """Verify that all expected tables have been created."""
    try:
        # Expected auth tables (actual table names)
        auth_tables = [
            "employees", "auth_logs", "camera_configs", 
            "system_logs", "auth_sessions"
        ]
        
        # Expected main tables (actual table names)
        main_tables = [
            "sites", "cameras", "camera_configs", "employee_sites",
            "alert_rules", "alert_logs", "system_metrics"
        ]
        
        # Check auth database tables
        with auth_engine.connect() as conn:
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            auth_created = [row[0] for row in result.fetchall()]
            
        # Check main database tables
        with main_engine.connect() as conn:
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            main_created = [row[0] for row in result.fetchall()]
        
        # Verify all expected tables exist
        auth_missing = set(auth_tables) - set(auth_created)
        main_missing = set(main_tables) - set(main_created)
        
        if auth_missing:
            logging.error(f"Missing auth tables: {auth_missing}")
            return False
            
        if main_missing:
            logging.error(f"Missing main tables: {main_missing}")
            return False
        
        logging.info(f"Auth tables created: {len(auth_created)}")
        logging.info(f"Main tables created: {len(main_created)}")
        return True
        
    except Exception as e:
        logging.error(f"Error verifying table creation: {e}")
        return False

def migrate_database():
    """Run database migrations and updates."""
    try:
        logging.info("Starting database migration...")
        
        # For now, just create tables
        # In the future, this would handle schema updates
        return create_all_tables()
        
    except Exception as e:
        logging.error(f"Error during database migration: {e}")
        return False

def reset_database():
    """Reset database by dropping and recreating all tables."""
    try:
        logging.warning("Resetting database...")
        
        # Drop all tables
        if not drop_all_tables():
            return False
        
        # Create all tables
        if not create_all_tables():
            return False
        
        logging.info("Database reset completed successfully")
        return True
        
    except Exception as e:
        logging.error(f"Error resetting database: {e}")
        return False

def get_database_info():
    """Get information about the current database state."""
    try:
        info = {
            "auth_database": {
                "url": str(auth_engine.url),
                "tables": [],
                "size_mb": 0
            },
            "main_database": {
                "url": str(main_engine.url),
                "tables": [],
                "size_mb": 0
            }
        }
        
        # Get auth database info
        with auth_engine.connect() as conn:
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            info["auth_database"]["tables"] = [row[0] for row in result.fetchall()]
            
            # Get database file size
            db_path = auth_engine.url.database
            if Path(db_path).exists():
                info["auth_database"]["size_mb"] = round(Path(db_path).stat().st_size / (1024 * 1024), 2)
        
        # Get main database info
        with main_engine.connect() as conn:
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            info["main_database"]["tables"] = [row[0] for row in result.fetchall()]
            
            # Get database file size
            db_path = main_engine.url.database
            if Path(db_path).exists():
                info["main_database"]["size_mb"] = round(Path(db_path).stat().st_size / (1024 * 1024), 2)
        
        return info
        
    except Exception as e:
        logging.error(f"Error getting database info: {e}")
        return None

if __name__ == "__main__":
    # Run table creation when script is executed directly
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "create":
            success = create_all_tables()
            sys.exit(0 if success else 1)
        elif command == "drop":
            success = drop_all_tables()
            sys.exit(0 if success else 1)
        elif command == "reset":
            success = reset_database()
            sys.exit(0 if success else 1)
        elif command == "info":
            info = get_database_info()
            if info:
                print("Database Information:")
                print(f"Auth DB: {info['auth_database']['url']} ({info['auth_database']['size_mb']} MB)")
                print(f"Main DB: {info['main_database']['url']} ({info['main_database']['size_mb']} MB)")
                print(f"Auth Tables: {len(info['auth_database']['tables'])}")
                print(f"Main Tables: {len(info['main_database']['tables'])}")
            sys.exit(0)
        else:
            print("Usage: python create_tables.py [create|drop|reset|info]")
            sys.exit(1)
    else:
        # Default: create tables
        success = create_all_tables()
        sys.exit(0 if success else 1)
