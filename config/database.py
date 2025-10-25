"""
OCCUR-CAM Database Configuration
Database connection and session management.
"""

from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from typing import Generator
import logging

from .settings import config

# Database engines
auth_engine = create_engine(
    config.database.AUTH_DATABASE_URL,
    poolclass=StaticPool,
    echo=False  # Set to True for SQL debugging
)

main_engine = create_engine(
    config.database.DATABASE_URL,
    poolclass=StaticPool,
    echo=False  # Set to True for SQL debugging
)

# Session makers
AuthSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=auth_engine)
MainSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=main_engine)

# Base classes for different databases
AuthBase = declarative_base()
MainBase = declarative_base()

# Metadata for table creation
auth_metadata = MetaData()
main_metadata = MetaData()

@contextmanager
def get_auth_db() -> Generator:
    """Get authentication database session with automatic cleanup."""
    db = AuthSessionLocal()
    try:
        yield db
    except Exception as e:
        db.rollback()
        logging.error(f"Database error in auth session: {e}")
        raise
    finally:
        db.close()

@contextmanager
def get_main_db() -> Generator:
    """Get main database session with automatic cleanup."""
    db = MainSessionLocal()
    try:
        yield db
    except Exception as e:
        db.rollback()
        logging.error(f"Database error in main session: {e}")
        raise
    finally:
        db.close()

def create_tables():
    """Create all database tables."""
    try:
        # Import models to register them with the base classes
        from database.schemas import auth_schemas, main_schemas
        
        # Create tables
        auth_metadata.create_all(bind=auth_engine)
        main_metadata.create_all(bind=main_engine)
        
        logging.info("Database tables created successfully")
        return True
    except Exception as e:
        logging.error(f"Error creating database tables: {e}")
        return False

def drop_tables():
    """Drop all database tables (use with caution)."""
    try:
        auth_metadata.drop_all(bind=auth_engine)
        main_metadata.drop_all(bind=main_engine)
        logging.warning("All database tables dropped")
        return True
    except Exception as e:
        logging.error(f"Error dropping database tables: {e}")
        return False

def test_connections():
    """Test database connections."""
    try:
        # Test auth database
        with auth_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logging.info("Auth database connection successful")
        
        # Test main database
        with main_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logging.info("Main database connection successful")
        
        return True
    except Exception as e:
        logging.error(f"Database connection test failed: {e}")
        return False

# Database health check
def check_database_health():
    """Check database health and return status."""
    health_status = {
        "auth_db": False,
        "main_db": False,
        "overall": False
    }
    
    try:
        # Check auth database
        with auth_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        health_status["auth_db"] = True
    except Exception as e:
        logging.error(f"Auth database health check failed: {e}")
    
    try:
        # Check main database
        with main_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        health_status["main_db"] = True
    except Exception as e:
        logging.error(f"Main database health check failed: {e}")
    
    health_status["overall"] = health_status["auth_db"] and health_status["main_db"]
    return health_status
