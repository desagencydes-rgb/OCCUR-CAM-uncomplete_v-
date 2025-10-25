#!/usr/bin/env python3
"""Check database tables."""

from config.database import auth_engine, main_engine
from sqlalchemy import text

print("Auth tables:")
with auth_engine.connect() as conn:
    result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
    auth_tables = [row[0] for row in result.fetchall()]
    for table in auth_tables:
        print(f"  {table}")

print("\nMain tables:")
with main_engine.connect() as conn:
    result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
    main_tables = [row[0] for row in result.fetchall()]
    for table in main_tables:
        print(f"  {table}")

print(f"\nAuth tables count: {len(auth_tables)}")
print(f"Main tables count: {len(main_tables)}")
