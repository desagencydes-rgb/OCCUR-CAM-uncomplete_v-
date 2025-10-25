#!/usr/bin/env python3
"""
Script to remove all users with first name 'Rayane' from the database.
"""

def remove_rayane_users():
    """Remove all users with first name 'Rayane' from the database."""
    try:
        from database.schemas.auth_schemas import Employee
        from config.database import get_auth_db
        
        print("🔍 Searching for users with first name 'Rayane'...")
        
        with get_auth_db() as db:
            # Find all users with first name 'Rayane'
            rayane_users = db.query(Employee).filter(Employee.first_name == 'Rayane').all()
            
            if not rayane_users:
                print("✅ No users found with first name 'Rayane'")
                return
            
            print(f"📋 Found {len(rayane_users)} users with first name 'Rayane':")
            for user in rayane_users:
                print(f"   - ID: {user.employee_id}, Name: {user.first_name} {user.last_name}, Email: {user.email}")
            
            # Ask for confirmation
            confirm = input(f"\n❓ Do you want to delete these {len(rayane_users)} users? (yes/no): ").lower().strip()
            
            if confirm in ['yes', 'y']:
                # Delete the users
                for user in rayane_users:
                    db.delete(user)
                
                db.commit()
                print(f"✅ Successfully deleted {len(rayane_users)} users with first name 'Rayane'")
            else:
                print("❌ Operation cancelled")
                
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    remove_rayane_users()

