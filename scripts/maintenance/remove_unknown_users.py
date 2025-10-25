#!/usr/bin/env python3
"""
Script to remove all users with first name 'Unknown' from the database.
"""

def remove_unknown_users():
    """Remove all users with first name 'Unknown' from the database."""
    try:
        from database.schemas.auth_schemas import Employee
        from config.database import get_auth_db
        
        print("🔍 Searching for users with first name 'Unknown'...")
        
        with get_auth_db() as db:
            # Find all users with first name 'Unknown'
            unknown_users = db.query(Employee).filter(Employee.first_name == 'Unknown').all()
            
            if not unknown_users:
                print("✅ No users found with first name 'Unknown'")
                return
            
            print(f"📋 Found {len(unknown_users)} users with first name 'Unknown':")
            for user in unknown_users:
                print(f"   - ID: {user.employee_id}, Name: {user.first_name} {user.last_name}, Email: {user.email}")
            
            # Ask for confirmation
            confirm = input(f"\n❓ Do you want to delete these {len(unknown_users)} users? (yes/no): ").lower().strip()
            
            if confirm in ['yes', 'y']:
                # Delete the users
                for user in unknown_users:
                    db.delete(user)
                
                db.commit()
                print(f"✅ Successfully deleted {len(unknown_users)} users with first name 'Unknown'")
            else:
                print("❌ Operation cancelled")
                
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    remove_unknown_users()

