import sqlite3
import os

def migrate():
    db_path = 'instance/diabetes_system.db'
    
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("Starting migration for Doctors table...")
    
    try:
        cursor.execute("ALTER TABLE doctors ADD COLUMN password_hash TEXT;")
        print("Added column: password_hash to doctors table")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e).lower():
            print("Column 'password_hash' already exists.")
        else:
            print(f"Error adding 'password_hash': {e}")

    conn.commit()
    conn.close()
    print("Migration finished!")

if __name__ == '__main__':
    migrate()
