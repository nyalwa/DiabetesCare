import sqlite3
import os

def migrate():
    db_path = 'instance/diabetes_system.db'
    
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("Starting migration for Patients table...")
    
    try:
        cursor.execute("ALTER TABLE patients ADD COLUMN user_id INTEGER REFERENCES users(id);")
        print("Added column: user_id to patients table")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e).lower():
            print("Column 'user_id' already exists.")
        else:
            print(f"Error adding 'user_id': {e}")

    # Also update existing records to link them via email
    print("Linking existing patients to users via email...")
    cursor.execute("""
        UPDATE patients 
        SET user_id = (SELECT id FROM users WHERE users.email = patients.email)
        WHERE user_id IS NULL
    """)
    print(f"Updated {cursor.rowcount} records.")

    conn.commit()
    conn.close()
    print("Migration finished!")

if __name__ == '__main__':
    migrate()
