import sqlite3
import os

def migrate():
    # Path to your database
    db_path = 'instance/diabetes_system.db'
    
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    columns_to_add = [
        ("smoking", "TEXT"),
        ("activity_level", "TEXT"),
        ("family_history", "TEXT"),
        ("medications", "TEXT")
    ]
    
    print("Starting migration...")
    
    for col_name, col_type in columns_to_add:
        try:
            cursor.execute(f"ALTER TABLE general_triage ADD COLUMN {col_name} {col_type};")
            print(f"Added column: {col_name}")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e).lower():
                print(f"Column '{col_name}' already exists.")
            else:
                print(f"Error adding '{col_name}': {e}")

    conn.commit()
    conn.close()
    print("Migration finished!")

if __name__ == '__main__':
    migrate()
