import sqlite3

def migrate():
    conn = sqlite3.connect('instance/diabetes_system.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute("ALTER TABLE appointments ADD COLUMN admin_notes TEXT;")
        conn.commit()
        print("Migration successful: admin_notes added.")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e).lower():
            print("Column already exists. No migration needed.")
        else:
            print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == '__main__':
    migrate()
