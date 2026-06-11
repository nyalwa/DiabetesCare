from app import app, db
from database import GeneralTriage
import sqlite3

def migrate():
    with app.app_context():
        # create_all will only create tables that do not exist yet.
        db.create_all()
        print("GeneralTriage table created successfully!")

if __name__ == "__main__":
    migrate()
