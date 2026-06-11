from app import app
from database import db, Admin

def reset_admins():
    with app.app_context():
        print("Cleaning Admin table...")
        Admin.query.delete()
        
        # Account 1: Administrator
        admin1 = Admin(
            username='admin',
            full_name='System Administrator',
            role='Administrator'
        )
        admin1.set_password('admin123')

        # Account 2: Receptionist
        admin2 = Admin(
            username='receptionist',
            full_name='Front Desk Receptionist',
            role='Receptionist'
        )
        admin2.set_password('receptionist123')

        db.session.add(admin1)
        db.session.add(admin2)
        db.session.commit()
        print("Success! Admins reset.")
        print("Username: admin | Password: admin123")
        print("Username: receptionist | Password: receptionist123")

if __name__ == '__main__':
    reset_admins()
