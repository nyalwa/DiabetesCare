from app import app
from database import db, Doctor

def update_passwords():
    with app.app_context():
        doctors = Doctor.query.all()
        print(f"Updating passwords for {len(doctors)} doctors...")
        for doctor in doctors:
            if not doctor.password_hash:
                doctor.set_password('doctor123')
                print(f"Set password for: {doctor.full_name}")
        db.session.commit()
        print("Success! All doctors now have passwords.")

if __name__ == '__main__':
    update_passwords()
