# ============================================================
# database.py — Database Models
# Defines all 5 tables in our system
# ============================================================

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

# Create the database object
# This is shared across the whole app
db = SQLAlchemy()


# ============================================================
# TABLE 1: Patient
# Stores every patient who goes through triage
# ============================================================
class Patient(db.Model):
    __tablename__ = 'patients'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    full_name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(100), nullable=True)
    pregnancies = db.Column(db.Float, nullable=False)
    glucose = db.Column(db.Float, nullable=False)
    blood_pressure = db.Column(db.Float, nullable=False)
    skin_thickness = db.Column(db.Float, nullable=False)
    insulin = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    dpf = db.Column(db.Float, nullable=False)
    prediction = db.Column(db.Integer, nullable=False)
    probability = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(20), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # One patient has one appointment
    appointment = db.relationship('Appointment',
                                   backref='patient',
                                   uselist=False)

    def __repr__(self):
        return f'<Patient {self.full_name} - {self.risk_level}>'


# ============================================================
# TABLE 2: Doctor
# Stores all doctors in the system
# ============================================================
class Doctor(db.Model):
    __tablename__ = 'doctors'

    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100), nullable=False)
    specialization = db.Column(db.String(100), nullable=False)
    hospital = db.Column(db.String(150), nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    experience_years = db.Column(db.Integer, nullable=False)
    bio = db.Column(db.Text, nullable=True)
    available = db.Column(db.Boolean, default=True)
    password_hash = db.Column(db.String(200), nullable=True) # Optional for now, will be set on creation

    # One doctor has many appointments
    appointments = db.relationship('Appointment',
                                    backref='doctor',
                                    lazy=True)

    def set_password(self, password):
        from werkzeug.security import generate_password_hash
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        if not self.password_hash:
            return False
        from werkzeug.security import check_password_hash
        try:
            return check_password_hash(self.password_hash, password)
        except (ValueError, TypeError):
            return self.password_hash == password

    def __repr__(self):
        return f'<Doctor {self.full_name} - {self.specialization}>'


# ============================================================
# TABLE 3: Appointment
# Links patients to doctors
# ============================================================
class Appointment(db.Model):
    __tablename__ = 'appointments'

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer,
                           db.ForeignKey('patients.id'),
                           nullable=False)
    doctor_id = db.Column(db.Integer,
                          db.ForeignKey('doctors.id'),
                          nullable=False)
    appointment_date = db.Column(db.String(20), nullable=False)
    appointment_time = db.Column(db.String(20), nullable=False)
    notes = db.Column(db.Text, nullable=True)
    admin_notes = db.Column(db.Text, nullable=True)
    status = db.Column(db.String(20), default='Confirmed')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Appointment Patient:{self.patient_id} Doctor:{self.doctor_id}>'


# ============================================================
# TABLE 4: Admin
# Stores admin accounts
# ============================================================
class Admin(db.Model):
    __tablename__ = 'admins'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    role = db.Column(db.String(50), default='Admin')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        if not self.password_hash:
            return False
        from werkzeug.security import check_password_hash
        try:
            return check_password_hash(self.password_hash, password)
        except (ValueError, TypeError):
            return self.password_hash == password

    def __repr__(self):
        return f'<Admin {self.username} - {self.role}>'


# ============================================================
# TABLE 5: User
# Stores patient login accounts
# ============================================================
class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    profile_picture = db.Column(db.String(200), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        # Hash password before saving
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        # Verify password against hash
        if not self.password_hash:
            return False
        from werkzeug.security import check_password_hash
        try:
            return check_password_hash(self.password_hash, password)
        except (ValueError, TypeError):
            return self.password_hash == password

    def __init__(self, full_name, email, phone):
        self.full_name = full_name
        self.email = email
        self.phone = phone

    def __repr__(self):
        return f'<User {self.email}>'

# ============================================================
# TABLE 6: GeneralTriage
# Stores initial vitals and symptoms for patients
# ============================================================
class GeneralTriage(db.Model):
    __tablename__ = 'general_triage'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    weight = db.Column(db.Float, nullable=False)
    height = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    blood_pressure = db.Column(db.String(20), nullable=False)
    smoking = db.Column(db.String(10), nullable=True)
    activity_level = db.Column(db.String(50), nullable=True)
    family_history = db.Column(db.String(10), nullable=True)
    medications = db.Column(db.Text, nullable=True)
    symptoms = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


    # Establish relationship to User
    user = db.relationship('User', backref=db.backref('general_triages', lazy=True))

    def __repr__(self):
        return f'<GeneralTriage User:{self.user_id} - {self.created_at.strftime("%Y-%m-%d")}>'


# ============================================================
# TABLE 7: SMSLog
# Records every SMS sent through the system
# ============================================================
class SMSLog(db.Model):
    __tablename__ = 'sms_logs'

    id          = db.Column(db.Integer, primary_key=True)
    recipient   = db.Column(db.String(30), nullable=False)
    message     = db.Column(db.Text, nullable=False)
    status      = db.Column(db.String(10), default='sent')   # 'sent' | 'failed'
    category    = db.Column(db.String(50), nullable=True)    # e.g. 'OTP', 'Appointment', 'Bulk'
    sent_at     = db.Column(db.DateTime, default=datetime.utcnow)

    def __init__(self, recipient, message, status='sent', category=None):
        self.recipient = recipient
        self.message = message
        self.status = status
        self.category = category

    def __repr__(self):
        return f'<SMSLog to:{self.recipient} status:{self.status}>'
