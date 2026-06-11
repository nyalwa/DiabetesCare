# ============================================================
# app.py — Main Flask Application
# The engine of the entire DiabetesCare system
# ============================================================

from flask import (Flask, render_template, request, redirect, url_for, session, send_file, flash)
from database import db, Patient, Doctor, Appointment, Admin, User, GeneralTriage, SMSLog
import joblib
import numpy as np
import io
import os
from datetime import datetime
import random
from flask_mail import Mail, Message
from functools import wraps
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle)

# Safely load environment variables from .env file if python-dotenv is installed
try:
    # pyrefly: ignore [missing-import]
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ============================================================
# APP CONFIGURATION
# ============================================================

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'diabetescare_secret_key_2024_local')

# Database — uses DATABASE_URL on Render, SQLite locally
db_url = os.environ.get('DATABASE_URL', 'sqlite:///diabetes_system.db')
# Render gives postgres:// but SQLAlchemy needs postgresql://
if db_url.startswith('postgres://'):
    db_url = db_url.replace('postgres://', 'postgresql://', 1)
app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Email Configuration (Gmail)
app.config['MAIL_SERVER']   = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT']     = int(os.environ.get('MAIL_PORT', 465))
app.config['MAIL_USE_SSL']  = os.environ.get('MAIL_USE_SSL', 'true').lower() == 'true'
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME', 'diabetescare15@gmail.com')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD', '')

db.init_app(app)
mail = Mail(app)

# ============================================================
# AFRICA'S TALKING SMS CONFIGURATION
# ============================================================
# Sign up free at: https://account.africastalking.com/auth/register
# Only an email address is needed — no credit card required.
# Use 'sandbox' as the username + AT_API_KEY for sandbox testing.
# For production: set your real username and live API key.
AT_USERNAME = os.environ.get('AT_USERNAME', 'sandbox')
AT_API_KEY  = os.environ.get('AT_API_KEY', '')

_at_sms = None
try:
    import africastalking
    if AT_API_KEY:
        africastalking.initialize(AT_USERNAME, AT_API_KEY)
        _at_sms = africastalking.SMS
        print(f"[SMS] Africa's Talking initialized (username={AT_USERNAME})")
    else:
        print("[SMS] Africa's Talking not configured — AT_API_KEY missing.")
except Exception as _at_err:
    print(f"[SMS] Africa's Talking init error: {_at_err}")


def format_phone(phone: str) -> str:
    """
    Ensures a phone number is in E.164 international format.
    If it starts with '0', replaces it with '+256' (Uganda).
    If it already starts with '+', leaves it unchanged.
    Adjust the country prefix below to match your region.
    """
    phone = str(phone).strip().replace(' ', '').replace('-', '')
    if phone.startswith('+'):
        return phone
    if phone.startswith('0'):
        return '+256' + phone[1:]
    return '+256' + phone


def send_sms(to_phone: str, body: str, category: str = 'General') -> bool:
    """
    Send an SMS via Africa's Talking and log the result.
    Returns True on success, False on failure.
    Never raises — the app keeps running even if SMS fails.
    """
    formatted = format_phone(to_phone)
    status = 'failed'
    try:
        if _at_sms:
            response = _at_sms.send(body, [formatted])
            # Africa's Talking returns a recipients list; status 'Success' means delivered
            recipients = response.get('SMSMessageData', {}).get('Recipients', [])
            if recipients and recipients[0].get('status') == 'Success':
                status = 'sent'
                print(f"[SMS] Sent to {formatted}: {body[:60]}...")
            else:
                err = recipients[0].get('status', 'Unknown') if recipients else 'No recipients'
                print(f"[SMS] AT delivery issue to {formatted}: {err}")
        else:
            print("[SMS] Africa's Talking not configured — skipping SMS.")
    except Exception as e:
        print(f"[SMS] Failed to send to {formatted}: {e}")

    # Log to DB (best-effort — don't crash if DB isn't ready yet)
    try:
        log = SMSLog(recipient=formatted, message=body,
                     status=status, category=category)
        db.session.add(log)
        db.session.commit()
    except Exception:
        pass

    return status == 'sent'


# ============================================================
# LOAD ML MODEL
# ============================================================

model = joblib.load('models/diabetes_model.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_names = joblib.load('models/feature_names.pkl')

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_risk_level(probability):
    """
    Converts probability into risk category
    >=50% = Positive
    30-49% = Negative - High Chances
    <30% = Negative - Low Chances
    """
    percentage = probability * 100

    if percentage >= 50:
        return {
            'level': 'Positive',
            'label': 'Positive',
            'color': 'danger',
            'percentage': round(percentage, 1),
            'message': 'Strong indicators of diabetes detected.',
            'recommendation': 'Immediate consultation with an Endocrinologist is strongly advised.'
        }
    elif percentage >= 30:
        return {
            'level': 'High Chances',
            'label': 'Negative - High Chances',
            'color': 'warning',
            'percentage': round(percentage, 1),
            'message': 'You are currently negative, but at high chances of developing diabetes.',
            'recommendation': 'Lifestyle changes recommended. Consult a GP soon.'
        }
    else:
        return {
            'level': 'Low Chances',
            'label': 'Negative - Low Chances',
            'color': 'success',
            'percentage': round(percentage, 1),
            'message': 'No strong indicators of diabetes detected. You have low chances.',
            'recommendation': 'Maintain a healthy lifestyle. Annual checkup recommended.'
        }


def get_health_tips(patient):
    """
    Generates personalized health tips based on
    the patient's specific clinical values.
    This is one of our unique features!
    """
    tips = []

    # Glucose based tips
    if patient.glucose > 140:
        tips.append({
            'icon': '🍬',
            'title': 'High Glucose Alert',
            'tip': f'Your glucose level is {patient.glucose} mg/dL '
                   f'which is above normal. Reduce sugar and '
                   f'refined carbohydrates immediately.'
        })
    elif patient.glucose > 100:
        tips.append({
            'icon': '🍎',
            'title': 'Borderline Glucose',
            'tip': f'Your glucose of {patient.glucose} mg/dL is '
                   f'slightly elevated. Limit sugary drinks '
                   f'and processed foods.'
        })
    else:
        tips.append({
            'icon': '✅',
            'title': 'Good Glucose Level',
            'tip': f'Your glucose of {patient.glucose} mg/dL '
                   f'is in a healthy range. Keep maintaining '
                   f'your current diet.'
        })

    # BMI based tips
    if patient.bmi >= 30:
        tips.append({
            'icon': '⚖️',
            'title': 'High BMI',
            'tip': f'Your BMI of {patient.bmi} indicates obesity. '
                   f'Losing even 5-10% of body weight can '
                   f'significantly reduce diabetes risk.'
        })
    elif patient.bmi >= 25:
        tips.append({
            'icon': '🏃',
            'title': 'Overweight',
            'tip': f'Your BMI of {patient.bmi} is slightly high. '
                   f'30 minutes of walking daily can help '
                   f'bring this down.'
        })
    else:
        tips.append({
            'icon': '💪',
            'title': 'Healthy Weight',
            'tip': f'Your BMI of {patient.bmi} is in a '
                   f'healthy range. Keep up your '
                   f'physical activity.'
        })

    # Blood pressure based tips
    if patient.blood_pressure > 90:
        tips.append({
            'icon': '❤️',
            'title': 'High Blood Pressure',
            'tip': f'Your blood pressure of '
                   f'{patient.blood_pressure} mm Hg is high. '
                   f'Reduce salt intake and manage stress.'
        })
    else:
        tips.append({
            'icon': '❤️',
            'title': 'Good Blood Pressure',
            'tip': f'Your blood pressure of '
                   f'{patient.blood_pressure} mm Hg is normal. '
                   f'Keep avoiding salty and processed foods.'
        })

    # Age based tips
    if patient.age > 45:
        tips.append({
            'icon': '📅',
            'title': 'Age Risk Factor',
            'tip': 'Being over 45 increases diabetes risk. '
                   'Schedule regular checkups every 6 months '
                   'and monitor your blood sugar at home.'
        })

    # Family history based tips
    if patient.dpf > 0.5:
        tips.append({
            'icon': '👨‍👩‍👧',
            'title': 'Family History Risk',
            'tip': f'Your diabetes pedigree score of {patient.dpf} '
                   f'suggests family history of diabetes. '
                   f'Regular screening is essential.'
        })

    # General tips always shown
    tips.append({
        'icon': '💧',
        'title': 'Stay Hydrated',
        'tip': 'Drink at least 8 glasses of water daily. '
               'Proper hydration helps regulate blood sugar levels.'
    })

    tips.append({
        'icon': '😴',
        'title': 'Quality Sleep',
        'tip': 'Get 7-8 hours of sleep per night. '
               'Poor sleep is linked to increased diabetes risk.'
    })

    return tips


def recommend_doctors(risk_level):
    """
    Matches patient risk level to doctor specialization
    """
    if risk_level == 'High Risk':
        doctors = Doctor.query.filter_by(
            specialization='Endocrinologist',
            available=True
        ).all()
    elif risk_level == 'Medium Risk':
        doctors = Doctor.query.filter(
            Doctor.specialization.in_(
                ['General Practitioner', 'Nutritionist']
            ),
            Doctor.available == True
        ).all()
    else:
        doctors = Doctor.query.filter_by(
            specialization='Nutritionist',
            available=True
        ).all()
    return doctors


def login_required(f):
    """
    Protects patient routes
    Redirects to login if not logged in
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('user_login'))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    """
    Protects admin routes
    Redirects to admin login if not logged in
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_id' not in session:
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function


def doctor_required(f):
    """
    Protects doctor routes
    Redirects to doctor login if not logged in
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'doctor_id' not in session:
            return redirect(url_for('doctor_login'))
        return f(*args, **kwargs)
    return decorated_function


def any_auth_required(f):
    """
    Allows patients, admins, and doctors to access the route
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session and 'admin_id' not in session and 'doctor_id' not in session:
            return redirect(url_for('user_login'))
        return f(*args, **kwargs)
    return decorated_function


# ============================================================
# PUBLIC ROUTES
# ============================================================

@app.route('/')
def index():
    """Homepage"""
    return render_template('index.html')


@app.route('/explore')
def explore():
    """Data insights page"""
    return render_template('explore.html')


@app.route('/calculator')
def calculator():
    """Quick risk calculator widget"""
    return render_template('calculator.html')


@app.route('/api/predict_quick', methods=['POST'])
def predict_quick():
    """
    Rapid JSON prediction endpoint for the calculator.
    No login required, no database saving.
    """
    try:
        data = request.get_json()
        
        # Extract features
        age = int(data.get('age', 0))
        pregnancies = float(data.get('pregnancies', 0))
        glucose = float(data.get('glucose', 0))
        blood_pressure = float(data.get('blood_pressure', 0))
        skin_thickness = float(data.get('skin_thickness', 0))
        insulin = float(data.get('insulin', 0))
        bmi = float(data.get('bmi', 0))
        dpf = float(data.get('dpf', 0))

        # Synchronize with model's expected features (including interactions)
        glucose_bmi = glucose * bmi
        glucose_age = glucose * age
        input_data = np.array([[
            pregnancies, glucose, blood_pressure,
            skin_thickness, insulin, bmi, dpf, age,
            glucose_bmi, glucose_age
        ]])

        # Scale and Predict
        input_scaled = scaler.transform(input_data)
        probability = float(model.predict_proba(input_scaled)[0][1])
        risk_info = get_risk_level(probability)

        return {
            'success': True,
            'probability': round(probability * 100, 1),
            'risk_level': risk_info['level'],
            'label': risk_info['label'],
            'color': risk_info['color'],
            'message': risk_info['message'],
            'recommendation': risk_info['recommendation']
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }, 400


# ============================================================
# PATIENT AUTH ROUTES
# ============================================================

@app.route('/register', methods=['GET', 'POST'])
def register():
    """
    Patient registration — Step 1.
    Validates the form, generates a 6-digit OTP,
    sends it via email AND SMS, then redirects to
    the email verification page. The account is NOT
    created in the database until the OTP is confirmed.
    """
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    error = None
    if request.method == 'POST':
        full_name = request.form['full_name'].strip()
        email     = request.form['email'].strip().lower()
        phone     = request.form['phone'].strip()
        password  = request.form['password']
        confirm   = request.form['confirm_password']

        if password != confirm:
            error = 'Passwords do not match!'
        elif User.query.filter_by(email=email).first():
            error = 'Email already registered. Please login.'
        else:
            # Generate a 6-digit OTP
            otp = str(random.randint(100000, 999999))

            # Store pending registration data in session
            # (account is only saved to DB after OTP is verified)
            session['reg_pending_name']  = full_name
            session['reg_pending_email'] = email
            session['reg_pending_phone'] = phone
            session['reg_pending_pwd']   = password   # plain; hashed after verify
            session['reg_otp']           = otp

            # ── Send OTP via Email ──────────────────────────────
            email_sent = True
            try:
                msg = Message(
                    subject="Your DiabetesCare Verification Code",
                    sender=("Diabetes Care", "diabetescare15@gmail.com"),
                    recipients=[email]
                )
                msg.body = (
                    f"Hello {full_name},\n\n"
                    f"Thank you for registering with DiabetesCare!\n\n"
                    f"Your verification code is: {otp}\n\n"
                    f"Enter this code on the verification page to activate your account.\n"
                    f"This code expires in 10 minutes.\n\n"
                    f"If you did not register, please ignore this email.\n\n"
                    f"– The DiabetesCare Team"
                )
                mail.send(msg)
            except Exception as e:
                print(f"[Email] Failed to send verification email: {e}")
                email_sent = False

            # ── Send OTP via SMS ────────────────────────────────
            if phone:
                sms_body = (
                    f"DiabetesCare: Your verification code is {otp}.\n"
                    f"Enter it to activate your account. Expires in 10 mins."
                )
                send_sms(phone, sms_body, category='OTP')

            if not email_sent:
                error = 'Failed to send the verification email. Please try again.'
                return render_template('register.html', error=error)

            return redirect(url_for('verify_registration'))

    return render_template('register.html', error=error)


@app.route('/register/verify-email', methods=['GET', 'POST'])
def verify_registration():
    """
    Patient registration — Step 2.
    The user enters the 6-digit OTP that was sent to
    their email and phone. On success the User record
    is created and the user is logged in automatically.
    """
    # Guard: must have a pending registration in session
    reg_email = session.get('reg_pending_email')
    stored_otp = session.get('reg_otp')
    if not reg_email or not stored_otp:
        return redirect(url_for('register'))

    error = None

    if request.method == 'POST':
        entered_otp = request.form.get('otp', '').strip()

        if entered_otp == stored_otp:
            # ── Create the account now that OTP is confirmed ──
            # Double-check email wasn't registered while user was on the verify page
            if User.query.filter_by(email=reg_email).first():
                session.pop('reg_pending_name', None)
                session.pop('reg_pending_email', None)
                session.pop('reg_pending_phone', None)
                session.pop('reg_pending_pwd', None)
                session.pop('reg_otp', None)
                return render_template(
                    'register.html',
                    error='Email already registered. Please login.'
                )

            new_user = User(
                full_name=session['reg_pending_name'],
                email=reg_email,
                phone=session['reg_pending_phone']
            )
            new_user.set_password(session['reg_pending_pwd'])
            db.session.add(new_user)
            db.session.commit()

            # Clear pending registration keys
            session.pop('reg_pending_name', None)
            session.pop('reg_pending_email', None)
            session.pop('reg_pending_phone', None)
            session.pop('reg_pending_pwd', None)
            session.pop('reg_otp', None)

            # Log the new user in
            session['user_id']   = new_user.id
            session['user_name'] = new_user.full_name
            session['user_email'] = new_user.email
            session['is_new_user'] = True

            return redirect(url_for('dashboard'))
        else:
            error = 'Incorrect code. Please try again.'

    return render_template(
        'verify_email.html',
        email=reg_email,
        phone=session.get('reg_pending_phone', ''),
        error=error
    )


@app.route('/user/login', methods=['GET', 'POST'])
def user_login():
    """Redirect old patient login URL to unified login page."""
    return redirect(url_for('login', role='patient'))


@app.route('/user/verify-login', methods=['GET', 'POST'])
def verify_login():
    """
    Patient login — Step 2.
    The user enters the 6-digit OTP sent to their email
    and phone. On success the session is granted.
    """
    pending_user_id = session.get('login_pending_user_id')
    stored_otp      = session.get('login_otp')

    if not pending_user_id or not stored_otp:
        return redirect(url_for('user_login'))

    error = None

    if request.method == 'POST':
        entered_otp = request.form.get('otp', '').strip()

        if entered_otp == stored_otp:
            # OTP correct — grant full session
            user_name  = session.get('login_pending_user_name')
            user_email = session.get('login_pending_email')

            # Clear login-pending keys
            session.pop('login_pending_user_id', None)
            session.pop('login_pending_user_name', None)
            session.pop('login_pending_email', None)
            session.pop('login_pending_phone', None)
            session.pop('login_otp', None)

            session['user_id']    = pending_user_id
            session['user_name']  = user_name
            session['user_email'] = user_email
            session['is_new_user'] = False

            return redirect(url_for('dashboard'))
        else:
            error = 'Incorrect code. Please try again.'

    return render_template(
        'verify_login.html',
        email=session.get('login_pending_email', ''),
        phone=session.get('login_pending_phone', ''),
        error=error
    )



@app.route('/user/logout')
def user_logout():
    """Patient logout"""
    session.pop('user_id', None)
    session.pop('user_name', None)
    session.pop('user_email', None)
    session.pop('is_new_user', None)
    session.pop('patient_id', None)
    session.pop('appointment_id', None)
    return redirect(url_for('index'))


# ============================================================
# UNIFIED LOGIN
# ============================================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    """
    Single login page for Patients, Doctors, and Admins.
    Role is determined by the 'role' field in the form or query string.
    """
    # If already logged in, redirect appropriately
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    if 'admin_id' in session:
        return redirect(url_for('admin_dashboard'))
    if 'doctor_id' in session:
        return redirect(url_for('doctor_dashboard'))

    error = None
    # Pre-select role from query param (e.g. /login?role=doctor)
    selected_role = request.args.get('role', 'patient')

    if request.method == 'POST':
        role     = request.form.get('role', 'patient')
        password = request.form.get('password', '')
        selected_role = role

        # ── Patient ─────────────────────────────────────────────
        if role == 'patient':
            email = request.form.get('identifier', '').strip().lower()
            user  = User.query.filter_by(email=email).first()
            if user and user.check_password(password):
                session['user_id']    = user.id
                session['user_name']  = user.full_name
                session['user_email'] = user.email
                session['is_new_user'] = False
                return redirect(url_for('dashboard'))
            else:
                error = 'Invalid email or password.'

        # ── Doctor ──────────────────────────────────────────────
        elif role == 'doctor':
            email  = request.form.get('identifier', '').strip().lower()
            doctor = Doctor.query.filter_by(email=email).first()
            if doctor and doctor.check_password(password):
                session.clear()
                session['doctor_id']   = doctor.id
                session['doctor_name'] = doctor.full_name
                return redirect(url_for('doctor_dashboard'))
            else:
                error = 'Invalid email or password.'

        # ── Admin ───────────────────────────────────────────────
        elif role == 'admin':
            username = request.form.get('identifier', '').strip()
            admin    = Admin.query.filter_by(username=username).first()
            if admin and admin.check_password(password):
                session.clear()
                session['admin_id']   = admin.id
                session['admin_name'] = admin.full_name
                session['admin_role'] = admin.role
                return redirect(url_for('admin_dashboard'))
            else:
                error = 'Invalid username or password.'

    return render_template('login.html', error=error, selected_role=selected_role)


# ============================================================
# FORGOT PASSWORD ROUTES
# ============================================================

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    """
    Step 1 — Patient enters their email.
    If found, a 6-digit security code is generated,
    stored in the session, and sent to their email.
    """
    error = None

    if request.method == 'POST':
        email = request.form['email'].strip().lower()
        user = User.query.filter_by(email=email).first()

        if user:
            # Generate a 6-digit OTP and store in session
            otp = str(random.randint(100000, 999999))
            session['reset_email'] = email
            session['reset_otp'] = otp
            session['otp_verified'] = False

            # Send OTP via Email
            try:
                msg = Message(
                    subject="Your DiabetesCare Security Code",
                    sender=("Diabetes Care", "diabetescare15@gmail.com"),
                    recipients=[email]
                )
                msg.body = f"Hello {user.full_name},\n\nSomeone requested a password reset for your DiabetesCare account.\n\nYour 6-digit security code is: {otp}\n\nIf this wasn't you, please ignore this email.\n\nThanks,\nDiabetesCare Team"
                mail.send(msg)
            except Exception as e:
                print(f"Error sending email: {e}")
                error = "Failed to send the email. Please try again later."
                return render_template('forgot_password.html', error=error)

            # ── Send reset code via SMS ─────────────────────────
            if user.phone:
                sms_body = (
                    f"DiabetesCare: Your password reset code is {otp}.\n"
                    f"Valid for 10 minutes. If this wasn't you, ignore this."
                )
                send_sms(user.phone, sms_body, category='OTP')

            return redirect(url_for('verify_reset_code'))
        else:
            error = 'No account found with that email address.'

    return render_template('forgot_password.html', error=error)


@app.route('/verify-reset-code', methods=['GET', 'POST'])
def verify_reset_code():
    """
    Step 2 — Patient enters the 6-digit security code.
    Only if it matches do they proceed to reset their password.
    """
    reset_email = session.get('reset_email')
    stored_otp = session.get('reset_otp')

    if not reset_email or not stored_otp:
        return redirect(url_for('forgot_password'))

    error = None

    if request.method == 'POST':
        entered_code = request.form['otp'].strip()

        if entered_code == stored_otp:
            # Mark as verified — allow entry to reset page
            session['otp_verified'] = True
            return redirect(url_for('reset_password'))
        else:
            error = 'Incorrect code. Please try again.'

    # Pass the OTP so it can be displayed on the page
    # (in a real app you would email it instead)
    user = User.query.filter_by(email=reset_email).first()
    phone = user.phone if user else ''
    return render_template('verify_code.html',
                           email=reset_email,
                           phone=phone,
                           otp=stored_otp,
                           error=error)


@app.route('/reset-password', methods=['GET', 'POST'])
def reset_password():
    """
    Step 3 — Patient sets a new password.
    Only reachable after OTP has been verified.
    """
    reset_email = session.get('reset_email')
    otp_verified = session.get('otp_verified', False)

    if not reset_email or not otp_verified:
        return redirect(url_for('forgot_password'))

    error = None

    if request.method == 'POST':
        new_password = request.form['new_password']
        confirm = request.form['confirm_password']

        if new_password != confirm:
            error = 'Passwords do not match!'
        elif len(new_password) < 6:
            error = 'Password must be at least 6 characters.'
        else:
            user = User.query.filter_by(email=reset_email).first()
            user.set_password(new_password)
            db.session.commit()

            # Clear all reset session keys
            session.pop('reset_email', None)
            session.pop('reset_otp', None)
            session.pop('otp_verified', None)

            return redirect(url_for('reset_success'))

    return render_template('reset_password.html',
                           email=reset_email,
                           error=error)


@app.route('/reset-success')
def reset_success():
    """
    Step 4 — Success confirmation page.
    """
    return render_template('reset_success.html')


@app.route('/my-appointments')
@login_required
def my_appointments():
    """Shows all user appointments"""
    user_id = session['user_id']
    # Get all patients for this user, then their appointments
    patients = Patient.query.filter_by(user_id=user_id).all()
    patient_ids = [p.id for p in patients]
    
    # Query appointments linked to these patient records
    appointments = Appointment.query.filter(Appointment.patient_id.in_(patient_ids)).order_by(Appointment.created_at.desc()).all()
    
    return render_template('my_appointments.html', appointments=appointments)

@app.route('/dashboard')
@login_required
def dashboard():
    """
    Smart patient dashboard
    New user    → Welcome + Start Triage
    Returning   → Past results + appointments
    """
    user_id = session.get('user_id')
    user = User.query.get(user_id)

    past_results = Patient.query.filter_by(
        user_id=user_id
    ).order_by(Patient.created_at.desc()).all()

    is_new = len(past_results) == 0

    return render_template('dashboard.html',
                           user=user,
                           past_results=past_results,
                           is_new=is_new)


# ============================================================
# PATIENT CLINICAL ROUTES
# ============================================================

@app.route('/doctors')
@login_required
def browse_doctors():
    """Browse and book doctors directly without triage"""
    doctors = Doctor.query.filter_by(available=True).all()
    return render_template('doctors.html', doctors=doctors)

@app.route('/book/direct/<int:doctor_id>')
@login_required
def direct_booking(doctor_id):
    """Creates a baseline patient record to allow direct booking"""
    user_id = session.get('user_id')
    user = User.query.get(user_id)
    
    # Create baseline record
    baseline_patient = Patient(
        full_name=user.full_name,
        age=0,
        gender="Not Specified",
        phone=user.phone,
        email=user.email,
        pregnancies=0.0,
        glucose=0.0,
        blood_pressure=0.0,
        skin_thickness=0.0,
        insulin=0.0,
        bmi=0.0,
        dpf=0.0,
        prediction=-1,
        probability=0.0,
        risk_level="Direct Booking"
    )
    db.session.add(baseline_patient)
    db.session.commit()
    
    # Set the patient ID in session and redirect to the standard booking page
    session['patient_id'] = baseline_patient.id
    return redirect(url_for('book_appointment', doctor_id=doctor_id))


@app.route('/general-triage', methods=['GET', 'POST'])
@login_required
def general_triage():
    user = User.query.get(session['user_id'])
    
    if request.method == 'POST':
        weight = float(request.form['weight'])
        height = float(request.form['height']) 
        height_m = height / 100
        bmi = weight / (height_m * height_m)
        
        gt = GeneralTriage(
            user_id=user.id,
            age=int(request.form['age']),
            gender=request.form['gender'],
            weight=weight,
            height=height,
            bmi=round(bmi, 1),
            blood_pressure=request.form['blood_pressure'],
            smoking=request.form.get('smoking'),
            activity_level=request.form.get('activity_level'),
            family_history=request.form.get('family_history'),
            medications=request.form.get('medications', ''),
            symptoms=", ".join(request.form.getlist('symptoms'))
        )
        db.session.add(gt)
        db.session.commit()

        
        session['last_general_triage_id'] = gt.id
        return redirect(url_for('general_triage_result'))

    return render_template('general_triage.html', user=user)

@app.route('/general-triage-result')
@login_required
def general_triage_result():
    gt_id = session.get('last_general_triage_id')
    if not gt_id:
        return redirect(url_for('dashboard'))
        
    gt = GeneralTriage.query.get(gt_id)
    return render_template('general_triage_result.html', gt=gt)

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    user = User.query.get(session['user_id'])
    if request.method == 'POST':
        phone = request.form.get('phone')
        if phone:
            user.phone = phone
            db.session.commit()
            flash('Profile updated successfully.', 'profile_success')
        return redirect(url_for('profile'))
    return render_template('profile.html', user=user)

@app.route('/change-password', methods=['POST'])
@login_required
def change_password():
    user = User.query.get(session['user_id'])
    current_password = request.form.get('current_password')
    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_password')

    if not user.check_password(current_password):
        flash('Incorrect current password.', 'security_danger')
    elif new_password != confirm_password:
        flash('New passwords do not match.', 'security_danger')
    else:
        user.set_password(new_password)
        db.session.commit()
        flash('Password updated successfully. You can log in with your new password.', 'security_success')
        
    return redirect(url_for('profile'))

def parse_diastolic(bp_string):
    """Extracts diastolic value from a string like '120/80'"""
    try:
        if '/' in str(bp_string):
            return float(str(bp_string).split('/')[-1].strip())
        return float(bp_string)
    except:
        return 0.0

@app.route('/triage', methods=['GET', 'POST'])
@login_required
def triage():
    """
    GET  — shows triage form
    POST — runs ML prediction and saves patient
    """
    user = User.query.get(session['user_id'])
    latest_gt = GeneralTriage.query.filter_by(user_id=session['user_id']).order_by(GeneralTriage.created_at.desc()).first()
    
    # Pre-parse BP for the clinical form if it exists
    parsed_bp = 0
    if latest_gt:
        parsed_bp = parse_diastolic(latest_gt.blood_pressure)


    if request.method == 'POST':
        # Get details from session + form
        full_name = session.get('user_name')
        email = session.get('user_email')
        age = int(request.form['age'])
        gender = request.form['gender']
        phone = request.form['phone']
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['blood_pressure'])
        skin_thickness = float(request.form['skin_thickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])

        # Prepare input for ML model (with interaction features)
        glucose_bmi = glucose * bmi
        glucose_age = glucose * age
        input_data = np.array([[
            pregnancies, glucose, blood_pressure,
            skin_thickness, insulin, bmi, dpf, age,
            glucose_bmi, glucose_age
        ]])

        # Scale and predict
        input_scaled = scaler.transform(input_data)
        prediction = int(model.predict(input_scaled)[0])
        probability = float(
            model.predict_proba(input_scaled)[0][1]
        )

        risk_info = get_risk_level(probability)

        # Save patient to database
        new_patient = Patient(
            user_id=session.get('user_id'),
            full_name=full_name,
            age=age,
            gender=gender,
            phone=phone,
            email=email,
            pregnancies=pregnancies,
            glucose=glucose,
            blood_pressure=blood_pressure,
            skin_thickness=skin_thickness,
            insulin=insulin,
            bmi=bmi,
            dpf=dpf,
            prediction=prediction,
            probability=probability,
            risk_level=risk_info['level']
        )
        db.session.add(new_patient)
        db.session.commit()

        session['patient_id'] = new_patient.id
        return redirect(url_for('result'))

    return render_template('triage.html', latest_gt=latest_gt, user=user, parsed_bp=parsed_bp)



@app.route('/result')
@login_required
def result():
    """Shows ML prediction result"""
    patient_id = session.get('patient_id')
    if not patient_id:
        return redirect(url_for('triage'))

    patient = Patient.query.get(patient_id)
    risk_info = get_risk_level(patient.probability)
    doctors = recommend_doctors(patient.risk_level)
    health_tips = get_health_tips(patient)

    return render_template('result.html',
                           patient=patient,
                           risk_info=risk_info,
                           doctors=doctors,
                           health_tips=health_tips)


@app.route('/book/<int:doctor_id>', methods=['GET', 'POST'])
@login_required
def book_appointment(doctor_id):
    """Appointment booking"""
    patient_id = session.get('patient_id')
    if not patient_id:
        return redirect(url_for('triage'))

    doctor = Doctor.query.get_or_404(doctor_id)
    patient = Patient.query.get(patient_id)

    if request.method == 'POST':
        appointment = Appointment(
            patient_id=patient_id,
            doctor_id=doctor_id,
            appointment_date=request.form['date'],
            appointment_time=request.form['time'],
            notes=request.form.get('notes', ''),
            status='Pending'
        )
        db.session.add(appointment)
        db.session.commit()
        session['appointment_id'] = appointment.id
        return redirect(url_for('confirmation'))

    return render_template('appointment.html',
                           doctor=doctor,
                           patient=patient)


@app.route('/confirmation')
@login_required
def confirmation():
    """Booking confirmation"""
    appointment_id = session.get('appointment_id')
    if not appointment_id:
        return redirect(url_for('index'))

    appointment = Appointment.query.get(appointment_id)
    return render_template('confirmation.html',
                           appointment=appointment)


@app.route('/report/<int:patient_id>')
@any_auth_required
def generate_report(patient_id):
    """Generates PDF report"""
    patient = Patient.query.get_or_404(patient_id)
    risk_info = get_risk_level(patient.probability)
    health_tips = get_health_tips(patient)
    appointment = patient.appointment

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=50, leftMargin=50,
                            topMargin=50, bottomMargin=50)
    elements = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=22,
        textColor=colors.HexColor('#1a3c5e'),
        spaceAfter=10
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#27ae60'),
        spaceBefore=15,
        spaceAfter=8
    )

    # Title
    elements.append(Paragraph(
        "DiabetesCare Detection Report", title_style
    ))
    elements.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
        styles['Normal']
    ))
    elements.append(Spacer(1, 20))

    # Patient Information
    elements.append(Paragraph(
        "Patient Information", heading_style
    ))
    patient_data = [
        ['Full Name', patient.full_name],
        ['Age', str(patient.age)],
        ['Gender', patient.gender],
        ['Phone', patient.phone],
        ['Email', patient.email or 'N/A'],
        ['Assessment Date',
         patient.created_at.strftime('%B %d, %Y')]
    ]
    pt = Table(patient_data, colWidths=[150, 350])
    pt.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1),
         colors.HexColor('#EBF5FB')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(pt)
    elements.append(Spacer(1, 15))

    # Diagnosis
    elements.append(Paragraph("Diagnosis Result", heading_style))
    diagnosis_data = [
        ['Diagnosis', risk_info['label']],
        ['Risk Level', risk_info['level']],
        ['Confidence Score', f"{risk_info['percentage']}%"],
        ['Assessment', risk_info['message']]
    ]
    dt = Table(diagnosis_data, colWidths=[150, 350])
    dt.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1),
         colors.HexColor('#EBF5FB')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(dt)
    elements.append(Spacer(1, 15))

    # Clinical Measurements
    elements.append(Paragraph(
        "Clinical Measurements", heading_style
    ))
    measurements_data = [
        ['Measurement', 'Value'],
        ['Glucose Level', f"{patient.glucose} mg/dL"],
        ['Blood Pressure', f"{patient.blood_pressure} mm Hg"],
        ['BMI', f"{patient.bmi} kg/m²"],
        ['Insulin', f"{patient.insulin} µU/mL"],
        ['Skin Thickness', f"{patient.skin_thickness} mm"],
        ['Diabetes Pedigree', str(patient.dpf)],
        ['Pregnancies', str(int(patient.pregnancies))]
    ]
    mt = Table(measurements_data, colWidths=[200, 300])
    mt.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0),
         colors.HexColor('#1a3c5e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(mt)
    elements.append(Spacer(1, 15))

    # Personalized Health Tips
    elements.append(Paragraph(
        "Personalized Health Tips", heading_style
    ))
    for tip in health_tips[:5]:
        elements.append(Paragraph(
            f"{tip['icon']} {tip['title']}: {tip['tip']}",
            styles['Normal']
        ))
        elements.append(Spacer(1, 5))

    # Appointment
    if appointment:
        elements.append(Spacer(1, 15))
        elements.append(Paragraph(
            "Appointment Details", heading_style
        ))
        appt_data = [
            ['Doctor', appointment.doctor.full_name],
            ['Specialization',
             appointment.doctor.specialization],
            ['Hospital', appointment.doctor.hospital],
            ['Date', appointment.appointment_date],
            ['Time', appointment.appointment_time],
            ['Status', appointment.status]
        ]
        at = Table(appt_data, colWidths=[150, 350])
        at.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1),
             colors.HexColor('#EBF5FB')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(at)

    # Disclaimer
    elements.append(Spacer(1, 30))
    elements.append(Paragraph(
        "⚠️ Disclaimer: This report is generated by an "
        "AI-assisted system and is for informational "
        "purposes only. Please consult a qualified medical "
        "professional for diagnosis and treatment.",
        ParagraphStyle('Disclaimer',
                       parent=styles['Normal'],
                       fontSize=9,
                       textColor=colors.grey)
    ))

    doc.build(elements)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name=f'report_{patient.full_name.replace(" ", "_")}.pdf',
        mimetype='application/pdf'
    )


# ============================================================
# ADMIN ROUTES
# ============================================================

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    """Redirect to unified login page (admin tab)."""
    if 'admin_id' in session:
        return redirect(url_for('admin_dashboard'))
    return redirect(url_for('login', role='admin'))


@app.route('/admin/logout')
def admin_logout():
    """Admin logout"""
    session.pop('admin_id', None)
    session.pop('admin_name', None)
    session.pop('admin_role', None)
    return redirect(url_for('login', role='admin'))


@app.route('/doctor/login', methods=['GET', 'POST'])
def doctor_login():
    """Redirect to unified login page (doctor tab)."""
    if 'doctor_id' in session:
        return redirect(url_for('doctor_dashboard'))
    return redirect(url_for('login', role='doctor'))


@app.route('/doctor/logout')
def doctor_logout():
    """Doctor logout"""
    session.pop('doctor_id', None)
    session.pop('doctor_name', None)
    return redirect(url_for('login', role='doctor'))


@app.route('/doctor/dashboard')
@doctor_required
def doctor_dashboard():
    """Doctor dashboard showing their appointments and patient results"""
    doctor_id = session['doctor_id']
    doctor = Doctor.query.get(doctor_id)
    
    # Get only confirmed appointments for this doctor
    appointments = Appointment.query.filter_by(
        doctor_id=doctor_id, 
        status='Confirmed'
    ).order_by(Appointment.created_at.desc()).all()
    
    # Get patient IDs from appointments
    patient_ids = [appt.patient_id for appt in appointments]
    patients = Patient.query.filter(Patient.id.in_(patient_ids)).all() if patient_ids else []
    
    # Create a map of patients for easy access
    patient_map = {p.id: p for p in patients}
    
    return render_template('doctor/dashboard.html', 
                           doctor=doctor, 
                           appointments=appointments,
                           patient_map=patient_map)


@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    """Admin dashboard with statistics based on unique patients"""
    all_assessments = Patient.query.order_by(Patient.created_at.desc()).all()
    total_appointments = Appointment.query.count()
    total_doctors = Doctor.query.count()
    total_users = User.query.count()

    # Distinguish between Assessment Count and Patient Count
    total_assessments = len(all_assessments)
    
    # Group by email to get unique patient statuses (latest assessment first)
    unique_patients_data = {}
    for p in all_assessments:
        if p.email not in unique_patients_data:
            unique_patients_data[p.email] = p.risk_level
            
    total_patients = len(unique_patients_data)
    
    # Count risk levels among the unique patients
    high_risk = list(unique_patients_data.values()).count('Positive')
    medium_risk = list(unique_patients_data.values()).count('High Chances')
    low_risk = list(unique_patients_data.values()).count('Low Chances')

    recent_appointments = Appointment.query.order_by(
        Appointment.created_at.desc()
    ).limit(5).all()

    return render_template('admin/dashboard.html',
                           total_patients=total_patients,
                           total_assessments=total_assessments,
                           total_appointments=total_appointments,
                           total_doctors=total_doctors,
                           total_users=total_users,
                           high_risk=high_risk,
                           medium_risk=medium_risk,
                           low_risk=low_risk,
                           recent_appointments=recent_appointments)


@app.route('/admin/appointments')
@admin_required
def admin_appointments():
    """View all appointments"""
    appointments = Appointment.query.order_by(
        Appointment.created_at.desc()
    ).all()
    return render_template('admin/appointments.html',
                           appointments=appointments)


@app.route('/admin/appointment/update/<int:appt_id>', methods=['POST'])
@admin_required
def update_appointment(appt_id):
    """Update appointment status and admin notes"""
    appointment = Appointment.query.get_or_404(appt_id)
    old_status = appointment.status
    new_status = request.form['status']
    appointment.status = new_status
    
    # Save cancellation reasons/notes if provided
    admin_notes = request.form.get('admin_notes', '').strip()
    if admin_notes:
        appointment.admin_notes = admin_notes
    elif 'admin_notes' in request.form:  # Clear it if it was emptied
        appointment.admin_notes = None

    db.session.commit()

    if new_status != old_status and new_status in ['Confirmed', 'Cancelled'] and appointment.patient.email:
        try:
            msg = Message(
                subject=f"Appointment {new_status} - DiabetesCare",
                sender=("Diabetes Care", "diabetescare15@gmail.com"),
                recipients=[appointment.patient.email]
            )
            if new_status == 'Confirmed':
                msg.body = (f"Hello {appointment.patient.full_name},\n\n"
                            f"Your appointment has been confirmed and scheduled.\n\n"
                            f"--- Appointment Details ---\n"
                            f"Doctor: {appointment.doctor.full_name}\n"
                            f"Specialization: {appointment.doctor.specialization}\n"
                            f"Hospital: {appointment.doctor.hospital}\n"
                            f"Date: {appointment.appointment_date}\n"
                            f"Time: {appointment.appointment_time}\n")
                if admin_notes:
                    msg.body += f"\nAdmin Notes: {admin_notes}\n"
                msg.body += f"\nThank you,\nDiabetesCare Team"
                
            elif new_status == 'Cancelled':
                msg.body = (f"Hello {appointment.patient.full_name},\n\n"
                            f"We regret to inform you that your appointment has been cancelled.\n\n"
                            f"--- Appointment Details ---\n"
                            f"Doctor: {appointment.doctor.full_name}\n"
                            f"Date: {appointment.appointment_date}\n"
                            f"Time: {appointment.appointment_time}\n")
                if admin_notes:
                    msg.body += f"\nReason for Cancellation: {admin_notes}\n"
                else:
                    msg.body += f"\nReason: No specific reason provided. Please contact the hospital for details.\n"
                msg.body += f"\nIf you have any questions, please reach out to us.\n\nThank you,\nDiabetesCare Team"
            
            mail.send(msg)
        except Exception as e:
            print(f"Error sending email: {e}")

    # ── Send SMS notification to patient ───────────────────
    if appointment.patient.phone:
        if new_status == 'Confirmed':
            sms_body = (
                f"DiabetesCare: Your appointment is CONFIRMED.\n"
                f"Doctor: {appointment.doctor.full_name}\n"
                f"Date: {appointment.appointment_date} at {appointment.appointment_time}\n"
                f"Hospital: {appointment.doctor.hospital}"
            )
        elif new_status == 'Cancelled':
            reason = admin_notes if admin_notes else 'Please contact the hospital for details.'
            sms_body = (
                f"DiabetesCare: Your appointment has been CANCELLED.\n"
                f"Doctor: {appointment.doctor.full_name} | Date: {appointment.appointment_date}\n"
                f"Reason: {reason}"
            )
        else:
            sms_body = None

        if sms_body:
            send_sms(appointment.patient.phone, sms_body, category='Appointment')

    return redirect(url_for('admin_appointments'))


@app.route('/admin/patients')
@admin_required
def admin_patients():
    """View all patients grouped by email to show history"""
    all_assessments = Patient.query.order_by(Patient.created_at.desc()).all()
    
    # Group assessments by email
    grouped = {}
    for p in all_assessments:
        email = p.email or 'N/A'
        if email not in grouped:
            grouped[email] = []
        grouped[email].append(p)
    
    # Create a list of the latest assessment for each unique email
    # for the main table display
    unique_patients = []
    for email in grouped:
        latest = grouped[email][0]
        # Attach the history to the object for easy template access
        latest.history = grouped[email]
        
        # Also fetch GeneralTriage history for this user
        user = User.query.filter_by(email=email).first()
        if user:
            latest.gt_history = GeneralTriage.query.filter_by(user_id=user.id).order_by(GeneralTriage.created_at.desc()).all()
        else:
            latest.gt_history = []
            
        unique_patients.append(latest)
        
    return render_template('admin/patients.html', 
                           patients=unique_patients)



@app.route('/admin/doctors')
@admin_required
def admin_doctors():
    """Manage doctors"""
    if session.get('admin_role') != 'Administrator':
        flash('Access restricted to Administrators.')
        return redirect(url_for('admin_dashboard'))
    doctors = Doctor.query.order_by(
        Doctor.specialization, Doctor.full_name
    ).all()
    return render_template('admin/doctors.html',
                           doctors=doctors)


@app.route('/admin/doctors/add', methods=['GET', 'POST'])
@admin_required
def add_doctor():
    """Add new doctor"""
    if session.get('admin_role') != 'Administrator':
        flash('Access restricted to Administrators.')
        return redirect(url_for('admin_dashboard'))
    if request.method == 'POST':
        new_doctor = Doctor(
            full_name=request.form['full_name'],
            specialization=request.form['specialization'],
            hospital=request.form['hospital'],
            phone=request.form['phone'],
            email=request.form['email'],
            experience_years=int(
                request.form['experience_years']
            ),
            bio=request.form['bio'],
            available=True
        )
        new_doctor.set_password(request.form['password'])
        db.session.add(new_doctor)
        db.session.commit()
        flash(f"Doctor {new_doctor.full_name} added successfully.", "success")
        return redirect(url_for('admin_doctors'))
    return render_template('admin/add_doctor.html')


@app.route('/admin/doctors/edit/<int:doctor_id>', methods=['POST'])
@admin_required
def edit_doctor(doctor_id):
    """Edit existing doctor"""
    if session.get('admin_role') != 'Administrator':
        flash('Access restricted to Administrators.')
        return redirect(url_for('admin_dashboard'))
    doctor = Doctor.query.get_or_404(doctor_id)
    doctor.full_name = request.form['full_name']
    doctor.specialization = request.form['specialization']
    doctor.hospital = request.form['hospital']
    doctor.phone = request.form['phone']
    doctor.email = request.form['email']
    doctor.experience_years = int(request.form['experience_years'])
    doctor.bio = request.form.get('bio', '')
    
    db.session.commit()
    flash(f"Doctor {doctor.full_name}'s profile has been perfectly updated.", 'success')
    return redirect(url_for('admin_doctors'))


@app.route('/admin/doctors/toggle/<int:doctor_id>')
@admin_required
def toggle_doctor(doctor_id):
    """Toggle doctor availability"""
    if session.get('admin_role') != 'Administrator':
        flash('Access restricted to Administrators.')
        return redirect(url_for('admin_dashboard'))
    doctor = Doctor.query.get_or_404(doctor_id)
    doctor.available = not doctor.available
    db.session.commit()
    return redirect(url_for('admin_doctors'))


@app.route('/admin/doctors/remove/<int:doctor_id>')
@admin_required
def remove_doctor(doctor_id):
    """Remove doctor"""
    if session.get('admin_role') != 'Administrator':
        flash('Access restricted to Administrators.')
        return redirect(url_for('admin_dashboard'))
    doctor = Doctor.query.get_or_404(doctor_id)
    db.session.delete(doctor)
    db.session.commit()
    return redirect(url_for('admin_doctors'))


@app.route('/admin/doctors/reset_password/<int:doctor_id>', methods=['POST'])
@admin_required
def reset_doctor_password(doctor_id):
    """Reset a doctor's login password"""
    if session.get('admin_role') != 'Administrator':
        flash('Access restricted to Administrators.')
        return redirect(url_for('admin_dashboard'))
        
    doctor = Doctor.query.get_or_404(doctor_id)
    new_password = request.form.get('new_password')
    
    if new_password:
        doctor.set_password(new_password)
        db.session.commit()
        flash(f"Password for {doctor.full_name} has been successfully reset.", "success")
    else:
        flash("Password cannot be empty.", "danger")
        
    return redirect(url_for('admin_doctors'))


@app.route('/admin/reports')
@admin_required
def admin_reports():
    """System reports"""
    if session.get('admin_role') != 'Administrator':
        flash('Access restricted to Administrators.')
        return redirect(url_for('admin_dashboard'))
    total_patients = Patient.query.count()
    high_risk = Patient.query.filter_by(
        risk_level='High Risk').count()
    medium_risk = Patient.query.filter_by(
        risk_level='Medium Risk').count()
    low_risk = Patient.query.filter_by(
        risk_level='Low Risk').count()
    total_appointments = Appointment.query.count()
    pending = Appointment.query.filter_by(
        status='Pending').count()
    confirmed = Appointment.query.filter_by(
        status='Confirmed').count()
    completed = Appointment.query.filter_by(
        status='Completed').count()
    cancelled = Appointment.query.filter_by(
        status='Cancelled').count()
    total_doctors = Doctor.query.count()
    available_doctors = Doctor.query.filter_by(
        available=True).count()

    doctors = Doctor.query.all()
    doctor_stats = []
    for doc in doctors:
        count = Appointment.query.filter_by(
            doctor_id=doc.id).count()
        doctor_stats.append({
            'name': doc.full_name,
            'specialization': doc.specialization,
            'appointments': count
        })
    doctor_stats.sort(
        key=lambda x: x['appointments'], reverse=True
    )

    return render_template('admin/reports.html',
                           total_patients=total_patients,
                           high_risk=high_risk,
                           medium_risk=medium_risk,
                           low_risk=low_risk,
                           total_appointments=total_appointments,
                           pending=pending,
                           confirmed=confirmed,
                           completed=completed,
                           cancelled=cancelled,
                           total_doctors=total_doctors,
                           available_doctors=available_doctors,
                           doctor_stats=doctor_stats)


@app.route('/admin/reports/download')
@admin_required
def download_system_report():
    """Download system PDF report — supports ?period=daily|weekly|monthly|annual|all"""
    if session.get('admin_role') != 'Administrator':
        flash('Access restricted to Administrators.')
        return redirect(url_for('admin_dashboard'))

    from datetime import timedelta
    period = request.args.get('period', 'all')
    now = datetime.now()

    # ── Determine the date filter window ────────────────────────
    if period == 'daily':
        since = now - timedelta(days=1)
        period_label = 'Daily Report'
        period_desc  = f"Last 24 hours — {now.strftime('%B %d, %Y')}"
    elif period == 'weekly':
        since = now - timedelta(weeks=1)
        period_label = 'Weekly Report'
        period_desc  = f"{(now - timedelta(weeks=1)).strftime('%b %d')} – {now.strftime('%b %d, %Y')}"
    elif period == 'monthly':
        since = now - timedelta(days=30)
        period_label = 'Monthly Report'
        period_desc  = f"{(now - timedelta(days=30)).strftime('%b %d')} – {now.strftime('%b %d, %Y')}"
    elif period == 'annual':
        since = now - timedelta(days=365)
        period_label = 'Annual Report'
        period_desc  = f"{(now - timedelta(days=365)).strftime('%b %Y')} – {now.strftime('%b %Y')}"
    else:
        since = None
        period_label = 'All-Time Report'
        period_desc  = f"Complete system history as of {now.strftime('%B %d, %Y')}"

    # ── Query data ───────────────────────────────────────────────
    total_patients = Patient.query.count()
    high_risk   = Patient.query.filter_by(risk_level='High Risk').count()
    medium_risk = Patient.query.filter_by(risk_level='Medium Risk').count()
    low_risk    = Patient.query.filter_by(risk_level='Low Risk').count()
    total_doctors = Doctor.query.count()

    appt_q = Appointment.query
    if since:
        appt_q = appt_q.filter(Appointment.created_at >= since)
    total_appointments = appt_q.count()
    appointments = appt_q.order_by(Appointment.created_at.desc()).all()

    # ── Build PDF ────────────────────────────────────────────────
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=50, leftMargin=50,
                            topMargin=50, bottomMargin=50)
    elements = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'Title', parent=styles['Title'],
        fontSize=22,
        textColor=colors.HexColor('#1a3c5e'),
        spaceAfter=5
    )
    heading_style = ParagraphStyle(
        'Heading', parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#27ae60'),
        spaceBefore=15, spaceAfter=8
    )
    sub_style = ParagraphStyle(
        'Sub', parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#7f8c8d'),
        spaceAfter=4
    )

    elements.append(Paragraph("DiabetesCare System Report", title_style))
    elements.append(Paragraph(period_label, heading_style))
    elements.append(Paragraph(period_desc, sub_style))
    elements.append(Paragraph(
        f"Generated: {now.strftime('%B %d, %Y at %I:%M %p')}",
        styles['Normal']
    ))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("System Overview", heading_style))
    summary_data = [
        ['Metric', 'Count'],
        ['Total Patients', str(total_patients)],
        ['Total Doctors', str(total_doctors)],
        [f'Appointments ({period_label})', str(total_appointments)],
        ['High Risk Patients', str(high_risk)],
        ['Medium Risk Patients', str(medium_risk)],
        ['Low Risk Patients', str(low_risk)],
    ]
    st = Table(summary_data, colWidths=[300, 200])
    st.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a3c5e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(st)
    elements.append(Spacer(1, 15))

    elements.append(Paragraph(
        f"Appointments — {period_label}", heading_style
    ))
    if appointments:
        appt_data = [['Patient', 'Doctor', 'Date', 'Time', 'Status']]
        for a in appointments:
            appt_data.append([
                a.patient.full_name,
                a.doctor.full_name,
                a.appointment_date,
                a.appointment_time,
                a.status
            ])
        at = Table(appt_data, colWidths=[110, 110, 80, 70, 80])
        at.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a3c5e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(at)
    else:
        elements.append(Paragraph(
            f"No appointments found for this period.", styles['Normal']
        ))

    doc.build(elements)
    buffer.seek(0)

    filename = f'system_report_{period}_{now.strftime("%Y%m%d")}.pdf'
    return send_file(
        buffer,
        as_attachment=True,
        download_name=filename,
        mimetype='application/pdf'
    )


# ============================================================
# ADMIN SMS CONTROL PANEL
# ============================================================

@app.route('/admin/sms', methods=['GET', 'POST'])
@admin_required
def admin_sms():
    """
    Admin SMS Control Panel.
    Lets admins send:
      - A bulk SMS to all patients of a given risk group
      - A custom SMS to any single phone number
    All sends are logged in the SMSLog table.
    """
    flash_msg = None
    flash_type = None

    if request.method == 'POST':
        action = request.form.get('action')

        # ── Single SMS ─────────────────────────────────────────
        if action == 'single':
            to_phone = request.form.get('to_phone', '').strip()
            message  = request.form.get('message', '').strip()
            if to_phone and message:
                success = send_sms(to_phone, message, category='Manual')
                if success:
                    flash_msg  = f'SMS sent successfully to {to_phone}.'
                    flash_type = 'success'
                else:
                    flash_msg  = f'SMS to {to_phone} failed. Check Twilio config.'
                    flash_type = 'danger'
            else:
                flash_msg  = 'Phone number and message are required.'
                flash_type = 'warning'

        # ── Bulk SMS ───────────────────────────────────────────
        elif action == 'bulk':
            risk_group = request.form.get('risk_group', '').strip()
            message    = request.form.get('bulk_message', '').strip()
            if not message:
                flash_msg  = 'Message body is required.'
                flash_type = 'warning'
            else:
                # Collect unique phone numbers from the User table
                # linked to patients with the selected risk level
                if risk_group == 'All':
                    users = User.query.filter(User.phone != None).all()
                else:
                    # Find user_ids whose latest patient record matches the risk level
                    patients = Patient.query.filter_by(risk_level=risk_group).all()
                    user_ids = list({p.user_id for p in patients if p.user_id})
                    users = User.query.filter(User.id.in_(user_ids)).all()

                sent_count = 0
                fail_count = 0
                for u in users:
                    if u.phone:
                        ok = send_sms(u.phone, message, category='Bulk')
                        if ok:
                            sent_count += 1
                        else:
                            fail_count += 1

                flash_msg  = (f'Bulk SMS: {sent_count} sent'
                              f'{(", " + str(fail_count) + " failed") if fail_count else ""}.'
                              f' Group: {risk_group}.')
                flash_type = 'success' if sent_count > 0 else 'danger'

    # Fetch recent SMS logs (last 100)
    logs = SMSLog.query.order_by(SMSLog.sent_at.desc()).limit(100).all()
    sms_stats = {
        'total'  : SMSLog.query.count(),
        'sent'   : SMSLog.query.filter_by(status='sent').count(),
        'failed' : SMSLog.query.filter_by(status='failed').count(),
    }

    return render_template('admin/sms.html',
                           logs=logs,
                           sms_stats=sms_stats,
                           flash_msg=flash_msg,
                           flash_type=flash_type)


# ============================================================
# DATABASE SEEDING
# ============================================================

def seed_doctors():
    """Adds 9 doctors on first run"""
    if Doctor.query.count() == 0:
        doctors = [
            Doctor(
                full_name='Dr. Sarah Mitchell',
                specialization='Endocrinologist',
                hospital='City General Hospital',
                phone='+256-700-100001',
                email='s.mitchell@citygeneral.com',
                experience_years=15,
                bio='Specialist in diabetes management '
                    'with 15 years of clinical experience.',
                available=True
            ),
            Doctor(
                full_name='Dr. James Okonkwo',
                specialization='Endocrinologist',
                hospital='St. Mary Medical Center',
                phone='+256-700-100002',
                email='j.okonkwo@stmary.com',
                experience_years=12,
                bio='Expert in Type 1 and Type 2 diabetes '
                    'and metabolic disorders.',
                available=True
            ),
            Doctor(
                full_name='Dr. Linda Zhao',
                specialization='Endocrinologist',
                hospital='University Health Center',
                phone='+256-700-100003',
                email='l.zhao@uhc.edu',
                experience_years=18,
                bio='Research focused endocrinologist '
                    'specializing in diabetes prevention.',
                available=True
            ),
            Doctor(
                full_name='Dr. Michael Torres',
                specialization='General Practitioner',
                hospital='Downtown Family Clinic',
                phone='+256-700-200001',
                email='m.torres@downtownfamily.com',
                experience_years=10,
                bio='Family medicine physician focused '
                    'on preventive care.',
                available=True
            ),
            Doctor(
                full_name='Dr. Amelia Foster',
                specialization='General Practitioner',
                hospital='Westside Health Clinic',
                phone='+256-700-200002',
                email='a.foster@westside.com',
                experience_years=8,
                bio='Dedicated to early intervention '
                    'for metabolic conditions.',
                available=True
            ),
            Doctor(
                full_name='Dr. Robert Nkemdirim',
                specialization='General Practitioner',
                hospital='Community Care Center',
                phone='+256-700-200003',
                email='r.nkemdirim@communitycare.com',
                experience_years=14,
                bio='Community health physician focused '
                    'on diabetes prevention.',
                available=True
            ),
            Doctor(
                full_name='Dr. Priya Sharma',
                specialization='Nutritionist',
                hospital='Wellness & Nutrition Center',
                phone='+256-700-300001',
                email='p.sharma@wellnesscenter.com',
                experience_years=9,
                bio='Clinical nutritionist specializing '
                    'in diabetic diets.',
                available=True
            ),
            Doctor(
                full_name='Dr. Kevin Adeyemi',
                specialization='Nutritionist',
                hospital='Holistic Health Clinic',
                phone='+256-700-300002',
                email='k.adeyemi@holistichealth.com',
                experience_years=7,
                bio='Expert in therapeutic nutrition '
                    'for diabetes prevention.',
                available=True
            ),
            Doctor(
                full_name='Dr. Susan Park',
                specialization='Nutritionist',
                hospital='City General Hospital',
                phone='+256-700-300003',
                email='s.park@citygeneral.com',
                experience_years=11,
                bio='Hospital based nutritionist managing '
                    'diabetic patients through diet.',
                available=True
            ),
        ]
        db.session.add_all(doctors)
        db.session.commit()
        print("✅ 9 doctors added!")


def seed_admin():
    """Creates admin accounts on first run"""
    if Admin.query.count() == 0:
        # Account 1: Main Admin
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
        print("[SUCCESS] Admin and Receptionist accounts created!")



# ============================================================
# RUN THE APP
# ============================================================

# Ensure database is ready before serving
with app.app_context():
    db.create_all()   # also creates sms_logs table
    seed_doctors()
    seed_admin()

if __name__ == '__main__':
    # Use PORT from environment or default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)