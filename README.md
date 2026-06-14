# Diabetes Detection & Care Management System

An advanced, end-to-end medical web application designed for early diabetes detection, patient triage tracking, clinical appointment scheduling, and communications management.

The system features a custom-engineered, high-accuracy machine learning ensemble model to predict diabetes risks and connects patients to specialized doctors via automated SMS and secure email verification flows.

---

## 🚀 Key Features

### 1. 🧠 ML-Powered Diabetes Risk Assessment
* **Ensemble Prediction Engine**: Uses a soft-voting classifier combining three powerful estimators:
  * **Random Forest Classifier**
  * **HistGradientBoosting Classifier**
  * **Extra Trees Classifier**
* **Robust Imputation**: Utilizes Scikit-Learn's `IterativeImputer` to address missing physiological values (e.g., Blood Pressure, Insulin, BMI) rather than discarding data.
* **Class Balancing**: Applies Synthetic Minority Over-sampling Technique (`SMOTE`) to handle class imbalance, ensuring high sensitivity and specificity.
* **Feature Engineering**: Integrates custom interactive medical metrics (`Glucose_BMI` and `Glucose_Age`) to elevate detection accuracy.
* **Risk Categorization**: Classifies patient results into HSL-themed dynamic indicators:
  * **Normal / Low Risk** (Green)
  * **Moderate Risk** (Orange)
  * **High Risk** (Red)

### 2. 🏥 Automated Medical Triage & Vitals Tracker
* Tracks comprehensive patient physiological vitals (Weight, Height, BMI, Blood Pressure, Smoking, Activity Levels, Family History, Medications, and active Symptoms).
* Maintains historical medical timelines of patient triage evaluations.

### 3. 👥 Unified Multi-Portal Access
* **Patient Portal**: Register, verify accounts via OTP, perform self-assessments, download medical PDF reports, schedule appointments, and update profiles.
* **Doctor Portal**: Manage personal schedules, view assigned patients, review triage vitals, write clinical notes, and manage appointments.
* **Admin Control Panel**: Full dashboard providing overview analytics of patient/doctor counts, doctor management, appointment approval workflows, and system settings.

### 4. ✉️ Communication & Verification Systems
* **Secure OTP Verification**: 2-factor email and SMS verification during registration and password recovery.
* **SMS Gateway Integration**: Real-time SMS notifications via **Africa's Talking SMS API** for appointment confirmations and security OTPs.
* **SMTP Email Server**: Fully integrates with SSL/TLS protocols to dispatch emails securely through SMTP (e.g., Gmail App Passwords).

---

## 🛠️ System Architecture

### 📊 Database Schema (SQLAlchemy Models)
The system operates on 7 relational tables defined in `database.py`:
1. **`User`**: Core patient authentication details (hashes, phone, email).
2. **`Patient`**: Contains medical vitals used specifically by the machine learning engine for diabetes predictions.
3. **`Doctor`**: Professional details, specializations, experiences, bios, and login hashes.
4. **`Admin`**: Administrator login details and role categories.
5. **`Appointment`**: Connects patients with doctors, tracking dates, times, status, and clinical notes.
6. **`GeneralTriage`**: Records vital signs and active symptoms logged by users.
7. **`SMSLog`**: A centralized audit trail logging all SMS messages, categories, and delivery statuses.

### 📁 Project Structure
```bash
diabetes_system/
│
├── app.py                      # Main Flask application (routing, business logic, endpoints)
├── database.py                 # SQLAlchemy relational schemas & authentication methods
├── model.py                    # Machine learning pipeline training script (imputation, SMOTE, Ensemble)
│
├── data/
│   └── diabetes.csv            # Training dataset (Pima Indians Dataset)
│
├── models/                     # Serialized Model Artifacts
│   ├── diabetes_model.pkl      # Trained Ensemble Voting Classifier
│   ├── scaler.pkl              # Fitted StandardScaler
│   └── feature_names.pkl       # Saved feature column structures
│
├── static/                     # Assets & Styling
│   ├── css/                    # Custom CSS files
│   ├── js/                     # Client-side validation & interactivity scripts
│   └── images/                 # SVG assets and system logos
│
├── templates/                  # Jinja2 HTML Front-ends
│   ├── admin/                  # Admin portal pages
│   ├── doctor/                 # Doctor portal pages
│   └── layout files...         # Common pages (login, registration, dashboards)
│
├── requirements.txt            # Project dependencies
├── runtime.txt                 # Target runtime configuration
└── .env                        # Local environment variables
```

---

## 💻 Local Setup & Installation

### 1. Prerequisites
Ensure you have **Python 3.12+** installed on your system.

### 2. Installation Steps
1. Clone the repository and navigate to the project directory:
   ```bash
   cd diabetes_system
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Unix/macOS:
   source venv/bin/activate
   ```
3. Install all dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and configure your credentials:
   ```bash
   cp .env.example .env
   ```
   *Modify the values inside `.env` to match your configurations (see Environment Variables below).*

### 3. Running the Server Locally
To start the local development server:
```bash
python -m flask --app app run --port 5000 --debug
```
*Access the application at `http://127.0.0.1:5000`.*

---

## ⚙️ Environment Variables (`.env`)

Configure the following variables in your local `.env` file or cloud dashboard:

| Variable | Description | Example / Default |
| :--- | :--- | :--- |
| `SECRET_KEY` | Flask session cookie encryption key | `your_secret_key` |
| `DATABASE_URL` | Production PostgreSQL/SQLAlchemy URI | `postgresql://...` |
| `POSTGRES_URL` | Vercel Postgres database URL | `postgres://...` |
| `MAIL_SERVER` | SMTP host server | `smtp.gmail.com` |
| `MAIL_PORT` | SMTP connection port | `465` (SSL) / `587` (TLS) |
| `MAIL_USE_TLS` | Enables explicit TLS/STARTTLS | `true` or `false` |
| `MAIL_USE_SSL` | Enables implicit SSL | `true` or `false` |
| `MAIL_USERNAME` | SMTP email address | `diabetescare15@gmail.com` |
| `MAIL_PASSWORD` | App password for Gmail SMTP account | `asynxmmkwevpodad` |
| `AT_USERNAME` | Africa's Talking API username | `sandbox` (development) |
| `AT_API_KEY` | Africa's Talking API key | `your_api_key_here` |

---

## ☁️ Production Deployment

### Vercel Deployment
The application is pre-configured to run seamlessly on **Vercel** serverless functions:
1. Connect your GitHub repository containing the codebase to Vercel.
2. In the Vercel project settings, configure the **Environment Variables** listed above.
3. Vercel will build the app and connect it to your configured serverless PostgreSQL database (e.g., Neon Postgres).

### Render Deployment
This project includes a `render.yaml` blueprint:
1. Connect your repository to Render.
2. Deploy the blueprint, which automatically provisions a Web Service running Gunicorn and a PostgreSQL database instance.
