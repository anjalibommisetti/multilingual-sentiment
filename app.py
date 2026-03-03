# app.py - Complete Fixed Version with Auto Logout
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
import sqlite3
import pandas as pd
import joblib
import os
from datetime import datetime, timedelta
from langdetect import detect, DetectorFactory
from dotenv import load_dotenv
import security  # Our encryption module
from flask_wtf.csrf import CSRFProtect
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default-secret-key-change-this')
app.config['SESSION_COOKIE_SECURE'] = False  # False for development (HTTP)
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# ========== AUTO LOGOUT CONFIGURATION ==========
from datetime import timedelta
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=5)  # Session expires in 5 minutes
app.config['REMEMBER_COOKIE_DURATION'] = timedelta(days=0)  # No remember me
app.config['SESSION_REFRESH_EACH_REQUEST'] = True  # Refresh session on each request
# ===============================================

# Make session permanent
@app.before_request
def make_session_permanent():
    session.permanent = True
    app.permanent_session_lifetime = timedelta(minutes=5)

# Initialize CSRF protection
csrf = CSRFProtect(app)

# ========== FIXED LOGIN MANAGER ==========
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'  # Route name for login page
login_manager.login_message = "🔐 Please login to access the dashboard."
login_manager.login_message_category = "info"
# =========================================

class User(UserMixin):
    def __init__(self, id):
        self.id = id

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# Load ML model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def init_db():
    conn = sqlite3.connect(os.getenv('DATABASE_PATH', 'database.db'))
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS feedback
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  encrypted_text BLOB,
                  sentiment TEXT,
                  confidence REAL,
                  language TEXT,
                  timestamp DATETIME,
                  ip_address TEXT)''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    return render_template('index.html')

# ========== FIXED LOGIN ROUTE ==========
@app.route('/login', methods=['GET', 'POST'])
def login():
    # If user is already logged in, go to dashboard
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Simple admin check
        if username == 'admin' and password == 'admin123':
            user = User(1)
            login_user(user, remember=False)  # remember=False is important!
            
            # Get the page they tried to access (if any)
            next_page = request.args.get('next')
            if next_page:
                return redirect(next_page)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')
# =======================================

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()  # Clear all session data
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

# ========== LANGUAGE DETECTION FUNCTION ==========
def detect_indian_language(text):
    """Detect Indian languages with extensive word lists"""
    DetectorFactory.seed = 0
    
    try:
        detected_code = detect(text)
        language_map = {
            'te': 'తెలుగు',
            'hi': 'हिन्दी',
            'ta': 'தமிழ்',
            'en': 'English',
            'ml': 'മലയാളം',
            'kn': 'ಕನ್ನಡ',
            'bn': 'বাংলা',
            'gu': 'ગુજરાતી',
            'mr': 'मराठी',
            'ur': 'اردو',
            'pa': 'ਪੰਜਾਬੀ',
            'or': 'ଓଡ଼ିଆ',
        }
        return language_map.get(detected_code, 'English')
    except:
        text_lower = text.lower()
        
        telugu_words = ['చాలా', 'బాగుంది', 'ధన్యవాదాలు', 'నాకు', 'ఇది', 'సంతోషం', 'చెత్త', 'లేదు']
        hindi_words = ['बहुत', 'अच्छा', 'धन्यवाद', 'मुझे', 'यह', 'खराब', 'नहीं']
        tamil_words = ['நன்றி', 'நல்ல', 'எனக்கு', 'இது', 'மோசம்', 'இல்லை']
        
        telugu_count = sum(1 for word in telugu_words if word in text_lower)
        hindi_count = sum(1 for word in hindi_words if word in text_lower)
        tamil_count = sum(1 for word in tamil_words if word in text_lower)
        
        if telugu_count > 0:
            return 'తెలుగు'
        elif hindi_count > 0:
            return 'हिन्दी'
        elif tamil_count > 0:
            return 'தமிழ்'
        else:
            return 'English'
# ================================================

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        feedback_text = request.form['feedback']
        
        lang = detect_indian_language(feedback_text)
        
        text_vectorized = vectorizer.transform([feedback_text])
        prediction = model.predict(text_vectorized)[0]
        probabilities = model.predict_proba(text_vectorized)[0]
        confidence = max(probabilities) * 100
        
        sentiment_map = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
        sentiment = sentiment_map[prediction]
        
        encrypted_text = security.encrypt_text(feedback_text)
        ip_address = request.remote_addr
        
        conn = sqlite3.connect(os.getenv('DATABASE_PATH', 'database.db'))
        c = conn.cursor()
        c.execute('''INSERT INTO feedback 
                     (encrypted_text, sentiment, confidence, language, timestamp, ip_address) 
                     VALUES (?, ?, ?, ?, ?, ?)''',
                  (encrypted_text, sentiment, confidence, lang, datetime.now(), ip_address))
        conn.commit()
        conn.close()
        
        return render_template('feedback.html', 
                             submitted=True,
                             feedback=feedback_text,
                             sentiment=sentiment,
                             confidence=round(confidence, 2),
                             language=lang)
    
    return render_template('feedback.html', submitted=False)

# ========== PROTECTED DASHBOARD ==========
@app.route('/dashboard')
@login_required  # This ensures redirect to login if not authenticated
def dashboard():
    conn = sqlite3.connect(os.getenv('DATABASE_PATH', 'database.db'))
    
    df = pd.read_sql_query("SELECT * FROM feedback ORDER BY timestamp DESC LIMIT 100", conn)
    
    if not df.empty:
        df['text'] = df['encrypted_text'].apply(lambda x: security.decrypt_text(x) if x else "")
    
    total = len(df)
    positive = len(df[df['sentiment'] == 'Positive']) if total > 0 else 0
    negative = len(df[df['sentiment'] == 'Negative']) if total > 0 else 0
    neutral = len(df[df['sentiment'] == 'Neutral']) if total > 0 else 0
    
    lang_stats = df['language'].value_counts().to_dict() if total > 0 else {}
    recent = df.head(10).to_dict('records') if total > 0 else []
    
    conn.close()
    
    return render_template('dashboard.html',
                         total=total,
                         positive=positive,
                         negative=negative,
                         neutral=neutral,
                         lang_stats=lang_stats,
                         recent=recent,
                         user=current_user)
# =========================================

@app.route('/api/analyze', methods=['POST'])
@csrf.exempt
def api_analyze():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]
    sentiment_map = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
    
    lang = detect_indian_language(text)
    
    return jsonify({
        'sentiment': sentiment_map[prediction],
        'language': lang,
        'encrypted': True
    })

if __name__ == '__main__':
    app.run(debug=True)