# app.py - Complete Version with HTTPS Support
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
import sqlite3
import pandas as pd
import joblib
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import security
from flask_wtf.csrf import CSRFProtect
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

# ========== LANGUAGE DETECTION ==========
from langdetect import detect, DetectorFactory
# ========================================

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default-secret-key')
app.config['SESSION_COOKIE_SECURE'] = True  # HTTPS only
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=5)
app.config['REMEMBER_COOKIE_DURATION'] = timedelta(days=0)

@app.before_request
def make_session_permanent():
    session.permanent = True

csrf = CSRFProtect(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = "🔐 Please login to access the dashboard."
login_manager.login_message_category = "info"

class User(UserMixin):
    def __init__(self, id):
        self.id = id

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

try:
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"⚠️ Could not load models: {e}")
    model = None
    vectorizer = None

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
    print("✅ Database initialized!")

init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username == 'admin' and password == 'admin123':
            user = User(1)
            login_user(user, remember=False)
            next_page = request.args.get('next')
            if next_page:
                return redirect(next_page)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

# ========== LANGUAGE DETECTION FUNCTION ==========
def detect_indian_language(text):
    """Detect Indian languages using langdetect + native script detection"""
    DetectorFactory.seed = 0
    
    # 1. Check for native scripts (Unicode characters)
    if any('\u0C00' <= c <= '\u0C7F' for c in text):
        return 'Telugu'
    if any('\u0900' <= c <= '\u097F' for c in text):
        return 'Hindi'
    if any('\u0B80' <= c <= '\u0BFF' for c in text):
        return 'Tamil'
    if any('\u0D00' <= c <= '\u0D7F' for c in text):
        return 'Malayalam'
    if any('\u0C80' <= c <= '\u0CFF' for c in text):
        return 'Kannada'
    if any('\u0980' <= c <= '\u09FF' for c in text):
        return 'Bengali'
    if any('\u0A80' <= c <= '\u0AFF' for c in text):
        return 'Gujarati'
    if any('\u0600' <= c <= '\u06FF' for c in text):
        return 'Urdu'
    if any('\u0A00' <= c <= '\u0A7F' for c in text):
        return 'Punjabi'
    
    # 2. Use langdetect for transliterated text
    try:
        lang_code = detect(text)
        lang_map = {
            'te': 'Telugu',
            'hi': 'Hindi',
            'ta': 'Tamil',
            'ml': 'Malayalam',
            'kn': 'Kannada',
            'bn': 'Bengali',
            'gu': 'Gujarati',
            'mr': 'Marathi',
            'ur': 'Urdu',
            'pa': 'Punjabi',
            'en': 'English'
        }
        return lang_map.get(lang_code, 'English')
    except Exception as e:
        print(f"⚠️ Language detection error: {e}")
        return 'English'
# ====================================================

# ========== SIMPLE TRANSLITERATION ==========
def transliterate_to_native(text, lang):
    """Simple transliteration - returns original text with language name"""
    return text
# ===========================================

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        original_text = request.form['feedback']
        
        # Detect language
        lang = detect_indian_language(original_text)
        
        # Transliterate (simple version - just returns original)
        display_text = transliterate_to_native(original_text, lang)
        
        # Check if models are loaded
        if model is None or vectorizer is None:
            flash('ML models not available. Please try again later.', 'danger')
            return render_template('feedback.html', submitted=False)
        
        # Predict sentiment
        try:
            text_vectorized = vectorizer.transform([original_text])
            prediction = model.predict(text_vectorized)[0]
            probabilities = model.predict_proba(text_vectorized)[0]
            confidence = max(probabilities) * 100
        except Exception as e:
            flash(f'Error processing sentiment: {e}', 'danger')
            return render_template('feedback.html', submitted=False)
        
        sentiment_map = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
        sentiment = sentiment_map[prediction]
        
        # Encrypt and save
        try:
            encrypted_text = security.encrypt_text(original_text)
        except Exception as e:
            print(f"⚠️ Encryption error: {e}")
            encrypted_text = original_text.encode()
        
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
                             feedback=display_text,
                             original_text=original_text,
                             sentiment=sentiment,
                             confidence=round(confidence, 2),
                             language=lang)
    
    return render_template('feedback.html', submitted=False)

# ========== PROTECTED DASHBOARD ==========
@app.route('/dashboard')
@login_required
def dashboard():
    conn = sqlite3.connect(os.getenv('DATABASE_PATH', 'database.db'))
    
    df = pd.read_sql_query("SELECT * FROM feedback ORDER BY timestamp DESC LIMIT 100", conn)
    
    if not df.empty:
        try:
            df['text'] = df['encrypted_text'].apply(lambda x: security.decrypt_text(x) if x else "")
        except:
            df['text'] = 'Encrypted data'
    
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
    
    if model is None or vectorizer is None:
        return jsonify({'error': 'ML models not available'}), 503
    
    try:
        text_vectorized = vectorizer.transform([text])
        prediction = model.predict(text_vectorized)[0]
        sentiment_map = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
        sentiment = sentiment_map[prediction]
    except:
        sentiment = 'Neutral'
    
    lang = detect_indian_language(text)
    
    return jsonify({
        'sentiment': sentiment,
        'language': lang,
        'encrypted': True
    })

# ========== HTTPS SUPPORT ==========
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    
    # Check if pyOpenSSL is installed
    try:
        import ssl
        # Run with HTTPS
        app.run(host='0.0.0.0', port=port, debug=True, ssl_context='adhoc')
    except Exception as e:
        print(f"⚠️ HTTPS not available: {e}")
        print("📌 Running with HTTP...")
        app.run(host='0.0.0.0', port=port, debug=True)
# =====================================