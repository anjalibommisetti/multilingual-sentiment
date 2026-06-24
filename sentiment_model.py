# sentiment_model.py - Diagnostic Version
import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

print("🔄 Diagnostic: Checking dataset...")

def load_from_txt(filename, max_rows=1000):
    """Load data from FastText format txt file"""
    data = []
    count = 0
    
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if count >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('__label__'):
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    label_text = parts[0]
                    text = parts[1]
                    
                    if '__label__1' in label_text:
                        label = 1  # Positive
                    elif '__label__2' in label_text:
                        label = 2  # Neutral
                    else:
                        label = 0  # Negative
                    
                    data.append({'text': text, 'label': label})
                    count += 1
                    print(f"  Sample {count}: Label={label}, Text={text[:50]}...")
    
    return pd.DataFrame(data)

# ========== LOAD DATA ==========
if os.path.exists('train.ft.txt'):
    print("📖 Reading train.ft.txt (first 1000 rows)...")
    df = load_from_txt('train.ft.txt', max_rows=1000)
    print(f"📊 Loaded: {len(df)} rows")
else:
    print("❌ train.ft.txt not found!")
    exit()

if len(df) == 0:
    print("❌ No data loaded! Check file format.")
    print("📄 First line of train.ft.txt:")
    with open('train.ft.txt', 'r', encoding='utf-8') as f:
        print(f.readline())
    exit()

print(f"📊 Labels distribution: {df['label'].value_counts().to_dict()}")

# ========== TRAIN MODEL ==========
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🔄 Vectorizing...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("🔄 Training model...")
model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
accuracy = (y_pred == y_test).mean()
print(f"✅ Model accuracy: {accuracy*100:.2f}%")

joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("💾 Model saved!")
print("✅ Done!")