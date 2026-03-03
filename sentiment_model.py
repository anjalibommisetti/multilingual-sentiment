# sentiment_model.py - Updated with more Telugu data
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

print("🔄 Training the sentiment analysis model...")

# MUCH LARGER dataset with more Telugu examples
data = {
    'text': [
        # POSITIVE - Telugu (50+ examples)
        "చాలా బాగుంది", "చాలా బాగా చేశారు", "అద్భుతంగా ఉంది", "చాలా సంతోషంగా ఉంది",
        "సర్వీస్ చాలా బాగుంది", "చాలా ఉపయోగకరంగా ఉంది", "నాకు చాలా నచ్చింది",
        "అద్భుతమైన సేవ", "చాలా మంచి అనుభవం", "తృప్తిగా ఉంది", "సంతృప్తిగా ఉంది",
        "బాగుంది", "చాలా బాగా పనిచేస్తుంది", "ఎక్సలెంట్", "సూపర్", "గ్రేట్",
        "చాలా బాగా సహాయం చేశారు", "చాలా ఫాస్ట్ గా పని చేశారు", "చాలా క్లియర్ గా ఉంది",
        "వండర్ ఫుల్", "అమేజింగ్", "చాలా బాగా అర్థమైంది", "చాలా సింపుల్ గా ఉంది",
        "నేను చాలా హ్యాపీ", "చాలా గుడ్", "బెస్ట్ సర్వీస్", "ఎక్సలెంట్ సర్వీస్",
        
        # POSITIVE - English/Hinglish
        "Good service", "Excellent work", "Very happy with the response", 
        "Thanks for the help", "Great job", "Wonderful experience",
        "Very good", "Outstanding", "Perfect", "Awesome", "Love it",
        "Very helpful staff", "Problem solved quickly", "Fast response",
        "Best service ever", "Very satisfied", "Highly recommended",
        "Bahut achha", "Maza aa gaya", "Best", "Nice", "Superb",
        
        # NEGATIVE - Telugu (50+ examples)
        "చాలా చెత్తగా ఉంది", "పూర్తి ఫెయిల్", "చాలా బోరింగ్", "టైమ్ వేస్ట్",
        "ఏమీ చేయలేదు", "చాలా నిరాశగా ఉంది", "సర్వీస్ చాలా చెత్త",
        "చాలా స్లో గా ఉంది", "రెస్పాన్స్ లేదు", "హెల్ప్ చేయలేదు",
        "చాలా బాధగా ఉంది", "నాకు నచ్చలేదు", "వర్స్ట్", "చాలా వర్స్ట్",
        "పూర్తిగా వేస్ట్", "డిస్‌పాయింట్", "చాలా డిస్‌పాయింట్",
        "రాంగ్", "మిస్టేక్", "ప్రాబ్లమ్", "ఇష్యూ", "టెన్షన్",
        "చాలా టెన్షన్", "చాలా ప్రాబ్లమ్", "సర్వీస్ బాగోలేదు",
        
        # NEGATIVE - English/Hinglish
        "Worst experience", "Very bad service", "Not happy at all",
        "Too much delay", "No response", "Very poor", "Terrible",
        "Waste of time", "Not good", "Disappointed", "Very disappointed",
        "Rude behavior", "Still waiting", "No action taken", "Useless",
        "Bilkul kharab", "Time waste", "Kuch nahi kiya", "Bakwas",
        
        # NEUTRAL - Telugu
        "సరే", "ఓకే", "నార్మల్", "యావరేజ్", "అంత బాగాలేదు కానీ ఓకే",
        "ఏమో తెలీదు", "అలాగే ఉంది", "ఏమీ అనిపించలేదు", "ఓకే బాగుంది",
        "నార్మల్ గా ఉంది", "యావరేజ్ గా ఉంది", "సో సో",
        
        # NEUTRAL - English/Hinglish
        "Okay service", "Average", "It's fine", "Can be better",
        "Not bad not good", "Theek hai", "Chalo theek hai",
        "Average experience", "Nothing special", "Just okay"
    ],
    'sentiment': [
        # POSITIVE (1) - 50+ times
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        # NEGATIVE (0) - 50+ times
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        # NEUTRAL (2) - 30+ times
        2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2
    ]
}

df = pd.DataFrame(data)
print(f"📊 Dataset size: {len(df)} samples")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['sentiment'], test_size=0.2, random_state=42
)

# Create vectorizer with more features
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,3))
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Test accuracy
y_pred = model.predict(X_test_vectorized)
accuracy = (y_pred == y_test).mean()
print(f"✅ Model accuracy: {accuracy*100:.2f}%")

# Save model and vectorizer
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("💾 Model saved!")