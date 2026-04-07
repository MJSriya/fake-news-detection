import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

def train_model():
    print("Loading data...")
    try:
        df = pd.read_csv('fake_news_dataset.csv')
    except Exception as e:
        print(f"Error loading csv: {e}")
        return

    # Feature extraction
    # Combine title and text for a richer feature set, handle missing values
    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')
    df['content'] = df['title'] + " " + df['text']
    
    # Target variable setup
    # Labels are 'Fake' or 'Real'
    labels = df['label']
    
    print("Splitting dataset...")
    x_train, x_test, y_train, y_test = train_test_split(df['content'], labels, test_size=0.2, random_state=42)
    
    print("Vectorizing text...")
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(x_train) 
    tfidf_test = tfidf_vectorizer.transform(x_test)
    
    print("Training model...")
    pac = LogisticRegression()
    pac.fit(tfidf_train, y_train)
    
    print("Evaluating model...")
    y_pred = pac.predict(tfidf_test)
    score = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {round(score*100,2)}%')
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Saving model and vectorizer...")
    with open('model.pkl', 'wb') as model_file:
        pickle.dump(pac, model_file)
    with open('vectorizer.pkl', 'wb') as vec_file:
        pickle.dump(tfidf_vectorizer, vec_file)
        
    print("Done!")

if __name__ == "__main__":
    train_model()
