import joblib


vectorizer = joblib.load('tfidf_vectorizer.pkl')
classifier = joblib.load('email_classifier.pkl')

def extract_sentence(text, word):
    sentences = text.split('.')
    for sentence in sentences:
        if word in sentence:
            return sentence.strip()
    return None

def predict_email_fraud():
 
    email_text = input("Enter email: ")


    X = vectorizer.transform([email_text])
    

    prediction_prob = classifier.predict_proba(X)
    fraud_prob = prediction_prob[0][1]
    is_fraud = 1 if fraud_prob > 0.5 else 0


    print(f"Is fraud email: {is_fraud}")
    print(f"Fraud Possibility: {fraud_prob * 100:.2f}%")

    tfidf_result = X.toarray()
    suspicious_word_index = tfidf_result.argmax()
    suspicious_word = vectorizer.get_feature_names_out()[suspicious_word_index]

    suspicious_sentence = extract_sentence(email_text, suspicious_word)
    
    if suspicious_sentence:
        print(f"Most sespicious part: {suspicious_sentence}")
    else:
        print(f"No sespicious content")

if __name__ == '__main__':
    predict_email_fraud()
