import joblib
from flask import Flask, request, jsonify

vectorizer = joblib.load('tfidf_vectorizer.pkl')
classifier = joblib.load('email_classifier.pkl')

app = Flask(__name__)

def extract_sentence(text, word):
    sentences = text.split('.')
    for sentence in sentences:
        if word in sentence:
            return sentence.strip()
    return None;

@app.route('/scan-api/email', methods=['GET'])
def predict_email_fraud():

   
    email_text = request.args["content"]

    X = vectorizer.transform([email_text])

    prediction_prob = classifier.predict_proba(X)
    fraud_prob = prediction_prob[0][1]
    is_fraud = 1 if fraud_prob > 0.5 else 0


    

    tfidf_result = X.toarray()
    suspicious_word_index = tfidf_result.argmax()
    suspicious_word = vectorizer.get_feature_names_out()[suspicious_word_index]

    suspicious_sentence = extract_sentence(email_text, suspicious_word)
    
    _fraud_prob = round(fraud_prob * 100,2);
    return jsonify({
      'is_fraud': is_fraud ,
      'FraudPossibility': str(_fraud_prob)+"%",
      "suspicious_sentence":suspicious_sentence,
      "email_text":email_text
    }), 200

if __name__ == '__main__':
  # predict_email_fraud()
  app.run(host="0.0.0.0", port=5000)