from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('disease_prediction_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.json['symptoms']
    symptoms_vectorized = vectorizer.transform([symptoms])
    prediction = model.predict(symptoms_vectorized)
    return jsonify({'disease': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
