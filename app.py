from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
import os
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from io import BytesIO

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Helper function to process data
def process_data(file_path):
    data = pd.read_csv(file_path)
    if 'age' not in data.columns or 'lab_test_result' not in data.columns:
        raise ValueError("CSV must contain 'age' and 'lab_test_result' columns.")
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[['age', 'lab_test_result']])
    iso_forest = IsolationForest(contamination=0.2, random_state=42)
    data['anomaly'] = iso_forest.fit_predict(data_scaled)
    return data

# Train a Random Forest model
def train_model(data):
    if 'diagnosis' not in data.columns:
        raise ValueError("CSV must contain 'diagnosis' column for model training.")
    
    X_train, X_test, y_train, y_test = train_test_split(
        data[['age', 'lab_test_result']], data['diagnosis'], test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(random_state=42, n_estimators=50)
    model.fit(X_train, y_train)
    return model, X_test, y_test

# Create a pie chart for anomaly counts
def create_pie_chart(anomaly_counts):
    labels = ['Normal', 'Anomalous']
    sizes = [anomaly_counts.get(1, 0), anomaly_counts.get(-1, 0)]
    
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#FF5722'])
    ax.axis('equal')
    
    img_io = BytesIO()
    plt.savefig(img_io, format='png')
    plt.close(fig)
    img_io.seek(0)
    return img_io

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    try:
        data = process_data(file_path)
        anomaly_counts = data['anomaly'].value_counts().to_dict()
        img_io = create_pie_chart(anomaly_counts)
        model, X_test, y_test = train_model(data)
        model_score = model.score(X_test, y_test)
        
        return jsonify({
            "anomaly_counts": anomaly_counts,
            "model_score": model_score,
            "pie_chart_url": "/chart"
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@app.route('/chart')
def serve_pie_chart():
    img_io = create_pie_chart({1: 100, -1: 50})
    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
