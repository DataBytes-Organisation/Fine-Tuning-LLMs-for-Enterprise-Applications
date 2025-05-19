from flask import Flask, render_template, request, jsonify
from model_loader import predict, qa
import os

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
        sentiment = predict(text)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"sentiment": sentiment})


@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "No query provided"}), 400
    try:
        answer = qa(query)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"answer": answer})


@app.route("/upload", methods=["POST"])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if not file.filename.lower().endswith('.txt'):
        return jsonify({"error": "Only .txt files are supported"}), 400
    try:
        content = file.read().decode('utf-8')
        sentiment = predict(content)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"sentiment": sentiment})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
