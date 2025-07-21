from flask import Flask, request, jsonify
from flask_cors import CORS
from qa_logic import process_question, save_document

app = Flask(__name__)
CORS(app)

@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "")
    answer = process_question(question)
    return jsonify({"answer": answer})

@app.route("/api/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if file and file.filename.endswith(".txt"):
        save_document(file)
        return jsonify({"message": "Document uploaded successfully."})
    return jsonify({"error": "Invalid file type"}), 400

if __name__ == "__main__":
    app.run(debug=True)
