from flask import Flask, request, render_template, redirect, url_for
from qa_logic import process_file, answer_query, load_index, get_uploaded_docs
import logging
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
load_index()

@app.route('/')

def home():
    return render_template('index.html')



@app.route('/ask', methods=['POST'])
def ask():
    query = request.form.get('query')  # not 'question'
    answer = answer_query(query)

    if not query or query.strip() == "":
        return render_template('index.html', answer="Please enter a valid query.")

    answer = answer_query(query)
    return render_template('index.html', answer=answer)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename.endswith('.txt'):
            text = file.read().decode('utf-8')
            process_file(text)
            file.save(f'data/{file.filename}')  # Save original for viewing
            return render_template('upload.html', message='File uploaded and indexed!')
        return render_template('upload.html', message='Please upload a .txt file.')
    return render_template('upload.html')

@app.route('/docs')
def docs():
    files = get_uploaded_docs()
    return render_template('docs.html', files=files)

if __name__ == "__main__":
   app.run(debug=True, port=5001)
