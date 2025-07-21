import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import pickle

# Directory setup
DATA_DIR = "data"
INDEX_DIR = "index"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# Load SentenceTransformer model safely
try:
   model = SentenceTransformer("all-MiniLM-L6-v2")  # or your preferred model
except Exception as e:
    print("Error loading embedding model:", e)
    model = None

# Load FLAN-T5 model
qa_model = pipeline("text2text-generation", model="google/flan-t5-base", device=-1)

# Index files
index_file = os.path.join(INDEX_DIR, "faiss.index")
text_data_file = os.path.join(INDEX_DIR, "texts.pkl")

texts = []
index = None

def load_index():
    global index, texts
    if os.path.exists(index_file):
        index = faiss.read_index(index_file)
        with open(text_data_file, "rb") as f:
            texts = pickle.load(f)
    else:
        index = faiss.IndexFlatL2(384)  # 384 = dim of all-MiniLM-L6-v2

def save_index():
    faiss.write_index(index, index_file)
    with open(text_data_file, "wb") as f:
        pickle.dump(texts, f)

def process_file(text):
    global index, texts, model
    if model is None:
        raise ValueError("Embedding model not loaded.")
    sentences = text.split("\n")
    embeddings = model.encode(sentences)
    index.add(np.array(embeddings).astype("float32"))
    texts.extend(sentences)
    save_index()

def answer_query(query):
    global index, texts, model
    if model is None:
        raise ValueError("Embedding model not loaded.")
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding).astype("float32"), k=5)
    context = " ".join([texts[i] for i in I[0]])
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    answer = qa_model(prompt, max_length=100, do_sample=False)[0]['generated_text']
    return answer.strip()

def get_uploaded_docs():
    files = []
    for f in os.listdir(DATA_DIR):
        if f.endswith(".txt"):
            with open(os.path.join(DATA_DIR, f), 'r', encoding='utf-8') as file:
                content = file.read()
                files.append({"name": f, "content": content})
    return files
