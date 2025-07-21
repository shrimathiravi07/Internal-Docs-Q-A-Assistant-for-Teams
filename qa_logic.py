import os
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = "documents/"
INDEX_PATH = "embeddings_store/"

os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(INDEX_PATH, exist_ok=True)

embedding = OpenAIEmbeddings()

def load_index():
    if os.path.exists(os.path.join(INDEX_PATH, "index.faiss")):
        return FAISS.load_local(INDEX_PATH, embedding)
    else:
        return None

def create_index():
    docs = []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(DATA_PATH, file))
            data = loader.load()
            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs.extend(splitter.split_documents(data))
    return FAISS.from_documents(docs, embedding)

def save_document(file):
    path = os.path.join(DATA_PATH, file.filename)
    file.save(path)
    # Recreate index after new upload
    index = create_index()
    index.save_local(INDEX_PATH)

def process_question(question):
    index = load_index()
    if index is None:
        return "No documents found. Please upload documents first."
    docs = index.similarity_search(question)
    chain = load_qa_chain(OpenAI(), chain_type="stuff")
    return chain.run(input_documents=docs, question=question)
