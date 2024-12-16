import os
import faiss
import torch
import chromadb
import transformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

class DocumentChatSystem:
    def __init__(self, 
                 pdf_directory='./documents', 
                 embedding_model='all-MiniLM-L6-v2',
                 llm_model='facebook/opt-350m'):
                     
        # Ensure directory exists
        os.makedirs(pdf_directory, exist_ok=True)     
                 
                     
        # Use smaller, more compatible models
        self.pdf_directory = pdf_directory
        
        # Embedding Model Setup
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
               
        except Exception as e:
            print(f"Embedding model error: {e}")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Vector Database Setup
        try:
            self.vector_db = chromadb.PersistentClient(path="./chroma_db")
            self.collection = self.vector_db.get_or_create_collection("document_collection")
           
        except Exception as e:
            print(f"Vector database error: {e}")
            self.vector_db = None
            self.collection = None
        
        # LLM Setup with error handling
        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(llm_model)
            self.llm = transformers.AutoModelForCausalLM.from_pretrained(llm_model)
 
        except Exception as e:
            print(f"Model loading error: {e}")
            # Fallback to a smaller model if needed
            self.tokenizer = transformers.AutoTokenizer.from_pretrained('distilgpt2')
            self.llm = transformers.AutoModelForCausalLM.from_pretrained('distilgpt2')
        
    def load_documents(self):
   
        """
        Load and process PDF documents from the specified directory
        """
        # Check if vector database is initialized
        if self.collection is None:
            print("Vector database not initialized")
            return 0
        
        # Ensure documents directory exists
        if not os.path.exists(self.pdf_directory):
            os.makedirs(self.pdf_directory)
            
        document_chunks = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=100
        )
        
        # Check if directory is empty
        pdf_files = [f for f in os.listdir(self.pdf_directory) if f.endswith('.PDF')]
        
        if not pdf_files:
            print(f"No PDF files found in {self.pdf_directory}")
            return 0
        
        for filename in pdf_files:
            filepath = os.path.join(self.pdf_directory, filename)
            try:
                loader = PyPDFLoader(filepath)
                pages = loader.load()
                
                for page in pages:
                    chunks = text_splitter.split_text(page.page_content)
                    document_chunks.extend(chunks)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        
        # Validate chunks before embedding
        if not document_chunks:
            print("No text chunks extracted from PDFs")
            return 0
        
        try:
            # Batch embedding to prevent memory issues
            batch_size = 100
            total_chunks = 0
            for i in range(0, len(document_chunks), batch_size):
                batch_chunks = document_chunks[i:i+batch_size]
                
                # Generate embeddings
                embeddings = self.embedding_model.encode(batch_chunks)
                
                 
                # Add to collection
                self.collection.add(
                    embeddings=embeddings.tolist(),
                    documents=batch_chunks,
                    ids=[f"chunk_{j}" for j in range(total_chunks, total_chunks+len(batch_chunks))]
                )
                    
                total_chunks += len(batch_chunks)
            
            print(f"Processed {total_chunks} document chunks")
            return total_chunks
        
        except Exception as e:
            print(f"Embedding or storage error: {e}")
            return 0
    
    def search_documents(self, query, top_k=5):
        """
        Search documents based on query
        """
        if self.collection is None:
            return []
        
        query_embedding = self.embedding_model.encode([query])[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        return results['documents'][0]
    
    def generate_response(self, context, query):
        """
        Generate response based on context and query
        """
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.llm.generate(
            inputs.input_ids, 
            max_length=2000, 
            num_return_sequences=1
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Flask Web Server for Browser Interaction
app = Flask(__name__, static_folder='frontend')
CORS(app)

document_system = DocumentChatSystem()
document_system.load_documents()

# Serve Frontend Routes
@app.route('/')
def serve_index():
    return send_file('frontend/index.html')

@app.route('/frontend/<path:path>')
def serve_frontend(path):
    return send_from_directory('frontend', path)

# Search and Chat API Routes
@app.route('/search', methods=['POST'])
def search_documents():
    data = request.json
    query = data.get('query', '')
    
    context = document_system.search_documents(query)
    response = document_system.generate_response('\n'.join(context), query)
    
    return jsonify({
        'context': context,
        'response': response
    })

# Additional API Endpoints
@app.route('/documents', methods=['GET'])
def list_documents():
    documents = os.listdir('./documents')
    return jsonify({
        'documents': [doc for doc in documents if doc.endswith('.pdf')]
    })

@app.route('/upload', methods=['POST'])
def upload_document():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.endswith('.pdf'):
        filepath = os.path.join('./documents', file.filename)
        file.save(filepath)
        return jsonify({'message': 'File uploaded successfully'})
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    # Ensure documents directory exists
    os.makedirs('./documents', exist_ok=True)
    os.makedirs('./app/frontend', exist_ok=True)
    
    app.run(host='0.0.0.0', port=5000, debug=True)