from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import hashlib

from utils.config import settings
from utils.logger import logger
from utils.vector_store import add_to_faiss_index
from utils.rag import answer_question

app = FastAPI(title="CV Chat API", version="1.0")

# Enable CORS properly for FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "data"
FAISS_DIR = "faiss_index"
HASH_INDEX_FILE = "hash_index.json"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FAISS_DIR, exist_ok=True)

# Initialize candidate manager
from utils.candidate_manager import CandidateManager
candidate_manager = CandidateManager(HASH_INDEX_FILE)

def compute_pdf_hash(file_path: str) -> str:
    """Compute a SHA256 hash of the PDF content."""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

@app.post("/upload-pdf")
async def upload_only_pdf(file: UploadFile = File(...)):
    """Upload a PDF, append embeddings, and remove file after indexing."""
    try:
        if not file.filename.endswith(".pdf"):
            return JSONResponse(status_code=400, content={"error": "Only PDF files are allowed."})

        file_path = os.path.join(UPLOAD_DIR, file.filename)

        # Save temporarily
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Compute file hash
        file_hash = compute_pdf_hash(file_path)

        # Build/append embeddings with candidate management
        msg,status_code = add_to_faiss_index(file_path, file_hash, FAISS_DIR, HASH_INDEX_FILE)

        # Remove the uploaded file after processing
        os.remove(file_path)

        # Get candidate info for response
        hash_index = candidate_manager.load_hash_index()
        candidate_id = hash_index["file_hashes"].get(file_hash)
        candidate_data = hash_index["candidates"].get(candidate_id, {})
        
        return {
            "message": msg,
            "candidate_name": candidate_data.get("candidate_name", file.filename),
            "status_code": status_code
        }

    except Exception as e:
        logger.error(f"PDF upload failed: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/ask")
async def ask_question(question: str = Form(...)):
    """Ask a question - simplified LLM-driven approach"""
    try:
        # Always call answer_question - it handles both scenarios internally
        answer = answer_question(question)

        if not answer or answer.strip() == "":
            return {"question": question, "answer": "I couldn't generate a response. Please try rephrasing your question."}

        return {"question": question, "answer": answer}

    except Exception as e:
        logger.error(f"Question answering failed: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/reset")
async def reset_vector_store():
    """Completely reset FAISS and hash store."""
    try:
        # Delete FAISS and hash index
        if os.path.exists(FAISS_DIR):
            shutil.rmtree(FAISS_DIR)
            os.makedirs(FAISS_DIR, exist_ok=True)

        if os.path.exists(HASH_INDEX_FILE):
            os.remove(HASH_INDEX_FILE)

        logger.warning("Vector store and candidate index reset")
        return {"message": "Vector store and hash index reset successfully."}

    except Exception as e:
        logger.error(f"Reset failed: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Reset failed: {str(e)}"})

# @app.get("/debug-chunks")
# async def debug_chunks():
#     """Debug endpoint to see what's actually in the vector store"""
#     try:
#         if not os.path.exists(FAISS_DIR) or not os.listdir(FAISS_DIR):
#             return {"message": "No documents uploaded"}
        
#         from langchain_community.embeddings import HuggingFaceEmbeddings
#         from langchain_community.vectorstores import FAISS
        
#         embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#         vector_store = FAISS.load_local(FAISS_DIR, embedding_model, allow_dangerous_deserialization=True)
        
#         # Get all documents
#         all_docs = vector_store.similarity_search("", k=20)  # Empty query to get all
        
#         chunk_info = []
#         for i, doc in enumerate(all_docs):
#             chunk_info.append({
#                 "chunk_number": i + 1,
#                 "candidate_name": doc.metadata.get('candidate_name', 'Unknown'),
#                 "content_length": len(doc.page_content),
#                 "content_preview": doc.page_content[:100] + "...",
#                 "metadata": doc.metadata
#             })
        
#         return {
#             "total_chunks": len(all_docs),
#             "total_candidates": len(set([doc.metadata.get('candidate_name', 'Unknown') for doc in all_docs])),
#             "chunks": chunk_info
#         }
        
#     except Exception as e:
#         return {"error": str(e)}