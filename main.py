from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import shutil
import hashlib
import uuid

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
UPLOADED_CVS_DIR = "uploaded_cvs"  # NEW: Permanent storage
UPLOADED_CVS_SERVER = "10.0.6.22:7878"
FAISS_DIR = "faiss_index"
HASH_INDEX_FILE = "hash_index.json"

# Create directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(UPLOADED_CVS_DIR, exist_ok=True)  # NEW
os.makedirs(FAISS_DIR, exist_ok=True)

# Initialize candidate manager
from utils.candidate_manager import CandidateManager
candidate_manager = CandidateManager(HASH_INDEX_FILE)

# Pydantic models for request/response schemas
class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    question: str
    answer: str

class UploadResponse(BaseModel):
    message: str
    candidate_name: str
    status_code: int

class ResetResponse(BaseModel):
    message: str

class ErrorResponse(BaseModel):
    error: str

def compute_pdf_hash(file_path: str) -> str:
    """Compute a SHA256 hash of the PDF content."""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def save_permanent_pdf(file_content: bytes, original_filename: str) -> str:
    """Save PDF permanently with original filename and return the path."""
    # Use original filename for permanent storage
    permanent_path = os.path.join(UPLOADED_CVS_DIR, original_filename)
    
    # Write the file
    with open(permanent_path, "wb") as f:
        f.write(file_content)
    
    logger.info(f"PDF saved permanently: {permanent_path}")
    return permanent_path

@app.post("/upload-pdf", response_model=UploadResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def upload_only_pdf(file: UploadFile = File(...)):
    """Upload a PDF, save permanently, append embeddings, and store real path"""
    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

        # Read file content once
        file_content = await file.read()
        
        # Save to temporary location for processing
        temp_file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(temp_file_path, "wb") as f:
            f.write(file_content)

        # Compute file hash from temporary file
        file_hash = compute_pdf_hash(temp_file_path)

        # Save to permanent storage with original filename
        permanent_path = save_permanent_pdf(file_content, file.filename)

        pdf_name = permanent_path.split("/")[-1]

        pdf_server_path = f"10.0.6.22:7878/{pdf_name}"

        # Build/append embeddings with PERMANENT file path
        msg, status_code = add_to_faiss_index(
            pdf_path=temp_file_path,  # Process from temp location
            file_hash=file_hash, 
            faiss_dir=FAISS_DIR, 
            hash_index_file=HASH_INDEX_FILE,
            permanent_storage_path=pdf_server_path  # Store permanent path
        )

        # Remove the temporary file after processing
        os.remove(temp_file_path)

        # Get candidate info for response
        hash_index = candidate_manager.load_hash_index()
        candidate_id = hash_index["file_hashes"].get(file_hash)
        candidate_data = hash_index["candidates"].get(candidate_id, {})
        
        return UploadResponse(
            message=msg,
            candidate_name=candidate_data.get("candidate_name", file.filename),
            status_code=status_code
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF upload failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PDF upload failed: {str(e)}")

@app.post("/ask", response_model=QuestionResponse, responses={500: {"model": ErrorResponse}})
async def ask_question(request: QuestionRequest):
    """Ask a question - accepts JSON body"""
    try:
        # Always call answer_question - it handles both scenarios internally
        answer = answer_question(request.question)

        if not answer or answer.strip() == "":
            return QuestionResponse(
                question=request.question,
                answer="I couldn't generate a response. Please try rephrasing your question."
            )

        return QuestionResponse(
            question=request.question,
            answer=answer
        )

    except Exception as e:
        logger.error(f"Question answering failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Question answering failed: {str(e)}")

@app.post("/reset", response_model=ResetResponse, responses={500: {"model": ErrorResponse}})
async def reset_vector_store():
    """Completely reset FAISS, hash store, and uploaded PDFs."""
    try:
        # Delete FAISS index
        if os.path.exists(FAISS_DIR):
            shutil.rmtree(FAISS_DIR)
            os.makedirs(FAISS_DIR, exist_ok=True)

        # Delete hash index
        if os.path.exists(HASH_INDEX_FILE):
            os.remove(HASH_INDEX_FILE)

        # NEW: Delete all uploaded PDFs
        if os.path.exists(UPLOADED_CVS_DIR):
            shutil.rmtree(UPLOADED_CVS_DIR)
            os.makedirs(UPLOADED_CVS_DIR, exist_ok=True)

        logger.warning("Vector store, candidate index, and uploaded PDFs reset")
        return ResetResponse(message="Vector store, hash index, and uploaded PDFs reset successfully.")

    except Exception as e:
        logger.error(f"Reset failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=3737)