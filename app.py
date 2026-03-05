# app.py
from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
import uuid
import logging
import tempfile
import wave
import numpy as np

from src.document_processor import DocumentProcessor
from src.embedding_service import EmbeddingService
from src.vector_db import VectorDatabase
from src.memory_system import MemorySystem
from src.qa_engine import QAEngine
from src.learning_pipeline import LearningPipeline
from src.indexer import index_document

# Import faster-whisper for better performance
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logging.warning("faster-whisper not installed. Install with: pip install faster-whisper")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Quiet mode for libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

app = FastAPI(title="Intelligent Document Q&A System", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Initialize components
# -------------------------
document_processor = DocumentProcessor()
embedding_service = EmbeddingService()
vector_db = VectorDatabase()
memory_system = MemorySystem(vector_db=vector_db, embedding_service=embedding_service)
qa_engine = QAEngine(
    embedding_service=embedding_service, 
    vector_db=vector_db, 
    memory_system=memory_system
)
learning_pipeline = LearningPipeline(vector_db, qa_engine)

# Initialize Whisper model for transcription (if available)
whisper_model = None
if WHISPER_AVAILABLE:
    try:
        # You can change model size: "tiny", "base", "small", "medium", "large"
        # Larger models are more accurate but slower
        whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        logger.info("Whisper model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Whisper model: {e}")

# -------------------------
# Data models
# -------------------------
class QueryRequest(BaseModel):
    query: str
    session_id: str
    document_filters: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    processing_time: float
    session_id: str
    interaction_id: Optional[str] = None

class FeedbackRequest(BaseModel):
    interaction_id: str
    feedback_type: str
    feedback_data: Dict[str, Any]
    corrected_answer: Optional[str] = None

class UploadResponse(BaseModel):
    document_id: str
    chunks_processed: int
    status: str

class RenameSessionRequest(BaseModel):
    name: str

class TranscriptionResponse(BaseModel):
    text: str
    language: Optional[str] = None
    duration: Optional[float] = None

# -------------------------
# Startup
# -------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("Intelligent Document Q&A System starting up...")
    os.makedirs("./data/documents", exist_ok=True)
    os.makedirs("./data/feedback", exist_ok=True)
    os.makedirs("./data/models", exist_ok=True)

# -------------------------
# Routes
# -------------------------
@app.post("/upload-document", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    try:
        doc_id = str(uuid.uuid4())
        safe_name = file.filename.replace("/", "_").replace("\\", "_")
        file_path = f"./data/documents/{doc_id}_{safe_name}"

        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")
            
        with open(file_path, "wb") as f:
            f.write(content)

        stats = index_document(
            file_path=file_path,
            doc_id=doc_id,
            processor=document_processor,
            embedder=embedding_service,
            vector_db=vector_db,
            extra_metadata={"original_filename": safe_name},
        )

        return UploadResponse(
            document_id=doc_id,
            chunks_processed=int(stats.get("chunks_indexed", 0)),
            status=str(stats.get("status", "success")),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error uploading document")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        # Log the document filters
        logger.info(f"Processing query with filters: {request.document_filters}")
        
        result = qa_engine.process_query(
            query=request.query,
            session_id=request.session_id,
            document_filters=request.document_filters,
        )
        
        logger.info(f"Returning interaction_id: {result.get('interaction_id')}")
        
        return QueryResponse(
            answer=result["answer"],
            confidence=float(result["confidence"]),
            sources=result["sources"],
            processing_time=float(result["processing_time"]),
            session_id=request.session_id,
            interaction_id=result.get("interaction_id"),
        )

    except Exception as e:
        logger.exception("Error processing query")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    try:
        logger.info(f"Received feedback for interaction: {request.interaction_id}")
        
        success = vector_db.store_feedback(
            interaction_id=request.interaction_id,
            feedback_type=request.feedback_type,
            feedback_data=request.feedback_data,
            corrected_answer=request.corrected_answer,
        )
        
        if success:
            return {"status": "feedback_received", "interaction_id": request.interaction_id}
        else:
            raise HTTPException(status_code=500, detail="Failed to store feedback")
            
    except Exception as e:
        logger.exception("Error processing feedback")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe audio file using Whisper speech-to-text
    """
    if not WHISPER_AVAILABLE or whisper_model is None:
        raise HTTPException(
            status_code=501, 
            detail="Whisper is not available. Please install faster-whisper: pip install faster-whisper"
        )
    
    try:
        # Read audio file
        audio_bytes = await file.read()
        
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        logger.info(f"Received audio file: {file.filename}, size: {len(audio_bytes)} bytes")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        try:
            # Transcribe with Whisper
            # segments is a generator, info contains metadata
            segments, info = whisper_model.transcribe(
                tmp_path, 
                beam_size=5,
                language="en",  # You can set to None for auto-detection
                task="transcribe",
                vad_filter=True,  # Voice Activity Detection - filters out non-speech
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            # Collect transcription
            transcription = " ".join([segment.text for segment in segments])
            
            logger.info(f"Transcribed: '{transcription}' (language: {info.language}, duration: {info.duration:.2f}s)")
            
            return TranscriptionResponse(
                text=transcription.strip(),
                language=info.language,
                duration=info.duration
            )
            
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
        
    except Exception as e:
        logger.exception("Error transcribing audio")
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")

@app.get("/conversation-history/{session_id}")
async def get_conversation_history(session_id: str):
    try:
        history = memory_system.get_episodic_memory(session_id)
        return {"session_id": session_id, "history": history}
    except Exception as e:
        logger.exception("Error getting conversation history")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions")
async def get_all_sessions():
    try:
        collection = vector_db.collections["user_interactions"]
        results = collection.get(include=["metadatas"])
        
        sessions_dict = {}
        metadatas = results.get("metadatas", [])
        
        for metadata in metadatas:
            session_id = metadata.get("session_id")
            if session_id:
                if session_id not in sessions_dict:
                    sessions_dict[session_id] = {
                        "session_id": session_id,
                        "name": f"Session {session_id[:8]}",
                        "message_count": 0,
                        "last_updated": metadata.get("timestamp", 0),
                        "first_message": metadata.get("query", "")
                    }
                
                sessions_dict[session_id]["message_count"] += 1
                current_ts = metadata.get("timestamp", 0)
                if current_ts > sessions_dict[session_id]["last_updated"]:
                    sessions_dict[session_id]["last_updated"] = current_ts
        
        sessions_list = list(sessions_dict.values())
        sessions_list.sort(key=lambda x: x["last_updated"], reverse=True)
        
        for session in sessions_list:
            first_msg = session.get("first_message", "")
            if first_msg:
                short_msg = first_msg[:30] + "..." if len(first_msg) > 30 else first_msg
                session["name"] = f"{short_msg}"
        
        logger.info(f"Found {len(sessions_list)} sessions")
        return {"sessions": sessions_list}
        
    except Exception as e:
        logger.exception("Error getting sessions list")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/session/{session_id}/rename")
async def rename_session(session_id: str, request: RenameSessionRequest):
    try:
        new_name = request.name
        if not new_name:
            raise HTTPException(status_code=400, detail="Name is required")
        
        logger.info(f"Renamed session {session_id} to {new_name}")
        return {"status": "success", "session_id": session_id, "name": new_name}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error renaming session")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/learn-from-feedback")
async def trigger_learning(background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(learning_pipeline.process_feedback_batch)
        return {"status": "learning_triggered"}
    except Exception as e:
        logger.exception("Error triggering learning")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "document_qa_system"}

@app.get("/metrics")
async def get_system_metrics():
    try:
        doc_collection = vector_db.collections["document_chunks"]
        interaction_collection = vector_db.collections["user_interactions"]

        return {
            "chunks_indexed": doc_collection.count(),
            "total_interactions": interaction_collection.count(),
            "active_sessions": len(memory_system.sessions),
            "cache_size": len(qa_engine.qa_cache),
        }

    except Exception as e:
        logger.exception("Error getting metrics")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/documents/{doc_id}")
async def debug_document(doc_id: str):
    """Debug endpoint to check document chunks"""
    try:
        collection = vector_db.collections["document_chunks"]
        results = collection.get(
            where={"doc_id": doc_id},
            limit=5,
            include=["metadatas", "documents"]
        )
        
        chunks = []
        ids = results.get("ids", [])
        metadatas = results.get("metadatas", [])
        documents = results.get("documents", [])
        
        for i in range(len(ids)):
            chunks.append({
                "id": ids[i],
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "content_preview": documents[i][:200] + "..." if i < len(documents) else ""
            })
        
        return {
            "doc_id": doc_id,
            "chunk_count": len(ids),
            "chunks": chunks
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/debug/interaction/{interaction_id}")
async def debug_interaction(interaction_id: str):
    """Debug endpoint to check a specific interaction"""
    try:
        collection = vector_db.collections["user_interactions"]
        results = collection.get(
            ids=[interaction_id],
            include=["metadatas", "documents"]
        )
        
        return {
            "interaction_id": interaction_id,
            "found": len(results.get("ids", [])) > 0,
            "metadata": results.get("metadatas", [{}])[0] if results.get("metadatas") else None,
            "document": results.get("documents", [None])[0] if results.get("documents") else None
        }
    except Exception as e:
        return {"error": str(e)}

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)