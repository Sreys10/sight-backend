import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from routes.face_routes import router as face_router
from routes.forensic_routes import router as forensic_router
from database import init_db

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("face_recognition")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions
    logger.info("Initializing Face Recognition & Forensic Backend...")
    
    # 1. Initialize PostgreSQL Database & pgvector Tables
    init_db()
    
    # 2. Initialize InsightFace buffalo_l model once
    try:
        import onnxruntime
        from insightface.app import FaceAnalysis
        
        # Check GPU availability
        available_providers = onnxruntime.get_available_providers()
        logger.info(f"Available ONNXRuntime providers: {available_providers}")
        
        if "CUDAExecutionProvider" in available_providers:
            ctx_id = 0
            logger.info("✓ CUDA GPU detected! InsightFace will run on GPU.")
        else:
            ctx_id = -1
            logger.info("CUDA GPU not detected. InsightFace will run on CPU.")
            
        # Initialize FaceAnalysis with the buffalo_l model bundle
        face_app = FaceAnalysis(name="buffalo_l", root="~/.insightface")
        face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        
        # Store in app state to be accessed as a singleton in routes
        app.state.face_analysis = face_app
        logger.info("✓ InsightFace model successfully loaded in memory.")
        
    except Exception as e:
        logger.critical(f"✗ Failed to load InsightFace model: {str(e)}")
        app.state.face_analysis = None
        
    yield
    
    # Shutdown actions
    logger.info("Shutting down Face Recognition & Forensic Backend...")

# Create FastAPI app
app = FastAPI(
    title="Evidence.ai Core API Service",
    description="FastAPI service mapping image tampering detection, weapon detection, metadata forensics, and pgvector face recognition.",
    version="1.0.0",
    lifespan=lifespan
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the storage directory statically so uploaded face images are viewable
os.makedirs("storage", exist_ok=True)
app.mount("/static", StaticFiles(directory="storage"), name="static")

# Include routes
app.include_router(face_router)
app.include_router(forensic_router)

@app.get("/health")
def health_check():
    """Health check endpoint."""
    model_loaded = False
    if hasattr(app.state, "face_analysis") and app.state.face_analysis is not None:
        model_loaded = True
        
    return {
        "status": "healthy" if model_loaded else "degraded",
        "service": "face_recognition",
        "model_loaded": model_loaded,
        "onnx_providers": getattr(sys.modules.get("onnxruntime"), "get_available_providers", lambda: [])()
    }

if __name__ == "__main__":
    import uvicorn
    # Allow port to be configurable via PORT environment variable, defaults to 8000
    # In production, Render/Railway injects PORT
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting server on port {port}...")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
