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

def download_model_if_needed(model_name: str, root_dir: str):
    """
    Download and extract the InsightFace model package in chunks to keep memory usage low.
    """
    import zipfile
    import urllib.request
    
    # Expand user home if ~ is used
    expanded_root = os.path.expanduser(root_dir)
    models_dir = os.path.join(expanded_root, "models")
    model_dir = os.path.join(models_dir, model_name)
    
    # Key files to check if already present
    key_files = ["det_500m.onnx" if model_name == "buffalo_s" else "det_10g.onnx", 
                 "w600k_mobi.onnx" if model_name == "buffalo_s" else "w600k_r50.onnx"]
    
    files_exist = os.path.exists(model_dir) and all(
        os.path.exists(os.path.join(model_dir, f)) for f in key_files
    )
    
    if not files_exist:
        logger.info(f"Model '{model_name}' not found locally at '{model_dir}'. Downloading...")
        os.makedirs(models_dir, exist_ok=True)
        zip_path = os.path.join(models_dir, f"{model_name}.zip")
        url = f"https://github.com/deepinsight/insightface/releases/download/v0.7/{model_name}.zip"
        
        try:
            logger.info(f"Downloading model zip from {url}...")
            with urllib.request.urlopen(url) as response, open(zip_path, 'wb') as out_file:
                chunk_size = 1024 * 1024  # 1MB chunks
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    
            logger.info("Download complete. Extracting model files...")
            os.makedirs(model_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(model_dir)
                
            os.remove(zip_path)
            logger.info(f"✓ Model '{model_name}' successfully downloaded and extracted.")
        except Exception as e:
            logger.error(f"Failed to download/extract model '{model_name}': {str(e)}")
            if os.path.exists(zip_path):
                try:
                    os.remove(zip_path)
                except:
                    pass
    else:
        logger.info(f"✓ Model '{model_name}' is already cached in '{model_dir}'.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions
    logger.info("Initializing Face Recognition & Forensic Backend...")
    
    # 1. Initialize PostgreSQL Database & pgvector Tables
    init_db()
    
    # 2. Get model configuration from environment
    model_name = os.getenv("INSIGHTFACE_MODEL", "buffalo_s")
    model_root = os.getenv("INSIGHTFACE_ROOT", "./.insightface")
    
    # 3. Initialize InsightFace model
    try:
        import onnxruntime
        
        # Pre-download the model if not present using our memory-efficient chunked downloader
        download_model_if_needed(model_name, model_root)
        
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
            
        # Initialize FaceAnalysis with the dynamic model, only loading detection and recognition
        face_app = FaceAnalysis(name=model_name, root=model_root, allowed_modules=['detection', 'recognition'])
        face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        
        # Store in app state to be accessed as a singleton in routes
        app.state.face_analysis = face_app
        logger.info(f"✓ InsightFace model '{model_name}' successfully loaded in memory.")
        
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
