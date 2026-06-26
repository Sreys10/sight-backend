import os
import sys
import psutil

def print_memory(label):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    print(f"[{label}] Memory usage: {mem_mb:.2f} MB")

print_memory("Step 1: Process Startup")

import fastapi
from fastapi import FastAPI
print_memory("Step 2: After importing FastAPI")

import sqlalchemy
print_memory("Step 3: After importing SQLAlchemy")

import onnxruntime
print_memory("Step 4: After importing ONNXRuntime")

from insightface.app import FaceAnalysis
print_memory("Step 5: After importing InsightFace")

from database import init_db
print_memory("Step 6: After importing database modules")

from routes.face_routes import router as face_router
print_memory("Step 7: After importing face_routes")

from routes.forensic_routes import router as forensic_router
print_memory("Step 8: After importing forensic_routes")

# Initialize database
init_db()
print_memory("Step 9: After executing init_db()")

# Initialize model
model_name = "buffalo_s"
model_root = "./.insightface"

face_app = FaceAnalysis(name=model_name, root=model_root, allowed_modules=['detection', 'recognition'])
print_memory("Step 10: After instantiating FaceAnalysis")

face_app.prepare(ctx_id=-1, det_size=(640, 640))
print_memory("Step 11: After preparing FaceAnalysis (ONNX models loaded)")
