#!/usr/bin/env python3
"""
Script to index database images to FAISS
Run this script to populate FAISS index with face embeddings from database images
"""
import os
import sys
from face_matcher import FaceMatcher

def main():
    # Get configuration from environment or use defaults
    database_path = os.getenv('DATABASE_PATH', 'database/')
    model_name = os.getenv('MODEL_NAME', 'ArcFace')
    detector_backend = os.getenv('DETECTOR_BACKEND', 'retinaface')
    faiss_index_path = os.getenv('FAISS_INDEX_PATH', 'faiss_index.bin')
    metadata_path = os.getenv('METADATA_PATH', 'faiss_metadata.json')
    
    print("=" * 60)
    print("Face Database Indexing Script (FAISS)")
    print("=" * 60)
    print(f"Database Path: {database_path}")
    print(f"Model: {model_name}")
    print(f"Detector: {detector_backend}")
    print(f"FAISS Index: {faiss_index_path}")
    print(f"Metadata: {metadata_path}")
    print("=" * 60)
    print()
    
    # Initialize face matcher
    try:
        face_matcher = FaceMatcher(
            default_database_path=database_path,
            faiss_index_path=faiss_index_path,
            metadata_path=metadata_path
        )
    except Exception as e:
        print(f"Error initializing FaceMatcher: {str(e)}")
        sys.exit(1)
    
    # Index database images
    print("Starting indexing process...")
    print()
    
    result = face_matcher.index_database_images(
        database_path=database_path,
        model_name=model_name,
        detector_backend=detector_backend
    )
    
    print()
    print("=" * 60)
    print("Indexing Results")
    print("=" * 60)
    print(f"Success: {result['success']}")
    print(f"Total images: {result.get('total', 0)}")
    print(f"Indexed: {result.get('indexed', 0)}")
    print(f"Failed: {result.get('failed', 0)}")
    
    if result.get('errors'):
        print("\nErrors:")
        for error in result['errors']:
            print(f"  - {error}")
    
    print("=" * 60)
    
    if result['success'] and result.get('indexed', 0) > 0:
        print("\n✅ Database indexing completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Database indexing completed with errors.")
        sys.exit(1)

if __name__ == '__main__':
    main()



