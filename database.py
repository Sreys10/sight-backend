import os
import logging
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger("face_recognition")

# Load database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    # Fallback to local default database URL for testing
    DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/postgres"

# For Supabase/Neon transaction pooler, sometimes we need to tweak parameters.
# However, for direct connections, standard engine is perfect.
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def init_db() -> None:
    """Initialize database extensions and create tables."""
    try:
        # Enable pgvector extension inside PostgreSQL database
        with engine.begin() as conn:
            logger.info("Enabling pgvector extension in database...")
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        
        # Auto-migrate: check if the face_embedding table already exists and has the old 512 dimension
        try:
            with engine.begin() as conn:
                table_exists = conn.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'face_embedding'
                    );
                """)).scalar()
                
                if table_exists:
                    dim = conn.execute(text("""
                        SELECT atttypmod FROM pg_attribute 
                        WHERE attrelid = 'face_embedding'::regclass AND attname = 'embedding';
                    """)).scalar()
                    
                    if dim is not None and dim != 128:
                        logger.warning(f"Database pgvector dimension mismatch (found {dim}, expected 128). Dropping tables for migration...")
                        conn.execute(text("DROP TABLE IF EXISTS face_embedding CASCADE;"))
                        conn.execute(text("DROP TABLE IF EXISTS person CASCADE;"))
                        logger.info("✓ Old 512-dimensional tables dropped successfully.")
        except Exception as check_err:
            logger.warning(f"Could not perform database dimension check: {str(check_err)}")
            
        # Create all tables defined in models.py
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        
        # Try to create HNSW index for optimized vector cosine similarity search
        with engine.begin() as conn:
            logger.info("Ensuring HNSW index exists on face_embedding...")
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS face_embedding_hnsw_idx 
                ON face_embedding USING hnsw (embedding vector_cosine_ops);
            """))
            
        logger.info("✓ Database initialized successfully.")
    except Exception as e:
        logger.error(f"✗ Database initialization failed: {str(e)}")
        # Don't re-raise if vector extension or hnsw index is already present 
        # but fails on non-superuser, standard tables are still created.
        try:
            Base.metadata.create_all(bind=engine)
            logger.info("✓ Created tables as fallback without index/extension setup.")
        except Exception as fallback_err:
            logger.critical(f"Critical DB error during fallback: {str(fallback_err)}")
            raise fallback_err

def get_db():
    """Dependency for getting DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
