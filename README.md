# Evidence.ai Face Recognition Service (FastAPI)

This directory contains the face recognition system integrated into the backend service using FastAPI, InsightFace (SCRFD & ArcFace Buffalo_L), and PostgreSQL pgvector.

---

## Architecture Overview

```
backend-service/
├── main.py            # FastAPI initialization & singleton model lifecycle
├── database.py        # SQLAlchemy & pgvector DB engine configuration
├── models.py          # Person and FaceEmbedding DB tables mapping
├── schemas.py         # Pydantic validation and response schemas
├── routes/
│   └── face_routes.py # API Endpoints (/register, /search)
└── services/
    ├── face_detection.py # SCRFD detection wrapper
    ├── face_embedding.py # ArcFace Buffalo_L feature extractor
    └── face_matching.py  # Cosine similarity search using pgvector
```

---

## Setup & Installation

### 1. Install Dependencies
Install all package dependencies via `pip`:
```bash
pip install -r requirements.txt
```

### 2. Database Setup (Neon / Supabase)
The database must be a PostgreSQL instance with the `pgvector` extension installed.
1. Connect to your database.
2. Enable the vector extension:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
3. The FastAPI service automatically executes this command and creates the tables (`person` and `face_embedding`) on startup. It also creates an optimized HNSW index:
   ```sql
   CREATE INDEX IF NOT EXISTS face_embedding_hnsw_idx 
   ON face_embedding USING hnsw (embedding vector_cosine_ops);
   ```

### 3. InsightFace Model Download
By default, the service uses the **Buffalo_L** model bundle.
- **Automatic**: On first start, the backend will automatically download the models to `~/.insightface/models/` from the official repository.
- **Manual**: Download the `buffalo_l.zip` archive and extract its contents (`det_10g.onnx`, `w600k_r50.onnx`, etc.) into `C:\Users\<Username>\.insightface\models\buffalo_l\` (Windows) or `~/.insightface/models/buffalo_l/` (Linux).

---

## Environment Variables

Ensure your environment variables are configured (e.g. in `.env` inside `backend-service/`):

```ini
# PostgreSQL connection string
DATABASE_URL=postgresql://postgres:<password>@<host>:5432/postgres

# API port for FastAPI
PORT=8000
```

---

## Running Locally

To start the FastAPI service on port 8000:

```bash
python main.py
```

---

## API Endpoints

### 1. Register Person
* **Endpoint**: `POST /api/faces/register`
* **Content-Type**: `multipart/form-data`
* **Form Parameters**:
  * `full_name` (string, required): Full name of the subject.
  * `gender` (string, optional): Male / Female / Other.
  * `age` (integer, optional): Subject age.
  * `case_number` (string, optional): Associated Case File ID.
  * `notes` (string, optional): Background info/notes.
  * `images` (List of files, required): Portrait photographs.

### 2. Search & Match Faces
* **Endpoint**: `POST /api/faces/search`
* **Content-Type**: `multipart/form-data`
* **Form Parameters**:
  * `image` (file, required): Image containing one or more faces.
  * `threshold` (float, optional, default: 0.60): Cosine similarity threshold (0.0 to 1.0).

---

## Testing

To run the automated endpoint validation tests:
```bash
python tests/test_api.py
```
