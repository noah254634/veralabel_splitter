# VeraLabel Dataset Splitter & Assembler Service

VeraLabel Splitter is a high-performance **FastAPI** service engineered to offload computationally heavy data operations from the main backend. It manages large-scale dataset ingestion, validation, splitting (by audio chunks, text lines, or RLHF nodes), and subsequent compilation (consensus adjudication, zipping, and secure upload to Cloudflare R2).

---

## Key Capabilities

* **Asynchronous Processing:** Long-running dataset splitting operations are executed via background tasks, preventing HTTP timeouts.
* **Format-Specific Splitters:**
  * **Text:** Lines/paragraphs extraction and validation.
  * **Audio:** Chunking audio files by duration and formatting.
  * **RLHF:** Multi-layer visual and conversational task distribution.
* **Assembler Service:** Pulls verified contributor contributions, executes consensus mechanisms, builds ZIP packages, and provisions secure retrieval links.
* **Cloudflare R2 Integration:** Frictionless read/write pipelines via tokenized presigned URLs.

---

## Technology Stack

* **Core:** Python 3.11+, FastAPI
* **AWS S3/R2 Client:** Boto3
* **ASGI Server:** Uvicorn
* **Database Driver:** Motor (MongoDB async)

---

## Configuration (`.env`)

Create a `.env` file in the project root with the following parameters:

```env
# Security secret for verification checks
INTERNAL_SECRET=your_jwt_signing_or_hmac_secret

# Cloudflare R2 Credentials
R2_ACCESS_KEY=your_r2_access_key
R2_SECRET_KEY=your_r2_secret_key
R2_ENDPOINT=https://your_cloudflare_account_id.r2.cloudflarestorage.com
R2_BUCKET_NAME=veralabel-bucket

# Main Backend Integration
BACKEND_API=https://api.veralabel.com/api/v1
BACKEND_TOKEN=backend_transient_auth_token
HANDSHAKE_URL=https://api.veralabel.com/api/v1/handshake

# Chunk Configuration
AUDIO_CHUNK_DURATION=30.0
```

---

## API Documentation

### 1. Health Status Check
* **Endpoint:** `GET /api/v1/health`
* **Description:** Inspects system uptime, PID status, and validates Cloudflare R2 connectivity.
* **Responses:**
  * `200 OK` — All integrations are healthy and connected.
  * `503 Service Unavailable` — Degradation detected (e.g., misconfigured R2 client).

### 2. Trigger Dataset Split
* **Endpoint:** `POST /api/v1/datasets`
* **Description:** Initiates splitting on a recently uploaded dataset.
* **Payload:**
  ```json
  {
    "dataset_id": "dataset_id_uuid",
    "project_id": "project_id_uuid",
    "r2_key": "raw/dataset.zip",
    "data_type": "audio",
    "download_url": "https://presigned-link.com"
  }
  ```
* **Response:** `202 Accepted`

### 3. Assemble Dataset
* **Endpoint:** `POST /api/v1/datasets/assemble`
* **Description:** Adjudicates contributor submissions and bundles the compiled assets.
* **Response:** `200 OK` with R2 storage key and size parameters.

---

## Run Locally

1. **Create and Activate Virtual Environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the Development Server:**
   ```bash
   uvicorn main:app --reload --port 8002
   ```
   * Interactive OpenAPI docs: `http://localhost:8002/docs`
   * Health status endpoint: `http://localhost:8002/api/v1/health`

---

## Container Deployment

### Local Compose Run
To launch the service locally in a container alongside its environment variables:
```bash
docker-compose up --build
```

### Production Build
```bash
docker build -t veralabel-splitter .
docker run -p 8002:8002 --env-file .env veralabel-splitter
```
