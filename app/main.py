import logging
# pyrefly: ignore [missing-import]
from fastapi import FastAPI
from app.api.routes import datasets, health
from app.core.config import settings

# Configure logging format and level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("veralabel-splitter")

app = FastAPI(
    title="VeraLabel Dataset Splitter",
    description="High-performance FastAPI service for splitting large datasets and uploading elements to Cloudflare R2",
    version="1.0.0"
)

# Mount datasets splitter router
app.include_router(
    datasets.router,
    prefix="/api/v1/datasets",
    tags=["datasets"]
)

# Mount health check router
app.include_router(
    health.router,
    prefix="/api/v1/health",
    tags=["health"]
)

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "VeraLabel Splitter Service",
        "api_prefix": "/api/v1"
    }
