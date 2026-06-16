from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from app.services.r2_service import r2_service
from app.core.config import settings
import time
import os

router = APIRouter()

# Record startup time
START_TIME = time.time()

@router.get("", response_model=dict)
def check_health():
    """
    Check the health of the FastAPI service and its integrations.
    Returns 200 OK if the system is healthy and R2 integration is configured,
    or 503 SERVICE_UNAVAILABLE if critical integrations are misconfigured.
    """
    uptime = time.time() - START_TIME
    
    # Check if R2 client is initialized
    r2_healthy = r2_service.s3_client is not None
    
    # Check environment variable presence
    env_status = {
        "INTERNAL_SECRET": bool(settings.INTERNAL_SECRET),
        "R2_ACCESS_KEY": bool(settings.R2_ACCESS_KEY),
        "R2_SECRET_KEY": bool(settings.R2_SECRET_KEY),
        "R2_ENDPOINT": bool(settings.R2_ENDPOINT),
        "R2_BUCKET_NAME": bool(settings.R2_BUCKET_NAME),
        "BACKEND_API": bool(settings.BACKEND_API)
    }
    
    all_critical_env_set = all([
        settings.R2_ACCESS_KEY,
        settings.R2_SECRET_KEY,
        settings.R2_ENDPOINT,
        settings.R2_BUCKET_NAME
    ])

    health_status = "healthy" if (r2_healthy and all_critical_env_set) else "degraded"
    
    response_payload = {
        "status": health_status,
        "uptime_seconds": round(uptime, 2),
        "pid": os.getpid(),
        "integrations": {
            "r2_s3_client": "connected" if r2_healthy else "disconnected",
            "environment_check": env_status
        }
    }
    
    if health_status == "healthy":
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=response_payload
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=response_payload
        )
