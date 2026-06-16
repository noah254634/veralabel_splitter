from fastapi import Header, HTTPException, status
from app.core.config import settings

async def verify_signature(x_vera_signature: str = Header(None, alias="X-Vera-Signature")):
    if not settings.INTERNAL_SECRET:
        # If no secret is configured, deny access by default for safety
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Worker secret signature is not configured"
        )
    
    if not x_vera_signature or x_vera_signature != settings.INTERNAL_SECRET:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized signature request"
        )
    return x_vera_signature
