import urllib.request
import json
import time
import logging
import re
from app.core.config import settings

logger = logging.getLogger("veralabel-splitter")

def register_tasks_with_backend(payload: dict) -> dict:
    """
    Register a batch of tasks with the backend tasks/register endpoint.
    Performs retries with exponential backoff on retryable HTTP errors.
    """
    backend_api = settings.BACKEND_API
    backend_token = settings.BACKEND_TOKEN
    handshake_url = settings.HANDSHAKE_URL
    
    if not backend_api:
        logger.warning("BACKEND_API not configured. Skipping backend task registration.")
        return {"ok": False, "reason": "missing_backend_api"}

    # Resolve register endpoint defensively
    base_api = backend_api.rstrip('/')
    base_api = re.sub(r'(/createTasks|/register-task|/register|/progress)$', '', base_api, flags=re.IGNORECASE)
    
    if '/api/v1' not in base_api:
        base_api = f"{base_api}/api/v1"
        
    if not base_api.endswith('/tasks'):
        base_api = f"{base_api}/tasks"
    endpoint = f"{base_api}/register"

    headers = {
        "Content-Type": "application/json"
    }
    if backend_token:
        headers["Authorization"] = f"Bearer {backend_token}"
    if handshake_url:
        headers["handshake-url"] = handshake_url

    max_attempts = 3
    task_count = len(payload.get("tasks", []))
    
    for attempt in range(1, max_attempts + 1):
        try:
            data_bytes = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(endpoint, data=data_bytes, headers=headers, method="POST")
            
            with urllib.request.urlopen(req, timeout=10) as response:
                # Accept both 201 (Created) and 202 (Accepted)
                if response.status in (200, 201, 202):
                    return {"ok": True, "status": response.status}
                
                body_text = response.read().decode("utf-8", errors="ignore")
                logger.warning(
                    f"Backend registration returned status {response.status} for batch ({task_count} tasks), "
                    f"attempt {attempt}/{max_attempts}: {body_text}"
                )
                
        except Exception as e:
            logger.error(
                f"Failed to register batch ({task_count} tasks), attempt {attempt}/{max_attempts}: {e}"
            )
            
        if attempt < max_attempts:
            backoff_ms = 300 * (2 ** (attempt - 1))
            time.sleep(backoff_ms / 1000.0)

    return {"ok": False, "reason": "registration_failed_after_retries"}
