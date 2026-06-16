import urllib.request
import json
import time
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from app.core.config import settings

logger = logging.getLogger("veralabel-splitter")

# Background thread pool for non-blocking HTTP requests to the backend
executor = ThreadPoolExecutor(max_workers=3)

def send_request_background(url: str, headers: dict, data_bytes: bytes):
    """Sync target for background execution in ThreadPoolExecutor"""
    try:
        req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status not in (200, 201, 202):
                logger.warning(f"Progress update webhook returned status {response.status}")
    except Exception as e:
        logger.error(f"Failed to send progress update to backend: {e}")

class ProgressLogger:
    def __init__(self, project_id: str, dataset_id: str):
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.backend_api = settings.BACKEND_API
        self.backend_token = settings.BACKEND_TOKEN
        self.handshake_url = settings.HANDSHAKE_URL
        
        self.events = []
        self.start_time = time.time()
        
    def _queue_event(self, event: dict, flush_immediately: bool = False):
        self.events.append(event)
        if flush_immediately or len(self.events) >= 50:
            self.flush(is_final=False)

    def log(self, message: str, metadata: dict = None):
        if metadata is None:
            metadata = {}
        event = {
            "type": "progress",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "message": message,
            "metadata": metadata
        }
        logger.info(f"[Progress] {message} {metadata}")
        self._queue_event(event)

    def error(self, message: str, error_details: dict = None):
        if error_details is None:
            error_details = {}
        event = {
            "type": "error",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "message": message,
            "error": error_details,
            "severity": error_details.get("severity", "error")
        }
        logger.error(f"[Error] {message} {error_details}")
        self._queue_event(event, flush_immediately=True)

    def checkpoint(self, label: str, metrics: dict = None):
        if metrics is None:
            metrics = {}
        event = {
            "type": "checkpoint",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "label": label,
            "metrics": {
                **metrics,
                "elapsedMs": int((time.time() - self.start_time) * 1000)
            }
        }
        logger.info(f"[Checkpoint] {label} {event['metrics']}")
        self._queue_event(event, flush_immediately=True)

    def complete(self, summary: dict = None):
        if summary is None:
            summary = {}
        event = {
            "type": "complete",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "summary": {
                **summary,
                "totalDurationMs": int((time.time() - self.start_time) * 1000)
            }
        }
        logger.info(f"[Complete] {event['summary']}")
        self._queue_event(event, flush_immediately=True)
        self.flush(is_final=True)

    def flush(self, is_final: bool = False):
        if not self.events:
            return
            
        events_to_send = list(self.events)
        self.events.clear()

        if not self.backend_api:
            logger.info("Backend API not configured, logging events locally only")
            return

        # Format endpoint (must target /tasks/progress)
        base_api = self.backend_api.rstrip('/')
        # Strip tasks endpoints if present to construct base tasks path
        import re
        base_api = re.sub(r'(/createTasks|/register-task|/register|/progress)$', '', base_api, flags=re.IGNORECASE)
        if not base_api.endswith('/tasks'):
            base_api = f"{base_api}/tasks"
        url = f"{base_api}/progress"

        payload = {
            "projectId": self.project_id,
            "datasetId": self.dataset_id,
            "events": events_to_send,
            "isFinal": is_final,
            "sentAt": datetime.utcnow().isoformat() + "Z"
        }

        headers = {
            "Content-Type": "application/json"
        }
        if self.backend_token:
            headers["Authorization"] = f"Bearer {self.backend_token}"
        if self.handshake_url:
            headers["handshake-url"] = self.handshake_url

        try:
            data_bytes = json.dumps(payload).encode("utf-8")
            # Submit to background executor to avoid blocking the caller
            executor.submit(send_request_background, url, headers, data_bytes)
        except Exception as e:
            logger.error(f"Failed to submit progress flush job to executor: {e}")
