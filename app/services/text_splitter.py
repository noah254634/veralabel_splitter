import uuid
import logging
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.services.r2_service import r2_service
from app.utils.progress_logger import ProgressLogger
from app.utils.batch_registration import register_tasks_with_backend

logger = logging.getLogger("veralabel-splitter")

def fast_split(line: str) -> str:
    """Synchronous djb2-based deterministic split"""
    h = 5381
    for char in line:
        h = ((h * 33) & 0xFFFFFFFF) ^ ord(char)
    score = h % 100
    if score < 70:
        return 'train'
    if score < 85:
        return 'validation'
    return 'test'

def process_text_file(file_bytes: bytes, project_id: str, dataset_id: str) -> dict:
    """
    Process plain text or JSONL dataset, splitting by line, uploading to R2, and registering batches.
    """
    progress_logger = ProgressLogger(project_id, dataset_id)
    progress_logger.log("Starting text file processing (Python FastAPI)")
    
    try:
        content = file_bytes.decode('utf-8', errors='ignore')
        lines = content.split('\n')
        
        # Filter valid non-empty lines
        valid_lines = [l.strip() for l in lines if l.strip()]
        total_items = len(valid_lines)
        progress_logger.log(f"Text file parsed: found {total_items} lines")
        
        count = 0
        task_buffer = []
        failed_items = 0
        failed_batches = 0
        BATCH_SIZE = 200

        def flush_text_batch(is_last: bool = False):
            nonlocal failed_batches
            if not task_buffer:
                return
            buffer_to_flush = list(task_buffer)
            task_buffer.clear()
            
            progress_logger.log(f"[Text] Flushing batch", {"batchSize": len(buffer_to_flush), "isLast": is_last})
            
            result = register_tasks_with_backend({
                "datasetId": dataset_id,
                "projectId": project_id,
                "tasks": buffer_to_flush,
                "isLastBatch": is_last
            })
            
            if not result.get("ok"):
                failed_batches += 1
                progress_logger.error("Failed to register batch with backend", {
                    "batchSize": len(buffer_to_flush),
                    "isLastBatch": is_last,
                    "result": result
                })
            else:
                progress_logger.log(f"[Text] Batch registered successfully", {"taskCount": len(buffer_to_flush)})

        if total_items == 0:
            progress_logger.checkpoint("flushed_final_batch", {"totalCount": 0, "failedItems": failed_items, "failedBatches": failed_batches})
            result = {
                "success": True,
                "processed": 0,
                "datasetId": dataset_id,
                "failedBatches": 0,
                "failedItems": failed_items,
                "message": "No valid text lines to process"
            }
            progress_logger.complete(result)
            return result

        # Thread pool to speed up R2 uploads
        with ThreadPoolExecutor(max_workers=10) as upload_executor:
            futures = {}
            for line in valid_lines:

                is_json = False
                content_type = "text/plain"
                ext = "txt"
                if line.startswith('{'):
                    try:
                        json.loads(line)
                        is_json = True
                        content_type = "application/json"
                        ext = "json"
                    except json.JSONDecodeError:
                        pass
                
                split_type = fast_split(line)
                task_id = str(uuid.uuid4())
                r2_key = f"projects/{project_id}/{dataset_id}/{split_type}/{task_id}.{ext}"
                
                # Submit R2 upload in background
                future = upload_executor.submit(
                    r2_service.upload_file, r2_key, line.encode('utf-8'), content_type, split_type
                )
                futures[future] = {
                    "task_id": task_id,
                    "r2_key": r2_key,
                    "split_type": split_type,
                    "content_type": content_type,
                    "preview": line[:100],
                }

            # Gather upload tasks as they complete
            processed_count = 0
            for future in as_completed(futures):
                meta = futures[future]
                processed_count += 1
                is_last = (processed_count == total_items)
                
                try:
                    uploaded = future.result()
                    if uploaded:
                        task_buffer.append({
                            "taskId": meta["task_id"],
                            "taskType": "text",
                            "r2_url": meta["r2_key"],
                            "split": meta["split_type"],
                            "contentType": meta["content_type"],
                            "contentPreview": meta["preview"],
                        })
                        count += 1
                        if count % 500 == 0:
                            progress_logger.log(f"Processed {count} items so far")
                    else:
                        failed_items += 1
                        progress_logger.error(f"R2 Upload failed for text key: {meta['r2_key']}")
                except Exception as e:
                    failed_items += 1
                    progress_logger.error(f"Exception during R2 upload of {meta['r2_key']}", {"error": str(e)})

                # Flush if buffer is full, or if we have reached the end and buffer is not empty
                if len(task_buffer) >= BATCH_SIZE or (is_last and task_buffer):
                    flush_text_batch(is_last=is_last)

        # Ensure progress logger completes
        progress_logger.checkpoint("flushed_final_batch", {"totalCount": count, "failedItems": failed_items, "failedBatches": failed_batches})
        
        result = {
            "success": True,
            "processed": count,
            "datasetId": dataset_id,
            "failedBatches": failed_batches,
            "failedItems": failed_items,
            "message": "Text dataset processed and registered (Python FastAPI)"
        }
        progress_logger.complete(result)
        return result
        
    except Exception as e:
        logger.error(f"Text file processing unhandled exception: {e}")
        progress_logger.error("Text file processing failed", {"error": str(e)})
        progress_logger.complete({"success": False, "error": str(e)})
        raise e
