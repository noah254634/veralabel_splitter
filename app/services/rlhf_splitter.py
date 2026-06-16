import uuid
import logging
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.services.r2_service import r2_service
from app.utils.progress_logger import ProgressLogger
from app.utils.batch_registration import register_tasks_with_backend

logger = logging.getLogger("veralabel-splitter")

def fast_split(prompt: str) -> str:
    """Synchronous djb2-based deterministic split based on prompt"""
    h = 5381
    for char in prompt:
        h = ((h * 33) & 0xFFFFFFFF) ^ ord(char)
    score = h % 100
    if score < 70:
        return 'train'
    if score < 85:
        return 'validation'
    return 'test'

def parse_json_or_jsonl(content_str: str) -> list:
    """
    Parses content_str which could be a JSON array, a single JSON object, or JSON Lines (JSONL).
    Returns a list of parsed dictionaries/objects.
    """
    content_str = content_str.strip()
    if not content_str:
        return []
        

    try:
        data = json.loads(content_str)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass
        

    lines = content_str.split('\n')
    parsed_items = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            parsed_items.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse line as JSON: {line[:100]}")
            
    return parsed_items

def process_rlhf_file(file_bytes: bytes, project_id: str, dataset_id: str) -> dict:
    """
    Process RLHF prompt-responses dataset (JSON/JSONL), validating, uploading normalized JSON records to R2, and registering batches.
    """
    progress_logger = ProgressLogger(project_id, dataset_id)
    progress_logger.log("Starting RLHF file processing (Python FastAPI)")
    
    try:
        content = file_bytes.decode('utf-8', errors='ignore')
        entries = parse_json_or_jsonl(content)
        
        # Parse and validate entries
        valid_entries = []
        malformed_lines = 0
        
        for entry in entries:
            if not isinstance(entry, dict):
                malformed_lines += 1
                continue
                
            prompt = entry.get("prompt", "")
            if not isinstance(prompt, str) or not prompt.strip():
                malformed_lines += 1
                continue
                
            responses = entry.get("responses")
            response = entry.get("response")
            
            final_responses = []
            if isinstance(responses, list):
                final_responses = responses
            elif isinstance(response, str):
                final_responses = [response]
                
            if len(final_responses) == 0:
                malformed_lines += 1
                continue
                
            # Normalize
            normalized_entry = {
                "prompt": prompt,
                "responses": final_responses,
                "metadata": {
                    "datasetId": dataset_id,
                    "projectId": project_id,
                    "source_language": "en-KE",
                    "processed_at": datetime.utcnow().isoformat() + "Z"
                }
            }
            valid_entries.append((prompt, normalized_entry))
            
        total_items = len(valid_entries)
        progress_logger.log(f"RLHF file parsed: found {total_items} valid records, skipped {malformed_lines} malformed items")
        
        count = 0
        task_buffer = []
        failed_entries = 0
        failed_batches = 0
        BATCH_SIZE = 100

        def flush_rlhf_batch(is_last: bool = False):
            nonlocal failed_batches
            if not task_buffer:
                return
            buffer_to_flush = list(task_buffer)
            task_buffer.clear()
            
            progress_logger.log(f"[RLHF] Flushing batch", {"batchSize": len(buffer_to_flush), "isLast": is_last})
            
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
                progress_logger.log(f"[RLHF] Batch registered successfully", {"taskCount": len(buffer_to_flush)})

        if total_items == 0:
            progress_logger.checkpoint("flushed_final_batch", {"totalCount": 0, "failedItems": failed_entries, "failedBatches": failed_batches})
            result = {
                "success": True,
                "processed": 0,
                "datasetId": dataset_id,
                "failedBatches": 0,
                "failedItems": failed_entries,
                "message": "No valid RLHF entries to process"
            }
            progress_logger.complete(result)
            return result

        # Thread pool to speed up R2 uploads
        with ThreadPoolExecutor(max_workers=10) as upload_executor:
            futures = {}
            for prompt, entry in valid_entries:
                split_type = fast_split(prompt)
                task_id = str(uuid.uuid4())
                r2_key = f"projects/{project_id}/{dataset_id}/{split_type}/{task_id}.json"
                entry_bytes = json.dumps(entry).encode('utf-8')
                
                # Submit R2 upload in background
                future = upload_executor.submit(
                    r2_service.upload_file, r2_key, entry_bytes, "application/json", split_type
                )
                futures[future] = {
                    "task_id": task_id,
                    "r2_key": r2_key,
                    "split_type": split_type,
                    "preview": prompt[:100],
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
                            "contentType": "text",
                            "taskType": "text",
                            "r2_url": meta["r2_key"],
                            "split": meta["split_type"],
                            "contentPreview": meta["preview"],
                        })
                        count += 1
                        if count % 500 == 0:
                            progress_logger.log(f"Processed {count} RLHF items so far")
                    else:
                        failed_entries += 1
                        progress_logger.error(f"R2 Upload failed for RLHF key: {meta['r2_key']}")
                except Exception as e:
                    failed_entries += 1
                    progress_logger.error(f"Exception during R2 upload of {meta['r2_key']}", {"error": str(e)})

                # Flush if buffer is full, or if we have reached the end and buffer is not empty
                if len(task_buffer) >= BATCH_SIZE or (is_last and task_buffer):
                    flush_rlhf_batch(is_last=is_last)

        # Ensure progress logger completes
        progress_logger.checkpoint("flushed_final_batch", {"totalCount": count, "failedItems": failed_entries, "failedBatches": failed_batches})
        
        result = {
            "success": True,
            "processed": count,
            "datasetId": dataset_id,
            "failedBatches": failed_batches,
            "failedItems": failed_entries,
            "message": "RLHF dataset processed and registered (Python FastAPI)"
        }
        progress_logger.complete(result)
        return result
        
    except Exception as e:
        logger.error(f"RLHF file processing unhandled exception: {e}")
        progress_logger.error("RLHF file processing failed", {"error": str(e)})
        progress_logger.complete({"success": False, "error": str(e)})
        raise e
