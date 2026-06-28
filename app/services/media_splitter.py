import io
import os
import zipfile
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.services.r2_service import r2_service
from app.utils.progress_logger import ProgressLogger
from app.utils.batch_registration import register_tasks_with_backend

logger = logging.getLogger("veralabel-splitter")

def fast_split(filename: str) -> str:
    """Synchronous djb2-based deterministic split"""
    h = 5381
    for char in filename:
        h = ((h * 33) & 0xFFFFFFFF) ^ ord(char)
    score = h % 100
    if score < 70:
        return 'train'
    if score < 85:
        return 'validation'
    return 'test'

def get_media_mime_type(ext: str) -> str:
    map = {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'webp': 'image/webp',
        'mp4': 'video/mp4',
        'mov': 'video/quicktime',
        'gif': 'image/gif',
        'pdf': 'application/pdf',
        'dcm': 'image/dicom',
        'dicom': 'image/dicom',
        'mp3': 'audio/mpeg',
        'wav': 'audio/wav',
    }
    return map.get(ext, 'application/octet-stream')

def infer_task_type(content_type: str) -> str:
    if content_type.startswith('image/'):
        return 'image'
    if content_type.startswith('audio/'):
        return 'audio'
    if content_type.startswith('video/'):
        return 'video'
    return 'text'

def process_media_zip(zip_path: str, project_id: str, dataset_id: str) -> dict:
    """
    Process zipped media dataset (images, videos, etc.), extracting, uploading to R2, and registering batches.
    """
    from threading import BoundedSemaphore
    progress_logger = ProgressLogger(project_id, dataset_id)
    progress_logger.log("Starting media ZIP processing (Python FastAPI)")
    
    try:
        # Open zip directly from local disk path to avoid loading zip bytes in memory
        zip_file = zipfile.ZipFile(zip_path)
        file_list = zip_file.namelist()
        
        # Filter valid files
        valid_files = []
        for name in file_list:
            if name.endswith('/') or name.startswith('__MACOSX') or '.DS_Store' in name:
                continue
            parts = name.split('.')
            if len(parts) > 1:
                ext = parts[-1].lower()
                mime = get_media_mime_type(ext)
                if mime != 'application/octet-stream':
                    valid_files.append((name, ext, mime))
                    
        total_files = len(valid_files)
        progress_logger.log(f"Archive structure parsed: found {total_files} media files")
        
        count = 0
        task_buffer = []
        failed_items = 0
        failed_batches = 0
        BATCH_SIZE = 100

        # Limit memory footprint: at most 15 files read into RAM at any one time (10 active threads + 5 queue buffer)
        semaphore = BoundedSemaphore(15)

        def flush_media_batch(is_last: bool = False):
            nonlocal failed_batches
            if not task_buffer:
                return
            buffer_to_flush = list(task_buffer)
            task_buffer.clear()
            
            progress_logger.log(f"[Media] Flushing batch", {"batchSize": len(buffer_to_flush), "isLast": is_last})
            
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
                progress_logger.log(f"[Media] Batch registered successfully", {"taskCount": len(buffer_to_flush)})

        if total_files == 0:
            progress_logger.checkpoint("flushed_final_batch", {"totalCount": 0, "failedItems": failed_items, "failedBatches": failed_batches})
            result = {
                "success": True,
                "processed": 0,
                "datasetId": dataset_id,
                "failedBatches": 0,
                "failedItems": failed_items,
                "message": "No valid media files to process"
            }
            progress_logger.complete(result)
            return result

        def upload_worker(r2_key, data, content_type, split):
            try:
                return r2_service.upload_file(r2_key, data, content_type, split)
            finally:
                semaphore.release()

        # Thread pool to speed up R2 uploads
        with ThreadPoolExecutor(max_workers=10) as upload_executor:
            futures = {}
            for name, ext, mime in valid_files:
                semaphore.acquire() # block if we already have 15 files loaded in memory
                
                try:
                    file_data = zip_file.read(name)
                except Exception as e:
                    semaphore.release()
                    logger.error(f"Failed to read file {name} from zip: {e}")
                    failed_items += 1
                    continue

                split_type = fast_split(name)
                r2_key = f"projects/{project_id}/{dataset_id}/{split_type}/{name}"
                
                # Submit R2 upload in background
                future = upload_executor.submit(
                    upload_worker, r2_key, file_data, mime, split_type
                )
                futures[future] = {
                    "name": name,
                    "r2_key": r2_key,
                    "split_type": split_type,
                    "content_type": mime,
                    "size": len(file_data)
                }

            # Gather upload tasks as they complete
            processed_count = 0
            for future in as_completed(futures):
                meta = futures[future]
                processed_count += 1
                is_last = (processed_count == total_files)
                
                try:
                    uploaded = future.result()
                    if uploaded:
                        task_type = infer_task_type(meta["content_type"])
                        task_buffer.append({
                            "taskId": str(uuid.uuid4()),
                            "taskType": task_type,
                            "r2_url": meta["r2_key"],
                            "split": meta["split_type"],
                            "fileName": meta["name"],
                            "name": meta["name"],
                            "fileSize": meta["size"],
                            "contentType": meta["content_type"]
                        })
                        count += 1
                    else:
                        failed_items += 1
                        progress_logger.error(f"R2 Upload failed for file: {meta['name']}")
                except Exception as e:
                    failed_items += 1
                    progress_logger.error(f"Exception during R2 upload of {meta['name']}", {"error": str(e)})

                # Flush if buffer is full, or if we have reached the end and buffer is not empty
                if len(task_buffer) >= BATCH_SIZE or (is_last and task_buffer):
                    flush_media_batch(is_last=is_last)

        # Ensure progress logger completes
        progress_logger.checkpoint("flushed_final_batch", {"totalCount": count, "failedItems": failed_items, "failedBatches": failed_batches})
        
        result = {
            "success": True,
            "processed": count,
            "datasetId": dataset_id,
            "failedBatches": failed_batches,
            "failedItems": failed_items,
            "message": "Media dataset processed and registered (Python FastAPI)"
        }
        progress_logger.complete(result)
        return result
        
    except Exception as e:
        logger.error(f"Media ZIP processing unhandled exception: {e}")
        progress_logger.error("Media ZIP processing failed", {"error": str(e)})
        progress_logger.complete({"success": False, "error": str(e)})
        raise e
