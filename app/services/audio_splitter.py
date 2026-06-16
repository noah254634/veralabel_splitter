import io
import os
import wave
import zipfile
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.services.r2_service import r2_service
from app.utils.progress_logger import ProgressLogger
from app.utils.batch_registration import register_tasks_with_backend
from app.core.config import settings

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

def get_audio_mime_type(ext: str) -> str:
    mime_map = {
        'mp3': 'audio/mpeg',
        'wav': 'audio/wav',
        'flac': 'audio/flac',
        'ogg': 'audio/ogg',
        'm4a': 'audio/m4a',
        'aac': 'audio/aac',
    }
    return mime_map.get(ext, 'audio/mpeg')

def slice_wav_file(file_data: bytes, chunk_duration_sec: float) -> list[tuple[bytes, float]]:
    """
    Slices WAV file bytes into multiple chunks, each up to chunk_duration_sec.
    Returns a list of tuples: (chunk_bytes, duration_seconds)
    """
    if chunk_duration_sec <= 0:
        return []
        
    try:
        wav_in = wave.open(io.BytesIO(file_data), 'rb')
    except Exception as e:
        logger.warning(f"Could not parse file as WAV: {e}")
        return []

    try:
        params = wav_in.getparams()
        nchannels, sampwidth, framerate, nframes, comptype, compname = params
        
        # Calculate total duration in seconds
        total_duration = nframes / float(framerate)
        if total_duration <= chunk_duration_sec:
            # No need to slice
            return [(file_data, total_duration)]
            
        chunks = []
        frames_per_chunk = int(framerate * chunk_duration_sec)
        
        frames_read = 0
        while frames_read < nframes:
            chunk_frames = wav_in.readframes(frames_per_chunk)
            if not chunk_frames:
                break
                
            actual_frames = len(chunk_frames) // (nchannels * sampwidth)
            if actual_frames == 0:
                break
                
            chunk_duration = actual_frames / float(framerate)
            
            chunk_io = io.BytesIO()
            wav_out = wave.open(chunk_io, 'wb')
            wav_out.setparams((nchannels, sampwidth, framerate, actual_frames, comptype, compname))
            wav_out.writeframes(chunk_frames)
            wav_out.close()
            
            chunks.append((chunk_io.getvalue(), chunk_duration))
            frames_read += actual_frames
            
        return chunks
    except Exception as e:
        logger.error(f"Error during WAV slicing: {e}. Returning original.")
        return []
    finally:
        wav_in.close()

def process_audio_zip(zip_bytes: bytes, project_id: str, dataset_id: str) -> dict:
    """
    Process zipped audio dataset, extracting, uploading to R2, and registering batches.
    """
    progress_logger = ProgressLogger(project_id, dataset_id)
    progress_logger.log("Starting audio ZIP processing (Python FastAPI)")
    
    try:
        # Load zip in memory
        zip_file = zipfile.ZipFile(io.BytesIO(zip_bytes))
        file_list = zip_file.namelist()
        
        # Filter valid files
        valid_files = []
        for name in file_list:
            if name.endswith('/') or name.startswith('__MACOSX') or '.DS_Store' in name:
                continue
            parts = name.split('.')
            if len(parts) > 1:
                ext = parts[-1].lower()
                if ext in ('mp3', 'wav', 'flac', 'ogg', 'm4a', 'aac'):
                    valid_files.append((name, ext))
                    
        total_files = len(valid_files)
        progress_logger.log(f"Archive structure parsed: found {total_files} audio files")
        
        count = 0
        task_buffer = []
        failed_items = 0
        failed_batches = 0
        BATCH_SIZE = 100

        def flush_audio_batch(is_last: bool = False):
            nonlocal failed_batches
            if not task_buffer:
                return
            buffer_to_flush = list(task_buffer)
            task_buffer.clear()
            
            progress_logger.log(f"[Audio] Flushing batch", {"batchSize": len(buffer_to_flush), "isLast": is_last})
            
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
                progress_logger.log(f"[Audio] Batch registered successfully", {"taskCount": len(buffer_to_flush)})

        # Prepare upload items (either sliced wav or original files)
        upload_items = []
        for name, ext in valid_files:
            try:
                file_data = zip_file.read(name)
            except Exception as e:
                logger.error(f"Failed to read file {name} from zip: {e}")
                failed_items += 1
                continue

            sliced_chunks = []
            if ext == 'wav' and settings.AUDIO_CHUNK_DURATION > 0:
                sliced_chunks = slice_wav_file(file_data, settings.AUDIO_CHUNK_DURATION)

            if sliced_chunks:
                # Keep subdirectories if present, but swap backslashes for forward slashes
                dir_name, file_base = os.path.split(name)
                base_name, file_ext = os.path.splitext(file_base)
                for idx, (chunk_bytes, chunk_duration) in enumerate(sliced_chunks):
                    chunk_name = os.path.join(dir_name, f"{base_name}_chunk_{idx}{file_ext}").replace("\\", "/")
                    upload_items.append({
                        "name": chunk_name,
                        "original_name": name,
                        "file_data": chunk_bytes,
                        "ext": ext,
                        "is_sliced": True,
                        "chunk_index": idx,
                        "duration": chunk_duration
                    })
            else:
                upload_items.append({
                    "name": name,
                    "original_name": name,
                    "file_data": file_data,
                    "ext": ext,
                    "is_sliced": False,
                    "chunk_index": 0,
                    "duration": 0.0
                })

        total_uploads = len(upload_items)
        progress_logger.log(f"Prepared upload queue: {total_uploads} total items (after potential slicing)")

        if total_uploads == 0:
            progress_logger.checkpoint("flushed_final_batch", {"totalCount": 0, "failedItems": failed_items, "failedBatches": failed_batches})
            result = {
                "success": True,
                "processed": 0,
                "datasetId": dataset_id,
                "failedBatches": 0,
                "failedItems": failed_items,
                "message": "No valid audio files to process"
            }
            progress_logger.complete(result)
            return result

        # Thread pool for R2 uploads
        with ThreadPoolExecutor(max_workers=10) as upload_executor:
            futures = {}
            for item in upload_items:
                split_type = fast_split(item["original_name"]) # keeps chunks of same file in same split
                content_type = get_audio_mime_type(item["ext"])
                r2_key = f"projects/{project_id}/{dataset_id}/{split_type}/{item['name']}"
                
                # Submit R2 upload in background
                future = upload_executor.submit(
                    r2_service.upload_file, r2_key, item["file_data"], content_type, split_type
                )
                futures[future] = {
                    "name": item["name"],
                    "r2_key": r2_key,
                    "split_type": split_type,
                    "content_type": content_type,
                    "size": len(item["file_data"])
                }

            # Gather upload tasks as they complete
            processed_count = 0
            for future in as_completed(futures):
                meta = futures[future]
                processed_count += 1
                is_last = (processed_count == total_uploads)
                
                try:
                    uploaded = future.result()
                    if uploaded:
                        task_buffer.append({
                            "taskId": str(uuid.uuid4()),
                            "taskType": "audio",
                            "r2_url": meta["r2_key"],
                            "split": meta["split_type"],
                            "fileName": meta["name"],
                            "name": meta["name"], # mapping to taskName
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
                    flush_audio_batch(is_last=is_last)

        # Ensure progress logger completes
        progress_logger.checkpoint("flushed_final_batch", {"totalCount": count, "failedItems": failed_items, "failedBatches": failed_batches})
        
        result = {
            "success": True,
            "processed": count,
            "datasetId": dataset_id,
            "failedBatches": failed_batches,
            "failedItems": failed_items,
            "message": "Audio dataset processed and registered (Python FastAPI)"
        }
        progress_logger.complete(result)
        return result
        
    except Exception as e:
        logger.error(f"Audio ZIP processing unhandled exception: {e}")
        progress_logger.error("Audio ZIP processing failed", {"error": str(e)})
        progress_logger.complete({"success": False, "error": str(e)})
        raise e

