import logging
import os
import tempfile
from app.services.r2_service import r2_service
from app.services.audio_splitter import process_audio_zip
from app.services.media_splitter import process_media_zip
from app.services.text_splitter import process_text_file
from app.services.rlhf_splitter import process_rlhf_file

logger = logging.getLogger("veralabel-splitter")

class SplitterService:
    def process_split_job(self, r2_key: str, project_id: str, dataset_id: str, data_type: str, download_url: str = None) -> dict:
        """
        Main orchestrator that downloads the source file from R2 to disk and routes the disk path to the specific splitter.
        """
        logger.info(f"Processing split job: dataType={data_type}, datasetId={dataset_id}, r2Key={r2_key}")
        
        # Create a secure temporary file to buffer the archive to disk
        temp_dir = tempfile.gettempdir()
        temp_fd, temp_path = tempfile.mkstemp(dir=temp_dir)
        os.close(temp_fd) # Close file descriptor; we'll write/read via standard file paths

        try:
            r2_service.download_file_to_disk(r2_key, temp_path, download_url)
        except Exception as e:
            logger.error(f"Failed to fetch source archive from R2: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return {
                "success": False,
                "error": f"Failed to download source archive from R2: {str(e)}",
                "status": 400
            }
            
        # Route based on data type
        normalized_type = str(data_type).lower().strip()
        
        try:
            if normalized_type == "audio":
                result = process_audio_zip(temp_path, project_id, dataset_id)
                return {**result, "status": 200}
            elif normalized_type == "media":
                result = process_media_zip(temp_path, project_id, dataset_id)
                return {**result, "status": 200}
            elif normalized_type == "text":
                result = process_text_file(temp_path, project_id, dataset_id)
                return {**result, "status": 200}
            elif normalized_type == "rlhf":
                result = process_rlhf_file(temp_path, project_id, dataset_id)
                return {**result, "status": 200}
            else:
                logger.error(f"Unsupported data type: {normalized_type}")
                return {
                    "success": False,
                    "error": f"Unsupported split type: {normalized_type}",
                    "status": 400
                }
        except Exception as e:
            logger.error(f"Error during splitting: {e}")
            return {
                "success": False,
                "error": str(e),
                "status": 500
            }
        finally:
            # Always clean up the temporary file from local storage
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    logger.info(f"Successfully cleaned up temporary archive path: {temp_path}")
            except Exception as cleanup_err:
                logger.error(f"Failed to remove temporary file {temp_path}: {cleanup_err}")

splitter_service = SplitterService()

