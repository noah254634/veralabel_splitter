import logging
from app.services.r2_service import r2_service
from app.services.audio_splitter import process_audio_zip
from app.services.media_splitter import process_media_zip
from app.services.text_splitter import process_text_file
from app.services.rlhf_splitter import process_rlhf_file

logger = logging.getLogger("veralabel-splitter")

class SplitterService:
    def process_split_job(self, r2_key: str, project_id: str, dataset_id: str, data_type: str, download_url: str = None) -> dict:
        """
        Main orchestrator that downloads the source file from R2 and routes it to the specific splitter.
        """
        logger.info(f"Processing split job: dataType={data_type}, datasetId={dataset_id}, r2Key={r2_key}")
        

        try:
            file_bytes = r2_service.download_file(r2_key, download_url)
        except Exception as e:
            logger.error(f"Failed to fetch source archive from R2: {e}")
            return {
                "success": False,
                "error": f"Failed to download source archive from R2: {str(e)}",
                "status": 400
            }
            
        # Route based on data type
        normalized_type = str(data_type).lower().strip()
        
        try:
            if normalized_type == "audio":
                result = process_audio_zip(file_bytes, project_id, dataset_id)
                return {**result, "status": 200}
            elif normalized_type == "media":
                result = process_media_zip(file_bytes, project_id, dataset_id)
                return {**result, "status": 200}
            elif normalized_type == "text":
                result = process_text_file(file_bytes, project_id, dataset_id)
                return {**result, "status": 200}
            elif normalized_type == "rlhf":
                result = process_rlhf_file(file_bytes, project_id, dataset_id)
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

splitter_service = SplitterService()

