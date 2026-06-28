import urllib.request
import boto3
from botocore.client import Config
from app.core.config import settings
import logging

logger = logging.getLogger("veralabel-splitter")

class R2Service:
    def __init__(self):
        self.bucket_name = settings.R2_BUCKET_NAME
        self.s3_client = None
        
        # Only initialize boto3 client if endpoint and credentials are provided
        if settings.R2_ENDPOINT and settings.R2_ACCESS_KEY:
            self.s3_client = boto3.client(
                "s3",
                endpoint_url=settings.R2_ENDPOINT,
                aws_access_key_id=settings.R2_ACCESS_KEY,
                aws_secret_access_key=settings.R2_SECRET_KEY,
                config=Config(signature_version="s3v4"),
                region_name="auto"
            )
        else:
            logger.warning("R2 S3 Client not initialized: missing R2 credentials in environment")

    def download_file(self, r2_key: str, download_url: str = None) -> bytes:
        """
        Download a file either from a presigned GET URL (if provided) 
        or directly from the R2 bucket using S3 API.
        """
        if download_url:
            logger.info(f"Downloading file from presigned URL: {r2_key}")
            try:
                with urllib.request.urlopen(download_url, timeout=30) as response:
                    return response.read()
            except Exception as e:
                logger.error(f"Failed to fetch file from download_url: {e}")
                # Fall back to direct R2 get if possible
                if not self.s3_client:
                    raise e
        
        if not self.s3_client:
            raise ValueError("S3 client not initialized and no downloadUrl provided")
            
        logger.info(f"Downloading key {r2_key} directly from R2 bucket {self.bucket_name}")
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=r2_key)
            return response['Body'].read()
        except Exception as e:
            logger.error(f"Failed to download key {r2_key} from R2: {e}")
            raise e

    def download_file_to_disk(self, r2_key: str, dest_path: str, download_url: str = None) -> None:
        """
        Download a file to disk path chunk-by-chunk using a presigned URL or directly via s3 client.
        """
        if download_url:
            logger.info(f"Downloading key {r2_key} from presigned URL to disk: {dest_path}")
            try:
                with urllib.request.urlopen(download_url, timeout=60) as response:
                    with open(dest_path, 'wb') as out_file:
                        chunk_size = 1024 * 1024 # 1MB chunks
                        while True:
                            chunk = response.read(chunk_size)
                            if not chunk:
                                break
                            out_file.write(chunk)
                return
            except Exception as e:
                logger.error(f"Failed to download file from download_url to disk: {e}")
                if not self.s3_client:
                    raise e

        if not self.s3_client:
            raise ValueError("S3 client not initialized and no downloadUrl provided")

        logger.info(f"Downloading key {r2_key} directly from R2 bucket {self.bucket_name} to disk {dest_path}")
        try:
            self.s3_client.download_file(self.bucket_name, r2_key, dest_path)
        except Exception as e:
            logger.error(f"Failed to download key {r2_key} directly to disk: {e}")
            raise e

    def upload_file(self, r2_key: str, body: bytes, content_type: str, split_type: str) -> bool:
        """
        Upload extracted file directly to R2 bucket with appropriate metadata.
        """
        if not self.s3_client:
            logger.error("Cannot upload file: S3 client not initialized")
            return False
            
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=r2_key,
                Body=body,
                ContentType=content_type,
                Metadata={
                    "vera-split": split_type
                }
            )
            return True
        except Exception as e:
            logger.error(f"Failed to upload key {r2_key} to R2: {e}")
            return False

r2_service = R2Service()
