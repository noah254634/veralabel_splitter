import os
from pathlib import Path
from dotenv import load_dotenv

# Load env variables from .env in the project root
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

class Settings:
    INTERNAL_SECRET: str = os.getenv("INTERNAL_SECRET", "")
    
    # R2 bindings/credentials
    R2_ACCESS_KEY: str = os.getenv("R2_ACCESS_KEY", "")
    R2_SECRET_KEY: str = os.getenv("R2_SECRET_KEY", "")
    R2_ENDPOINT: str = os.getenv("R2_ENDPOINT", "")
    R2_BUCKET_NAME: str = os.getenv("R2_BUCKET_NAME", "veralabel-bucket")
    
    # Main backend configuration
    BACKEND_API: str = os.getenv("BACKEND_API", "")
    BACKEND_TOKEN: str = os.getenv("BACKEND_TOKEN", "")
    HANDSHAKE_URL: str = os.getenv("HANDSHAKE_URL", "")
    
    # Audio splitting configurations
    AUDIO_CHUNK_DURATION: float = float(os.getenv("AUDIO_CHUNK_DURATION", "30.0"))


settings = Settings()
