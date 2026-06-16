from pydantic import BaseModel, Field
from typing import Optional

class SplitJobRequest(BaseModel):
    r2_key: str = Field(..., alias="r2Key")
    project_id: str = Field(..., alias="projectId")
    dataset_id: str = Field(..., alias="datasetId")
    data_type: str = Field(..., alias="dataType")
    download_url: Optional[str] = Field(None, alias="downloadUrl")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "r2Key": "datasets/6a158ea9dd11827240bf7af9/d44181a9-907b-43d8-af04-1d62c6c584b0",
                "projectId": "6a16b8bd5bf13861ef4d178c",
                "datasetId": "6a2046122b6b4eacab311b07",
                "dataType": "audio",
                "downloadUrl": "https://presigned-url-from-r2"
            }
        }
