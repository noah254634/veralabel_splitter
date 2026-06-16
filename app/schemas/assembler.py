from pydantic import BaseModel, Field
from typing import List, Optional

class SubmissionItem(BaseModel):
    submission_id: str = Field(..., alias="submissionId")
    output_url: str = Field(..., alias="outputUrl")
    labeller_trust_score: float = Field(0.5, alias="labellerTrustScore")

    class Config:
        populate_by_name = True

class TaskItem(BaseModel):
    task_id: str = Field(..., alias="taskId")
    task_name: str = Field(..., alias="taskName")
    split: str = Field(..., alias="split")
    input_url: str = Field(..., alias="inputUrl")
    file_name: Optional[str] = Field(None, alias="fileName")
    submissions: List[SubmissionItem] = []

    class Config:
        populate_by_name = True

class AssembleRequest(BaseModel):
    dataset_id: str = Field(..., alias="datasetId")
    dataset_name: str = Field(..., alias="datasetName")
    data_type: str = Field(..., alias="dataType")
    labelling_method: str = Field(..., alias="labellingMethod")
    tasks: List[TaskItem]

    class Config:
        populate_by_name = True

class AssembleResponse(BaseModel):
    success: bool
    message: str
    r2_key: Optional[str] = Field(None, alias="r2Key")
    size_bytes: Optional[int] = Field(None, alias="sizeBytes")
