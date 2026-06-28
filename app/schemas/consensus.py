from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Optional

class SubmissionItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    submission_id: str = Field(..., alias="submissionId")
    output_url: str = Field(..., alias="outputUrl")
    labeller_trust_score: float = Field(0.5, alias="labellerTrustScore")

class TaskItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    task_id: str = Field(..., alias="taskId")
    submissions: List[SubmissionItem] = []

class ConsensusRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    data_type: str = Field(..., alias="dataType")
    labelling_method: str = Field(..., alias="labellingMethod")
    match_threshold: float = Field(0.5, alias="matchThreshold")
    tasks: List[TaskItem]

class TaskConsensusResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    task_id: str = Field(..., alias="taskId")
    consensus_score: float = Field(..., alias="consensusScore")
    pairwise_iou: Dict[str, float] = Field(..., alias="pairwiseIoU")
    outliers: List[str]

class ConsensusResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    success: bool
    message: str
    results: List[TaskConsensusResult]
