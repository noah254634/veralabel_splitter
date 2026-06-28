# pyrefly: ignore [missing-import]
from fastapi import APIRouter, Depends, BackgroundTasks, status
# pyrefly: ignore [missing-import]
from fastapi.responses import JSONResponse
from app.schemas.dataset import SplitJobRequest
from app.schemas.assembler import AssembleRequest, AssembleResponse
from app.schemas.consensus import ConsensusRequest, ConsensusResponse
from app.core.security import verify_signature
from app.services.splitter_service import splitter_service
from app.services.assembler_service import assembler_service
from app.services.consensus_service import consensus_service
import logging

logger = logging.getLogger("veralabel-splitter")

router = APIRouter()

@router.post("", response_model=dict)
def trigger_split(
    payload: SplitJobRequest, 
    background_tasks: BackgroundTasks,
    _signature: str = Depends(verify_signature)
):
    """
    HTTP route handler triggered by the backend to initiate splitting.
    Offloads the process to a background task so it responds immediately,
    preventing timeouts/hangs for large datasets.
    """
    logger.info(f"Received split trigger request for dataset {payload.dataset_id}")
    
    background_tasks.add_task(
        splitter_service.process_split_job,
        r2_key=payload.r2_key,
        project_id=payload.project_id,
        dataset_id=payload.dataset_id,
        data_type=payload.data_type,
        download_url=payload.download_url
    )
    
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "success": True,
            "message": "Dataset splitting initiated successfully in background."
        }
    )

@router.post("/assemble", response_model=AssembleResponse)
def assemble_dataset(
    payload: AssembleRequest,
    _signature: str = Depends(verify_signature)
):
    """
    HTTP route handler triggered by the backend to compile a dataset.
    Downloads resources, runs consensus, builds ZIP package, and uploads to R2.
    """
    logger.info(f"Received assemble request for dataset {payload.dataset_id}")
    result = assembler_service.assemble_dataset(payload)
    if not result.get("success"):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "message": result.get("message")
            }
        )
    return AssembleResponse(
        success=True,
        message=result.get("message"),
        r2Key=result.get("r2Key"),
        sizeBytes=result.get("sizeBytes")
    )

@router.post("/consensus/evaluate", response_model=ConsensusResponse)
def evaluate_consensus(
    payload: ConsensusRequest,
    _signature: str = Depends(verify_signature)
):
    """
    HTTP route handler triggered by the backend to evaluate consensus (IoU, agreement, outliers)
    across multiple submissions of a task.
    """
    logger.info(f"Received consensus evaluation request for {len(payload.tasks)} tasks.")
    result = consensus_service.evaluate(payload)
    return result



