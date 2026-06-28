import io
import json
import logging
import urllib.request
import numpy as np
from PIL import Image, ImageDraw
from typing import List, Dict, Any
from scipy.optimize import linear_sum_assignment
from app.schemas.consensus import ConsensusRequest, ConsensusResponse, TaskConsensusResult

logger = logging.getLogger("veralabel-splitter")

def download_file_bytes(url: str) -> bytes:
    """Download helper utility with a 15-second timeout."""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'VeraLabel-Splitter-Worker'})
        with urllib.request.urlopen(req, timeout=15) as response:
            return response.read()
    except Exception as e:
        logger.error(f"Download failed for URL: {url[:100]}... Error: {e}")
        raise e

def get_box_iou(box1: Dict[str, Any], box2: Dict[str, Any]) -> float:
    """
    Computes Intersection over Union (IoU) between two bounding boxes.
    Boxes are in normalized coordinates format: {'x': x, 'y': y, 'w': w, 'h': h}
    """
    x1_1, y1_1 = float(box1.get('x', 0)), float(box1.get('y', 0))
    w1_1, h1_1 = float(box1.get('w', 0)), float(box1.get('h', 0))
    x2_1, y2_1 = x1_1 + w1_1, y1_1 + h1_1
    
    x1_2, y1_2 = float(box2.get('x', 0)), float(box2.get('y', 0))
    w2_2, h2_2 = float(box2.get('w', 0)), float(box2.get('h', 0))
    x2_2, y2_2 = x1_2 + w2_2, y1_2 + h2_2
    
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    area1 = w1_1 * h1_1
    area2 = w2_2 * h2_2
    union_area = area1 + area2 - inter_area
    
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area

def get_polygon_iou(poly1: List[List[float]], poly2: List[List[float]]) -> float:
    """
    Computes IoU between two polygons by rasterizing them to binary masks using Pillow.
    Uses bounding box cropping and scaling to avoid heavy memory footprints.
    """
    xs1 = [float(pt[0]) for pt in poly1]
    ys1 = [float(pt[1]) for pt in poly1]
    xs2 = [float(pt[0]) for pt in poly2]
    ys2 = [float(pt[1]) for pt in poly2]
    
    if not xs1 or not ys1 or not xs2 or not ys2:
        return 0.0
        
    min_x = min(min(xs1), min(xs2))
    max_x = max(max(xs1), max(xs2))
    min_y = min(min(ys1), min(ys2))
    max_y = max(max(ys1), max(ys2))
    
    bbox_w = max_x - min_x
    bbox_h = max_y - min_y
    
    if bbox_w <= 0.0 or bbox_h <= 0.0:
        return 0.0
        
    # Scale canvas to always target max_dim for high-precision rasterization.
    max_dim = 1000.0
    scale = max_dim / max(bbox_w, bbox_h)
        
    scaled_w = int(bbox_w * scale) + 2
    scaled_h = int(bbox_h * scale) + 2
    
    img1 = Image.new("1", (scaled_w, scaled_h), 0)
    img2 = Image.new("1", (scaled_w, scaled_h), 0)
    
    draw1 = ImageDraw.Draw(img1)
    draw2 = ImageDraw.Draw(img2)
    
    scaled_poly1 = [((pt[0] - min_x) * scale, (pt[1] - min_y) * scale) for pt in poly1]
    scaled_poly2 = [((pt[0] - min_x) * scale, (pt[1] - min_y) * scale) for pt in poly2]
    
    draw1.polygon(scaled_poly1, fill=1)
    draw2.polygon(scaled_poly2, fill=1)
    
    mask1 = np.array(img1)
    mask2 = np.array(img2)
    
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    return float(intersection) / float(union)

def levenshtein_ratio(s1: str, s2: str) -> float:
    """Normalized string similarity based on Levenshtein distance."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + 1)
                
    return 1.0 - (dp[m][n] / max(m, n))

def get_rlhf_similarity(anno1: Dict[str, Any], anno2: Dict[str, Any]) -> float:
    """Computes similarity score between two RLHF annotations."""
    scores = []
    
    # 1. Preference choice (1.0 if identical, 0.0 otherwise)
    pref1 = anno1.get('preference')
    pref2 = anno2.get('preference')
    if pref1 is not None or pref2 is not None:
        scores.append(1.0 if pref1 == pref2 else 0.0)
        
    # 2. Ratings (absolute difference divided by scale range)
    ratings1 = anno1.get('ratings', {})
    ratings2 = anno2.get('ratings', {})
    all_keys = set(ratings1.keys()) | set(ratings2.keys())
    
    for key in all_keys:
        val1 = ratings1.get(key)
        val2 = ratings2.get(key)
        if val1 is not None and val2 is not None:
            # Assume 1-5 scale range (max diff = 4.0)
            diff = abs(float(val1) - float(val2))
            scores.append(max(0.0, 1.0 - (diff / 4.0)))
            
    # 3. Rationale text similarity (using Levenshtein similarity)
    rat1 = anno1.get('rationale', '')
    rat2 = anno2.get('rationale', '')
    if rat1 or rat2:
        scores.append(levenshtein_ratio(rat1, rat2))
        
    return sum(scores) / len(scores) if scores else 0.0

def match_shapes(shapes1: List[Dict[str, Any]], shapes2: List[Dict[str, Any]], shape_type: str, threshold: float) -> float:
    """
    Solves optimal assignment using Hungarian algorithm (scipy linear_sum_assignment).
    Groups shapes by class label first, then computes pairwise IoU.
    """
    if not shapes1 and not shapes2:
        return 1.0
    if not shapes1 or not shapes2:
        return 0.0
        
    # Group shapes by class label
    grouped1 = {}
    grouped2 = {}
    all_labels = set()
    
    for s in shapes1:
        lbl = str(s.get('label', '')).strip()
        grouped1.setdefault(lbl, []).append(s)
        all_labels.add(lbl)
        
    for s in shapes2:
        lbl = str(s.get('label', '')).strip()
        grouped2.setdefault(lbl, []).append(s)
        all_labels.add(lbl)
        
    total_matched_iou = 0.0
    total_shapes = 0
    
    for label in all_labels:
        list1 = grouped1.get(label, [])
        list2 = grouped2.get(label, [])
        
        m, n = len(list1), len(list2)
        if m == 0 or n == 0:
            total_shapes += (m + n)
            continue
            
        cost_matrix = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                if shape_type == 'polygons':
                    iou = get_polygon_iou(list1[i].get('polygon', []), list2[j].get('polygon', []))
                else:
                    iou = get_box_iou(list1[i], list2[j])
                cost_matrix[i, j] = 1.0 - iou
                
        # Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched_rows = set()
        matched_cols = set()
        
        for r, c in zip(row_ind, col_ind):
            iou = 1.0 - cost_matrix[r, c]
            if iou >= threshold:
                total_matched_iou += iou
                matched_rows.add(r)
                matched_cols.add(c)
                
        unmatched_count = (m - len(matched_rows)) + (n - len(matched_cols))
        total_shapes += (len(matched_rows) + unmatched_count)
        
    if total_shapes == 0:
        return 1.0
        
    return total_matched_iou / total_shapes

class ConsensusService:
    def evaluate(self, payload: ConsensusRequest) -> Dict[str, Any]:
        """
        Evaluate consensus metrics for all tasks in the request.
        """
        logger.info(f"Consensus evaluation triggered for {len(payload.tasks)} tasks. Data type={payload.data_type}")
        
        results = []
        for task in payload.tasks:
            # 1. Download all submission JSON payloads
            submissions_data = []
            for sub in task.submissions:
                try:
                    bytes_data = download_file_bytes(sub.output_url)
                    annotation = json.loads(bytes_data.decode('utf-8', errors='ignore'))
                    submissions_data.append({
                        'submissionId': sub.submission_id,
                        'trust': sub.labeller_trust_score,
                        'annotation': annotation
                    })
                except Exception as e:
                    logger.error(f"Error downloading/parsing submission {sub.submission_id}: {e}")
                    
            if not submissions_data:
                results.append(TaskConsensusResult(
                    taskId=task.task_id,
                    consensusScore=1.0,
                    pairwiseIoU={},
                    outliers=[]
                ))
                continue
                
            num_subs = len(submissions_data)
            
            # Handle 1 Labeller case
            if num_subs == 1:
                results.append(TaskConsensusResult(
                    taskId=task.task_id,
                    consensusScore=1.0,
                    pairwiseIoU={},
                    outliers=[]
                ))
                continue
                
            # Compute pairwise similarities
            pairwise_scores = {}
            logger.info(f"Computing pairwise scores for task {task.task_id} with {num_subs} submissions.")
            for i in range(num_subs):
                for j in range(i + 1, num_subs):
                    sub1 = submissions_data[i]
                    sub2 = submissions_data[j]
                    
                    sim = 0.0
                    method = str(payload.labelling_method).lower().strip()
                    
                    if method == "classification":
                        lbl1 = str(sub1['annotation'].get('classificationLabel', '')).strip()
                        lbl2 = str(sub2['annotation'].get('classificationLabel', '')).strip()
                        sim = 1.0 if lbl1 == lbl2 else 0.0
                        logger.info(f"Pairwise classification similarity between {sub1['submissionId']} and {sub2['submissionId']}: {sim}")
                        
                    elif method == "transcription":
                        txt1 = str(sub1['annotation'].get('transcription', '')).strip()
                        txt2 = str(sub2['annotation'].get('transcription', '')).strip()
                        sim = levenshtein_ratio(txt1, txt2)
                        logger.info(f"Pairwise transcription similarity between {sub1['submissionId']} and {sub2['submissionId']}: {sim:.4f}")
                        
                    elif method == "rlhf":
                        sim = get_rlhf_similarity(sub1['annotation'], sub2['annotation'])
                        logger.info(f"Pairwise RLHF similarity between {sub1['submissionId']} and {sub2['submissionId']}: {sim:.4f}")
                        
                    elif method == "annotation":
                        has_boxes = any(s['annotation'].get('boundingBoxes') for s in (sub1, sub2))
                        has_polys = any(s['annotation'].get('polygons') for s in (sub1, sub2))
                        
                        if has_polys:
                            sim = match_shapes(
                                sub1['annotation'].get('polygons', []),
                                sub2['annotation'].get('polygons', []),
                                'polygons',
                                payload.match_threshold
                            )
                            logger.info(f"Pairwise polygon annotation IoU between {sub1['submissionId']} and {sub2['submissionId']}: {sim:.4f}")
                        elif has_boxes:
                            sim = match_shapes(
                                sub1['annotation'].get('boundingBoxes', []),
                                sub2['annotation'].get('boundingBoxes', []),
                                'boundingBoxes',
                                payload.match_threshold
                            )
                            logger.info(f"Pairwise bounding box annotation IoU between {sub1['submissionId']} and {sub2['submissionId']}: {sim:.4f}")
                        else:
                            sim = 1.0  # Both empty annotations
                            logger.info(f"Pairwise annotation similarity (both empty) between {sub1['submissionId']} and {sub2['submissionId']}: {sim}")
                            
                    key = f"{sub1['submissionId']}_{sub2['submissionId']}"
                    pairwise_scores[key] = round(sim, 4)
                    
            # Consensus score is the average of all pairwise similarities
            all_scores = list(pairwise_scores.values())
            consensus_score = round(sum(all_scores) / len(all_scores), 4) if all_scores else 1.0
            logger.info(f"Task {task.task_id} consensus score (average pairwise similarity): {consensus_score}")
            
            # Compute individual score for outlier detection (mean score per labeller with others)
            labeller_scores = {}
            for sub in submissions_data:
                sub_id = sub['submissionId']
                scores = []
                for key, val in pairwise_scores.items():
                    if sub_id in key:
                        scores.append(val)
                labeller_scores[sub_id] = np.mean(scores) if scores else 1.0
                logger.info(f"Labeller average agreement score for {sub_id}: {labeller_scores[sub_id]:.4f}")
                
            # Perform MAD (Median Absolute Deviation) outlier detection (K >= 3)
            outliers = []
            if num_subs >= 3:
                X = list(labeller_scores.values())
                med = np.median(X)
                mad = np.median([abs(x - med) for x in X])
                
                # Scale factors to prevent division by zero or overly sensitive flags on small dev
                mad_val = max(mad, 0.05)
                logger.info(f"MAD Outlier detection for task {task.task_id}: median={med:.4f}, MAD={mad:.4f}, effective_MAD={mad_val:.4f}")
                
                for sub_id, score in labeller_scores.items():
                    # Modified Z-score calculation
                    modified_z = (0.6745 * (score - med)) / mad_val
                    logger.info(f"Labeller {sub_id} Z-Score: {modified_z:.4f} (Agreement score={score:.4f})")
                    # Flag as outlier if performance is significantly below median
                    if modified_z < -3.5:
                        outliers.append(sub_id)
                        logger.warn(f"Flagged outlier: Labeller submission {sub_id} (Z-Score={modified_z:.4f} < -3.5)")
                        
            results.append(TaskConsensusResult(
                taskId=task.task_id,
                consensusScore=consensus_score,
                pairwiseIoU=pairwise_scores,
                outliers=outliers
            ))
            
        return {
            "success": True,
            "message": f"Consensus completed across {len(payload.tasks)} tasks.",
            "results": results
        }

consensus_service = ConsensusService()
