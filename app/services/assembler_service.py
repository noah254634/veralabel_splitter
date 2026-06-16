import io
import json
import logging
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Dict, Any

from app.services.r2_service import r2_service
from app.schemas.assembler import AssembleRequest, TaskItem

logger = logging.getLogger("veralabel-splitter")

def compute_iou(box1: Dict[str, Any], box2: Dict[str, Any]) -> float:
    """
    Computes Intersection over Union (IoU) between two bounding boxes.
    Boxes are in normalized coordinates format: {'x': x, 'y': y, 'w': w, 'h': h}
    """
    x1_1, y1_1 = float(box1.get('x', 0)), float(box1.get('y', 0))
    x2_1, y2_1 = x1_1 + float(box1.get('w', 0)), y1_1 + float(box1.get('h', 0))
    
    x1_2, y1_2 = float(box2.get('x', 0)), float(box2.get('y', 0))
    x2_2, y2_2 = x1_2 + float(box2.get('w', 0)), y1_2 + float(box2.get('h', 0))
    
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    area1 = float(box1.get('w', 0)) * float(box1.get('h', 0))
    area2 = float(box2.get('w', 0)) * float(box2.get('h', 0))
    union_area = area1 + area2 - inter_area
    
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area

def cluster_bounding_boxes(boxes: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    Groups bounding boxes of the same label that overlap with average IoU >= 0.5.
    """
    clusters = []
    for item in boxes:
        placed = False
        for cluster in clusters:
            # check average IoU with all boxes in the cluster
            avg_iou = sum(compute_iou(item, c_item) for c_item in cluster) / len(cluster)
            if avg_iou >= 0.5:
                cluster.append(item)
                placed = True
                break
        if not placed:
            clusters.append([item])
    return clusters

def run_bounding_box_consensus(submissions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merges bounding boxes across multiple submissions using IoU clustering.
    Averages coordinates of matching boxes, weighted by the labellers' trust scores.
    """
    # Flatten all bounding boxes from all submissions, tracking trust score
    all_boxes = []
    for sub in submissions:
        trust = max(float(sub.get('trust', 0.5)), 0.1)
        sub_boxes = sub.get('annotation', {}).get('boundingBoxes')
        if sub_boxes and isinstance(sub_boxes, list):
            for box in sub_boxes:
                if isinstance(box, dict):
                    all_boxes.append({
                        'x': box.get('x'),
                        'y': box.get('y'),
                        'w': box.get('w'),
                        'h': box.get('h'),
                        'label': str(box.get('label', '')).strip(),
                        'trust': trust
                    })

    if not all_boxes:
        return []

    # Group boxes by label
    by_label = {}
    for box in all_boxes:
        label = box['label']
        if label not in by_label:
            by_label[label] = []
        by_label[label].append(box)

    consensus_boxes = []
    for label, label_boxes in by_label.items():
        clusters = cluster_bounding_boxes(label_boxes)
        for cluster in clusters:
            total_trust = sum(box['trust'] for box in cluster)
            avg_x = sum(box['x'] * box['trust'] for box in cluster) / total_trust
            avg_y = sum(box['y'] * box['trust'] for box in cluster) / total_trust
            avg_w = sum(box['w'] * box['trust'] for box in cluster) / total_trust
            avg_h = sum(box['h'] * box['trust'] for box in cluster) / total_trust
            
            consensus_boxes.append({
                'x': round(avg_x, 2),
                'y': round(avg_y, 2),
                'w': round(avg_w, 2),
                'h': round(avg_h, 2),
                'label': label
            })

    return consensus_boxes

def run_classification_consensus(submissions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Resolves classification label via majority vote and trust score tie-breaker.
    """
    votes = {}
    trust_sums = {}
    for sub in submissions:
        trust = max(float(sub.get('trust', 0.5)), 0.1)
        label = sub.get('annotation', {}).get('classificationLabel')
        if label:
            label_str = str(label).strip()
            votes[label_str] = votes.get(label_str, 0) + 1
            trust_sums[label_str] = trust_sums.get(label_str, 0.0) + trust

    if not votes:
        return {'classificationLabel': None, 'agreementRate': 0.0}

    max_votes = max(votes.values())
    candidates = [label for label, count in votes.items() if count == max_votes]
    
    if len(candidates) == 1:
        consensus_label = candidates[0]
    else:
        # Tie-breaker by trust sum
        consensus_label = max(candidates, key=lambda l: trust_sums.get(l, 0.0))

    agreement = votes[consensus_label] / len(submissions) if len(submissions) > 0 else 0.0
    return {
        'classificationLabel': consensus_label,
        'agreementRate': round(agreement, 2)
    }

def run_transcription_consensus(submissions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Resolves transcription selection by picking the text of the highest-trust labeller.
    """
    valid_subs = []
    for sub in submissions:
        trust = max(float(sub.get('trust', 0.5)), 0.1)
        text = sub.get('annotation', {}).get('transcription')
        if text:
            valid_subs.append({
                'transcription': str(text).strip(),
                'trust': trust
            })

    if not valid_subs:
        return {'transcription': None}

    # Sort by trust desc, then by transcription length desc (longer transcriptions are usually more detailed)
    valid_subs.sort(key=lambda s: (s['trust'], len(s['transcription'])), reverse=True)
    return {
        'transcription': valid_subs[0]['transcription']
    }

def run_rlhf_consensus(submissions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Combines RLHF preference choices, ratings, rationales, and rubrics.
    """
    if not submissions:
        return {}

    # 1. Preference
    pref_votes = {}
    pref_trust = {}
    for sub in submissions:
        trust = max(float(sub.get('trust', 0.5)), 0.1)
        pref = sub.get('annotation', {}).get('preference')
        if pref:
            pref_str = str(pref).strip()
            pref_votes[pref_str] = pref_votes.get(pref_str, 0) + 1
            pref_trust[pref_str] = pref_trust.get(pref_str, 0.0) + trust

    consensus_pref = None
    if pref_votes:
        max_votes = max(pref_votes.values())
        candidates = [p for p, c in pref_votes.items() if c == max_votes]
        if len(candidates) == 1:
            consensus_pref = candidates[0]
        else:
            consensus_pref = max(candidates, key=lambda c: pref_trust.get(c, 0.0))

    # 2. Ratings (Weighted averages)
    ratings_grouped = {}
    for sub in submissions:
        trust = max(float(sub.get('trust', 0.5)), 0.1)
        ratings = sub.get('annotation', {}).get('ratings') or {}
        for key, val in ratings.items():
            if val is not None:
                if key not in ratings_grouped:
                    ratings_grouped[key] = []
                ratings_grouped[key].append((float(val), trust))

    consensus_ratings = {}
    for key, val_trusts in ratings_grouped.items():
        total_trust = sum(t for _, t in val_trusts)
        weighted_sum = sum(v * t for v, t in val_trusts)
        consensus_ratings[key] = round(weighted_sum / total_trust, 2) if total_trust > 0 else 0.0

    # 3. Rationale (Highest trust labeller)
    sorted_by_trust = sorted(submissions, key=lambda s: max(float(s.get('trust', 0.5)), 0.1), reverse=True)
    consensus_rationale = sorted_by_trust[0].get('annotation', {}).get('rationale') or ""

    # 4. Rubrics (Weighted majority vote)
    rubric_tags = set()
    for sub in submissions:
        rubrics = sub.get('annotation', {}).get('rubrics') or {}
        rubric_tags.update(rubrics.keys())

    consensus_rubrics = {}
    for tag in rubric_tags:
        votes_true = 0.0
        total_trust = 0.0
        for sub in submissions:
            trust = max(float(sub.get('trust', 0.5)), 0.1)
            rubrics = sub.get('annotation', {}).get('rubrics') or {}
            val = rubrics.get(tag)
            total_trust += trust
            if val is True or val == "true" or val == 1:
                votes_true += trust
        consensus_rubrics[tag] = votes_true >= (total_trust * 0.5) if total_trust > 0 else False

    return {
        'preference': consensus_pref,
        'ratings': consensus_ratings,
        'rationale': consensus_rationale,
        'rubrics': consensus_rubrics
    }

def download_file_bytes(url: str) -> bytes:
    """Download helper utility with a 15-second timeout."""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'VeraLabel-Splitter-Worker'})
        with urllib.request.urlopen(req, timeout=15) as response:
            return response.read()
    except Exception as e:
        logger.error(f"Download failed for URL: {url[:100]}... Error: {e}")
        raise e

class AssemblerService:
    def assemble_dataset(self, payload: AssembleRequest) -> Dict[str, Any]:
        """
        Gathers inputs and annotations, computes consensus, packages into a ZIP, and uploads to R2.
        """
        logger.info(f"Assembling dataset: {payload.dataset_id} ({payload.dataset_name}) Type={payload.data_type}")
        
        is_media = str(payload.data_type).lower().strip() in ('image', 'audio', 'video')
        
        # Parallel downloads of task inputs and submission annotations
        download_queue = []
        
        # We store references to map downloaded bytes back to tasks/submissions
        # key: url -> payload descriptor
        url_mapping = {}
        
        for task in payload.tasks:
            # Enqueue task input file download
            if task.input_url not in url_mapping:
                url_mapping[task.input_url] = {'type': 'input', 'task_id': task.task_id}
                download_queue.append(task.input_url)
                
            # Enqueue submission files downloads
            for sub in task.submissions:
                if sub.output_url not in url_mapping:
                    url_mapping[sub.output_url] = {
                        'type': 'submission',
                        'task_id': task.task_id,
                        'submission_id': sub.submission_id,
                        'trust': sub.labeller_trust_score
                    }
                    download_queue.append(sub.output_url)

        logger.info(f"Enqueued {len(download_queue)} downloads from R2 presigned URLs")
        
        # Download concurrently
        downloaded_bytes = {}
        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = {executor.submit(download_file_bytes, url): url for url in download_queue}
            for future in as_completed(futures):
                url = futures[future]
                try:
                    downloaded_bytes[url] = future.result()
                except Exception as e:
                    logger.error(f"Failed to fetch resource: {url[:80]} - Error: {e}")
                    # Return failure immediately if we cannot get critical training data
                    return {
                        'success': False,
                        'message': f"Failed to download resource: {str(e)}"
                    }

        logger.info("Downloads completed successfully. Merging annotations and running consensus algorithms...")

        # Process each task and compute consensus
        compiled_records = []
        zip_assets = [] # list of tuples: (filename, bytes)
        
        for task in payload.tasks:
            # 1. Parse original input content if text/rlhf, or reference if media
            input_bytes = downloaded_bytes.get(task.input_url)
            input_data = None
            
            if not input_bytes:
                logger.warning(f"No bytes downloaded for task input: {task.task_id}")
                continue
                
            if is_media:
                # Store asset bytes for zipping later
                # Determine asset path inside ZIP
                ext = task.input_url.split('?')[0].split('.')[-1].lower()
                # fallback extension if we cannot parse it
                if len(ext) > 4 or not ext.isalnum():
                    ext = "jpg" if str(payload.data_type).lower().strip() == "image" else "wav"
                
                # Clean up fileName
                asset_name = task.file_name or f"task_{task.task_id}.{ext}"
                zip_assets.append((f"assets/{asset_name}", input_bytes))
                input_data = f"assets/{asset_name}"
            else:
                # Parse as text or JSON
                try:
                    content_str = input_bytes.decode('utf-8', errors='ignore')
                    try:
                        input_data = json.loads(content_str)
                    except json.JSONDecodeError:
                        input_data = content_str # plain text
                except Exception as parse_err:
                    logger.warning(f"Could not parse task input data as string: {parse_err}")
                    input_data = None

            # 2. Parse and assemble submissions
            task_submissions = []
            for sub in task.submissions:
                sub_bytes = downloaded_bytes.get(sub.output_url)
                if not sub_bytes:
                    continue
                try:
                    annotation = json.loads(sub_bytes.decode('utf-8', errors='ignore'))
                    task_submissions.append({
                        'submissionId': sub.submission_id,
                        'trust': sub.labeller_trust_score,
                        'annotation': annotation
                    })
                except Exception as parse_err:
                    logger.warning(f"Could not parse submission annotation JSON: {parse_err}")

            # 3. Apply consensus based on labelling method
            consensus_result = {}
            method = str(payload.labelling_method).lower().strip()
            
            if method == "classification":
                consensus_result = run_classification_consensus(task_submissions)
            elif method == "transcription":
                consensus_result = run_transcription_consensus(task_submissions)
            elif method == "rlhf":
                consensus_result = run_rlhf_consensus(task_submissions)
            elif method == "annotation" and str(payload.data_type).lower().strip() == "image":
                consensus_result = {'boundingBoxes': run_bounding_box_consensus(task_submissions)}
            else:
                # Fallback: simple merge/list of annotations
                consensus_result = {
                    'rawSubmissions': [s.get('annotation') for s in task_submissions]
                }

            # 4. Save record
            record = {
                'taskId': task.task_id,
                'taskName': task.task_name,
                'split': task.split,
            }
            
            if is_media:
                record['file_name'] = input_data
            else:
                record['input'] = input_data
                
            record['consensus'] = consensus_result
            compiled_records.append(record)

        # Build ZIP Archive in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            # Write annotations JSON
            annotations_content = json.dumps({
                'info': {
                    'datasetId': payload.dataset_id,
                    'datasetName': payload.dataset_name,
                    'dataType': payload.data_type,
                    'labellingMethod': payload.labelling_method,
                    'compiledAt': datetime.utcnow().isoformat() + "Z"
                },
                'records': compiled_records
            }, indent=2)
            
            zip_file.writestr("dataset.json", annotations_content)
            
            # Write asset files if media
            if is_media:
                for path, bytes_data in zip_assets:
                    zip_file.writestr(path, bytes_data)

        # Get finished bytes
        zip_bytes = zip_buffer.getvalue()
        zip_buffer.close()
        
        # Upload compiled ZIP back to Cloudflare R2
        r2_key = f"compiled_datasets/{payload.dataset_id}.zip"
        content_type = "application/zip"
        
        logger.info(f"Uploading compiled ZIP ({len(zip_bytes)} bytes) to R2 at key: {r2_key}")
        upload_ok = r2_service.upload_file(r2_key, zip_bytes, content_type, "compiled")
        
        if not upload_ok:
            return {
                'success': False,
                'message': "Failed to upload compiled ZIP file to Cloudflare R2"
            }
            
        return {
            'success': True,
            'message': f"Dataset compiled successfully. Consensus completed across {len(payload.tasks)} tasks.",
            'r2Key': r2_key,
            'sizeBytes': len(zip_bytes)
        }

assembler_service = AssemblerService()
